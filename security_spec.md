# SynthForge AI — Firestore Zero-Trust Security Specification (`security_spec.md`)

## 1. Data Invariants & Master Source of Truth

1. **Authenticated & Email-Verified Identity**: Every write operation across all 12 collections requires `request.auth != null && request.auth.token.email_verified == true`.
2. **Strict Ownership Isolation (`ownerId` / `uid`)**:
   - Every `/users/{userId}` document can only be created, read, or updated by `request.auth.uid == userId`. No blanket `isSignedIn()` reads are permitted.
   - Every artifact document (`datasets`, `dataset_schemas`, `profiling_reports`, `generation_configs`, `generation_runs`, `synthetic_datasets`, `quality_assessments`, `privacy_assessments`, `ml_experiments`, `model_metrics`, `export_history`) MUST carry `ownerId == request.auth.uid`.
3. **Path Variable Hardening**: Every single-document operation (`get`, `create`, `update`, `delete`) validates its path ID variable with `isValidId(id)` (`id is string && id.size() <= 128 && id.matches('^[a-zA-Z0-9_\\-]+$')`).
4. **Anti-Update-Gap & Immutable Fields**:
   - All `create` and `update` blocks invoke the entity's `isValid[Entity](incoming())` blueprint helper first.
   - `ownerId` and `createdAt` are strictly immutable across all updates (`incoming().ownerId == existing().ownerId && incoming().createdAt == existing().createdAt`).
   - Updates use action-scoped `incoming().diff(existing()).affectedKeys().hasOnly(...)` allowlists.
5. **Query Enforcer (`allow list`)**: Every `allow list` rule enforces `resource.data.ownerId == request.auth.uid` without `get()` or `exists()` lookups inside list blocks.

---

## 2. The "Dirty Dozen" Adversarial Payloads

1. **Payload 01 — Unverified Email Write**: Authenticated user with `email_verified: false` attempting to create `/datasets/ds_01`. -> `PERMISSION_DENIED`
2. **Payload 02 — Cross-Tenant Owner Spoofing**: User `uid_A` creating `/datasets/ds_01` with `ownerId: "uid_B"`. -> `PERMISSION_DENIED`
3. **Payload 03 — Shadow Field Injection on Create**: Creating `/datasets/ds_01` with an undeclared field `isAdmin: true`. -> `PERMISSION_DENIED`
4. **Payload 04 — Shadow Field Injection on Update**: Updating `/datasets/ds_01` with `affectedKeys` containing `isVerified: true`. -> `PERMISSION_DENIED`
5. **Payload 05 — Immutable `ownerId` Tampering**: Updating `/datasets/ds_01` to transfer `ownerId` to another user. -> `PERMISSION_DENIED`
6. **Payload 06 — Immutable `createdAt` Tampering**: Updating `/generation_runs/run_01` with a modified `createdAt` timestamp. -> `PERMISSION_DENIED`
7. **Payload 07 — Path ID Poisoning**: Creating a document with an invalid ID containing spaces or special characters (`/datasets/bad id$!`). -> `PERMISSION_DENIED`
8. **Payload 08 — Denial-of-Wallet String Overflow**: Creating `/datasets/ds_01` with `name` exceeding `maxLength: 160`. -> `PERMISSION_DENIED`
9. **Payload 09 — Blanket List Scraping**: Querying `/generation_runs` without filtering `where('ownerId', '==', request.auth.uid)`. -> `PERMISSION_DENIED`
10. **Payload 10 — Unauthorized Profile Read (PII Isolation)**: User `uid_A` attempting `get(/users/uid_B)`. -> `PERMISSION_DENIED`
11. **Payload 11 — Role Escalation in User Profile**: User attempting to create or update `/users/uid_A` with `role: "superadmin"` (outside allowed enum `['engineer', 'researcher', 'auditor']`). -> `PERMISSION_DENIED`
12. **Payload 12 — Value Poisoning on Update**: Updating `/datasets/ds_01` whitelisted field `transformationsJson` with a number (`12345`) instead of a bounded string. -> `PERMISSION_DENIED`

---

## 3. Red Team Audit Summary

- **Shadow Update Test**: Passed across all 12 collections via `hasAll` + `hasOnly` in `isValid[Entity]` and `affectedKeys().hasOnly()` on updates.
- **Email Spoofing Test**: Passed via `isVerifiedUser()` requiring `request.auth.token.email_verified == true`.
- **PII Blanket Test**: Passed; `/users/{userId}` restricts `get` and `list` strictly to `request.auth.uid == userId`.
- **Query Trust Test**: Passed; every `allow list` checks `resource.data.ownerId == request.auth.uid`.
- **Value Poisoning Test**: Passed; every `allow update` begins with `isValid[Entity](incoming())`.
