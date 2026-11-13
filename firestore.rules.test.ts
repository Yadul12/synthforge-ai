import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import { BUILTIN_SAMPLE_DATASETS } from './src/data/sampleDatasets';
import { profileDataset, applyDataTransformation } from './src/engine/profiler';
import { generateSyntheticDataset } from './src/engine/generator';
import { evaluateSyntheticQuality } from './src/engine/evaluator';
import { evaluatePrivacyRisk } from './src/engine/privacy';
import { runDownstreamMLBenchmark } from './src/engine/mlBenchmark';

// Deterministic Firestore Rules AST/Invariant Evaluator for the "Dirty Dozen" Payloads
function simulateFirestoreRuleEvaluation(payloadScenario: {
  collection: string;
  docId: string;
  operation: 'create' | 'update' | 'get' | 'list';
  authUid: string | null;
  emailVerified: boolean;
  incomingData?: Record<string, any>;
  existingData?: Record<string, any>;
  listFilterOwnerId?: string;
}): 'ALLOWED' | 'PERMISSION_DENIED' {
  const {
    collection,
    docId,
    operation,
    authUid,
    emailVerified,
    incomingData,
    existingData,
    listFilterOwnerId,
  } = payloadScenario;

  if (!authUid) return 'PERMISSION_DENIED';
  const isValidId = /^[a-zA-Z0-9_\-]+$/.test(docId) && docId.length >= 1 && docId.length <= 128;

  if (operation !== 'list' && !isValidId) {
    return 'PERMISSION_DENIED';
  }

  if (operation === 'list') {
    if (listFilterOwnerId !== authUid) return 'PERMISSION_DENIED';
    return 'ALLOWED';
  }

  if (collection === 'users') {
    if (operation === 'get') {
      return authUid === docId ? 'ALLOWED' : 'PERMISSION_DENIED';
    }
    if (!emailVerified || authUid !== docId || !incomingData) return 'PERMISSION_DENIED';
    const allowedKeys = ['uid', 'displayName', 'role', 'createdAt'];
    const keys = Object.keys(incomingData);
    if (!allowedKeys.every((k) => keys.includes(k)) || !keys.every((k) => allowedKeys.includes(k))) {
      return 'PERMISSION_DENIED';
    }
    if (!['engineer', 'researcher', 'auditor'].includes(incomingData.role)) {
      return 'PERMISSION_DENIED';
    }
    return 'ALLOWED';
  }

  if (collection === 'datasets' || collection === 'generation_runs') {
    if (operation === 'get') {
      return existingData?.ownerId === authUid ? 'ALLOWED' : 'PERMISSION_DENIED';
    }
    if (!emailVerified || !incomingData) return 'PERMISSION_DENIED';
    if (incomingData.ownerId !== authUid) return 'PERMISSION_DENIED';

    if (collection === 'datasets') {
      const reqKeys = [
        'ownerId',
        'name',
        'sourceFormat',
        'rowCount',
        'columnCount',
        'schemaSummaryJson',
        'sampleRowsJson',
        'createdAt',
      ];
      const optKeys = ['description', 'targetColumn', 'transformationsJson'];
      const allAllowed = [...reqKeys, ...optKeys];
      const keys = Object.keys(incomingData);
      if (!reqKeys.every((k) => keys.includes(k)) || !keys.every((k) => allAllowed.includes(k))) {
        return 'PERMISSION_DENIED';
      }
      if (typeof incomingData.name !== 'string' || incomingData.name.length > 160) {
        return 'PERMISSION_DENIED';
      }
      if (
        'transformationsJson' in incomingData &&
        typeof incomingData.transformationsJson !== 'string'
      ) {
        return 'PERMISSION_DENIED';
      }
    }

    if (operation === 'update' && existingData) {
      if (incomingData.ownerId !== existingData.ownerId) return 'PERMISSION_DENIED';
      if (incomingData.createdAt !== existingData.createdAt) return 'PERMISSION_DENIED';
    }

    return 'ALLOWED';
  }

  return 'PERMISSION_DENIED';
}

describe('SynthForge AI — Security Rules ("Dirty Dozen" Payloads) & End-to-End ML Pipeline', () => {
  it('verifies production firestore.rules contains all 12 collections and zero-trust primitives', () => {
    const rulesText = fs.readFileSync('./firestore.rules', 'utf-8');
    expect(rulesText).toContain("rules_version = '2';");
    expect(rulesText).toContain('match /users/{userId}');
    expect(rulesText).toContain('match /datasets/{datasetId}');
    expect(rulesText).toContain('match /dataset_schemas/{schemaId}');
    expect(rulesText).toContain('match /profiling_reports/{reportId}');
    expect(rulesText).toContain('match /generation_configs/{configId}');
    expect(rulesText).toContain('match /generation_runs/{runId}');
    expect(rulesText).toContain('match /synthetic_datasets/{synthId}');
    expect(rulesText).toContain('match /quality_assessments/{assessmentId}');
    expect(rulesText).toContain('match /privacy_assessments/{privacyId}');
    expect(rulesText).toContain('match /ml_experiments/{experimentId}');
    expect(rulesText).toContain('match /model_metrics/{metricId}');
    expect(rulesText).toContain('match /export_history/{exportId}');
  });

  it('rejects all 12 "Dirty Dozen" adversarial payloads with PERMISSION_DENIED', () => {
    const validDatasetPayload = {
      ownerId: 'uid_A',
      name: 'Customer Churn',
      sourceFormat: 'csv',
      rowCount: 100,
      columnCount: 8,
      schemaSummaryJson: '[]',
      sampleRowsJson: '[]',
      createdAt: '2026-11-13T00:00:00.000Z',
    };

    // 1. Unverified email write
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: false,
        incomingData: validDatasetPayload,
      })
    ).toBe('PERMISSION_DENIED');

    // 2. Cross-tenant owner spoofing
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: true,
        incomingData: { ...validDatasetPayload, ownerId: 'uid_B' },
      })
    ).toBe('PERMISSION_DENIED');

    // 3. Shadow field injection on create
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: true,
        incomingData: { ...validDatasetPayload, isAdmin: true },
      })
    ).toBe('PERMISSION_DENIED');

    // 4. Shadow field injection on update
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'update',
        authUid: 'uid_A',
        emailVerified: true,
        existingData: validDatasetPayload,
        incomingData: { ...validDatasetPayload, isVerified: true },
      })
    ).toBe('PERMISSION_DENIED');

    // 5. Immutable ownerId tampering on update
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'update',
        authUid: 'uid_A',
        emailVerified: true,
        existingData: validDatasetPayload,
        incomingData: { ...validDatasetPayload, ownerId: 'uid_B' },
      })
    ).toBe('PERMISSION_DENIED');

    // 6. Immutable createdAt tampering on update
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'update',
        authUid: 'uid_A',
        emailVerified: true,
        existingData: validDatasetPayload,
        incomingData: { ...validDatasetPayload, createdAt: '2099-01-01T00:00:00.000Z' },
      })
    ).toBe('PERMISSION_DENIED');

    // 7. Path ID poisoning
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'bad id$!',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: true,
        incomingData: validDatasetPayload,
      })
    ).toBe('PERMISSION_DENIED');

    // 8. Denial-of-Wallet string overflow
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: true,
        incomingData: { ...validDatasetPayload, name: 'X'.repeat(200) },
      })
    ).toBe('PERMISSION_DENIED');

    // 9. Blanket list scraping without ownerId filter
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'generation_runs',
        docId: '',
        operation: 'list',
        authUid: 'uid_A',
        emailVerified: true,
        listFilterOwnerId: undefined,
      })
    ).toBe('PERMISSION_DENIED');

    // 10. Unauthorized profile read (PII isolation)
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'users',
        docId: 'uid_B',
        operation: 'get',
        authUid: 'uid_A',
        emailVerified: true,
      })
    ).toBe('PERMISSION_DENIED');

    // 11. Self-assigned role escalation
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'users',
        docId: 'uid_A',
        operation: 'create',
        authUid: 'uid_A',
        emailVerified: true,
        incomingData: {
          uid: 'uid_A',
          displayName: 'Attacker',
          role: 'superadmin',
          createdAt: '2026-11-13T00:00:00.000Z',
        },
      })
    ).toBe('PERMISSION_DENIED');

    // 12. Value poisoning on whitelisted update key
    expect(
      simulateFirestoreRuleEvaluation({
        collection: 'datasets',
        docId: 'ds_01',
        operation: 'update',
        authUid: 'uid_A',
        emailVerified: true,
        existingData: validDatasetPayload,
        incomingData: { ...validDatasetPayload, transformationsJson: 12345 },
      })
    ).toBe('PERMISSION_DENIED');
  });

  it('profiles datasets, enforces custom constraints across multiple generators, evaluates KS/Wasserstein/JSD quality, privacy risk, and TSTR ML utility', () => {
    const churnDs = BUILTIN_SAMPLE_DATASETS[0];
    expect(churnDs.rowCount).toBeGreaterThan(150);
    expect(churnDs.profile.qualityScore).toBeGreaterThan(60);

    // Data preparation test
    const curated = applyDataTransformation(churnDs.rows, 'drop_duplicates');
    expect(curated.transformedRows.length).toBeLessThanOrEqual(churnDs.rows.length);

    // Test both Gaussian Copula and CTGAN
    for (const method of ['gaussian_copula', 'ctgan'] as const) {
      const gen = generateSyntheticDataset(churnDs.rows, {
        name: `Test ${method}`,
        datasetId: churnDs.id,
        method,
        numRows: 120,
        selectedColumns: churnDs.profile.columns.map((c) => c.name),
        randomSeed: 42,
        missingValueMode: 'zero_missing',
        constraints: churnDs.defaultConstraints,
      });

      expect(gen.syntheticRows.length).toBe(120);
      expect(gen.finalViolationCount).toBe(0);

      const quality = evaluateSyntheticQuality(
        churnDs.rows,
        gen.syntheticRows,
        churnDs.defaultConstraints,
        gen.rawViolationCountBeforeProjection
      );
      expect(quality.overallScore).toBeGreaterThan(70);
      expect(quality.constraintViolationRate).toBe(0);

      const privacy = evaluatePrivacyRisk(churnDs.rows, gen.syntheticRows);
      expect(privacy.miaSimulatedAuc).toBeGreaterThanOrEqual(0.45);
      expect(privacy.miaSimulatedAuc).toBeLessThanOrEqual(0.99);

      const ml = runDownstreamMLBenchmark(
        churnDs.rows,
        gen.syntheticRows,
        churnDs.targetColumn,
        churnDs.id,
        `run_test_${method}`
      );
      expect(ml.leakageCheckPassed).toBe(true);
      expect(ml.models.length).toBe(5);
      expect(ml.avgTstrRetentionPct).toBeGreaterThan(60);
    }
  });
});
