# SynthForge AI — Synthetic Data Generation & Model Training Intelligence Platform

**SynthForge AI** is an end-to-end synthetic data engineering, statistical auditing, empirical privacy risk evaluation, and downstream machine learning utility benchmarking platform.

## 1. System Architecture

```text
Real Dataset (CSV / JSON / Excel / Built-in)
   │
   ▼
[1. Data Ingestion, Schema Explorer & Curation Workbench]
   ├── Type Inference (Numerical, Categorical, Datetime, Identifier)
   ├── Missingness, Duplicate & 1.5×IQR Outlier Detection
   ├── Class Imbalance & Pairwise Pearson Correlation Matrix
   └── Live Transformation Preview & Curation History (Impute, Clip, Z-Score, Min-Max, Oversample)
   │
   ▼
[2. Synthetic Data Generation Engine (Deterministic PRNG + Custom Constraints)]
   ├── Statistical: Multivariate Gaussian (Cholesky Covariance) & Target-Conditioned Sampling
   ├── Machine Learning / SDV: Gaussian Copula, CTGAN (Mode-Specific VGM), TVAE (Latent ELBO)
   └── Custom Domain Constraints & Post-Hoc Projection (Range, Non-Negative, Positive, Date Order, Category Allowlist)
   │
   ├──► [3. Statistical Quality Evaluator]
   │       ├── Two-Sample Kolmogorov-Smirnov (KS) Statistic & Asymptotic p-Value
   │       ├── Range-Normalized 1D Wasserstein-1 (Earth Mover's) Distance
   │       ├── Chi-Square (χ²) Goodness-of-Fit & Jensen-Shannon Divergence (JSD)
   │       └── Pairwise Correlation Matrix Drift & Explicit Metric Limitation Explanations
   │
   ├──► [4. Privacy & Disclosure Risk Auditor]
   │       ├── Exact Record Overlap & Near-Duplicate Rate (Gower Distance < 0.045)
   │       ├── Distance to Closest Record (DCR 5th Percentile vs Holdout Baseline)
   │       ├── Nearest-Neighbour Distance Ratio (NNDR = d₁/d₂ Memorisation Detector)
   │       └── Controlled Membership-Inference Attack (MIA) ROC-AUC & Attacker Advantage
   │
   └──► [5. Downstream ML Utility Benchmarking (Zero Data Leakage)]
           ├── Pre-Split Original Real Data → 75% Real Train | 25% Held-Out Real Test
           ├── Experiment A (TRTR): Train on Real Train → Test on Held-Out Real Test
           ├── Experiment B (TSTR): Train on Synthetic → Test on Held-Out Real Test
           ├── Experiment C (Augmented): Train on Real Train + Synthetic → Test on Held-Out Real Test
           └── Models: Logistic/Ridge Regression, Random Forest, XGBoost (Regularized GBDT), Gradient Boosting, CART Decision Tree
```

## 2. Built-In Sample Datasets

1. **Customer Churn Telemetry (`ds_customer_churn`)**: SaaS subscription dataset with registration and order dates, age, tenure, monthly charges, support ticket volume, contract type, plan tier, and binary churn target.
2. **Omnichannel Retail Transactions (`ds_retail_transactions`)**: Multi-region order records with customer registration vs order dates, basket size, unit price, promotional discounts, product categories, and high-value order target.
3. **Consumer Credit & Loan Underwriting (`ds_loan_applications`)**: Financial underwriting dataset with applicant age, employment tenure, annual income, FICO credit score, loan amount, debt-to-income ratio, and default risk target.
4. **Precision CNC Manufacturing Telemetry (`ds_manufacturing_ops`)**: Industrial sensor telemetry capturing spindle temperature, vibration amplitude, hydraulic pressure, RPM speed, alloy grade, and surface defect inspection outcome.

## 3. Generation Algorithms & Constraint Enforcement

- **Gaussian Copula**: Transforms each feature marginal $F_j(x_j)$ to uniform ranks $u_j \in (0, 1)$, applies the inverse standard normal CDF $z_j = \Phi^{-1}(u_j)$, estimates the latent correlation matrix $\Sigma_Z$, factors $\Sigma_Z = L L^\top$ via regularized Cholesky decomposition, draws correlated normal vectors, and maps back through $F_j^{-1}(\Phi(z_j))$.
- **CTGAN & TVAE**: Combines latent manifold prototypes with conditional categorical Gumbel-Softmax and variational Gaussian reparameterization ($z = \mu + \sigma \odot \epsilon$), logging epoch-by-epoch convergence telemetry.
- **Custom Domain Constraints**: Supports numerical ranges (`Age 18–90`), non-negativity (`Transaction Amount >= 0`), positivity (`Loan Amount > 0`), temporal ordering (`Order Date >= Registration Date`), and categorical allowlists.

## 4. Evaluation Methodology & Metric Limitations

- **Kolmogorov-Smirnov (KS) Test**: Measures maximum vertical distance between 1D empirical CDFs; less sensitive to tail anomalies and blind to joint multivariate interactions.
- **Wasserstein-1 Distance**: Quantifies minimum distribution transport cost; evaluates columns independently.
- **Chi-Square & Jensen-Shannon Divergence**: Evaluates categorical and discretized histogram overlap; high marginal similarity can occur even under trivial row copying.
- **TSTR ML Utility**: Evaluates whether decision boundaries learned from synthetic records generalize to unseen real holdout records without data leakage.

## 5. Privacy Limitations

Empirical distance metrics (Exact Overlap, DCR, NNDR, and simulated Membership-Inference ROC-AUC) are **statistical disclosure risk indicators**, not formal mathematical $(\epsilon, \delta)$-differential privacy guarantees.

## 6. Setup, Testing & Deployment

```bash
# Install dependencies
npm install

# Run ESLint security rules check & automated Vitest suite
npx eslint firestore.rules
npx vitest run firestore.rules.test.ts

# Start development server on port 3000
npm run dev

# Build for production or run with Docker Compose
npm run build
docker compose up --build
```
