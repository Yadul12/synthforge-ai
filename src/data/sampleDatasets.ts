import { DatasetRecord, CustomConstraint } from '../types/synthforge';
import { profileDataset } from '../engine/profiler';

// Deterministic Mulberry32 PRNG for reproducible built-in sample generation
function createRng(seed: number) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randNormal(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = Math.max(1e-7, rng());
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

function addDays(baseIso: string, days: number): string {
  const d = new Date(baseIso);
  d.setUTCDate(d.getUTCDate() + Math.round(days));
  return d.toISOString().slice(0, 10);
}

function buildCustomerChurnDataset(): DatasetRecord {
  const rng = createRng(1042);
  const rows: Record<string, any>[] = [];
  const contracts = ['Month-to-Month', 'One Year', 'Two Year'];
  const payments = ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Auto-Pay'];
  const tiers = ['Starter', 'Professional', 'Enterprise'];

  for (let i = 1; i <= 180; i++) {
    const age = Math.round(Math.max(19, Math.min(86, randNormal(rng, 42, 13))));
    const tenureMonths = Math.round(Math.max(1, Math.min(72, randNormal(rng, 26, 18))));
    const tierIdx = rng() < 0.45 ? 0 : rng() < 0.8 ? 1 : 2;
    const planTier = tiers[tierIdx];
    const baseCharge = tierIdx === 0 ? 39 : tierIdx === 1 ? 89 : 175;
    const monthlyCharges = Number(
      Math.max(18, baseCharge + randNormal(rng, 0, 14) + (age > 60 ? -5 : 4)).toFixed(2)
    );
    // Inject a few realistic outliers & missing values for profiling richness
    const totalSpend =
      i % 29 === 0
        ? null
        : Number(
            Math.max(
              monthlyCharges,
              tenureMonths * monthlyCharges * (0.92 + rng() * 0.16)
            ).toFixed(2)
          );
    const supportTickets = Math.max(0, Math.round(randNormal(rng, 2.1 + (monthlyCharges > 110 ? 1.4 : 0), 1.8)));
    const contractType =
      tenureMonths > 36
        ? contracts[rng() < 0.65 ? 2 : 1]
        : contracts[rng() < 0.6 ? 0 : 1];
    const paymentMethod = payments[Math.floor(rng() * payments.length)];

    const regDate = addDays('2022-01-15', Math.floor(rng() * 750));
    const lastOrderDate = addDays(regDate, Math.max(2, tenureMonths * 28 + Math.floor(rng() * 20)));

    // Logistic latent score for realistic ML signal
    const logit =
      -1.35 +
      (contractType === 'Month-to-Month' ? 1.45 : -0.8) +
      (supportTickets >= 4 ? 1.15 : -0.3) +
      (tenureMonths < 12 ? 0.95 : -0.65) +
      (monthlyCharges > 110 ? 0.55 : -0.2) +
      randNormal(rng, 0, 0.55);
    const prob = 1 / (1 + Math.exp(-logit));
    const churned = prob > 0.55 ? 'Yes' : 'No';

    rows.push({
      customer_id: `CUST-${String(1000 + i)}`,
      registration_date: regDate,
      last_order_date: lastOrderDate,
      age: i === 17 ? 94 : age, // 1 extreme outlier for profiler detection
      tenure_months: tenureMonths,
      monthly_charges: monthlyCharges,
      total_spend: totalSpend,
      support_tickets: supportTickets,
      contract_type: contractType,
      payment_method: paymentMethod,
      plan_tier: planTier,
      churned,
    });
  }

  // Add 2 exact duplicates so profiler detects duplicate records
  rows.push({ ...rows[4] });
  rows.push({ ...rows[19] });

  const defaultConstraints: CustomConstraint[] = [
    {
      id: 'c_churn_age',
      name: 'Age must be between 18 and 90',
      type: 'range',
      column: 'age',
      min: 18,
      max: 90,
      enforcePostHoc: true,
    },
    {
      id: 'c_churn_charges',
      name: 'Monthly charges must be non-negative',
      type: 'non_negative',
      column: 'monthly_charges',
      enforcePostHoc: true,
    },
    {
      id: 'c_churn_dates',
      name: 'Last order date cannot precede registration date',
      type: 'date_order',
      column: 'last_order_date',
      referenceColumn: 'registration_date',
      enforcePostHoc: true,
    },
    {
      id: 'c_churn_tier',
      name: 'Plan tier must belong to valid SaaS tier list',
      type: 'category_in',
      column: 'plan_tier',
      allowedValues: ['Starter', 'Professional', 'Enterprise'],
      enforcePostHoc: true,
    },
  ];

  const id = 'ds_customer_churn';
  return {
    id,
    ownerId: 'system',
    name: 'Customer Churn Telemetry (SaaS B2B)',
    description:
      'Subscription lifecycle dataset containing customer demographics, billing velocity, support ticket frequency, temporal registration/order dates, and binary churn outcome.',
    sourceFormat: 'built-in',
    rowCount: rows.length,
    columnCount: Object.keys(rows[0]).length,
    targetColumn: 'churned',
    rows,
    profile: profileDataset(id, rows, 'churned'),
    transformations: [],
    defaultConstraints,
    createdAt: '2026-11-01T09:00:00.000Z',
  };
}

function buildRetailTransactionsDataset(): DatasetRecord {
  const rng = createRng(2088);
  const rows: Record<string, any>[] = [];
  const categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Health & Beauty', 'Sports'];
  const channels = ['Web Store', 'Mobile App', 'POS Retail'];
  const regions = ['North America', 'Europe', 'Asia-Pacific'];

  for (let i = 1; i <= 175; i++) {
    const customerAge = Math.round(Math.max(18, Math.min(82, randNormal(rng, 38, 12))));
    const catIdx = Math.floor(rng() * categories.length);
    const productCategory = categories[catIdx];
    const baseUnitPrice =
      catIdx === 0 ? 185 : catIdx === 1 ? 54 : catIdx === 2 ? 92 : catIdx === 3 ? 38 : 76;
    const unitPrice = Number(Math.max(8, randNormal(rng, baseUnitPrice, baseUnitPrice * 0.28)).toFixed(2));
    const basketSize = Math.max(1, Math.round(randNormal(rng, 3.2, 1.7)));
    const discountPct = Number(Math.max(0, Math.min(40, randNormal(rng, 8, 6))).toFixed(1));
    const transactionAmount = Number(
      (unitPrice * basketSize * (1 - discountPct / 100)).toFixed(2)
    );
    const regDate = addDays('2023-02-01', Math.floor(rng() * 420));
    const orderDate = addDays(regDate, Math.max(1, Math.floor(rng() * 340)));
    const channel = channels[rng() < 0.48 ? 1 : rng() < 0.82 ? 0 : 2];
    const region = regions[Math.floor(rng() * regions.length)];
    const isHighValue = transactionAmount > 260 ? 'High-Value' : 'Standard';

    rows.push({
      transaction_id: `TXN-${String(5000 + i)}`,
      customer_reg_date: regDate,
      order_date: orderDate,
      customer_age: customerAge,
      basket_size: basketSize,
      unit_price: i % 35 === 0 ? null : unitPrice,
      discount_pct: discountPct,
      transaction_amount: transactionAmount,
      product_category: productCategory,
      channel,
      region,
      is_high_value: isHighValue,
    });
  }

  const defaultConstraints: CustomConstraint[] = [
    {
      id: 'c_ret_age',
      name: 'Customer age must be between 18 and 90',
      type: 'range',
      column: 'customer_age',
      min: 18,
      max: 90,
      enforcePostHoc: true,
    },
    {
      id: 'c_ret_amt',
      name: 'Transaction amount must be non-negative',
      type: 'non_negative',
      column: 'transaction_amount',
      enforcePostHoc: true,
    },
    {
      id: 'c_ret_dates',
      name: 'Order date cannot precede customer registration date',
      type: 'date_order',
      column: 'order_date',
      referenceColumn: 'customer_reg_date',
      enforcePostHoc: true,
    },
    {
      id: 'c_ret_cat',
      name: 'Product category must belong to valid catalog list',
      type: 'category_in',
      column: 'product_category',
      allowedValues: categories,
      enforcePostHoc: true,
    },
  ];

  const id = 'ds_retail_transactions';
  return {
    id,
    ownerId: 'system',
    name: 'Omnichannel Retail Transactions',
    description:
      'Multi-region retail order records capturing customer registration vs order dates, basket composition, promotional discounting, and high-value order classification.',
    sourceFormat: 'built-in',
    rowCount: rows.length,
    columnCount: Object.keys(rows[0]).length,
    targetColumn: 'is_high_value',
    rows,
    profile: profileDataset(id, rows, 'is_high_value'),
    transformations: [],
    defaultConstraints,
    createdAt: '2026-11-02T11:30:00.000Z',
  };
}

function buildLoanApplicationsDataset(): DatasetRecord {
  const rng = createRng(3099);
  const rows: Record<string, any>[] = [];
  const purposes = ['Debt Consolidation', 'Home Improvement', 'Small Business', 'Medical', 'Auto'];
  const ownerships = ['Mortgage', 'Rent', 'Own'];

  for (let i = 1; i <= 170; i++) {
    const applicantAge = Math.round(Math.max(21, Math.min(78, randNormal(rng, 41, 11))));
    const employmentYears = Math.max(0, Math.min(applicantAge - 18, Math.round(randNormal(rng, 7.5, 4.8))));
    const annualIncome = Number(
      Math.max(24000, randNormal(rng, 78000 + employmentYears * 2100, 26000)).toFixed(0)
    );
    const creditScore = Math.round(
      Math.max(520, Math.min(850, randNormal(rng, 685 + (annualIncome - 75000) / 1400, 52)))
    );
    const loanAmount = Number(
      Math.max(2500, Math.min(85000, randNormal(rng, 22000 + annualIncome * 0.12, 9500))).toFixed(0)
    );
    const debtToIncome = Number(
      Math.max(4, Math.min(58, (loanAmount / Math.max(1, annualIncome)) * 55 + randNormal(rng, 12, 5.5))).toFixed(2)
    );
    const loanPurpose = purposes[Math.floor(rng() * purposes.length)];
    const homeOwnership = ownerships[rng() < 0.46 ? 0 : rng() < 0.84 ? 1 : 2];
    const appDate = addDays('2024-01-10', Math.floor(rng() * 520));

    const riskLogit =
      -1.1 +
      (creditScore < 640 ? 1.55 : -0.85) +
      (debtToIncome > 32 ? 1.25 : -0.5) +
      (employmentYears < 2 ? 0.65 : -0.3) +
      randNormal(rng, 0, 0.5);
    const defaultRisk = 1 / (1 + Math.exp(-riskLogit)) > 0.54 ? 'Default' : 'Performing';

    rows.push({
      application_id: `LOAN-${String(8000 + i)}`,
      application_date: appDate,
      applicant_age: applicantAge,
      employment_years: i % 40 === 0 ? null : employmentYears,
      annual_income: annualIncome,
      credit_score: creditScore,
      loan_amount: loanAmount,
      debt_to_income: debtToIncome,
      loan_purpose: loanPurpose,
      home_ownership: homeOwnership,
      default_risk: defaultRisk,
    });
  }

  const defaultConstraints: CustomConstraint[] = [
    {
      id: 'c_loan_age',
      name: 'Applicant age must be between 18 and 90',
      type: 'range',
      column: 'applicant_age',
      min: 18,
      max: 90,
      enforcePostHoc: true,
    },
    {
      id: 'c_loan_pos',
      name: 'Loan amount must be strictly positive',
      type: 'positive',
      column: 'loan_amount',
      enforcePostHoc: true,
    },
    {
      id: 'c_loan_fico',
      name: 'Credit score must be between 300 and 850',
      type: 'range',
      column: 'credit_score',
      min: 300,
      max: 850,
      enforcePostHoc: true,
    },
  ];

  const id = 'ds_loan_applications';
  return {
    id,
    ownerId: 'system',
    name: 'Consumer Credit & Loan Underwriting',
    description:
      'Financial underwriting records featuring borrower income, FICO credit scores, debt-to-income ratios, collateral ownership, and credit default outcomes.',
    sourceFormat: 'built-in',
    rowCount: rows.length,
    columnCount: Object.keys(rows[0]).length,
    targetColumn: 'default_risk',
    rows,
    profile: profileDataset(id, rows, 'default_risk'),
    transformations: [],
    defaultConstraints,
    createdAt: '2026-11-03T14:15:00.000Z',
  };
}

function buildManufacturingOpsDataset(): DatasetRecord {
  const rng = createRng(4077);
  const rows: Record<string, any>[] = [];
  const shifts = ['Morning', 'Afternoon', 'Night'];
  const lines = ['CNC-Alpha', 'CNC-Beta', 'Laser-Gamma'];
  const grades = ['Ti-6Al-4V', 'Al-7075', 'Inconel-718'];

  for (let i = 1; i <= 165; i++) {
    const rpmSpeed = Math.round(Math.max(1200, Math.min(5400, randNormal(rng, 3200, 620))));
    const pressureBar = Number(Math.max(2.5, Math.min(14.5, randNormal(rng, 7.8, 1.6))).toFixed(2));
    const spindleTempC = Number(
      Math.max(38, 45 + (rpmSpeed - 2500) * 0.014 + pressureBar * 1.8 + randNormal(rng, 0, 4.2)).toFixed(2)
    );
    const vibrationMmS = Number(
      Math.max(0.4, 1.1 + Math.max(0, (rpmSpeed - 3400) * 0.0018) + randNormal(rng, 0, 0.65)).toFixed(2)
    );
    const powerKw = Number(
      Math.max(5, 8.5 + rpmSpeed * 0.0042 + pressureBar * 0.75 + randNormal(rng, 0, 1.1)).toFixed(2)
    );
    const shift = shifts[Math.floor(rng() * shifts.length)];
    const machineLine = lines[Math.floor(rng() * lines.length)];
    const materialGrade = grades[Math.floor(rng() * grades.length)];
    const dateStr = addDays('2025-01-05', Math.floor(i / 2));

    const defectLogit =
      -1.6 +
      (spindleTempC > 72 ? 1.5 : -0.6) +
      (vibrationMmS > 2.4 ? 1.45 : -0.55) +
      (shift === 'Night' ? 0.45 : -0.1) +
      randNormal(rng, 0, 0.5);
    const defectDetected = 1 / (1 + Math.exp(-defectLogit)) > 0.52 ? 'Defect' : 'Pass';

    rows.push({
      batch_id: `MFG-${String(9000 + i)}`,
      timestamp: dateStr,
      spindle_temp_c: spindleTempC,
      vibration_mm_s: vibrationMmS,
      pressure_bar: pressureBar,
      rpm_speed: rpmSpeed,
      power_kw: powerKw,
      shift,
      machine_line: machineLine,
      material_grade: materialGrade,
      defect_detected: defectDetected,
    });
  }

  const defaultConstraints: CustomConstraint[] = [
    {
      id: 'c_mfg_temp',
      name: 'Spindle temperature must be between 20°C and 120°C',
      type: 'range',
      column: 'spindle_temp_c',
      min: 20,
      max: 120,
      enforcePostHoc: true,
    },
    {
      id: 'c_mfg_vib',
      name: 'Vibration amplitude must be non-negative',
      type: 'non_negative',
      column: 'vibration_mm_s',
      enforcePostHoc: true,
    },
    {
      id: 'c_mfg_grade',
      name: 'Material grade must belong to certified alloy list',
      type: 'category_in',
      column: 'material_grade',
      allowedValues: grades,
      enforcePostHoc: true,
    },
  ];

  const id = 'ds_manufacturing_ops';
  return {
    id,
    ownerId: 'system',
    name: 'Precision CNC Manufacturing Telemetry',
    description:
      'Industrial sensor telemetry measuring spindle temperature, vibration harmonics, hydraulic pressure, power draw, alloy grade, and surface defect inspection results.',
    sourceFormat: 'built-in',
    rowCount: rows.length,
    columnCount: Object.keys(rows[0]).length,
    targetColumn: 'defect_detected',
    rows,
    profile: profileDataset(id, rows, 'defect_detected'),
    transformations: [],
    defaultConstraints,
    createdAt: '2026-11-04T16:45:00.000Z',
  };
}

export const BUILTIN_SAMPLE_DATASETS: DatasetRecord[] = [
  buildCustomerChurnDataset(),
  buildRetailTransactionsDataset(),
  buildLoanApplicationsDataset(),
  buildManufacturingOpsDataset(),
];
