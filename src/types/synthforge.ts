export type ColumnDataType = 'numerical' | 'categorical' | 'datetime' | 'identifier';

export interface ColumnProfile {
  name: string;
  type: ColumnDataType;
  missingCount: number;
  missingRate: number;
  uniqueCount: number;
  cardinalityRate: number;
  // Numerical stats
  mean?: number;
  std?: number;
  min?: number;
  q25?: number;
  median?: number;
  q75?: number;
  max?: number;
  skewness?: number;
  outlierCount?: number;
  outlierRate?: number;
  histogram?: { bin: string; count: number; minVal: number; maxVal: number }[];
  // Categorical / Datetime stats
  topCategories?: { value: string; count: number; frequency: number }[];
  minDate?: string;
  maxDate?: string;
}

export interface ClassImbalanceInfo {
  targetColumn: string;
  classes: { label: string; count: number; share: number }[];
  imbalanceRatio: number; // majority / minority
  minorityShare: number;
  isImbalanced: boolean;
  entropy: number;
}

export interface FeatureCorrelation {
  colA: string;
  colB: string;
  correlation: number;
}

export interface ProfilingReportData {
  datasetId: string;
  rowCount: number;
  columnCount: number;
  numericalCount: number;
  categoricalCount: number;
  datetimeCount: number;
  totalMissingCount: number;
  missingRate: number;
  duplicateRowCount: number;
  duplicateRate: number;
  totalOutlierCount: number;
  outlierRate: number;
  qualityScore: number;
  columns: ColumnProfile[];
  correlations: FeatureCorrelation[];
  classImbalance?: ClassImbalanceInfo;
  warnings: { severity: 'high' | 'medium' | 'info'; column?: string; message: string; remediation: string }[];
}

export type GenerationMethodId =
  | 'gaussian_copula'
  | 'ctgan'
  | 'tvae'
  | 'multivariate_normal'
  | 'conditional_sampling';

export type ConstraintType =
  | 'range'
  | 'non_negative'
  | 'positive'
  | 'date_order'
  | 'category_in'
  | 'column_comparison';

export interface CustomConstraint {
  id: string;
  name: string;
  type: ConstraintType;
  column: string;
  min?: number;
  max?: number;
  referenceColumn?: string;
  operator?: '>=' | '<=' | '>' | '<';
  allowedValues?: string[];
  enforcePostHoc: boolean;
}

export interface GenerationConfigParams {
  id?: string;
  name: string;
  datasetId: string;
  method: GenerationMethodId;
  numRows: number;
  selectedColumns: string[];
  randomSeed: number;
  missingValueMode: 'preserve_rate' | 'impute_complete' | 'zero_missing';
  conditionalColumn?: string;
  conditionalValue?: string;
  epochs?: number;
  batchSize?: number;
  embeddingDim?: number;
  constraints: CustomConstraint[];
}

export interface FeatureQualityMetric {
  column: string;
  type: ColumnDataType;
  // Numerical
  realMean?: number;
  synthMean?: number;
  realStd?: number;
  synthStd?: number;
  realMedian?: number;
  synthMedian?: number;
  realQ25?: number;
  synthQ25?: number;
  realQ75?: number;
  synthQ75?: number;
  ksStatistic?: number;
  ksPValue?: number;
  wassersteinDistance?: number;
  // Categorical & shared
  chiSquareStat?: number;
  chiSquarePValue?: number;
  jsDivergence: number;
  similarityScore: number; // 0 - 100
  distributionComparison: { label: string; realFreq: number; synthFreq: number }[];
}

export interface QualityEvaluationReport {
  overallScore: number;
  statisticalSimilarityScore: number;
  correlationPreservationScore: number;
  structuralQualityScore: number;
  meanKsComplement: number;
  meanJsSimilarity: number;
  meanWassersteinDrift: number;
  correlationFrobeniusDrift: number;
  realMissingRate: number;
  synthMissingRate: number;
  realDuplicateRate: number;
  synthDuplicateRate: number;
  constraintViolationCount: number;
  constraintViolationRate: number;
  rawViolationCountBeforeProjection: number;
  featureMetrics: FeatureQualityMetric[];
  correlationComparison: {
    colA: string;
    colB: string;
    realCorr: number;
    synthCorr: number;
    drift: number;
  }[];
  warnings: string[];
  metricLimitations: {
    metric: string;
    purpose: string;
    limitation: string;
  }[];
}

export interface PrivacyAssessmentReport {
  overallRiskScore: number; // 0 (lowest risk) to 100 (highest risk)
  riskLevel: 'Low' | 'Moderate' | 'Elevated' | 'Critical';
  exactOverlapCount: number;
  exactOverlapRate: number;
  nearDuplicateCount: number;
  nearDuplicateRate: number;
  dcrMean: number;
  dcrMedian: number;
  dcr5thPercentile: number;
  holdoutDcr5thPercentile: number;
  dcrRatioToHoldout: number; // >= 1.0 means synth is no closer than unseen real holdout
  nndrMean: number;
  nndr5thPercentile: number;
  memorisationSuspectCount: number;
  miaSimulatedAuc: number;
  miaAttackerAdvantage: number;
  miaPrecisionAt10Fpr: number;
  dcrHistogram: { distanceBin: string; synthToRealPct: number; holdoutToRealPct: number }[];
  nearestNeighbourPairs: {
    synthRowIndex: number;
    closestRealRowIndex: number;
    normalizedDistance: number;
    nndr: number;
    status: 'Safe' | 'Close' | 'Memorisation Warning';
  }[];
  warnings: string[];
}

export type MLModelType =
  | 'logistic_regression'
  | 'random_forest'
  | 'xgboost'
  | 'gradient_boosting'
  | 'decision_tree';

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: { tp: number; fp: number; tn: number; fn: number };
}

export interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ExperimentModelResult {
  modelId: MLModelType;
  modelName: string;
  taskType: 'classification' | 'regression';
  // Experiment A: Train Real, Test Held-Out Real (TRTR)
  experimentA_Real: ClassificationMetrics & RegressionMetrics;
  // Experiment B: Train Synthetic, Test Held-Out Real (TSTR)
  experimentB_Synthetic: ClassificationMetrics & RegressionMetrics;
  // Experiment C: Train Combined Real + Synthetic, Test Held-Out Real (Augmented)
  experimentC_Augmented: ClassificationMetrics & RegressionMetrics;
  utilityRetentionPct: number; // TSTR / TRTR
  augmentationGainPct: number; // (Augmented - TRTR)
  featureImportances: { feature: string; realImp: number; synthImp: number }[];
}

export interface MLBenchmarkReport {
  id: string;
  runId: string;
  datasetId: string;
  targetColumn: string;
  taskType: 'classification' | 'regression';
  realTrainSize: number;
  realHoldoutTestSize: number;
  syntheticTrainSize: number;
  augmentedTrainSize: number;
  leakageCheckPassed: boolean;
  overallUtilityScore: number;
  avgTstrRetentionPct: number;
  avgAugmentationDeltaPct: number;
  models: ExperimentModelResult[];
  summaryVerdict: string;
  createdAt: string;
}

export type TransformationActionType =
  | 'impute_missing'
  | 'drop_duplicates'
  | 'clip_outliers_iqr'
  | 'normalize_zscore'
  | 'normalize_minmax'
  | 'encode_categorical'
  | 'drop_column'
  | 'balance_classes_oversample';

export interface TransformationStep {
  id: string;
  action: TransformationActionType;
  column?: string;
  strategy?: string;
  description: string;
  rowsBefore: number;
  rowsAfter: number;
  appliedAt: string;
}

export interface DatasetRecord {
  id: string;
  ownerId: string;
  name: string;
  description: string;
  sourceFormat: 'csv' | 'json' | 'excel' | 'built-in';
  rowCount: number;
  columnCount: number;
  targetColumn: string;
  rows: Record<string, any>[];
  profile: ProfilingReportData;
  transformations: TransformationStep[];
  defaultConstraints: CustomConstraint[];
  createdAt: string;
}

export interface GenerationRunRecord {
  id: string;
  ownerId: string;
  datasetId: string;
  datasetName: string;
  config: GenerationConfigParams;
  durationMs: number;
  status: 'completed' | 'failed';
  syntheticRows: Record<string, any>[];
  qualityReport: QualityEvaluationReport;
  privacyReport: PrivacyAssessmentReport;
  mlReport?: MLBenchmarkReport;
  epochLogs: { epoch: number; generatorLoss: number; discriminatorLoss: number; klDivergence?: number }[];
  createdAt: string;
}

export interface ExportHistoryRecord {
  id: string;
  ownerId: string;
  runId: string;
  exportType: 'csv' | 'json' | 'pdf_report' | 'schema_json';
  fileName: string;
  rowCount: number;
  createdAt: string;
}
