import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import {
  Upload,
  Play,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  Plus,
  Trash2,
  Bookmark,
  Eye,
  RotateCcw,
  ArrowRight,
  Search,
} from 'lucide-react';
import {
  CustomConstraint,
  DatasetRecord,
  GenerationConfigParams,
  GenerationMethodId,
  GenerationRunRecord,
  TransformationActionType,
} from '../types/synthforge';
import { applyDataTransformation } from '../engine/profiler';

// ---------------------------------------------------------------------------
// 1. OVERVIEW VIEW
// ---------------------------------------------------------------------------
interface OverviewProps {
  activeDataset: DatasetRecord;
  activeRun: GenerationRunRecord;
  allRuns: GenerationRunRecord[];
  onNavigate: (tab: string) => void;
  onQuickGenerate: () => void;
}

export const OverviewView: React.FC<OverviewProps> = ({
  activeDataset,
  activeRun,
  allRuns,
  onNavigate,
  onQuickGenerate,
}) => {
  const mlRetention = activeRun.mlReport?.avgTstrRetentionPct ?? 0;

  const pipelineStages = [
    {
      step: '01. Real Dataset',
      metric: `${activeDataset.rowCount} rows · ${activeDataset.columnCount} cols`,
      sub: activeDataset.name,
      targetTab: 'explorer',
    },
    {
      step: '02. Data Profiling',
      metric: `${activeDataset.profile.qualityScore}% quality`,
      sub: `${(activeDataset.profile.missingRate * 100).toFixed(1)}% missing · ${activeDataset.profile.duplicateRowCount} dups`,
      targetTab: 'profiler',
    },
    {
      step: '03. Synthetic Generation',
      metric: `${activeRun.syntheticRows.length} rows (${activeRun.config.method})`,
      sub: `${activeRun.durationMs} ms · seed ${activeRun.config.randomSeed}`,
      targetTab: 'generator',
    },
    {
      step: '04. Quality Evaluation',
      metric: `${activeRun.qualityReport.overallScore}% fidelity`,
      sub: `1-KS ${(activeRun.qualityReport.meanKsComplement * 100).toFixed(1)}% · ${(activeRun.qualityReport.constraintViolationRate * 100).toFixed(1)}% violations`,
      targetTab: 'quality',
    },
    {
      step: '05. Privacy Assessment',
      metric: `${activeRun.privacyReport.riskLevel} Risk (${activeRun.privacyReport.overallRiskScore})`,
      sub: `DCR₅ ${activeRun.privacyReport.dcr5thPercentile} · MIA AUC ${activeRun.privacyReport.miaSimulatedAuc}`,
      targetTab: 'privacy',
    },
    {
      step: '06. ML Utility Benchmark',
      metric: `${mlRetention}% TSTR Parity`,
      sub: `Target: ${activeDataset.targetColumn} (Zero-Leakage Split)`,
      targetTab: 'ml',
    },
  ];

  const firstNumericFeature =
    activeRun.qualityReport.featureMetrics.find((f) => f.type === 'numerical') ||
    activeRun.qualityReport.featureMetrics[0];

  return (
    <div className="space-y-8">
      {/* Top Executive Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 pb-6 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-2xl space-y-2">
          <p className="text-xs text-slate-500 dark:text-slate-400">
            End-to-End Tabular Intelligence Pipeline · Active Dataset: {activeDataset.name}
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Synthetic Data Generation, Statistical Auditing & TSTR Model Validation
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
            All metrics below are computed live from your active dataset and synthetic generation runs.
            Inspect univariate Kolmogorov-Smirnov and Wasserstein tests, evaluate nearest-neighbour
            disclosure risks, and verify downstream predictive utility on strictly held-out real test splits.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            onClick={() => onNavigate('explorer')}
            className="px-4 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Upload / Select Dataset
          </button>
          <button
            onClick={onQuickGenerate}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            Run Full Pipeline Now
          </button>
        </div>
      </div>

      {/* 7 Real Pipeline KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-4">
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Original Dataset Size</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {activeDataset.rowCount.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {activeDataset.columnCount} features · {activeDataset.sourceFormat}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Synthetic Generated</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {activeRun.syntheticRows.length.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {activeRun.config.method}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Data Quality Score</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-emerald-600 dark:text-emerald-400">
            {activeRun.qualityReport.overallScore}%
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            1-JSD {(activeRun.qualityReport.meanJsSimilarity * 100).toFixed(1)}%
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Constraint Violation</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {(activeRun.qualityReport.constraintViolationRate * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {activeRun.qualityReport.rawViolationCountBeforeProjection} projected post-hoc
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Privacy Disclosure Risk</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-amber-600 dark:text-amber-400">
            {activeRun.privacyReport.riskLevel} ({activeRun.privacyReport.overallRiskScore})
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            Overlap {(activeRun.privacyReport.exactOverlapRate * 100).toFixed(1)}% · MIA {activeRun.privacyReport.miaSimulatedAuc}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Downstream ML Utility</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-blue-600 dark:text-blue-400">
            {mlRetention}%
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            TSTR vs TRTR parity
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Generation Duration</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {activeRun.durationMs} ms
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {allRuns.length} runs logged
          </div>
        </div>
      </div>

      {/* Interactive 6-Stage Pipeline Architecture Strip */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            End-to-End Pipeline Execution Chain
          </h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Click any stage to inspect detailed diagnostics
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-6 gap-3">
          {pipelineStages.map((st) => (
            <button
              key={st.step}
              onClick={() => onNavigate(st.targetTab)}
              className="text-left p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg hover:border-blue-500 dark:hover:border-blue-500 transition-colors group"
            >
              <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
                <span>{st.step}</span>
                <ArrowRight className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 group-hover:translate-x-0.5 transition-transform" />
              </div>
              <div className="mt-2 text-sm font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 truncate">
                {st.metric}
              </div>
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 truncate">
                {st.sub}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 2-Column Analytical Preview: Distribution Fidelity + Downstream ML Parity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Marginal Distribution Alignment ({firstNumericFeature?.column})
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                KS={firstNumericFeature?.ksStatistic ?? 'N/A'} · JSD={firstNumericFeature?.jsDivergence ?? 0}
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
              Comparison of empirical bin frequencies (%) between source real records and synthesized output.
            </p>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={firstNumericFeature?.distributionComparison || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="realFreq" name="Real Dataset (%)" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="synthFreq" name="Synthetic Dataset (%)" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Downstream ML Utility: TRTR vs TSTR vs Augmented
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                Held-out Real Test N={activeRun.mlReport?.realHoldoutTestSize || 0}
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
              Primary predictive score ({activeRun.mlReport?.taskType === 'classification' ? 'F1-Score' : 'R²'}) evaluated on unseen real holdout records.
            </p>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={(activeRun.mlReport?.models || []).map((m) => ({
                  name: m.modelName.split(' ')[0],
                  ExpA_Real:
                    m.taskType === 'classification'
                      ? m.experimentA_Real.f1Score
                      : m.experimentA_Real.r2,
                  ExpB_Synth:
                    m.taskType === 'classification'
                      ? m.experimentB_Synthetic.f1Score
                      : m.experimentB_Synthetic.r2,
                  ExpC_Augmented:
                    m.taskType === 'classification'
                      ? m.experimentC_Augmented.f1Score
                      : m.experimentC_Augmented.r2,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="ExpA_Real" name="Exp A: Train Real" fill="#64748b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="ExpB_Synth" name="Exp B: Train Synthetic" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="ExpC_Augmented" name="Exp C: Real + Synth" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 2. DATASET EXPLORER & DATA PREPARATION VIEW
// ---------------------------------------------------------------------------
interface DatasetExplorerProps {
  datasets: DatasetRecord[];
  activeDataset: DatasetRecord;
  onSelectDataset: (dsId: string) => void;
  onUploadFile: (file: File) => Promise<void>;
  onUpdateDatasetRows: (newRows: Record<string, any>[], stepDesc: any, targetCol?: string) => void;
  onResetDataset: () => void;
}

export const DatasetExplorerView: React.FC<DatasetExplorerProps> = ({
  datasets,
  activeDataset,
  onSelectDataset,
  onUploadFile,
  onUpdateDatasetRows,
  onResetDataset,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Data preparation & curation state
  const [prepAction, setPrepAction] = useState<TransformationActionType>('impute_missing');
  const [prepColumn, setPrepColumn] = useState<string>(
    activeDataset.profile.columns.find((c) => c.type === 'numerical')?.name ||
      activeDataset.profile.columns[0]?.name ||
      ''
  );
  const [prepStrategy, setPrepStrategy] = useState<string>('median');
  const [previewResult, setPreviewResult] = useState<{
    transformedRows: Record<string, any>[];
    step: any;
  } | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    setIsUploading(true);
    try {
      await onUploadFile(file);
    } catch (err: any) {
      setUploadError(err.message || 'Failed to parse uploaded dataset.');
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  const handlePreviewTransform = () => {
    const res = applyDataTransformation(
      activeDataset.rows,
      prepAction,
      prepColumn,
      prepStrategy
    );
    setPreviewResult(res);
  };

  const handleApplyTransform = () => {
    if (!previewResult) return;
    onUpdateDatasetRows(previewResult.transformedRows, previewResult.step);
    setPreviewResult(null);
  };

  const filteredRows = useMemo(() => {
    const source = previewResult ? previewResult.transformedRows : activeDataset.rows;
    if (!searchQuery.trim()) return source.slice(0, 25);
    const q = searchQuery.toLowerCase();
    return source
      .filter((r) =>
        Object.values(r).some((v) => String(v ?? '').toLowerCase().includes(q))
      )
      .slice(0, 25);
  }, [activeDataset.rows, previewResult, searchQuery]);

  const colNames = Object.keys((previewResult ? previewResult.transformedRows : activeDataset.rows)[0] || {});

  return (
    <div className="space-y-8">
      {/* Header & File Upload */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-5 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Dataset Ingestion, Schema Explorer & Curation Workbench
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Upload CSV, JSON, or Excel (.xlsx) datasets, inspect inferred column schemas, and preview data preparation transformations before applying them.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors whitespace-nowrap shrink-0">
            <Upload className="w-3.5 h-3.5" />
            <span>{isUploading ? 'Parsing Dataset...' : 'Upload CSV / JSON / Excel'}</span>
            <input
              type="file"
              accept=".csv,.json,.xlsx,.xls"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
          {activeDataset.transformations.length > 0 && (
            <button
              onClick={onResetDataset}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Revert Curation ({activeDataset.transformations.length})
            </button>
          )}
        </div>
      </div>

      {uploadError && (
        <div className="p-4 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-xs text-red-700 dark:text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Dataset Selector Strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {datasets.map((ds) => {
          const isSelected = ds.id === activeDataset.id;
          return (
            <button
              key={ds.id}
              onClick={() => {
                onSelectDataset(ds.id);
                setPreviewResult(null);
              }}
              className={`text-left p-4 rounded-lg border transition-colors ${
                isSelected
                  ? 'bg-blue-50/60 dark:bg-blue-950/30 border-blue-600 dark:border-blue-500'
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span>{ds.sourceFormat.toUpperCase()}</span>
                <span className="font-mono tabular-nums">
                  {ds.rowCount} rows · {ds.columnCount} cols
                </span>
              </div>
              <div className="mt-1.5 text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
                {ds.name}
              </div>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 line-clamp-2">
                {ds.description}
              </p>
              <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span>Target: {ds.targetColumn}</span>
                <span className="font-mono tabular-nums">
                  Quality {ds.profile.qualityScore}%
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Interactive Schema Explorer */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Interactive Schema Explorer ({activeDataset.profile.columns.length} Columns)
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Automatically inferred column data types, cardinality, missing rates, and empirical summary bounds.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
              Downstream Target Variable:
            </label>
            <select
              value={activeDataset.targetColumn}
              onChange={(e) =>
                onUpdateDatasetRows(activeDataset.rows, undefined, e.target.value)
              }
              className="px-2.5 py-1.5 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100"
            >
              {activeDataset.profile.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.type})
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                <th className="py-2.5 px-3 font-medium">Column Name</th>
                <th className="py-2.5 px-3 font-medium">Inferred Type</th>
                <th className="py-2.5 px-3 font-medium text-right">Missing</th>
                <th className="py-2.5 px-3 font-medium text-right">Unique / Cardinality</th>
                <th className="py-2.5 px-3 font-medium text-right">Mean / Mode</th>
                <th className="py-2.5 px-3 font-medium text-right">Min / Q25 / Median / Q75 / Max</th>
                <th className="py-2.5 px-3 font-medium text-right">IQR Outliers</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {activeDataset.profile.columns.map((col) => (
                <tr
                  key={col.name}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40"
                >
                  <td className="py-2.5 px-3 font-mono font-medium text-slate-900 dark:text-slate-100">
                    {col.name}
                    {col.name === activeDataset.targetColumn ? ' · Target' : ''}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 dark:text-slate-300 capitalize">
                    {col.type}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {col.missingCount} ({(col.missingRate * 100).toFixed(1)}%)
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {col.uniqueCount} ({(col.cardinalityRate * 100).toFixed(0)}%)
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {col.type === 'numerical'
                      ? `${col.mean} (±${col.std})`
                      : col.topCategories?.[0]
                      ? `${col.topCategories[0].value} (${(col.topCategories[0].frequency * 100).toFixed(0)}%)`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-500 dark:text-slate-400">
                    {col.type === 'numerical'
                      ? `${col.min} / ${col.q25} / ${col.median} / ${col.q75} / ${col.max}`
                      : col.type === 'datetime'
                      ? `${col.minDate} → ${col.maxDate}`
                      : col.topCategories
                          ?.slice(0, 3)
                          .map((c) => c.value)
                          .join(', ') || '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {col.type === 'numerical'
                      ? `${col.outlierCount} (${((col.outlierRate || 0) * 100).toFixed(1)}%)`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Preparation & Curation Workbench with Live Preview */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Data Preparation & Curation Workbench
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Configure missing-value imputation, IQR outlier clipping, Z-score/Min-Max scaling, categorical encoding, or class balancing. Preview changes before committing.
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
            Applied Transformations: {activeDataset.transformations.length}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Transformation Operation
            </label>
            <select
              value={prepAction}
              onChange={(e) => {
                setPrepAction(e.target.value as TransformationActionType);
                setPreviewResult(null);
              }}
              className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100"
            >
              <option value="impute_missing">Missing-Value Imputation (Median / Mean / Mode)</option>
              <option value="clip_outliers_iqr">Outlier Handling (1.5× IQR Fence Clipping)</option>
              <option value="normalize_zscore">Normalisation (Z-Score Standardization)</option>
              <option value="normalize_minmax">Normalisation (Min-Max [0, 1] Scaling)</option>
              <option value="encode_categorical">Categorical Ordinal Encoding</option>
              <option value="balance_classes_oversample">Class Balancing (Minority Oversampling)</option>
              <option value="drop_duplicates">Deduplicate Exact Records</option>
              <option value="drop_column">Feature Selection (Drop Column)</option>
            </select>
          </div>

          {prepAction !== 'drop_duplicates' && (
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Target Column
              </label>
              <select
                value={prepColumn}
                onChange={(e) => {
                  setPrepColumn(e.target.value);
                  setPreviewResult(null);
                }}
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100"
              >
                {activeDataset.profile.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.type})
                  </option>
                ))}
              </select>
            </div>
          )}

          {prepAction === 'impute_missing' && (
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Imputation Strategy
              </label>
              <select
                value={prepStrategy}
                onChange={(e) => {
                  setPrepStrategy(e.target.value);
                  setPreviewResult(null);
                }}
                className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100"
              >
                <option value="median">Median / Most Frequent Mode</option>
                <option value="mean">Arithmetic Mean</option>
              </select>
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              onClick={handlePreviewTransform}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-800 dark:text-slate-100 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 whitespace-nowrap shrink-0"
            >
              <Eye className="w-3.5 h-3.5" />
              Preview Transformation
            </button>
            {previewResult && (
              <button
                onClick={handleApplyTransform}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-emerald-600 rounded-lg hover:bg-emerald-500 whitespace-nowrap shrink-0"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Apply to Dataset
              </button>
            )}
          </div>
        </div>

        {previewResult && (
          <div className="p-3 bg-blue-50/70 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg flex items-center justify-between text-xs">
            <div>
              <span className="font-semibold text-blue-900 dark:text-blue-200">
                Preview Ready:{' '}
              </span>
              <span className="text-blue-800 dark:text-blue-300">
                {previewResult.step.description} (Rows: {previewResult.step.rowsBefore} →{' '}
                {previewResult.step.rowsAfter})
              </span>
            </div>
            <button
              onClick={() => setPreviewResult(null)}
              className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            >
              Discard Preview
            </button>
          </div>
        )}

        {activeDataset.transformations.length > 0 && (
          <div className="pt-2 border-t border-slate-100 dark:border-slate-800 space-y-1.5">
            <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Transformation History Log:
            </div>
            {activeDataset.transformations.map((t, idx) => (
              <div
                key={t.id}
                className="text-xs font-mono text-slate-600 dark:text-slate-300 flex items-center justify-between"
              >
                <span>
                  {idx + 1}. {t.description}
                </span>
                <span className="text-slate-400 tabular-nums">
                  {t.rowsBefore} → {t.rowsAfter} rows
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Data Records Table */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            {previewResult ? 'Previewing Transformed Records' : 'Source Dataset Records'} (Showing 25 of{' '}
            {previewResult ? previewResult.transformedRows.length : activeDataset.rowCount})
          </h2>
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter records..."
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                {colNames.map((col) => (
                  <th key={col} className="py-2 px-3 font-medium whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {filteredRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                  {colNames.map((col) => (
                    <td key={col} className="py-2 px-3 whitespace-nowrap">
                      {row[col] === null || row[col] === undefined ? (
                        <span className="text-amber-600 dark:text-amber-400 italic">NULL</span>
                      ) : (
                        String(row[col])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 3. DATA QUALITY PROFILER VIEW
// ---------------------------------------------------------------------------
export const DataQualityProfilerView: React.FC<{ activeDataset: DatasetRecord }> = ({
  activeDataset,
}) => {
  const { profile } = activeDataset;
  const numericCols = profile.columns.filter((c) => c.type === 'numerical');
  const [selectedNumCol, setSelectedNumCol] = useState<string>(
    numericCols[0]?.name || ''
  );

  const selectedColProfile =
    numericCols.find((c) => c.name === selectedNumCol) || numericCols[0];

  return (
    <div className="space-y-8">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Automated Data Quality & Statistical Profiler
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Deep inspection of missing values, duplicate records, IQR outlier distributions, class imbalance, and pairwise feature correlations for {activeDataset.name}.
          </p>
        </div>
        <div className="text-right font-mono tabular-nums">
          <div className="text-xs text-slate-500 dark:text-slate-400">Overall Source Quality</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
            {profile.qualityScore}%
          </div>
        </div>
      </div>

      {/* Summary Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Missing Values</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {profile.totalMissingCount} cells ({(profile.missingRate * 100).toFixed(2)}%)
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Duplicate Records</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {profile.duplicateRowCount} rows ({(profile.duplicateRate * 100).toFixed(2)}%)
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Detected IQR Outliers</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-amber-600 dark:text-amber-400">
            {profile.totalOutlierCount} values ({(profile.outlierRate * 100).toFixed(2)}%)
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Target Class Imbalance</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {profile.classImbalance
              ? `${profile.classImbalance.imbalanceRatio}:1 ratio`
              : 'Continuous'}
          </div>
        </div>
      </div>

      {/* Automated Quality Warnings */}
      {profile.warnings.length > 0 && (
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-3">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            Automated Diagnostic Warnings ({profile.warnings.length})
          </h2>
          <div className="divide-y divide-slate-100 dark:divide-slate-800">
            {profile.warnings.map((w, i) => (
              <div key={i} className="py-2.5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <AlertTriangle
                    className={`w-4 h-4 shrink-0 ${
                      w.severity === 'high' ? 'text-red-500' : 'text-amber-500'
                    }`}
                  />
                  <span className="font-medium text-slate-900 dark:text-slate-100">
                    {w.message}
                  </span>
                </div>
                <span className="text-slate-500 dark:text-slate-400">
                  {w.remediation}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2-Column Charts: Numerical Histogram + Target Class Imbalance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Univariate Numerical Distribution & Skewness
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
                Mean {selectedColProfile?.mean} · Median {selectedColProfile?.median} · Skewness {selectedColProfile?.skewness}
              </p>
            </div>
            <select
              value={selectedColProfile?.name || ''}
              onChange={(e) => setSelectedNumCol(e.target.value)}
              className="px-2.5 py-1.5 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md"
            >
              {numericCols.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={selectedColProfile?.histogram || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" name="Record Count" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Target Class Balance ({profile.classImbalance?.targetColumn || activeDataset.targetColumn})
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
              Imbalance Ratio {profile.classImbalance?.imbalanceRatio ?? 1}:1 · Shannon Entropy {profile.classImbalance?.entropy ?? 0} bits
            </p>
          </div>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={profile.classImbalance?.classes || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" name="Class Frequency" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Pairwise Pearson Correlation Matrix */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            Feature Relationship Pearson Correlation Matrix (r)
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Linear dependence across numerical features. Synthetic generators (Gaussian Copula, CTGAN, TVAE) preserve this covariance structure.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono tabular-nums border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800">
                <th className="py-2 px-3 text-left text-slate-500">Feature</th>
                {numericCols.map((c) => (
                  <th key={c.name} className="py-2 px-3 text-right text-slate-500">
                    {c.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {numericCols.map((rowCol) => (
                <tr key={rowCol.name}>
                  <td className="py-2 px-3 font-medium text-slate-900 dark:text-slate-100">
                    {rowCol.name}
                  </td>
                  {numericCols.map((colCol) => {
                    const cell = profile.correlations.find(
                      (item) => item.colA === rowCol.name && item.colB === colCol.name
                    );
                    const val = cell ? cell.correlation : 0;
                    const absVal = Math.abs(val);
                    return (
                      <td
                        key={colCol.name}
                        className={`py-2 px-3 text-right ${
                          absVal > 0.5 && rowCol.name !== colCol.name
                            ? 'font-bold text-blue-600 dark:text-blue-400'
                            : 'text-slate-600 dark:text-slate-300'
                        }`}
                      >
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 4. SYNTHETIC DATA GENERATOR VIEW
// ---------------------------------------------------------------------------
interface GeneratorViewProps {
  activeDataset: DatasetRecord;
  activeRun: GenerationRunRecord;
  savedConfigs: GenerationConfigParams[];
  onExecuteGeneration: (config: GenerationConfigParams) => void;
  onSaveConfig: (config: GenerationConfigParams) => void;
}

export const SyntheticGeneratorView: React.FC<GeneratorViewProps> = ({
  activeDataset,
  activeRun,
  savedConfigs,
  onExecuteGeneration,
  onSaveConfig,
}) => {
  const [method, setMethod] = useState<GenerationMethodId>(activeRun.config.method);
  const [numRows, setNumRows] = useState<number>(activeRun.config.numRows || 200);
  const [randomSeed, setRandomSeed] = useState<number>(activeRun.config.randomSeed || 42);
  const [missingValueMode, setMissingValueMode] = useState<
    'preserve_rate' | 'impute_complete' | 'zero_missing'
  >(activeRun.config.missingValueMode || 'preserve_rate');
  const [epochs, setEpochs] = useState<number>(30);
  const [selectedCols, setSelectedCols] = useState<string[]>(
    activeDataset.profile.columns.map((c) => c.name)
  );
  const [constraints, setConstraints] = useState<CustomConstraint[]>(
    activeDataset.defaultConstraints || []
  );

  // New constraint builder state
  const [newRuleCol, setNewRuleCol] = useState<string>(
    activeDataset.profile.columns.find((c) => c.type === 'numerical')?.name ||
      activeDataset.profile.columns[0]?.name ||
      ''
  );
  const [newRuleType, setNewRuleType] = useState<CustomConstraint['type']>('range');
  const [newRuleMin, setNewRuleMin] = useState<string>('18');
  const [newRuleMax, setNewRuleMax] = useState<string>('90');
  const [newRuleRefCol, setNewRuleRefCol] = useState<string>(
    activeDataset.profile.columns[1]?.name || ''
  );
  const [newRuleCats, setNewRuleCats] = useState<string>('Starter, Professional, Enterprise');

  const handleAddConstraint = () => {
    const id = `rule_${Date.now()}`;
    let name = '';
    if (newRuleType === 'range') {
      name = `${newRuleCol} must be between ${newRuleMin} and ${newRuleMax}`;
    } else if (newRuleType === 'non_negative') {
      name = `${newRuleCol} must be non-negative (>= 0)`;
    } else if (newRuleType === 'positive') {
      name = `${newRuleCol} must be strictly positive (> 0)`;
    } else if (newRuleType === 'date_order') {
      name = `${newRuleCol} cannot precede ${newRuleRefCol}`;
    } else if (newRuleType === 'category_in') {
      name = `${newRuleCol} must belong to [${newRuleCats}]`;
    } else {
      name = `${newRuleCol} >= ${newRuleRefCol}`;
    }

    const rule: CustomConstraint = {
      id,
      name,
      type: newRuleType,
      column: newRuleCol,
      min: newRuleType === 'range' ? Number(newRuleMin) : undefined,
      max: newRuleType === 'range' ? Number(newRuleMax) : undefined,
      referenceColumn:
        newRuleType === 'date_order' || newRuleType === 'column_comparison'
          ? newRuleRefCol
          : undefined,
      allowedValues:
        newRuleType === 'category_in'
          ? newRuleCats.split(',').map((s) => s.trim()).filter(Boolean)
          : undefined,
      enforcePostHoc: true,
    };
    setConstraints((prev) => [...prev, rule]);
  };

  const buildCurrentConfig = (): GenerationConfigParams => ({
    id: `cfg_${Date.now()}`,
    name: `${activeDataset.name} — ${method.toUpperCase()}`,
    datasetId: activeDataset.id,
    method,
    numRows,
    selectedColumns: selectedCols,
    randomSeed,
    missingValueMode,
    epochs,
    constraints,
  });

  const methodCatalog: {
    id: GenerationMethodId;
    category: string;
    title: string;
    desc: string;
  }[] = [
    {
      id: 'gaussian_copula',
      category: 'Machine Learning · SDV Copula',
      title: 'Gaussian Copula Synthesizer',
      desc: 'Transforms marginals via empirical rank CDFs into latent standard normal space, models joint dependencies via Cholesky correlation matrix, and maps back via inverse quantiles.',
    },
    {
      id: 'ctgan',
      category: 'Deep Generative · SDV Neural',
      title: 'CTGAN (Conditional Tabular GAN)',
      desc: 'Uses mode-specific variational Gaussian mixture normalization for continuous columns and conditional Gumbel-Softmax sampling for imbalanced categorical columns.',
    },
    {
      id: 'tvae',
      category: 'Deep Generative · SDV Neural',
      title: 'TVAE (Tabular Variational Autoencoder)',
      desc: 'Optimizes Evidence Lower Bound (ELBO) across a latent Gaussian bottleneck using reparameterized encoder-decoder projections for high fidelity on mixed tables.',
    },
    {
      id: 'multivariate_normal',
      category: 'Statistical · Parametric',
      title: 'Multivariate Gaussian & Frequency Sampler',
      desc: 'Draws continuous features from a Cholesky-decomposed covariance matrix and categorical variables from empirical multinomial frequencies.',
    },
    {
      id: 'conditional_sampling',
      category: 'Statistical · Stratified',
      title: 'Target-Conditioned Subgroup Sampler',
      desc: 'Conditions joint feature generation on target variable strata to preserve class-specific feature moments and contingency structures.',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-5 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Synthetic Data Generation Engine
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Configure statistical and deep generative synthesizers, define domain validation constraints, and execute reproducible synthetic generation runs.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => onSaveConfig(buildCurrentConfig())}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 whitespace-nowrap shrink-0"
          >
            <Bookmark className="w-3.5 h-3.5" />
            Save Configuration ({savedConfigs.length})
          </button>
          <button
            onClick={() => onExecuteGeneration(buildCurrentConfig())}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            Generate Synthetic Dataset
          </button>
        </div>
      </div>

      {/* Generation Algorithm Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-4">
        {methodCatalog.map((item) => {
          const active = method === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setMethod(item.id)}
              className={`text-left p-4 rounded-lg border transition-colors ${
                active
                  ? 'bg-blue-50/70 dark:bg-blue-950/30 border-blue-600 dark:border-blue-500'
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:border-slate-400'
              }`}
            >
              <div className="text-xs text-slate-500 dark:text-slate-400">{item.category}</div>
              <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">
                {item.title}
              </div>
              <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                {item.desc}
              </p>
            </button>
          );
        })}
      </div>

      {/* Hyperparameters & Column Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Synthesizer Hyperparameters & Reproducibility
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Number of Generated Rows
              </label>
              <input
                type="number"
                min={20}
                max={5000}
                value={numRows}
                onChange={(e) => setNumRows(Math.max(20, Math.min(5000, Number(e.target.value))))}
                className="w-full px-3 py-2 text-xs font-mono tabular-nums bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Deterministic Random Seed
              </label>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs font-mono tabular-nums bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Missing-Value Behaviour
              </label>
              <select
                value={missingValueMode}
                onChange={(e) => setMissingValueMode(e.target.value as any)}
                className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              >
                <option value="preserve_rate">Preserve Empirical Missing Rate</option>
                <option value="zero_missing">Synthesize Complete Records (0% Missing)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Training Epochs (CTGAN / TVAE)
              </label>
              <input
                type="number"
                min={10}
                max={200}
                value={epochs}
                onChange={(e) => setEpochs(Math.max(10, Math.min(200, Number(e.target.value))))}
                className="w-full px-3 py-2 text-xs font-mono tabular-nums bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              />
            </div>
          </div>

          {/* Active Columns Checklist */}
          <div className="pt-3 border-t border-slate-100 dark:border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
                Included Columns ({selectedCols.length} of {activeDataset.profile.columns.length})
              </span>
              <button
                onClick={() =>
                  setSelectedCols(activeDataset.profile.columns.map((c) => c.name))
                }
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
              >
                Select All Columns
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {activeDataset.profile.columns.map((c) => {
                const checked = selectedCols.includes(c.name);
                return (
                  <label
                    key={c.name}
                    className={`cursor-pointer inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded border transition-colors ${
                      checked
                        ? 'bg-blue-50 dark:bg-blue-950/40 border-blue-500 text-blue-900 dark:text-blue-200'
                        : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() =>
                        setSelectedCols((prev) =>
                          checked
                            ? prev.length > 2
                              ? prev.filter((item) => item !== c.name)
                              : prev
                            : [...prev, c.name]
                        )
                      }
                      className="rounded"
                    />
                    <span>{c.name}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        {/* Epoch Training Loss Telemetry (for CTGAN / TVAE) */}
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Generator Convergence Telemetry
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              {activeRun.epochLogs.length > 0
                ? `Epoch loss curve for ${activeRun.config.method.toUpperCase()} (${activeRun.epochLogs.length} epochs)`
                : 'Analytical Cholesky decomposition solved in closed form (0 iterative epochs).'}
            </p>
          </div>
          {activeRun.epochLogs.length > 0 ? (
            <div className="h-44 mt-3">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activeRun.epochLogs}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      color: '#f8fafc',
                      fontSize: '12px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="generatorLoss"
                    name="Generator / Recon Loss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                  {activeRun.config.method === 'ctgan' && (
                    <Line
                      type="monotone"
                      dataKey="discriminatorLoss"
                      name="Discriminator Loss"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="my-auto py-8 text-center text-xs text-slate-500 dark:text-slate-400 font-mono">
              Exact Copula / Multivariate Cholesky Factorization Active
              <div className="mt-1 text-slate-400">
                Select CTGAN or TVAE and run generation to view neural epoch loss curves.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Custom Domain Constraints Engine */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Custom Domain Constraints & Post-Hoc Projection Rules ({constraints.length})
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Enforce range bounds (e.g., Age 18–90), non-negative transaction amounts, temporal ordering (Order Date ≥ Registration Date), and valid category memberships.
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-emerald-600 dark:text-emerald-400">
            Active Run Violations: {(activeRun.qualityReport.constraintViolationRate * 100).toFixed(2)}%
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Rule Type
            </label>
            <select
              value={newRuleType}
              onChange={(e) => setNewRuleType(e.target.value as any)}
              className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
            >
              <option value="range">Numerical Range [Min, Max]</option>
              <option value="non_negative">Non-Negative (&gt;= 0)</option>
              <option value="positive">Strictly Positive (&gt; 0)</option>
              <option value="date_order">Date Order (Col ≥ Reference Date)</option>
              <option value="category_in">Valid Category Allowlist</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Constrained Column
            </label>
            <select
              value={newRuleCol}
              onChange={(e) => setNewRuleCol(e.target.value)}
              className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
            >
              {activeDataset.profile.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          {newRuleType === 'range' && (
            <>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  Minimum Value
                </label>
                <input
                  type="number"
                  value={newRuleMin}
                  onChange={(e) => setNewRuleMin(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  Maximum Value
                </label>
                <input
                  type="number"
                  value={newRuleMax}
                  onChange={(e) => setNewRuleMax(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
                />
              </div>
            </>
          )}

          {newRuleType === 'date_order' && (
            <div className="md:col-span-2">
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Preceding Reference Date Column
              </label>
              <select
                value={newRuleRefCol}
                onChange={(e) => setNewRuleRefCol(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              >
                {activeDataset.profile.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {newRuleType === 'category_in' && (
            <div className="md:col-span-2">
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                Allowed Categories (comma-separated)
              </label>
              <input
                type="text"
                value={newRuleCats}
                onChange={(e) => setNewRuleCats(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
              />
            </div>
          )}

          <div>
            <button
              onClick={handleAddConstraint}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 dark:bg-slate-700 rounded-lg hover:bg-slate-800 whitespace-nowrap shrink-0"
            >
              <Plus className="w-3.5 h-3.5" />
              Add Constraint Rule
            </button>
          </div>
        </div>

        <div className="divide-y divide-slate-100 dark:divide-slate-800">
          {constraints.map((c) => (
            <div
              key={c.id}
              className="py-2.5 flex items-center justify-between text-xs"
            >
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                <span className="font-medium text-slate-900 dark:text-slate-100">
                  {c.name}
                </span>
                <span className="text-slate-400 font-mono">
                  Column: {c.column} · Type: {c.type}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <label className="inline-flex items-center gap-1.5 text-slate-500 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={c.enforcePostHoc}
                    onChange={() =>
                      setConstraints((prev) =>
                        prev.map((item) =>
                          item.id === c.id
                            ? { ...item, enforcePostHoc: !item.enforcePostHoc }
                            : item
                        )
                      )
                    }
                  />
                  <span>Enforce Post-Hoc</span>
                </label>
                <button
                  onClick={() =>
                    setConstraints((prev) => prev.filter((item) => item.id !== c.id))
                  }
                  className="text-slate-400 hover:text-red-500"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Generated Synthetic Data Preview */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            Generated Synthetic Dataset Preview ({activeRun.syntheticRows.length} Rows · Method: {activeRun.config.method})
          </h2>
          <span className="text-xs font-mono tabular-nums text-slate-500">
            Showing first 15 synthesized records
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                {Object.keys(activeRun.syntheticRows[0] || {}).map((col) => (
                  <th key={col} className="py-2 px-3 font-medium whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {activeRun.syntheticRows.slice(0, 15).map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                  {Object.keys(activeRun.syntheticRows[0] || {}).map((col) => (
                    <td key={col} className="py-2 px-3 whitespace-nowrap">
                      {row[col] === null || row[col] === undefined ? 'NULL' : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 5. QUALITY EVALUATION VIEW
// ---------------------------------------------------------------------------
export const QualityEvaluationView: React.FC<{ activeRun: GenerationRunRecord }> = ({
  activeRun,
}) => {
  const { qualityReport } = activeRun;
  const [selectedFeature, setSelectedFeature] = useState<string>(
    qualityReport.featureMetrics[0]?.column || ''
  );

  const currentFeat =
    qualityReport.featureMetrics.find((f) => f.column === selectedFeature) ||
    qualityReport.featureMetrics[0];

  return (
    <div className="space-y-8">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Synthetic Data Statistical Quality Evaluation
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Rigorous comparison of real vs synthetic distributions using Kolmogorov-Smirnov tests, Wasserstein-1 distance, Chi-Square goodness-of-fit, Jensen-Shannon divergence, and correlation drift.
          </p>
        </div>
        <div className="text-right font-mono tabular-nums">
          <div className="text-xs text-slate-500 dark:text-slate-400">Composite Fidelity Score</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
            {qualityReport.overallScore}%
          </div>
        </div>
      </div>

      {/* Top Sub-Score Strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Univariate Similarity</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {qualityReport.statisticalSimilarityScore}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Mean 1-KS: {(qualityReport.meanKsComplement * 100).toFixed(1)}%
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Correlation Preservation</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {qualityReport.correlationPreservationScore}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Mean |Δr|: {qualityReport.correlationFrobeniusDrift}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Wasserstein-1 Drift</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {qualityReport.meanWassersteinDrift}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Range-normalized EMD
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Jensen-Shannon Similarity</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {(qualityReport.meanJsSimilarity * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            1 - JSD (bits)
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Structural Integrity</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {qualityReport.structuralQualityScore}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Violations: {(qualityReport.constraintViolationRate * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Distribution Comparison & Pairwise Correlation Drift */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Real vs Synthetic Distribution Overlay ({currentFeat?.column})
              </h3>
              <p className="text-xs text-slate-500 font-mono tabular-nums">
                {currentFeat?.type === 'numerical'
                  ? `KS=${currentFeat.ksStatistic} (p=${currentFeat.ksPValue}) · W1=${currentFeat.wassersteinDistance} · JSD=${currentFeat.jsDivergence}`
                  : `χ²=${currentFeat?.chiSquareStat} (p=${currentFeat?.chiSquarePValue}) · JSD=${currentFeat?.jsDivergence}`}
              </p>
            </div>
            <select
              value={currentFeat?.column || ''}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="px-2.5 py-1.5 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md"
            >
              {qualityReport.featureMetrics.map((f) => (
                <option key={f.column} value={f.column}>
                  {f.column} ({f.similarityScore}%)
                </option>
              ))}
            </select>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={currentFeat?.distributionComparison || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                <YAxis unit="%" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="realFreq" name="Real Distribution (%)" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="synthFreq" name="Synthetic Distribution (%)" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Pairwise Correlation Matrix Drift (Real r vs Synthetic r)
            </h3>
            <p className="text-xs text-slate-500">
              Top feature pairs ranked by correlation magnitude and absolute drift |r_real - r_synth|.
            </p>
          </div>
          <div className="overflow-y-auto max-h-64">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                  <th className="py-2 px-2">Feature Pair</th>
                  <th className="py-2 px-2 text-right">Real r</th>
                  <th className="py-2 px-2 text-right">Synth r</th>
                  <th className="py-2 px-2 text-right">|Δ Drift|</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
                {qualityReport.correlationComparison
                  .filter((c) => c.colA < c.colB)
                  .map((c, idx) => (
                    <tr key={idx}>
                      <td className="py-2 px-2 text-slate-900 dark:text-slate-100">
                        {c.colA} × {c.colB}
                      </td>
                      <td className="py-2 px-2 text-right">{c.realCorr.toFixed(3)}</td>
                      <td className="py-2 px-2 text-right">{c.synthCorr.toFixed(3)}</td>
                      <td
                        className={`py-2 px-2 text-right ${
                          c.drift > 0.15 ? 'text-amber-500 font-semibold' : 'text-emerald-500'
                        }`}
                      >
                        {c.drift.toFixed(3)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Feature-Level Statistical Test Table */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Feature-Level Statistical Hypothesis Tests & Quantile Comparison
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                <th className="py-2.5 px-3">Feature</th>
                <th className="py-2.5 px-3">Type</th>
                <th className="py-2.5 px-3 text-right">Real Mean (±Std)</th>
                <th className="py-2.5 px-3 text-right">Synth Mean (±Std)</th>
                <th className="py-2.5 px-3 text-right">Real Median [Q25, Q75]</th>
                <th className="py-2.5 px-3 text-right">Synth Median [Q25, Q75]</th>
                <th className="py-2.5 px-3 text-right">KS / χ² Stat (p-val)</th>
                <th className="py-2.5 px-3 text-right">W1 / JSD</th>
                <th className="py-2.5 px-3 text-right">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {qualityReport.featureMetrics.map((f) => (
                <tr key={f.column} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                  <td className="py-2.5 px-3 font-medium text-slate-900 dark:text-slate-100">
                    {f.column}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 capitalize">{f.type}</td>
                  <td className="py-2.5 px-3 text-right">
                    {f.type === 'numerical' ? `${f.realMean} (±${f.realStd})` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {f.type === 'numerical' ? `${f.synthMean} (±${f.synthStd})` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-500">
                    {f.type === 'numerical'
                      ? `${f.realMedian} [${f.realQ25}, ${f.realQ75}]`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-500">
                    {f.type === 'numerical'
                      ? `${f.synthMedian} [${f.synthQ25}, ${f.synthQ75}]`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {f.type === 'numerical'
                      ? `KS=${f.ksStatistic} (p=${f.ksPValue})`
                      : `χ²=${f.chiSquareStat} (p=${f.chiSquarePValue})`}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {f.type === 'numerical'
                      ? `W1=${f.wassersteinDistance} · JSD=${f.jsDivergence}`
                      : `JSD=${f.jsDivergence}`}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold text-emerald-600 dark:text-emerald-400">
                    {f.similarityScore}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Methodological Limitations of Statistical Quality Metrics */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-3">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Methodological Limitations & Interpretation Guidance
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          A high statistical similarity score does NOT guarantee privacy safety or downstream ML utility. Review the mathematical blind spots of each metric below:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {qualityReport.metricLimitations.map((lim) => (
            <div
              key={lim.metric}
              className="p-3.5 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 rounded-lg space-y-1"
            >
              <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                {lim.metric}
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-300">
                <span className="font-medium">What it measures: </span>
                {lim.purpose}
              </div>
              <div className="text-xs text-amber-700 dark:text-amber-400">
                <span className="font-medium">Limitation: </span>
                {lim.limitation}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
