import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  ShieldAlert,
  Play,
  Download,
  FileText,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Sun,
  Moon,
  Database,
} from 'lucide-react';
import {
  DatasetRecord,
  ExportHistoryRecord,
  GenerationRunRecord,
  MLModelType,
} from '../types/synthforge';

// ---------------------------------------------------------------------------
// 6. PRIVACY & DISCLOSURE RISK ASSESSMENT VIEW
// ---------------------------------------------------------------------------
export const PrivacyAssessmentView: React.FC<{ activeRun: GenerationRunRecord }> = ({
  activeRun,
}) => {
  const { privacyReport } = activeRun;

  return (
    <div className="space-y-8">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Privacy & Disclosure Risk Assessment
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Empirical disclosure auditing via exact record overlap, Gower nearest-neighbour distances (DCR), Nearest-Neighbour Distance Ratio (NNDR), and controlled Membership-Inference Attack (MIA) simulation.
          </p>
        </div>
        <div className="text-right font-mono tabular-nums">
          <div className="text-xs text-slate-500 dark:text-slate-400">Empirical Risk Indicator</div>
          <div
            className={`text-2xl font-bold ${
              privacyReport.riskLevel === 'Low'
                ? 'text-emerald-600 dark:text-emerald-400'
                : privacyReport.riskLevel === 'Moderate'
                ? 'text-amber-600 dark:text-amber-400'
                : 'text-red-600 dark:text-red-400'
            }`}
          >
            {privacyReport.riskLevel} ({privacyReport.overallRiskScore}/100)
          </div>
        </div>
      </div>

      {/* Explicit Non-Differential Privacy Scientific Disclaimer */}
      <div className="p-4 bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 dark:text-amber-200 space-y-1 leading-relaxed">
          <div className="font-semibold">
            Scientific Privacy Notice: Empirical Risk Indicators vs Formal Differential Privacy
          </div>
          <p>
            The metrics below (Exact Overlap, Distance to Closest Record, NNDR, and simulated Membership-Inference ROC-AUC) are empirical post-hoc disclosure risk indicators. They do <strong>NOT</strong> constitute a formal mathematical (ε, δ)-Differential Privacy guarantee, as the synthesizer gradients were not clipped or Perturbed via DP-SGD / PATE mechanisms.
          </p>
        </div>
      </div>

      {/* 6 Privacy KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Exact Record Overlap</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {(privacyReport.exactOverlapRate * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            {privacyReport.exactOverlapCount} identical rows
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Near-Duplicate Rate</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {(privacyReport.nearDuplicateRate * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Gower dist &lt; 0.045
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">5th Pct DCR (Synth)</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {privacyReport.dcr5thPercentile}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Holdout DCR₅: {privacyReport.holdoutDcr5thPercentile}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Mean NNDR (d₁ / d₂)</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {privacyReport.nndrMean}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            5th Pct NNDR: {privacyReport.nndr5thPercentile}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">MIA Attack ROC-AUC</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-blue-600 dark:text-blue-400">
            {privacyReport.miaSimulatedAuc}
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            0.50 = Random Guess
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
          <div className="text-xs text-slate-500 dark:text-slate-400">Attacker Advantage</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {(privacyReport.miaAttackerAdvantage * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 font-mono tabular-nums">
            Prec@10%FPR: {(privacyReport.miaPrecisionAt10Fpr * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* DCR Distribution Chart + Nearest-Neighbour Memorisation Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Distance to Closest Record (DCR): Synthetic-to-Train vs Holdout-to-Train
            </h3>
            <p className="text-xs text-slate-500">
              If Synthetic-to-Train distances are substantially smaller than Real Holdout-to-Train distances, the generator is memorizing training points.
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={privacyReport.dcrHistogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="distanceBin" tick={{ fontSize: 10 }} />
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
                <Bar
                  dataKey="synthToRealPct"
                  name="Synthetic → Real Train (%)"
                  fill="#3b82f6"
                  radius={[3, 3, 0, 0]}
                />
                <Bar
                  dataKey="holdoutToRealPct"
                  name="Real Holdout → Real Train (%)"
                  fill="#64748b"
                  radius={[3, 3, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Closest Synthetic-to-Source Record Pairs (Memorisation Inspector)
            </h3>
            <p className="text-xs text-slate-500">
              Pairs with low Gower distance (d₁ &lt; 0.055) and low NNDR (d₁/d₂ &lt; 0.45) indicate potential local memorisation of isolated outliers.
            </p>
          </div>
          <div className="overflow-y-auto max-h-64">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                  <th className="py-2 px-2">Synth Row #</th>
                  <th className="py-2 px-2">Closest Real Row #</th>
                  <th className="py-2 px-2 text-right">Gower Dist (d₁)</th>
                  <th className="py-2 px-2 text-right">NNDR (d₁/d₂)</th>
                  <th className="py-2 px-2 text-right">Audit Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
                {privacyReport.nearestNeighbourPairs.map((pair, idx) => (
                  <tr key={idx}>
                    <td className="py-2 px-2">Synth #{pair.synthRowIndex}</td>
                    <td className="py-2 px-2">Source #{pair.closestRealRowIndex}</td>
                    <td className="py-2 px-2 text-right">{pair.normalizedDistance}</td>
                    <td className="py-2 px-2 text-right">{pair.nndr}</td>
                    <td
                      className={`py-2 px-2 text-right font-sans font-medium ${
                        pair.status === 'Memorisation Warning'
                          ? 'text-red-500'
                          : pair.status === 'Close'
                          ? 'text-amber-500'
                          : 'text-emerald-500'
                      }`}
                    >
                      {pair.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 7. DOWNSTREAM ML UTILITY BENCHMARKING VIEW
// ---------------------------------------------------------------------------
interface MLBenchmarkViewProps {
  activeDataset: DatasetRecord;
  activeRun: GenerationRunRecord;
  onRunMLBenchmark: (targetCol: string, models: MLModelType[]) => void;
}

export const MLUtilityBenchmarkView: React.FC<MLBenchmarkViewProps> = ({
  activeDataset,
  activeRun,
  onRunMLBenchmark,
}) => {
  const [targetCol, setTargetCol] = useState<string>(
    activeRun.mlReport?.targetColumn || activeDataset.targetColumn
  );
  const [selectedModels, setSelectedModels] = useState<MLModelType[]>([
    'logistic_regression',
    'random_forest',
    'xgboost',
    'gradient_boosting',
    'decision_tree',
  ]);
  const [focusedModelIdx, setFocusedModelIdx] = useState<number>(0);

  const mlReport = activeRun.mlReport;
  const isClass = (mlReport?.taskType || 'classification') === 'classification';
  const focusedModel = mlReport?.models[focusedModelIdx] || mlReport?.models[0];

  const allModelOptions: { id: MLModelType; label: string }[] = [
    { id: 'logistic_regression', label: 'Logistic / Ridge Regression' },
    { id: 'random_forest', label: 'Random Forest' },
    { id: 'xgboost', label: 'XGBoost (Regularized GBDT)' },
    { id: 'gradient_boosting', label: 'Gradient Boosting' },
    { id: 'decision_tree', label: 'Decision Tree (CART)' },
  ];

  const toggleModel = (id: MLModelType) => {
    setSelectedModels((prev) =>
      prev.includes(id)
        ? prev.length > 1
          ? prev.filter((m) => m !== id)
          : prev
        : [...prev, id]
    );
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-5 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Downstream ML Utility Benchmarking (TRTR vs TSTR vs Augmented)
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Zero-leakage experimental design: Real data is partitioned into 75% Real Train and 25% Held-Out Real Test prior to evaluation. All three experiments are evaluated strictly on the held-out real test split.
          </p>
        </div>
        <button
          onClick={() => onRunMLBenchmark(targetCol, selectedModels)}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors whitespace-nowrap shrink-0"
        >
          <Play className="w-3.5 h-3.5" />
          Execute 3-Experiment ML Benchmark
        </button>
      </div>

      {/* Experiment Configuration Bar */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap">
              Target Variable:
            </label>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="px-3 py-1.5 text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg"
            >
              {activeDataset.profile.columns
                .filter((c) => c.type !== 'identifier' && c.type !== 'datetime')
                .map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.type === 'categorical' ? 'Classification' : 'Regression / Binary'})
                  </option>
                ))}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {allModelOptions.map((m) => {
              const checked = selectedModels.includes(m.id);
              return (
                <label
                  key={m.id}
                  className={`cursor-pointer inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border ${
                    checked
                      ? 'bg-blue-50 dark:bg-blue-950/40 border-blue-500 text-blue-900 dark:text-blue-200'
                      : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleModel(m.id)}
                  />
                  <span>{m.label}</span>
                </label>
              );
            })}
          </div>
        </div>

        {mlReport && (
          <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 font-mono tabular-nums">
            <span>
              Task: {mlReport.taskType.toUpperCase()} · Real Train N={mlReport.realTrainSize} · Synthetic Train N={mlReport.syntheticTrainSize} · Augmented Train N={mlReport.augmentedTrainSize}
            </span>
            <span className="text-emerald-600 dark:text-emerald-400 font-sans font-medium">
              Zero Data Leakage Verified (Held-Out Real Test N={mlReport.realHoldoutTestSize})
            </span>
          </div>
        )}
      </div>

      {mlReport && (
        <>
          {/* Verdict Banner */}
          <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="space-y-1">
              <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Predictive Utility Verdict (Target: {mlReport.targetColumn})
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300">
                {mlReport.summaryVerdict}
              </p>
            </div>
            <div className="flex items-center gap-6 shrink-0 font-mono tabular-nums">
              <div>
                <div className="text-xs text-slate-500">Avg TSTR Retention</div>
                <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                  {mlReport.avgTstrRetentionPct}%
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Avg Augmentation Delta</div>
                <div className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                  {mlReport.avgAugmentationDeltaPct >= 0 ? '+' : ''}
                  {mlReport.avgAugmentationDeltaPct}%
                </div>
              </div>
            </div>
          </div>

          {/* Side-by-Side Model Comparison Table across Experiments A, B, and C */}
          <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Side-by-Side Experiment Comparison on Held-Out Real Test Data
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                    <th className="py-2.5 px-3">Model Architecture</th>
                    {isClass ? (
                      <>
                        <th className="py-2.5 px-3 text-right">Exp A (Real) Acc / F1 / AUC</th>
                        <th className="py-2.5 px-3 text-right">Exp B (Synth) Acc / F1 / AUC</th>
                        <th className="py-2.5 px-3 text-right">Exp C (Aug) Acc / F1 / AUC</th>
                        <th className="py-2.5 px-3 text-right">Exp B Prec / Rec</th>
                      </>
                    ) : (
                      <>
                        <th className="py-2.5 px-3 text-right">Exp A (Real) MAE / RMSE / R²</th>
                        <th className="py-2.5 px-3 text-right">Exp B (Synth) MAE / RMSE / R²</th>
                        <th className="py-2.5 px-3 text-right">Exp C (Aug) MAE / RMSE / R²</th>
                      </>
                    )}
                    <th className="py-2.5 px-3 text-right">TSTR Parity</th>
                    <th className="py-2.5 px-3 text-right">Augmentation Gain</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
                  {mlReport.models.map((m, idx) => (
                    <tr
                      key={m.modelId}
                      onClick={() => setFocusedModelIdx(idx)}
                      className={`cursor-pointer ${
                        focusedModelIdx === idx
                          ? 'bg-blue-50/50 dark:bg-blue-950/30'
                          : 'hover:bg-slate-50/70 dark:hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-sans font-medium text-slate-900 dark:text-slate-100">
                        {m.modelName}
                      </td>
                      {isClass ? (
                        <>
                          <td className="py-2.5 px-3 text-right">
                            {m.experimentA_Real.accuracy} / {m.experimentA_Real.f1Score} / {m.experimentA_Real.rocAuc}
                          </td>
                          <td className="py-2.5 px-3 text-right text-blue-600 dark:text-blue-400 font-semibold">
                            {m.experimentB_Synthetic.accuracy} / {m.experimentB_Synthetic.f1Score} / {m.experimentB_Synthetic.rocAuc}
                          </td>
                          <td className="py-2.5 px-3 text-right text-emerald-600 dark:text-emerald-400 font-semibold">
                            {m.experimentC_Augmented.accuracy} / {m.experimentC_Augmented.f1Score} / {m.experimentC_Augmented.rocAuc}
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-500">
                            {m.experimentB_Synthetic.precision} / {m.experimentB_Synthetic.recall}
                          </td>
                        </>
                      ) : (
                        <>
                          <td className="py-2.5 px-3 text-right">
                            {m.experimentA_Real.mae} / {m.experimentA_Real.rmse} / {m.experimentA_Real.r2}
                          </td>
                          <td className="py-2.5 px-3 text-right text-blue-600 dark:text-blue-400 font-semibold">
                            {m.experimentB_Synthetic.mae} / {m.experimentB_Synthetic.rmse} / {m.experimentB_Synthetic.r2}
                          </td>
                          <td className="py-2.5 px-3 text-right text-emerald-600 dark:text-emerald-400 font-semibold">
                            {m.experimentC_Augmented.mae} / {m.experimentC_Augmented.rmse} / {m.experimentC_Augmented.r2}
                          </td>
                        </>
                      )}
                      <td className="py-2.5 px-3 text-right font-bold">
                        {m.utilityRetentionPct}%
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right font-semibold ${
                          m.augmentationGainPct >= 0 ? 'text-emerald-500' : 'text-amber-500'
                        }`}
                      >
                        {m.augmentationGainPct >= 0 ? '+' : ''}
                        {m.augmentationGainPct}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Confusion Matrix & Feature Importance Alignment for Focused Model */}
          {focusedModel && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    Feature Importance Alignment ({focusedModel.modelName})
                  </h3>
                  <span className="text-xs text-slate-500">Real Model vs Synthetic Model (%)</span>
                </div>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={focusedModel.featureImportances}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                      <XAxis type="number" unit="%" tick={{ fontSize: 10 }} />
                      <YAxis dataKey="feature" type="category" width={110} tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0f172a',
                          borderColor: '#334155',
                          color: '#f8fafc',
                          fontSize: '12px',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px' }} />
                      <Bar dataKey="realImp" name="Exp A (Real) Imp %" fill="#64748b" />
                      <Bar dataKey="synthImp" name="Exp B (Synth) Imp %" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Confusion Matrices on Held-Out Real Test Split ({focusedModel.modelName})
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono tabular-nums text-xs">
                  {[
                    { title: 'Exp A: Train Real', cm: focusedModel.experimentA_Real.confusionMatrix },
                    { title: 'Exp B: Train Synth', cm: focusedModel.experimentB_Synthetic.confusionMatrix },
                    { title: 'Exp C: Augmented', cm: focusedModel.experimentC_Augmented.confusionMatrix },
                  ].map((item) => (
                    <div
                      key={item.title}
                      className="p-3 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 rounded-lg space-y-2"
                    >
                      <div className="font-sans font-semibold text-slate-800 dark:text-slate-200">
                        {item.title}
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-center">
                        <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded">
                          <div className="text-[10px] text-slate-500">True Pos (TP)</div>
                          <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                            {item.cm.tp}
                          </div>
                        </div>
                        <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded">
                          <div className="text-[10px] text-slate-500">False Pos (FP)</div>
                          <div className="text-sm font-bold text-amber-600 dark:text-amber-400">
                            {item.cm.fp}
                          </div>
                        </div>
                        <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded">
                          <div className="text-[10px] text-slate-500">False Neg (FN)</div>
                          <div className="text-sm font-bold text-amber-600 dark:text-amber-400">
                            {item.cm.fn}
                          </div>
                        </div>
                        <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded">
                          <div className="text-[10px] text-slate-500">True Neg (TN)</div>
                          <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                            {item.cm.tn}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// 8. GENERATION HISTORY & RUN COMPARISON VIEW
// ---------------------------------------------------------------------------
interface HistoryViewProps {
  runs: GenerationRunRecord[];
  activeRun: GenerationRunRecord;
  onSelectRun: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
}

export const GenerationHistoryView: React.FC<HistoryViewProps> = ({
  runs,
  activeRun,
  onSelectRun,
  onDeleteRun,
}) => {
  const [compareIds, setCompareIds] = useState<string[]>(
    runs.slice(0, 3).map((r) => r.id)
  );

  const toggleCompare = (id: string) => {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const comparedRuns = runs.filter((r) => compareIds.includes(r.id));

  return (
    <div className="space-y-8">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
          Generation History & Multi-Run Comparison
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Inspect historical synthetic generation runs persisted in your workspace and compare statistical fidelity, privacy risk, and ML utility across algorithms.
        </p>
      </div>

      {/* Run History Table */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Executed Generation Runs ({runs.length})
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                <th className="py-2.5 px-3 font-sans">Compare</th>
                <th className="py-2.5 px-3 font-sans">Run ID & Timestamp</th>
                <th className="py-2.5 px-3 font-sans">Dataset</th>
                <th className="py-2.5 px-3 font-sans">Synthesizer</th>
                <th className="py-2.5 px-3 text-right">Rows / Seed</th>
                <th className="py-2.5 px-3 text-right">Quality Score</th>
                <th className="py-2.5 px-3 text-right">Privacy Risk</th>
                <th className="py-2.5 px-3 text-right">TSTR Utility</th>
                <th className="py-2.5 px-3 text-right font-sans">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {runs.map((r) => {
                const isCurrent = r.id === activeRun.id;
                return (
                  <tr
                    key={r.id}
                    className={
                      isCurrent
                        ? 'bg-blue-50/50 dark:bg-blue-950/30'
                        : 'hover:bg-slate-50/70 dark:hover:bg-slate-800/40'
                    }
                  >
                    <td className="py-2.5 px-3">
                      <input
                        type="checkbox"
                        checked={compareIds.includes(r.id)}
                        onChange={() => toggleCompare(r.id)}
                      />
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="font-medium text-slate-900 dark:text-slate-100">
                        {r.id}
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {r.createdAt.slice(0, 19).replace('T', ' ')}
                      </div>
                    </td>
                    <td className="py-2.5 px-3 font-sans">{r.datasetName}</td>
                    <td className="py-2.5 px-3 uppercase">{r.config.method}</td>
                    <td className="py-2.5 px-3 text-right">
                      {r.syntheticRows.length} (s={r.config.randomSeed})
                    </td>
                    <td className="py-2.5 px-3 text-right font-semibold text-emerald-600 dark:text-emerald-400">
                      {r.qualityReport.overallScore}%
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {r.privacyReport.riskLevel} ({r.privacyReport.overallRiskScore})
                    </td>
                    <td className="py-2.5 px-3 text-right font-semibold text-blue-600 dark:text-blue-400">
                      {r.mlReport ? `${r.mlReport.avgTstrRetentionPct}%` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <div className="inline-flex items-center gap-2">
                        <button
                          onClick={() => onSelectRun(r.id)}
                          className="px-2.5 py-1 text-xs font-sans font-medium bg-slate-100 dark:bg-slate-800 rounded hover:bg-blue-600 hover:text-white transition-colors"
                        >
                          {isCurrent ? 'Active' : 'Inspect'}
                        </button>
                        {runs.length > 1 && (
                          <button
                            onClick={() => onDeleteRun(r.id)}
                            className="text-slate-400 hover:text-red-500"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Multi-Run Side-by-Side Comparison Chart */}
      {comparedRuns.length > 0 && (
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-4">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Selected Run Trade-off Comparison (Quality Fidelity vs Privacy Safety vs ML Utility)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparedRuns.map((r) => ({
                  name: `${r.config.method} (${r.id.slice(-4)})`,
                  QualityScore: r.qualityReport.overallScore,
                  PrivacySafety: Number((100 - r.privacyReport.overallRiskScore).toFixed(1)),
                  MLUtility: r.mlReport?.avgTstrRetentionPct || 0,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 105]} unit="%" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="QualityScore" name="Statistical Quality (%)" fill="#10b981" />
                <Bar dataKey="PrivacySafety" name="Privacy Safety (100 - Risk)" fill="#f59e0b" />
                <Bar dataKey="MLUtility" name="Downstream TSTR Utility (%)" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// 9. REPORTS & EXPORT VIEW
// ---------------------------------------------------------------------------
interface ExportViewProps {
  activeDataset: DatasetRecord;
  activeRun: GenerationRunRecord;
  exportHistory: ExportHistoryRecord[];
  onExportAction: (type: 'csv' | 'json' | 'pdf_report' | 'schema_json') => void;
}

export const ReportsAndExportView: React.FC<ExportViewProps> = ({
  activeDataset,
  activeRun,
  exportHistory,
  onExportAction,
}) => {
  return (
    <div className="space-y-8">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
          Dataset Export, Schema Metadata & PDF Audit Reports
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Download synthesized tabular records in CSV or JSON format, export column schema definitions, or generate a formatted PDF evaluation report for {activeRun.datasetName}.
        </p>
      </div>

      {/* 4 Primary Export Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between space-y-4">
          <div className="space-y-1.5">
            <div className="text-xs font-mono text-blue-600 dark:text-blue-400">
              TABULAR ARTIFACT · CSV
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Synthetic Dataset (CSV)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Download all {activeRun.syntheticRows.length} constraint-validated synthetic rows ready for pandas / scikit-learn ingestion.
            </p>
          </div>
          <button
            onClick={() => onExportAction('csv')}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download CSV ({activeRun.syntheticRows.length} rows)
          </button>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between space-y-4">
          <div className="space-y-1.5">
            <div className="text-xs font-mono text-emerald-600 dark:text-emerald-400">
              TABULAR ARTIFACT · JSON
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Synthetic Dataset (JSON)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Export structured JSON array of synthesized records along with generation configuration parameters.
            </p>
          </div>
          <button
            onClick={() => onExportAction('json')}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold text-slate-900 dark:text-slate-100 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download JSON Records
          </button>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between space-y-4">
          <div className="space-y-1.5">
            <div className="text-xs font-mono text-amber-600 dark:text-amber-400">
              COMPLIANCE AUDIT · PDF
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Evaluation & Privacy Report (PDF)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Generate a multi-table PDF document containing KS/Chi-Square tests, DCR privacy metrics, and TSTR ML benchmark tables.
            </p>
          </div>
          <button
            onClick={() => onExportAction('pdf_report')}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-emerald-600 rounded-lg hover:bg-emerald-500 transition-colors"
          >
            <FileText className="w-3.5 h-3.5" />
            Download Audit Report PDF
          </button>
        </div>

        <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col justify-between space-y-4">
          <div className="space-y-1.5">
            <div className="text-xs font-mono text-slate-500">
              METADATA & SCHEMA · JSON
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Schema & Constraints Spec
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Export inferred column types, empirical quantiles, correlation matrix, and custom constraint definitions for {activeDataset.name}.
            </p>
          </div>
          <button
            onClick={() => onExportAction('schema_json')}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold text-slate-900 dark:text-slate-100 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Export Schema & Metadata
          </button>
        </div>
      </div>

      {/* Export History Audit Table */}
      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-3">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Export Audit History ({exportHistory.length})
        </h2>
        {exportHistory.length === 0 ? (
          <p className="text-xs text-slate-500 py-4">
            No exports downloaded yet in this session. Click any export button above to download artifacts and log the event.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                  <th className="py-2 px-3">Export ID</th>
                  <th className="py-2 px-3">Run ID</th>
                  <th className="py-2 px-3">Format</th>
                  <th className="py-2 px-3">File Name</th>
                  <th className="py-2 px-3 text-right">Records</th>
                  <th className="py-2 px-3 text-right">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
                {exportHistory.map((ex) => (
                  <tr key={ex.id}>
                    <td className="py-2 px-3">{ex.id}</td>
                    <td className="py-2 px-3">{ex.runId}</td>
                    <td className="py-2 px-3 uppercase">{ex.exportType}</td>
                    <td className="py-2 px-3 text-slate-900 dark:text-slate-100">{ex.fileName}</td>
                    <td className="py-2 px-3 text-right">{ex.rowCount}</td>
                    <td className="py-2 px-3 text-right text-slate-500">
                      {ex.createdAt.slice(0, 19).replace('T', ' ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 10. SETTINGS VIEW
// ---------------------------------------------------------------------------
interface SettingsViewProps {
  isDark: boolean;
  onToggleTheme: () => void;
  userEmail: string | null;
  onSignIn: () => void;
  onSignOut: () => void;
  defaultSeed: number;
  onChangeDefaultSeed: (s: number) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  isDark,
  onToggleTheme,
  userEmail,
  onSignIn,
  onSignOut,
  defaultSeed,
  onChangeDefaultSeed,
}) => {
  return (
    <div className="space-y-8 max-w-3xl">
      <div className="pb-5 border-b border-slate-200 dark:border-slate-800">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
          Platform Settings & Persistence Configuration
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Configure workspace theme preferences, deterministic PRNG defaults, and cloud database synchronization.
        </p>
      </div>

      <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Interface Color Theme
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Switch between high-contrast Dark Slate laboratory mode and crisp Light mode.
            </div>
          </div>
          <button
            onClick={onToggleTheme}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
            <span>{isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}</span>
          </button>
        </div>

        <div className="pt-5 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Default Reproducibility Seed
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Used for Mulberry32 / Box-Muller Gaussian sampling and train-test holdout splits.
            </div>
          </div>
          <input
            type="number"
            value={defaultSeed}
            onChange={(e) => onChangeDefaultSeed(Number(e.target.value))}
            className="w-28 px-3 py-1.5 text-xs font-mono tabular-nums bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-right"
          />
        </div>

        <div className="pt-5 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-start gap-3">
            <Database className="w-5 h-5 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
            <div>
              <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Cloud Database Persistence (Firebase Firestore & Auth)
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                {userEmail
                  ? `Signed in as ${userEmail}. All 12 entity collections (datasets, schemas, profiling reports, generation configs, runs, quality/privacy assessments, ML benchmarks, and exports) automatically sync to Firestore.`
                  : 'Sign in with Google to persist uploaded datasets, generation runs, ML benchmarks, and export history across sessions.'}
              </div>
            </div>
          </div>
          {userEmail ? (
            <button
              onClick={onSignOut}
              className="px-4 py-2 text-xs font-medium text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/40 whitespace-nowrap shrink-0"
            >
              Sign Out
            </button>
          ) : (
            <button
              onClick={onSignIn}
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 whitespace-nowrap shrink-0"
            >
              Sign In with Google
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
