/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import { onAuthStateChanged, User } from 'firebase/auth';
import {
  LayoutDashboard,
  Database,
  Activity,
  Cpu,
  CheckSquare,
  Shield,
  BarChart3,
  History,
  FileDown,
  Settings,
  LogIn,
  LogOut,
  Play,
} from 'lucide-react';
import { BUILTIN_SAMPLE_DATASETS } from './data/sampleDatasets';
import { profileDataset } from './engine/profiler';
import { generateSyntheticDataset } from './engine/generator';
import { evaluateSyntheticQuality } from './engine/evaluator';
import { evaluatePrivacyRisk } from './engine/privacy';
import { runDownstreamMLBenchmark } from './engine/mlBenchmark';
import {
  parseUploadedDatasetFile,
  downloadRowsAsCsv,
  downloadObjectAsJson,
  downloadEvaluationReportPdf,
} from './utils/exporters';
import {
  auth,
  signInWithGoogle,
  signOutUser,
  persistDatasetToFirestore,
  persistGenerationConfigToFirestore,
  persistGenerationRunToFirestore,
  persistMLBenchmarkToFirestore,
  persistExportEventToFirestore,
  loadUserWorkspaceFromFirestore,
  deleteRunFromFirestore,
} from './lib/firebase';
import {
  DatasetRecord,
  ExportHistoryRecord,
  GenerationConfigParams,
  GenerationRunRecord,
  MLModelType,
  TransformationStep,
} from './types/synthforge';
import {
  OverviewView,
  DatasetExplorerView,
  DataQualityProfilerView,
  SyntheticGeneratorView,
  QualityEvaluationView,
} from './components/PipelineViewsPart1';
import {
  PrivacyAssessmentView,
  MLUtilityBenchmarkView,
  GenerationHistoryView,
  ReportsAndExportView,
  SettingsView,
} from './components/PipelineViewsPart2';

function createInitialRunForDataset(
  ds: DatasetRecord,
  method: GenerationConfigParams['method'] = 'gaussian_copula',
  seed = 42
): GenerationRunRecord {
  const config: GenerationConfigParams = {
    id: `cfg_init_${ds.id}`,
    name: `${ds.name} (${method})`,
    datasetId: ds.id,
    method,
    numRows: ds.rowCount,
    selectedColumns: ds.profile.columns.map((c) => c.name),
    randomSeed: seed,
    missingValueMode: 'preserve_rate',
    epochs: 30,
    constraints: ds.defaultConstraints,
  };

  const genOut = generateSyntheticDataset(ds.rows, config);
  const qualityReport = evaluateSyntheticQuality(
    ds.rows,
    genOut.syntheticRows,
    config.constraints,
    genOut.rawViolationCountBeforeProjection
  );
  const privacyReport = evaluatePrivacyRisk(ds.rows, genOut.syntheticRows);
  const runId = `run_${ds.id}_${method}_${seed}`;
  const mlReport = runDownstreamMLBenchmark(
    ds.rows,
    genOut.syntheticRows,
    ds.targetColumn,
    ds.id,
    runId
  );

  return {
    id: runId,
    ownerId: 'system',
    datasetId: ds.id,
    datasetName: ds.name,
    config,
    durationMs: genOut.durationMs,
    status: 'completed',
    syntheticRows: genOut.syntheticRows,
    qualityReport,
    privacyReport,
    mlReport,
    epochLogs: genOut.epochLogs,
    createdAt: new Date().toISOString(),
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [isDark, setIsDark] = useState<boolean>(true);
  const [defaultSeed, setDefaultSeed] = useState<number>(42);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [statusToast, setStatusToast] = useState<string | null>(null);

  // Datasets & Original backup for curation reset
  const [datasets, setDatasets] = useState<DatasetRecord[]>(() => BUILTIN_SAMPLE_DATASETS);
  const [activeDatasetId, setActiveDatasetId] = useState<string>(
    BUILTIN_SAMPLE_DATASETS[0].id
  );

  // Initial Runs (pre-computed across Gaussian Copula & CTGAN on initial load)
  const initialRuns = useMemo(() => {
    const firstDs = BUILTIN_SAMPLE_DATASETS[0];
    const r1 = createInitialRunForDataset(firstDs, 'gaussian_copula', 42);
    const r2 = createInitialRunForDataset(firstDs, 'ctgan', 108);
    return [r1, r2];
  }, []);

  const [runs, setRuns] = useState<GenerationRunRecord[]>(initialRuns);
  const [activeRunId, setActiveRunId] = useState<string>(initialRuns[0].id);
  const [savedConfigs, setSavedConfigs] = useState<GenerationConfigParams[]>([
    initialRuns[0].config,
  ]);
  const [exportHistory, setExportHistory] = useState<ExportHistoryRecord[]>([]);

  const activeDataset = useMemo(
    () => datasets.find((d) => d.id === activeDatasetId) || datasets[0],
    [datasets, activeDatasetId]
  );

  const activeRun = useMemo(
    () => runs.find((r) => r.id === activeRunId) || runs[0],
    [runs, activeRunId]
  );

  const showNotice = (msg: string) => {
    setStatusToast(msg);
    setTimeout(() => setStatusToast(null), 4000);
  };

  // Sync dark class on <html>
  useEffect(() => {
    const rootEl = document.documentElement;
    if (isDark) {
      rootEl.classList.add('dark');
    } else {
      rootEl.classList.remove('dark');
    }
  }, [isDark]);

  // Listen to Firebase Auth state & load persisted workspace
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        try {
          const loaded = await loadUserWorkspaceFromFirestore(user.uid);
          if (loaded.runs.length > 0) {
            setRuns((prev) => {
              const existingIds = new Set(prev.map((r) => r.id));
              const merged = [
                ...loaded.runs.filter((r) => !existingIds.has(r.id)),
                ...prev,
              ];
              return merged;
            });
          }
          if (loaded.exports.length > 0) {
            setExportHistory(loaded.exports);
          }
          if (loaded.configs.length > 0) {
            setSavedConfigs((prev) => [...loaded.configs, ...prev]);
          }
        } catch (e) {
          console.warn('Firestore workspace sync notice:', e);
        }
      }
    });
    return () => unsub();
  }, []);

  const handleSelectDataset = (dsId: string) => {
    setActiveDatasetId(dsId);
    const ds = datasets.find((d) => d.id === dsId);
    if (!ds) return;
    const existingRunForDs = runs.find((r) => r.datasetId === dsId);
    if (existingRunForDs) {
      setActiveRunId(existingRunForDs.id);
    } else {
      const freshRun = createInitialRunForDataset(ds, 'gaussian_copula', defaultSeed);
      setRuns((prev) => [freshRun, ...prev]);
      setActiveRunId(freshRun.id);
    }
  };

  const handleUploadFile = async (file: File) => {
    const { rows, sourceFormat } = await parseUploadedDatasetFile(file);
    const id = `ds_upload_${Date.now()}`;
    const cols = Object.keys(rows[0] || {});
    const targetColumn = cols[cols.length - 1] || 'target';
    const profile = profileDataset(id, rows, targetColumn);

    const newDs: DatasetRecord = {
      id,
      ownerId: currentUser?.uid || 'local',
      name: file.name.replace(/\.[^/.]+$/, ''),
      description: `User-uploaded ${sourceFormat.toUpperCase()} dataset (${rows.length} rows, ${cols.length} columns).`,
      sourceFormat,
      rowCount: rows.length,
      columnCount: cols.length,
      targetColumn,
      rows,
      profile,
      transformations: [],
      defaultConstraints: [],
      createdAt: new Date().toISOString(),
    };

    setDatasets((prev) => [newDs, ...prev]);
    setActiveDatasetId(newDs.id);

    const autoRun = createInitialRunForDataset(newDs, 'gaussian_copula', defaultSeed);
    setRuns((prev) => [autoRun, ...prev]);
    setActiveRunId(autoRun.id);
    showNotice(`Uploaded & profiled "${newDs.name}" (${newDs.rowCount} rows).`);

    if (currentUser) {
      await persistDatasetToFirestore(newDs);
      await persistGenerationRunToFirestore(autoRun);
    }
  };

  const handleUpdateDatasetRows = async (
    newRows: Record<string, any>[],
    step?: TransformationStep,
    newTargetCol?: string
  ) => {
    const targetCol = newTargetCol || activeDataset.targetColumn;
    const updatedProfile = profileDataset(activeDataset.id, newRows, targetCol);
    const updatedDs: DatasetRecord = {
      ...activeDataset,
      rowCount: newRows.length,
      columnCount: Object.keys(newRows[0] || {}).length,
      targetColumn: targetCol,
      rows: newRows,
      profile: updatedProfile,
      transformations: step
        ? [...activeDataset.transformations, step]
        : activeDataset.transformations,
    };

    setDatasets((prev) =>
      prev.map((d) => (d.id === updatedDs.id ? updatedDs : d))
    );
    showNotice(
      step
        ? `Applied transformation: ${step.description}`
        : `Updated target column to "${targetCol}".`
    );

    if (currentUser) {
      await persistDatasetToFirestore(updatedDs);
    }
  };

  const handleResetDataset = () => {
    const orig = BUILTIN_SAMPLE_DATASETS.find((d) => d.id === activeDataset.id);
    if (orig) {
      setDatasets((prev) => prev.map((d) => (d.id === orig.id ? orig : d)));
      showNotice(`Reverted all transformations on "${orig.name}".`);
    }
  };

  const handleExecuteGeneration = async (config: GenerationConfigParams) => {
    const genOut = generateSyntheticDataset(activeDataset.rows, config);
    const qualityReport = evaluateSyntheticQuality(
      activeDataset.rows,
      genOut.syntheticRows,
      config.constraints,
      genOut.rawViolationCountBeforeProjection
    );
    const privacyReport = evaluatePrivacyRisk(
      activeDataset.rows,
      genOut.syntheticRows
    );
    const runId = `run_${Date.now()}`;
    const mlReport = runDownstreamMLBenchmark(
      activeDataset.rows,
      genOut.syntheticRows,
      activeDataset.targetColumn,
      activeDataset.id,
      runId
    );

    const newRun: GenerationRunRecord = {
      id: runId,
      ownerId: currentUser?.uid || 'local',
      datasetId: activeDataset.id,
      datasetName: activeDataset.name,
      config,
      durationMs: genOut.durationMs,
      status: 'completed',
      syntheticRows: genOut.syntheticRows,
      qualityReport,
      privacyReport,
      mlReport,
      epochLogs: genOut.epochLogs,
      createdAt: new Date().toISOString(),
    };

    setRuns((prev) => [newRun, ...prev]);
    setActiveRunId(newRun.id);
    showNotice(
      `Generated ${newRun.syntheticRows.length} synthetic rows (${config.method.toUpperCase()}) · Quality ${qualityReport.overallScore}% · TSTR ${mlReport.avgTstrRetentionPct}%`
    );

    if (currentUser) {
      await persistGenerationRunToFirestore(newRun);
      await persistMLBenchmarkToFirestore(newRun.id, mlReport);
    }
  };

  const handleSaveConfig = async (config: GenerationConfigParams) => {
    setSavedConfigs((prev) => [config, ...prev]);
    showNotice(`Saved generation configuration "${config.name}".`);
    if (currentUser) {
      await persistGenerationConfigToFirestore(config);
    }
  };

  const handleRunMLBenchmark = async (
    targetCol: string,
    selectedModels: MLModelType[]
  ) => {
    const mlReport = runDownstreamMLBenchmark(
      activeDataset.rows,
      activeRun.syntheticRows,
      targetCol,
      activeDataset.id,
      activeRun.id,
      selectedModels
    );

    const updatedRun: GenerationRunRecord = {
      ...activeRun,
      mlReport,
    };

    setRuns((prev) =>
      prev.map((r) => (r.id === updatedRun.id ? updatedRun : r))
    );
    showNotice(
      `Completed 3-Experiment ML Benchmark on "${targetCol}" (Avg TSTR Parity: ${mlReport.avgTstrRetentionPct}%).`
    );

    if (currentUser) {
      await persistMLBenchmarkToFirestore(updatedRun.id, mlReport);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    setRuns((prev) => {
      const remaining = prev.filter((r) => r.id !== runId);
      if (activeRunId === runId && remaining.length > 0) {
        setActiveRunId(remaining[0].id);
      }
      return remaining;
    });
    if (currentUser) {
      await deleteRunFromFirestore(runId);
    }
  };

  const handleExportAction = async (
    exportType: 'csv' | 'json' | 'pdf_report' | 'schema_json'
  ) => {
    let fileName = '';
    if (exportType === 'csv') {
      fileName = `${activeRun.datasetName.replace(/\s+/g, '_')}_Synthetic_${activeRun.config.method}.csv`;
      downloadRowsAsCsv(activeRun.syntheticRows, fileName);
    } else if (exportType === 'json') {
      fileName = `${activeRun.datasetName.replace(/\s+/g, '_')}_Synthetic_${activeRun.config.method}.json`;
      downloadObjectAsJson(
        {
          runId: activeRun.id,
          dataset: activeRun.datasetName,
          config: activeRun.config,
          rows: activeRun.syntheticRows,
        },
        fileName
      );
    } else if (exportType === 'pdf_report') {
      fileName = downloadEvaluationReportPdf(activeRun, activeDataset);
    } else {
      fileName = `${activeDataset.name.replace(/\s+/g, '_')}_Schema_Metadata.json`;
      downloadObjectAsJson(
        {
          datasetId: activeDataset.id,
          name: activeDataset.name,
          rowCount: activeDataset.rowCount,
          columnCount: activeDataset.columnCount,
          targetColumn: activeDataset.targetColumn,
          columns: activeDataset.profile.columns,
          constraints: activeDataset.defaultConstraints,
          transformations: activeDataset.transformations,
        },
        fileName
      );
    }

    const record: ExportHistoryRecord = {
      id: `exp_${Date.now()}`,
      ownerId: currentUser?.uid || 'local',
      runId: activeRun.id,
      exportType,
      fileName,
      rowCount: activeRun.syntheticRows.length,
      createdAt: new Date().toISOString(),
    };
    setExportHistory((prev) => [record, ...prev]);
    showNotice(`Exported "${fileName}".`);

    if (currentUser) {
      await persistExportEventToFirestore(record);
    }
  };

  const navItems = [
    { id: 'overview', label: '01. Overview', icon: LayoutDashboard },
    { id: 'explorer', label: '02. Dataset Explorer', icon: Database },
    { id: 'profiler', label: '03. Data Quality Profiler', icon: Activity },
    { id: 'generator', label: '04. Synthetic Generator', icon: Cpu },
    { id: 'quality', label: '05. Quality Evaluation', icon: CheckSquare },
    { id: 'privacy', label: '06. Privacy Assessment', icon: Shield },
    { id: 'ml', label: '07. ML Utility Benchmark', icon: BarChart3 },
    { id: 'history', label: '08. Generation History', icon: History },
    { id: 'export', label: '09. Reports & Export', icon: FileDown },
    { id: 'settings', label: '10. Settings', icon: Settings },
  ];

  const currentNavLabel =
    navItems.find((n) => n.id === activeTab)?.label.replace(/^\d+\.\s*/, '') ||
    'Overview';

  return (
    <div className="min-h-screen flex bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      {/* Sidebar Navigation (260px Fixed Desktop Width) */}
      <aside className="w-64 shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col justify-between">
        <div>
          <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800">
            <div className="text-base font-bold tracking-tight font-display text-slate-900 dark:text-slate-100 whitespace-nowrap">
              SynthForge AI
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Synthetic Data & ML Intelligence
            </div>
          </div>

          <nav className="p-3 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-lg transition-colors whitespace-nowrap ${
                    active
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Active Workspace Summary Footer */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-2 text-xs">
          <div className="text-slate-500 dark:text-slate-400">Active Dataset</div>
          <div className="font-semibold text-slate-900 dark:text-slate-100 truncate">
            {activeDataset.name}
          </div>
          <div className="font-mono tabular-nums text-slate-500 dark:text-slate-400">
            {activeDataset.rowCount} rows · {activeRun.config.method}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar Contract: Breadcrumb on Left, Primary Actions on Right */}
        <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 truncate">
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              SynthForge AI
            </span>
            <span>/</span>
            <span>{currentNavLabel}</span>
            <span>/</span>
            <span className="font-mono text-slate-700 dark:text-slate-300 truncate">
              {activeDataset.name}
            </span>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => handleExecuteGeneration(activeRun.config)}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" />
              Synthesize ({activeRun.config.method})
            </button>

            {currentUser ? (
              <button
                onClick={signOutUser}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors whitespace-nowrap shrink-0"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>{currentUser.displayName || currentUser.email || 'Signed In'}</span>
              </button>
            ) : (
              <button
                onClick={signInWithGoogle}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors whitespace-nowrap shrink-0"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sync to Cloud</span>
              </button>
            )}
          </div>
        </header>

        {statusToast && (
          <div className="mx-6 mt-4 px-4 py-2.5 bg-emerald-600 text-white text-xs font-medium rounded-lg flex items-center justify-between">
            <span>{statusToast}</span>
            <button onClick={() => setStatusToast(null)} className="text-white/80 hover:text-white">
              Dismiss
            </button>
          </div>
        )}

        <main className="flex-1 p-6 max-w-[1440px] w-full mx-auto">
          {activeTab === 'overview' && (
            <OverviewView
              activeDataset={activeDataset}
              activeRun={activeRun}
              allRuns={runs}
              onNavigate={setActiveTab}
              onQuickGenerate={() => handleExecuteGeneration(activeRun.config)}
            />
          )}

          {activeTab === 'explorer' && (
            <DatasetExplorerView
              datasets={datasets}
              activeDataset={activeDataset}
              onSelectDataset={handleSelectDataset}
              onUploadFile={handleUploadFile}
              onUpdateDatasetRows={handleUpdateDatasetRows}
              onResetDataset={handleResetDataset}
            />
          )}

          {activeTab === 'profiler' && (
            <DataQualityProfilerView activeDataset={activeDataset} />
          )}

          {activeTab === 'generator' && (
            <SyntheticGeneratorView
              activeDataset={activeDataset}
              activeRun={activeRun}
              savedConfigs={savedConfigs}
              onExecuteGeneration={handleExecuteGeneration}
              onSaveConfig={handleSaveConfig}
            />
          )}

          {activeTab === 'quality' && (
            <QualityEvaluationView activeRun={activeRun} />
          )}

          {activeTab === 'privacy' && (
            <PrivacyAssessmentView activeRun={activeRun} />
          )}

          {activeTab === 'ml' && (
            <MLUtilityBenchmarkView
              activeDataset={activeDataset}
              activeRun={activeRun}
              onRunMLBenchmark={handleRunMLBenchmark}
            />
          )}

          {activeTab === 'history' && (
            <GenerationHistoryView
              runs={runs}
              activeRun={activeRun}
              onSelectRun={(id) => {
                setActiveRunId(id);
                setActiveTab('quality');
              }}
              onDeleteRun={handleDeleteRun}
            />
          )}

          {activeTab === 'export' && (
            <ReportsAndExportView
              activeDataset={activeDataset}
              activeRun={activeRun}
              exportHistory={exportHistory}
              onExportAction={handleExportAction}
            />
          )}

          {activeTab === 'settings' && (
            <SettingsView
              isDark={isDark}
              onToggleTheme={() => setIsDark((prev) => !prev)}
              userEmail={currentUser?.email || null}
              onSignIn={signInWithGoogle}
              onSignOut={signOutUser}
              defaultSeed={defaultSeed}
              onChangeDefaultSeed={setDefaultSeed}
            />
          )}
        </main>
      </div>
    </div>
  );
}
