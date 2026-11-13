import { initializeApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  signOut as fbSignOut,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  getDocFromServer,
  setDoc,
  getDocs,
  deleteDoc,
  collection,
  query,
  where,
  updateDoc,
} from 'firebase/firestore';
import firebaseConfig from '../../firebase-applet-config.json';
import {
  DatasetRecord,
  GenerationRunRecord,
  GenerationConfigParams,
  ExportHistoryRecord,
  MLBenchmarkReport,
} from '../types/synthforge';

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app, firebaseConfig.firestoreDatabaseId);
export const auth = getAuth(app);
export const googleAuthProvider = new GoogleAuthProvider();

// Mandatory connection validation on boot
async function testConnection() {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.error('Please check your Firebase configuration.');
    }
  }
}
testConnection();

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
    tenantId?: string | null;
    providerInfo?: {
      providerId?: string | null;
      email?: string | null;
    }[];
  };
}

export function handleFirestoreError(
  error: unknown,
  operationType: OperationType,
  path: string | null
): never {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo:
        auth.currentUser?.providerData?.map((provider) => ({
          providerId: provider.providerId,
          email: provider.email,
        })) || [],
    },
    operationType,
    path,
  };
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

// Validation synchronicity constants matching firebase-blueprint.json & firestore.rules
export const BLUEPRINT_LIMITS = {
  ID_MAX_LEN: 128,
  ID_PATTERN: /^[a-zA-Z0-9_\-]+$/,
  NAME_MAX_LEN: 160,
  DESC_MAX_LEN: 1000,
  SCHEMA_SUMMARY_MAX: 60000,
  SAMPLE_ROWS_MAX: 250000,
  TRANSFORMATIONS_MAX: 50000,
  SYNTH_ROWS_MAX: 300000,
  REPORT_JSON_MAX: 150000,
};

export function sanitizeId(rawId: string): string {
  const cleaned = rawId.replace(/[^a-zA-Z0-9_\-]/g, '_').slice(0, BLUEPRINT_LIMITS.ID_MAX_LEN);
  return cleaned.length > 0 ? cleaned : `id_${Date.now()}`;
}

function safeJsonTruncate(obj: any, maxLen: number): string {
  const str = JSON.stringify(obj);
  if (str.length <= maxLen) return str;
  if (Array.isArray(obj)) {
    const ratio = maxLen / str.length;
    const reducedCount = Math.max(5, Math.floor(obj.length * ratio * 0.85));
    return JSON.stringify(obj.slice(0, reducedCount));
  }
  return str.slice(0, maxLen);
}

export async function signInWithGoogle() {
  const result = await signInWithPopup(auth, googleAuthProvider);
  if (result.user) {
    await ensureUserDocument(result.user.uid, result.user.displayName || 'ML Engineer');
  }
  return result.user;
}

export async function signOutUser() {
  await fbSignOut(auth);
}

export async function ensureUserDocument(uid: string, displayName: string) {
  const cleanUid = sanitizeId(uid);
  const path = `users/${cleanUid}`;
  try {
    await setDoc(
      doc(db, 'users', cleanUid),
      {
        uid: cleanUid,
        displayName: (displayName || 'ML Engineer').slice(0, 120),
        role: 'engineer',
        createdAt: new Date().toISOString(),
      },
      { merge: false }
    );
  } catch (error) {
    // Ignore if already exists and immutable createdAt differs, or report if permission error
    console.debug('User profile sync info:', error);
  }
  return path;
}

export async function persistDatasetToFirestore(dataset: DatasetRecord) {
  if (!auth.currentUser) return;
  const uid = auth.currentUser.uid;
  const datasetId = sanitizeId(dataset.id);
  const createdAt = dataset.createdAt || new Date().toISOString();

  // 1. /datasets/{datasetId}
  const datasetPath = `datasets/${datasetId}`;
  try {
    await setDoc(doc(db, 'datasets', datasetId), {
      ownerId: uid,
      name: dataset.name.slice(0, BLUEPRINT_LIMITS.NAME_MAX_LEN),
      description: (dataset.description || '').slice(0, BLUEPRINT_LIMITS.DESC_MAX_LEN),
      sourceFormat: dataset.sourceFormat,
      rowCount: dataset.rowCount,
      columnCount: dataset.columnCount,
      targetColumn: (dataset.targetColumn || '').slice(0, 120),
      schemaSummaryJson: safeJsonTruncate(
        dataset.profile.columns.map((c) => ({
          name: c.name,
          type: c.type,
          missingRate: c.missingRate,
          uniqueCount: c.uniqueCount,
        })),
        BLUEPRINT_LIMITS.SCHEMA_SUMMARY_MAX
      ),
      sampleRowsJson: safeJsonTruncate(dataset.rows.slice(0, 250), BLUEPRINT_LIMITS.SAMPLE_ROWS_MAX),
      transformationsJson: safeJsonTruncate(
        dataset.transformations || [],
        BLUEPRINT_LIMITS.TRANSFORMATIONS_MAX
      ),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, datasetPath);
  }

  // 2. /dataset_schemas/{schemaId}
  const schemaId = sanitizeId(`schema_${datasetId}`);
  try {
    await setDoc(doc(db, 'dataset_schemas', schemaId), {
      ownerId: uid,
      datasetId,
      columnsJson: safeJsonTruncate(dataset.profile.columns, 100000),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `dataset_schemas/${schemaId}`);
  }

  // 3. /profiling_reports/{reportId}
  const reportId = sanitizeId(`prof_${datasetId}`);
  try {
    await setDoc(doc(db, 'profiling_reports', reportId), {
      ownerId: uid,
      datasetId,
      qualityScore: dataset.profile.qualityScore,
      missingRate: dataset.profile.missingRate,
      duplicateRate: dataset.profile.duplicateRate,
      outlierRate: dataset.profile.outlierRate,
      reportJson: safeJsonTruncate(dataset.profile, 200000),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `profiling_reports/${reportId}`);
  }
}

export async function persistGenerationConfigToFirestore(config: GenerationConfigParams) {
  if (!auth.currentUser) return;
  const uid = auth.currentUser.uid;
  const configId = sanitizeId(config.id || `cfg_${Date.now()}`);
  const createdAt = new Date().toISOString();
  try {
    await setDoc(doc(db, 'generation_configs', configId), {
      ownerId: uid,
      datasetId: sanitizeId(config.datasetId),
      name: config.name.slice(0, BLUEPRINT_LIMITS.NAME_MAX_LEN),
      method: config.method,
      numRows: Math.min(50000, Math.max(10, config.numRows)),
      randomSeed: config.randomSeed,
      configJson: safeJsonTruncate(config, 100000),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `generation_configs/${configId}`);
  }
}

export async function persistGenerationRunToFirestore(run: GenerationRunRecord) {
  if (!auth.currentUser) return;
  const uid = auth.currentUser.uid;
  const runId = sanitizeId(run.id);
  const datasetId = sanitizeId(run.datasetId);
  const createdAt = run.createdAt || new Date().toISOString();

  // 1. /generation_runs/{runId}
  try {
    const payload: Record<string, any> = {
      ownerId: uid,
      datasetId,
      datasetName: run.datasetName.slice(0, BLUEPRINT_LIMITS.NAME_MAX_LEN),
      method: run.config.method,
      numRows: run.syntheticRows.length,
      randomSeed: run.config.randomSeed,
      durationMs: Math.max(0, run.durationMs),
      qualityScore: run.qualityReport.overallScore,
      privacyRiskScore: run.privacyReport.overallRiskScore,
      constraintViolationRate: run.qualityReport.constraintViolationRate,
      status: run.status,
      syntheticRowsJson: safeJsonTruncate(
        run.syntheticRows.slice(0, 300),
        BLUEPRINT_LIMITS.SYNTH_ROWS_MAX
      ),
      qualityReportJson: safeJsonTruncate(run.qualityReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX),
      privacyReportJson: safeJsonTruncate(run.privacyReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX),
      createdAt,
    };
    if (run.mlReport) {
      payload.mlUtilityDelta = run.mlReport.avgTstrRetentionPct;
      payload.mlReportJson = safeJsonTruncate(run.mlReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX);
    }
    await setDoc(doc(db, 'generation_runs', runId), payload);
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `generation_runs/${runId}`);
  }

  // 2. /synthetic_datasets/{synthId}
  const synthId = sanitizeId(`synth_${runId}`);
  try {
    await setDoc(doc(db, 'synthetic_datasets', synthId), {
      ownerId: uid,
      runId,
      datasetId,
      name: `${run.datasetName} (${run.config.method})`.slice(0, BLUEPRINT_LIMITS.NAME_MAX_LEN),
      method: run.config.method,
      rowCount: run.syntheticRows.length,
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `synthetic_datasets/${synthId}`);
  }

  // 3. /quality_assessments/{assessmentId}
  const qaId = sanitizeId(`qa_${runId}`);
  try {
    await setDoc(doc(db, 'quality_assessments', qaId), {
      ownerId: uid,
      runId,
      overallScore: run.qualityReport.overallScore,
      constraintViolationRate: run.qualityReport.constraintViolationRate,
      metricsJson: safeJsonTruncate(run.qualityReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `quality_assessments/${qaId}`);
  }

  // 4. /privacy_assessments/{privacyId}
  const privId = sanitizeId(`priv_${runId}`);
  try {
    await setDoc(doc(db, 'privacy_assessments', privId), {
      ownerId: uid,
      runId,
      overallRiskScore: run.privacyReport.overallRiskScore,
      exactOverlapRate: run.privacyReport.exactOverlapRate,
      nearDuplicateRate: run.privacyReport.nearDuplicateRate,
      miaAuc: run.privacyReport.miaSimulatedAuc,
      metricsJson: safeJsonTruncate(run.privacyReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `privacy_assessments/${privId}`);
  }
}

export async function persistMLBenchmarkToFirestore(
  runId: string,
  mlReport: MLBenchmarkReport
) {
  if (!auth.currentUser) return;
  const uid = auth.currentUser.uid;
  const cleanRunId = sanitizeId(runId);
  const expId = sanitizeId(mlReport.id);
  const createdAt = mlReport.createdAt || new Date().toISOString();

  // 1. Update generation_runs with mlUtilityDelta & mlReportJson
  try {
    await updateDoc(doc(db, 'generation_runs', cleanRunId), {
      mlUtilityDelta: mlReport.avgTstrRetentionPct,
      mlReportJson: safeJsonTruncate(mlReport, BLUEPRINT_LIMITS.REPORT_JSON_MAX),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.UPDATE, `generation_runs/${cleanRunId}`);
  }

  // 2. /ml_experiments/{experimentId}
  try {
    await setDoc(doc(db, 'ml_experiments', expId), {
      ownerId: uid,
      runId: cleanRunId,
      datasetId: sanitizeId(mlReport.datasetId),
      targetColumn: mlReport.targetColumn.slice(0, 120),
      taskType: mlReport.taskType,
      resultsJson: safeJsonTruncate(mlReport, 180000),
      createdAt,
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `ml_experiments/${expId}`);
  }

  // 3. /model_metrics/{metricId} for each trained model
  for (const m of mlReport.models) {
    const metricId = sanitizeId(`mm_${expId}_${m.modelId}`);
    const isClass = mlReport.taskType === 'classification';
    try {
      await setDoc(doc(db, 'model_metrics', metricId), {
        ownerId: uid,
        experimentId: expId,
        modelName: m.modelName.slice(0, 80),
        primaryScoreReal: isClass ? m.experimentA_Real.f1Score : m.experimentA_Real.r2,
        primaryScoreSynthetic: isClass
          ? m.experimentB_Synthetic.f1Score
          : m.experimentB_Synthetic.r2,
        primaryScoreAugmented: isClass
          ? m.experimentC_Augmented.f1Score
          : m.experimentC_Augmented.r2,
        createdAt,
      });
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, `model_metrics/${metricId}`);
    }
  }
}

export async function persistExportEventToFirestore(record: ExportHistoryRecord) {
  if (!auth.currentUser) return;
  const uid = auth.currentUser.uid;
  const exportId = sanitizeId(record.id);
  try {
    await setDoc(doc(db, 'export_history', exportId), {
      ownerId: uid,
      runId: sanitizeId(record.runId),
      exportType: record.exportType,
      fileName: record.fileName.slice(0, 200),
      rowCount: record.rowCount,
      createdAt: record.createdAt || new Date().toISOString(),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, `export_history/${exportId}`);
  }
}

export async function loadUserWorkspaceFromFirestore(uid: string): Promise<{
  runs: GenerationRunRecord[];
  exports: ExportHistoryRecord[];
  configs: GenerationConfigParams[];
}> {
  const runs: GenerationRunRecord[] = [];
  const exports: ExportHistoryRecord[] = [];
  const configs: GenerationConfigParams[] = [];

  try {
    const runsQ = query(collection(db, 'generation_runs'), where('ownerId', '==', uid));
    const snap = await getDocs(runsQ);
    snap.forEach((docSnap) => {
      const d = docSnap.data();
      try {
        const syntheticRows = JSON.parse(d.syntheticRowsJson || '[]');
        const qualityReport = JSON.parse(d.qualityReportJson || '{}');
        const privacyReport = JSON.parse(d.privacyReportJson || '{}');
        const mlReport = d.mlReportJson ? JSON.parse(d.mlReportJson) : undefined;
        runs.push({
          id: docSnap.id,
          ownerId: d.ownerId,
          datasetId: d.datasetId,
          datasetName: d.datasetName,
          config: {
            name: `${d.datasetName} (${d.method})`,
            datasetId: d.datasetId,
            method: d.method,
            numRows: d.numRows,
            selectedColumns: Object.keys(syntheticRows[0] || {}),
            randomSeed: d.randomSeed,
            missingValueMode: 'preserve_rate',
            constraints: [],
          },
          durationMs: d.durationMs,
          status: d.status,
          syntheticRows,
          qualityReport,
          privacyReport,
          mlReport,
          epochLogs: [],
          createdAt: d.createdAt,
        });
      } catch (e) {
        console.warn('Skipped malformed run doc', e);
      }
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.LIST, 'generation_runs');
  }

  try {
    const expQ = query(collection(db, 'export_history'), where('ownerId', '==', uid));
    const expSnap = await getDocs(expQ);
    expSnap.forEach((docSnap) => {
      const d = docSnap.data();
      exports.push({
        id: docSnap.id,
        ownerId: d.ownerId,
        runId: d.runId,
        exportType: d.exportType,
        fileName: d.fileName,
        rowCount: d.rowCount,
        createdAt: d.createdAt,
      });
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.LIST, 'export_history');
  }

  try {
    const cfgQ = query(collection(db, 'generation_configs'), where('ownerId', '==', uid));
    const cfgSnap = await getDocs(cfgQ);
    cfgSnap.forEach((docSnap) => {
      const d = docSnap.data();
      try {
        configs.push(JSON.parse(d.configJson));
      } catch (e) {
        console.warn('Skipped malformed config doc', e);
      }
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.LIST, 'generation_configs');
  }

  runs.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  exports.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return { runs, exports, configs };
}

export async function deleteRunFromFirestore(runId: string) {
  if (!auth.currentUser) return;
  const cleanId = sanitizeId(runId);
  try {
    await deleteDoc(doc(db, 'generation_runs', cleanId));
  } catch (error) {
    handleFirestoreError(error, OperationType.DELETE, `generation_runs/${cleanId}`);
  }
}
