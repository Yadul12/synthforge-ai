import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { DatasetRecord, GenerationRunRecord } from '../types/synthforge';

export async function parseUploadedDatasetFile(
  file: File
): Promise<{
  rows: Record<string, any>[];
  sourceFormat: 'csv' | 'json' | 'excel';
}> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  if (ext === 'json') {
    const text = await file.text();
    const parsed = JSON.parse(text);
    const rows = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed.data)
      ? parsed.data
      : Array.isArray(parsed.rows)
      ? parsed.rows
      : [];
    if (!rows.length || typeof rows[0] !== 'object') {
      throw new Error('JSON file must contain an array of record objects.');
    }
    return { rows, sourceFormat: 'json' };
  }

  if (ext === 'xlsx' || ext === 'xls') {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: 'array' });
    const firstSheetName = workbook.SheetNames[0];
    if (!firstSheetName) {
      throw new Error('Excel workbook contains no sheets.');
    }
    const worksheet = workbook.Sheets[firstSheetName];
    const rows = XLSX.utils.sheet_to_json<Record<string, any>>(worksheet, {
      defval: null,
    });
    if (!rows.length) {
      throw new Error('Excel sheet contains no data rows.');
    }
    return { rows, sourceFormat: 'excel' };
  }

  // Default to CSV
  const text = await file.text();
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, any>>(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (!results.data || results.data.length === 0) {
          reject(new Error('CSV file contains no valid data rows.'));
          return;
        }
        resolve({ rows: results.data, sourceFormat: 'csv' });
      },
      error: (err: Error) => {
        reject(new Error(`CSV Parse Error: ${err.message}`));
      },
    });
  });
}

function triggerBrowserDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadRowsAsCsv(rows: Record<string, any>[], filename: string) {
  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  triggerBrowserDownload(blob, filename);
}

export function downloadObjectAsJson(data: any, filename: string) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
  triggerBrowserDownload(blob, filename);
}

export function downloadEvaluationReportPdf(
  run: GenerationRunRecord,
  dataset?: DatasetRecord
): string {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const filename = `SynthForge_Evaluation_${run.id}.pdf`;

  // Header
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.setTextColor(15, 23, 42);
  doc.text('SynthForge AI — Synthetic Data Audit & ML Utility Report', 14, 18);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.setTextColor(71, 85, 105);
  doc.text(
    `Run ID: ${run.id}  |  Dataset: ${run.datasetName}  |  Engine: ${run.config.method.toUpperCase()}  |  Generated: ${run.createdAt.slice(0, 19)}`,
    14,
    25
  );

  // Executive KPI Summary Table
  autoTable(doc, {
    startY: 30,
    head: [
      [
        'Synthetic Rows',
        'Overall Quality Score',
        'Constraint Violations',
        'Privacy Risk Level',
        'MIA ROC-AUC',
        'Downstream TSTR Parity',
      ],
    ],
    body: [
      [
        String(run.syntheticRows.length),
        `${run.qualityReport.overallScore}%`,
        `${(run.qualityReport.constraintViolationRate * 100).toFixed(2)}%`,
        `${run.privacyReport.riskLevel} (${run.privacyReport.overallRiskScore}/100)`,
        String(run.privacyReport.miaSimulatedAuc),
        run.mlReport ? `${run.mlReport.avgTstrRetentionPct}%` : 'Not Run',
      ],
    ],
    theme: 'grid',
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontSize: 8.5 },
    bodyStyles: { fontSize: 9, textColor: [15, 23, 42] },
  });

  // Feature-Level Statistical Tests
  let finalY = (doc as any).lastAutoTable.finalY + 8;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text('1. Univariate & Distributional Statistical Tests', 14, finalY);

  autoTable(doc, {
    startY: finalY + 3,
    head: [
      [
        'Feature',
        'Type',
        'Real Mean/Top',
        'Synth Mean/Top',
        'KS / ChiSq Stat',
        'p-Value',
        'Wasserstein / JSD',
        'Fidelity',
      ],
    ],
    body: run.qualityReport.featureMetrics.map((f) => [
      f.column,
      f.type,
      f.type === 'numerical' ? `${f.realMean} (±${f.realStd})` : 'Categorical',
      f.type === 'numerical' ? `${f.synthMean} (±${f.synthStd})` : 'Categorical',
      f.type === 'numerical' ? `KS=${f.ksStatistic}` : `χ²=${f.chiSquareStat}`,
      String(f.type === 'numerical' ? f.ksPValue : f.chiSquarePValue),
      f.type === 'numerical'
        ? `W1=${f.wassersteinDistance} / JSD=${f.jsDivergence}`
        : `JSD=${f.jsDivergence}`,
      `${f.similarityScore}%`,
    ]),
    theme: 'striped',
    headStyles: { fillColor: [30, 41, 59], textColor: 255, fontSize: 8 },
    bodyStyles: { fontSize: 8 },
  });

  finalY = (doc as any).lastAutoTable.finalY + 8;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('2. Empirical Privacy & Disclosure Risk Indicators', 14, finalY);

  autoTable(doc, {
    startY: finalY + 3,
    head: [
      [
        'Exact Overlap Rate',
        'Near-Duplicate Rate',
        '5th Pct DCR (Synth)',
        '5th Pct DCR (Holdout)',
        'Mean NNDR (d1/d2)',
        'MIA Attacker Advantage',
      ],
    ],
    body: [
      [
        `${(run.privacyReport.exactOverlapRate * 100).toFixed(2)}% (${run.privacyReport.exactOverlapCount})`,
        `${(run.privacyReport.nearDuplicateRate * 100).toFixed(2)}% (${run.privacyReport.nearDuplicateCount})`,
        String(run.privacyReport.dcr5thPercentile),
        String(run.privacyReport.holdoutDcr5thPercentile),
        String(run.privacyReport.nndrMean),
        `${(run.privacyReport.miaAttackerAdvantage * 100).toFixed(1)}%`,
      ],
    ],
    theme: 'grid',
    headStyles: { fillColor: [30, 41, 59], textColor: 255, fontSize: 8 },
    bodyStyles: { fontSize: 8.5 },
  });

  finalY = (doc as any).lastAutoTable.finalY + 5;
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text(
    'Privacy Limitation Notice: Empirical DCR, NNDR, and Membership-Inference metrics are statistical disclosure risk indicators, NOT formal (ε, δ)-differential privacy guarantees.',
    14,
    finalY,
    { maxWidth: 180 }
  );

  if (run.mlReport) {
    finalY += 9;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.text(
      `3. Downstream ML Utility Benchmark (Target: ${run.mlReport.targetColumn}, Zero-Leakage Holdout Test N=${run.mlReport.realHoldoutTestSize})`,
      14,
      finalY
    );

    const isClass = run.mlReport.taskType === 'classification';
    autoTable(doc, {
      startY: finalY + 3,
      head: [
        [
          'Algorithm',
          isClass ? 'Exp A (Real) F1 / AUC' : 'Exp A (Real) R² / RMSE',
          isClass ? 'Exp B (Synth) F1 / AUC' : 'Exp B (Synth) R² / RMSE',
          isClass ? 'Exp C (Aug) F1 / AUC' : 'Exp C (Aug) R² / RMSE',
          'TSTR Parity',
          'Augmentation Delta',
        ],
      ],
      body: run.mlReport.models.map((m) => [
        m.modelName,
        isClass
          ? `${m.experimentA_Real.f1Score} / ${m.experimentA_Real.rocAuc}`
          : `${m.experimentA_Real.r2} / ${m.experimentA_Real.rmse}`,
        isClass
          ? `${m.experimentB_Synthetic.f1Score} / ${m.experimentB_Synthetic.rocAuc}`
          : `${m.experimentB_Synthetic.r2} / ${m.experimentB_Synthetic.rmse}`,
        isClass
          ? `${m.experimentC_Augmented.f1Score} / ${m.experimentC_Augmented.rocAuc}`
          : `${m.experimentC_Augmented.r2} / ${m.experimentC_Augmented.rmse}`,
        `${m.utilityRetentionPct}%`,
        `${m.augmentationGainPct >= 0 ? '+' : ''}${m.augmentationGainPct}%`,
      ]),
      theme: 'striped',
      headStyles: { fillColor: [15, 23, 42], textColor: 255, fontSize: 8 },
      bodyStyles: { fontSize: 8 },
    });
  }

  if (dataset && dataset.transformations.length > 0) {
    finalY = (doc as any).lastAutoTable.finalY + 8;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(15, 23, 42);
    doc.text(
      `Applied Data Preparation Steps (${dataset.transformations.length}): ${dataset.transformations.map((t) => t.description).join(' | ')}`,
      14,
      finalY,
      { maxWidth: 180 }
    );
  }

  doc.save(filename);
  return filename;
}
