import { PrivacyAssessmentReport } from '../types/synthforge';
import { inferColumnType, computeQuantile } from './profiler';

// Compute normalized Gower distance between two tabular records across non-identifier columns
function computeGowerDistance(
  rowA: Record<string, any>,
  rowB: Record<string, any>,
  cols: string[],
  colTypes: Map<string, string>,
  numRanges: Map<string, { min: number; span: number }>
): number {
  let sumDist = 0;
  let validCols = 0;

  for (const col of cols) {
    const va = rowA[col];
    const vb = rowB[col];
    if (va === null || va === undefined || vb === null || vb === undefined) {
      continue;
    }
    validCols++;
    const t = colTypes.get(col);
    if (t === 'numerical') {
      const na = Number(va);
      const nb = Number(vb);
      const rng = numRanges.get(col) || { min: 0, span: 1 };
      if (!Number.isNaN(na) && !Number.isNaN(nb)) {
        sumDist += Math.min(1, Math.abs(na - nb) / rng.span);
      }
    } else if (t === 'datetime') {
      const ta = Date.parse(String(va));
      const tb = Date.parse(String(vb));
      const rng = numRanges.get(col) || { min: 0, span: 86400000 * 365 };
      if (!Number.isNaN(ta) && !Number.isNaN(tb)) {
        sumDist += Math.min(1, Math.abs(ta - tb) / rng.span);
      }
    } else {
      sumDist += String(va) === String(vb) ? 0 : 1;
    }
  }

  return validCols > 0 ? sumDist / validCols : 1;
}

export function evaluatePrivacyRisk(
  realRows: Record<string, any>[],
  syntheticRows: Record<string, any>[]
): PrivacyAssessmentReport {
  const allCols = Object.keys(syntheticRows[0] || {});
  const colTypes = new Map<string, string>();
  const evalCols: string[] = [];
  const numRanges = new Map<string, { min: number; span: number }>();

  for (const col of allCols) {
    const t = inferColumnType(realRows, col);
    colTypes.set(col, t);
    if (t !== 'identifier') {
      evalCols.push(col);
      if (t === 'numerical' || t === 'datetime') {
        const vals = [...realRows, ...syntheticRows]
          .map((r) => (t === 'datetime' ? Date.parse(String(r[col])) : Number(r[col])))
          .filter((n) => !Number.isNaN(n) && Number.isFinite(n));
        const min = vals.length > 0 ? Math.min(...vals) : 0;
        const max = vals.length > 0 ? Math.max(...vals) : 1;
        numRanges.set(col, { min, span: Math.max(1e-6, max - min) });
      }
    }
  }

  // 1. Exact record overlap on non-identifier attributes
  const realSignatures = new Set<string>();
  for (const r of realRows) {
    const sig = evalCols.map((c) => String(r[c] ?? 'NULL')).join('|');
    realSignatures.add(sig);
  }

  let exactOverlapCount = 0;
  for (const s of syntheticRows) {
    const sig = evalCols.map((c) => String(s[c] ?? 'NULL')).join('|');
    if (realSignatures.has(sig)) {
      exactOverlapCount++;
    }
  }
  const exactOverlapRate = Number(
    (syntheticRows.length > 0 ? exactOverlapCount / syntheticRows.length : 0).toFixed(4)
  );

  // Split realRows into 70% Member (Train) and 30% Non-Member (Holdout) for DCR baseline & Membership-Inference demonstration
  const splitIdx = Math.max(5, Math.floor(realRows.length * 0.7));
  const realTrain = realRows.slice(0, splitIdx);
  const realHoldout = realRows.slice(splitIdx);

  // 2. Nearest-neighbour distances (DCR) and NNDR (d1 / d2) from Synthetic to Real Train
  const dcrValues: number[] = [];
  const nndrValues: number[] = [];
  let nearDuplicateCount = 0;
  let memorisationSuspectCount = 0;
  const nearestNeighbourPairs: PrivacyAssessmentReport['nearestNeighbourPairs'] = [];

  const maxEvalSynth = Math.min(syntheticRows.length, 220);
  for (let sIdx = 0; sIdx < maxEvalSynth; sIdx++) {
    const sRow = syntheticRows[sIdx];
    let d1 = Infinity;
    let d2 = Infinity;
    let closestRealIdx = 0;

    for (let rIdx = 0; rIdx < realTrain.length; rIdx++) {
      const dist = computeGowerDistance(sRow, realTrain[rIdx], evalCols, colTypes, numRanges);
      if (dist < d1) {
        d2 = d1;
        d1 = dist;
        closestRealIdx = rIdx;
      } else if (dist < d2) {
        d2 = dist;
      }
    }

    const nndr = d2 > 1e-6 && Number.isFinite(d2) ? Math.min(1, d1 / d2) : 1;
    dcrValues.push(d1);
    nndrValues.push(nndr);

    if (d1 < 0.045) {
      nearDuplicateCount++;
    }
    if (d1 < 0.055 && nndr < 0.45) {
      memorisationSuspectCount++;
    }

    if (nearestNeighbourPairs.length < 25) {
      const status =
        d1 < 0.045 || (d1 < 0.06 && nndr < 0.45)
          ? 'Memorisation Warning'
          : d1 < 0.09
          ? 'Close'
          : 'Safe';
      nearestNeighbourPairs.push({
        synthRowIndex: sIdx + 1,
        closestRealRowIndex: closestRealIdx + 1,
        normalizedDistance: Number(d1.toFixed(4)),
        nndr: Number(nndr.toFixed(4)),
        status,
      });
    }
  }

  nearestNeighbourPairs.sort((a, b) => a.normalizedDistance - b.normalizedDistance);

  // Compute Holdout-to-Real-Train DCR baseline
  const holdoutDcrs: number[] = [];
  for (const hRow of realHoldout) {
    let minD = Infinity;
    for (const rRow of realTrain) {
      const d = computeGowerDistance(hRow, rRow, evalCols, colTypes, numRanges);
      if (d < minD) minD = d;
    }
    holdoutDcrs.push(minD);
  }

  const sortedDcr = [...dcrValues].sort((a, b) => a - b);
  const sortedHoldoutDcr = [...holdoutDcrs].sort((a, b) => a - b);
  const sortedNndr = [...nndrValues].sort((a, b) => a - b);

  const dcrMean = Number(
    (sortedDcr.reduce((a, b) => a + b, 0) / Math.max(1, sortedDcr.length)).toFixed(4)
  );
  const dcrMedian = Number(computeQuantile(sortedDcr, 0.5).toFixed(4));
  const dcr5thPercentile = Number(computeQuantile(sortedDcr, 0.05).toFixed(4));
  const holdoutDcr5thPercentile = Number(
    Math.max(0.01, computeQuantile(sortedHoldoutDcr, 0.05)).toFixed(4)
  );
  const dcrRatioToHoldout = Number((dcr5thPercentile / holdoutDcr5thPercentile).toFixed(3));

  const nndrMean = Number(
    (sortedNndr.reduce((a, b) => a + b, 0) / Math.max(1, sortedNndr.length)).toFixed(4)
  );
  const nndr5thPercentile = Number(computeQuantile(sortedNndr, 0.05).toFixed(4));
  const nearDuplicateRate = Number((nearDuplicateCount / Math.max(1, maxEvalSynth)).toFixed(4));

  // 3. Controlled Membership-Inference Attack (MIA) Demonstration
  // Attack score for a candidate real record x is -min_{s in Synth} GowerDistance(x, s)
  const memberScores: number[] = [];
  const nonMemberScores: number[] = [];

  for (const mRow of realTrain.slice(0, 60)) {
    let minD = Infinity;
    for (let i = 0; i < maxEvalSynth; i++) {
      const d = computeGowerDistance(mRow, syntheticRows[i], evalCols, colTypes, numRanges);
      if (d < minD) minD = d;
    }
    memberScores.push(-minD);
  }

  for (const nmRow of realHoldout.slice(0, 60)) {
    let minD = Infinity;
    for (let i = 0; i < maxEvalSynth; i++) {
      const d = computeGowerDistance(nmRow, syntheticRows[i], evalCols, colTypes, numRanges);
      if (d < minD) minD = d;
    }
    nonMemberScores.push(-minD);
  }

  // Mann-Whitney U statistic for ROC-AUC of Membership Inference Attack
  let wins = 0;
  let totalPairs = 0;
  for (const ms of memberScores) {
    for (const nms of nonMemberScores) {
      totalPairs++;
      if (ms > nms) wins += 1;
      else if (ms === nms) wins += 0.5;
    }
  }
  const miaSimulatedAuc = Number(
    (totalPairs > 0 ? Math.max(0.45, Math.min(0.99, wins / totalPairs)) : 0.5).toFixed(4)
  );
  const miaAttackerAdvantage = Number(
    Math.max(0, (miaSimulatedAuc - 0.5) * 2).toFixed(4)
  );

  // Precision at 10% False Positive Rate
  const sortedNmScores = [...nonMemberScores].sort((a, b) => b - a);
  const fpr10Threshold =
    sortedNmScores[Math.max(0, Math.floor(sortedNmScores.length * 0.1))] ?? -0.05;
  const tpAt10 = memberScores.filter((s) => s >= fpr10Threshold).length;
  const fpAt10 = nonMemberScores.filter((s) => s >= fpr10Threshold).length;
  const miaPrecisionAt10Fpr = Number(
    (tpAt10 + fpAt10 > 0 ? tpAt10 / (tpAt10 + fpAt10) : 0.5).toFixed(4)
  );

  // Build DCR Histogram comparing Synth-to-Real vs Holdout-to-Real
  const bins = [0, 0.04, 0.08, 0.12, 0.16, 0.22, 0.3, 1.0];
  const dcrHistogram: PrivacyAssessmentReport['dcrHistogram'] = [];
  for (let b = 0; b < bins.length - 1; b++) {
    const low = bins[b];
    const high = bins[b + 1];
    const sCnt = sortedDcr.filter((v) => v >= low && (b === bins.length - 2 ? v <= high : v < high)).length;
    const hCnt = sortedHoldoutDcr.filter((v) => v >= low && (b === bins.length - 2 ? v <= high : v < high)).length;
    dcrHistogram.push({
      distanceBin: `${low.toFixed(2)}–${high.toFixed(2)}`,
      synthToRealPct: Number(((sCnt / Math.max(1, sortedDcr.length)) * 100).toFixed(1)),
      holdoutToRealPct: Number(((hCnt / Math.max(1, sortedHoldoutDcr.length)) * 100).toFixed(1)),
    });
  }

  // Overall Risk Score (0-100)
  const rawRisk =
    exactOverlapRate * 250 +
    nearDuplicateRate * 180 +
    Math.max(0, (miaSimulatedAuc - 0.5) * 140) +
    Math.max(0, (1 - Math.min(1.2, dcrRatioToHoldout)) * 45);
  const overallRiskScore = Number(Math.max(4, Math.min(98, rawRisk)).toFixed(1));

  const riskLevel: PrivacyAssessmentReport['riskLevel'] =
    overallRiskScore >= 65
      ? 'Critical'
      : overallRiskScore >= 40
      ? 'Elevated'
      : overallRiskScore >= 20
      ? 'Moderate'
      : 'Low';

  const warnings: string[] = [];
  if (exactOverlapCount > 0) {
    warnings.push(
      `Detected ${exactOverlapCount} exact record overlaps (${(exactOverlapRate * 100).toFixed(2)}%) between synthetic and source data.`
    );
  }
  if (nearDuplicateCount > 0) {
    warnings.push(
      `Found ${nearDuplicateCount} synthetic records within Gower distance < 0.045 of a source training record.`
    );
  }
  if (miaSimulatedAuc > 0.62) {
    warnings.push(
      `Distance-based Membership-Inference Attack achieved ROC-AUC of ${miaSimulatedAuc} (Attacker Advantage ${(miaAttackerAdvantage * 100).toFixed(1)}%), indicating potential overfitting to training members.`
    );
  }
  if (warnings.length === 0) {
    warnings.push(
      `Empirical distance-to-closest-record (5th percentile DCR = ${dcr5thPercentile}) and MIA AUC (${miaSimulatedAuc}) indicate low empirical memorisation risk.`
    );
  }

  return {
    overallRiskScore,
    riskLevel,
    exactOverlapCount,
    exactOverlapRate,
    nearDuplicateCount,
    nearDuplicateRate,
    dcrMean,
    dcrMedian,
    dcr5thPercentile,
    holdoutDcr5thPercentile,
    dcrRatioToHoldout,
    nndrMean,
    nndr5thPercentile,
    memorisationSuspectCount,
    miaSimulatedAuc,
    miaAttackerAdvantage,
    miaPrecisionAt10Fpr,
    dcrHistogram,
    nearestNeighbourPairs,
    warnings,
  };
}
