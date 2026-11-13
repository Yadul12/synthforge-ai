import {
  CustomConstraint,
  FeatureQualityMetric,
  QualityEvaluationReport,
} from '../types/synthforge';
import { inferColumnType, computeQuantile, profileDataset } from './profiler';
import { validateRowAgainstConstraint } from './generator';

// Two-sample Kolmogorov-Smirnov statistic & asymptotic p-value
export function computeTwoSampleKS(
  sampleA: number[],
  sampleB: number[]
): { ksStat: number; pValue: number } {
  if (sampleA.length === 0 || sampleB.length === 0) {
    return { ksStat: 1, pValue: 0 };
  }
  const sortedA = [...sampleA].sort((a, b) => a - b);
  const sortedB = [...sampleB].sort((a, b) => a - b);
  const n1 = sortedA.length;
  const n2 = sortedB.length;

  let i = 0;
  let j = 0;
  let dMax = 0;

  while (i < n1 && j < n2) {
    const v1 = sortedA[i];
    const v2 = sortedB[j];
    if (v1 <= v2) i++;
    if (v2 <= v1) j++;
    const d = Math.abs(i / n1 - j / n2);
    if (d > dMax) dMax = d;
  }

  const en = Math.sqrt((n1 * n2) / (n1 + n2));
  const lambda = (en + 0.12 + 0.11 / en) * dMax;
  let sum = 0;
  for (let k = 1; k <= 25; k++) {
    sum += Math.pow(-1, k - 1) * Math.exp(-2 * k * k * lambda * lambda);
  }
  const pValue = Math.max(0, Math.min(1, 2 * sum));
  return {
    ksStat: Number(dMax.toFixed(4)),
    pValue: Number(pValue.toFixed(4)),
  };
}

// Normalized 1D Wasserstein-1 (Earth Mover's) distance
export function computeNormalizedWasserstein(
  sampleA: number[],
  sampleB: number[]
): number {
  if (sampleA.length === 0 || sampleB.length === 0) return 1;
  const sortedA = [...sampleA].sort((a, b) => a - b);
  const sortedB = [...sampleB].sort((a, b) => a - b);
  const minVal = Math.min(sortedA[0], sortedB[0]);
  const maxVal = Math.max(sortedA[sortedA.length - 1], sortedB[sortedB.length - 1]);
  const range = Math.max(1e-6, maxVal - minVal);

  const steps = 50;
  let sumDiff = 0;
  for (let s = 0; s <= steps; s++) {
    const q = s / steps;
    const qa = computeQuantile(sortedA, q);
    const qb = computeQuantile(sortedB, q);
    sumDiff += Math.abs(qa - qb);
  }
  const rawW1 = sumDiff / (steps + 1);
  return Number(Math.min(1, rawW1 / range).toFixed(4));
}

// Jensen-Shannon Divergence (in base-2 bits, bounded in [0, 1])
export function computeJensenShannonDivergence(
  probP: number[],
  probQ: number[]
): number {
  const k = Math.min(probP.length, probQ.length);
  if (k === 0) return 0;
  let klPM = 0;
  let klQM = 0;
  for (let i = 0; i < k; i++) {
    const p = Math.max(1e-9, probP[i]);
    const q = Math.max(1e-9, probQ[i]);
    const m = 0.5 * (p + q);
    klPM += p * Math.log2(p / m);
    klQM += q * Math.log2(q / m);
  }
  const jsd = 0.5 * klPM + 0.5 * klQM;
  return Number(Math.max(0, Math.min(1, jsd)).toFixed(4));
}

// Chi-Square test for categorical frequency comparison
export function computeChiSquareTest(
  realProbs: number[],
  synthCounts: number[],
  totalSynth: number
): { chiSquare: number; pValue: number } {
  let chiSq = 0;
  const k = realProbs.length;
  for (let i = 0; i < k; i++) {
    const expected = Math.max(1, realProbs[i] * totalSynth);
    const observed = synthCounts[i] || 0;
    chiSq += Math.pow(observed - expected, 2) / expected;
  }
  const df = Math.max(1, k - 1);
  // Wilson-Hilferty normal approximation for Chi-Square p-value
  const z =
    (Math.pow(chiSq / df, 1 / 3) - (1 - 2 / (9 * df))) /
    Math.sqrt(2 / (9 * df));
  const pValue = Number(Math.max(0, Math.min(1, 1 - 0.5 * (1 + Math.tanh(z * 0.79788)))).toFixed(4));
  return {
    chiSquare: Number(chiSq.toFixed(3)),
    pValue,
  };
}

export function evaluateSyntheticQuality(
  realRows: Record<string, any>[],
  syntheticRows: Record<string, any>[],
  constraints: CustomConstraint[] = [],
  rawViolationCountBeforeProjection = 0
): QualityEvaluationReport {
  const synthCols = Object.keys(syntheticRows[0] || {});
  const featureMetrics: FeatureQualityMetric[] = [];
  const warnings: string[] = [];

  for (const col of synthCols) {
    const type = inferColumnType(realRows, col);
    if (type === 'identifier') continue;

    const realVals = realRows
      .map((r) => r[col])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');
    const synthVals = syntheticRows
      .map((r) => r[col])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');

    if (type === 'numerical') {
      const rNums = realVals.map(Number).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b);
      const sNums = synthVals.map(Number).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b);

      const rMean = rNums.reduce((a, b) => a + b, 0) / Math.max(1, rNums.length);
      const sMean = sNums.reduce((a, b) => a + b, 0) / Math.max(1, sNums.length);
      const rStd = Math.sqrt(
        rNums.reduce((a, b) => a + Math.pow(b - rMean, 2), 0) / Math.max(1, rNums.length - 1)
      );
      const sStd = Math.sqrt(
        sNums.reduce((a, b) => a + Math.pow(b - sMean, 2), 0) / Math.max(1, sNums.length - 1)
      );

      const { ksStat, pValue } = computeTwoSampleKS(rNums, sNums);
      const wDist = computeNormalizedWasserstein(rNums, sNums);

      // Build 8 shared bins for distribution comparison & JSD
      const minVal = Math.min(rNums[0] ?? 0, sNums[0] ?? 0);
      const maxVal = Math.max(rNums[rNums.length - 1] ?? 1, sNums[sNums.length - 1] ?? 1);
      const binCount = 8;
      const binWidth = (maxVal - minVal || 1) / binCount;

      const rProbs: number[] = [];
      const sProbs: number[] = [];
      const distributionComparison: { label: string; realFreq: number; synthFreq: number }[] = [];

      for (let b = 0; b < binCount; b++) {
        const low = minVal + b * binWidth;
        const high = b === binCount - 1 ? maxVal + 1e-6 : minVal + (b + 1) * binWidth;
        const rCnt = rNums.filter((v) => v >= low && v < high).length;
        const sCnt = sNums.filter((v) => v >= low && v < high).length;
        const rp = rCnt / Math.max(1, rNums.length);
        const sp = sCnt / Math.max(1, sNums.length);
        rProbs.push(rp);
        sProbs.push(sp);
        distributionComparison.push({
          label: `${low.toFixed(1)}`,
          realFreq: Number((rp * 100).toFixed(1)),
          synthFreq: Number((sp * 100).toFixed(1)),
        });
      }

      const jsd = computeJensenShannonDivergence(rProbs, sProbs);
      const simScore = Number(
        Math.max(0, Math.min(100, (1 - 0.45 * ksStat - 0.35 * jsd - 0.2 * wDist) * 100)).toFixed(1)
      );

      if (ksStat > 0.18) {
        warnings.push(
          `Column "${col}" shows elevated Kolmogorov-Smirnov drift (KS=${ksStat}, p=${pValue}).`
        );
      }

      featureMetrics.push({
        column: col,
        type,
        realMean: Number(rMean.toFixed(2)),
        synthMean: Number(sMean.toFixed(2)),
        realStd: Number(rStd.toFixed(2)),
        synthStd: Number(sStd.toFixed(2)),
        realMedian: Number(computeQuantile(rNums, 0.5).toFixed(2)),
        synthMedian: Number(computeQuantile(sNums, 0.5).toFixed(2)),
        realQ25: Number(computeQuantile(rNums, 0.25).toFixed(2)),
        synthQ25: Number(computeQuantile(sNums, 0.25).toFixed(2)),
        realQ75: Number(computeQuantile(rNums, 0.75).toFixed(2)),
        synthQ75: Number(computeQuantile(sNums, 0.75).toFixed(2)),
        ksStatistic: ksStat,
        ksPValue: pValue,
        wassersteinDistance: wDist,
        jsDivergence: jsd,
        similarityScore: simScore,
        distributionComparison,
      });
    } else {
      // Categorical or datetime
      const allCats = Array.from(
        new Set([...realVals.map(String), ...synthVals.map(String)])
      ).slice(0, 10);

      const rProbs: number[] = [];
      const sProbs: number[] = [];
      const sCounts: number[] = [];
      const distributionComparison: { label: string; realFreq: number; synthFreq: number }[] = [];

      for (const cat of allCats) {
        const rCnt = realVals.filter((v) => String(v) === cat).length;
        const sCnt = synthVals.filter((v) => String(v) === cat).length;
        const rp = rCnt / Math.max(1, realVals.length);
        const sp = sCnt / Math.max(1, synthVals.length);
        rProbs.push(rp);
        sProbs.push(sp);
        sCounts.push(sCnt);
        distributionComparison.push({
          label: cat.length > 14 ? `${cat.slice(0, 12)}…` : cat,
          realFreq: Number((rp * 100).toFixed(1)),
          synthFreq: Number((sp * 100).toFixed(1)),
        });
      }

      const jsd = computeJensenShannonDivergence(rProbs, sProbs);
      const { chiSquare, pValue } = computeChiSquareTest(rProbs, sCounts, synthVals.length || 1);
      const simScore = Number(Math.max(0, Math.min(100, (1 - jsd) * 100)).toFixed(1));

      featureMetrics.push({
        column: col,
        type,
        chiSquareStat: chiSquare,
        chiSquarePValue: pValue,
        jsDivergence: jsd,
        similarityScore: simScore,
        distributionComparison,
      });
    }
  }

  // Pairwise correlation comparison
  const realProf = profileDataset('real', realRows);
  const synthProf = profileDataset('synth', syntheticRows);

  const correlationComparison: QualityEvaluationReport['correlationComparison'] = [];
  let totalCorrDrift = 0;
  let offDiagCount = 0;

  for (const rc of realProf.correlations) {
    const sc = synthProf.correlations.find(
      (c) => c.colA === rc.colA && c.colB === rc.colB
    );
    const synthCorr = sc ? sc.correlation : 0;
    const drift = Number(Math.abs(rc.correlation - synthCorr).toFixed(3));
    correlationComparison.push({
      colA: rc.colA,
      colB: rc.colB,
      realCorr: rc.correlation,
      synthCorr,
      drift,
    });
    if (rc.colA !== rc.colB) {
      totalCorrDrift += drift;
      offDiagCount++;
    }
  }

  const correlationFrobeniusDrift = Number(
    (offDiagCount > 0 ? totalCorrDrift / offDiagCount : 0).toFixed(4)
  );
  const correlationPreservationScore = Number(
    Math.max(0, Math.min(100, (1 - correlationFrobeniusDrift * 1.35) * 100)).toFixed(1)
  );

  // Constraint violation verification on final synthetic rows
  let constraintViolationCount = 0;
  if (constraints.length > 0) {
    for (const r of syntheticRows) {
      const violated = constraints.some((c) => !validateRowAgainstConstraint(r, c));
      if (violated) constraintViolationCount++;
    }
  }
  const constraintViolationRate = Number(
    (syntheticRows.length > 0 ? constraintViolationCount / syntheticRows.length : 0).toFixed(4)
  );

  const numFeatures = featureMetrics.filter((f) => f.type === 'numerical');
  const meanKsComplement = Number(
    (
      numFeatures.reduce((acc, f) => acc + (1 - (f.ksStatistic || 0)), 0) /
      Math.max(1, numFeatures.length)
    ).toFixed(4)
  );
  const meanJsSimilarity = Number(
    (
      featureMetrics.reduce((acc, f) => acc + (1 - f.jsDivergence), 0) /
      Math.max(1, featureMetrics.length)
    ).toFixed(4)
  );
  const meanWassersteinDrift = Number(
    (
      numFeatures.reduce((acc, f) => acc + (f.wassersteinDistance || 0), 0) /
      Math.max(1, numFeatures.length)
    ).toFixed(4)
  );

  const statisticalSimilarityScore = Number(
    (
      featureMetrics.reduce((acc, f) => acc + f.similarityScore, 0) /
      Math.max(1, featureMetrics.length)
    ).toFixed(1)
  );

  const missingDrift = Math.abs(realProf.missingRate - synthProf.missingRate);
  const dupDrift = Math.abs(realProf.duplicateRate - synthProf.duplicateRate);
  const structuralQualityScore = Number(
    Math.max(
      0,
      Math.min(100, (1 - constraintViolationRate * 2 - missingDrift * 1.5 - dupDrift * 1.5) * 100)
    ).toFixed(1)
  );

  const overallScore = Number(
    (
      0.5 * statisticalSimilarityScore +
      0.3 * correlationPreservationScore +
      0.2 * structuralQualityScore
    ).toFixed(1)
  );

  const metricLimitations: QualityEvaluationReport['metricLimitations'] = [
    {
      metric: 'Kolmogorov-Smirnov (KS) Two-Sample Test',
      purpose: 'Measures maximum vertical distance between univariate empirical CDFs.',
      limitation:
        'Sensitive primarily near the median of a 1D marginal distribution; less sensitive to tail anomalies and completely blind to multivariate joint dependencies or privacy memorisation.',
    },
    {
      metric: 'Wasserstein-1 (Earth Mover’s) Distance',
      purpose: 'Quantifies minimum transport cost required to morph the synthetic marginal into the real marginal.',
      limitation:
        'Depends on feature scale (requires range normalization) and evaluates each column independently without testing cross-feature interaction terms needed by ML classifiers.',
    },
    {
      metric: 'Chi-Square (χ²) & Jensen-Shannon Divergence (JSD)',
      purpose: 'Compares categorical probability mass functions and discretized histogram overlap.',
      limitation:
        'Sensitive to rare categories and histogram binning choices. High marginal JSD similarity can be achieved even by trivial bootstrap copying of source rows (which violates privacy).',
    },
    {
      metric: 'Pairwise Pearson Correlation Drift',
      purpose: 'Evaluates preservation of linear bivariate relationships across numerical columns.',
      limitation:
        'Captures only linear second-order moments. Non-linear conditional manifolds and higher-order categorical-continuous interactions require TSTR downstream ML benchmarking.',
    },
  ];

  return {
    overallScore,
    statisticalSimilarityScore,
    correlationPreservationScore,
    structuralQualityScore,
    meanKsComplement,
    meanJsSimilarity,
    meanWassersteinDrift,
    correlationFrobeniusDrift,
    realMissingRate: realProf.missingRate,
    synthMissingRate: synthProf.missingRate,
    realDuplicateRate: realProf.duplicateRate,
    synthDuplicateRate: synthProf.duplicateRate,
    constraintViolationCount,
    constraintViolationRate,
    rawViolationCountBeforeProjection,
    featureMetrics,
    correlationComparison,
    warnings,
    metricLimitations,
  };
}
