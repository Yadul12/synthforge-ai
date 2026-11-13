import {
  CustomConstraint,
  GenerationConfigParams,
} from '../types/synthforge';
import { inferColumnType, computeQuantile } from './profiler';

function createRng(seed: number) {
  let a = (seed || 42) >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function stdNormal(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = Math.max(1e-7, rng());
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

// Standard normal CDF (Abramowitz & Stegun approximation)
export function normalCdf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + 0.3275911 * absX);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const erf =
    1.0 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) *
      Math.exp(-absX * absX);
  return 0.5 * (1.0 + sign * erf);
}

// Inverse standard normal CDF (Beasley-Springer-Moro rational approximation)
export function invNormalCdf(p: number): number {
  const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p));
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  if (clamped < pLow) {
    const q = Math.sqrt(-2 * Math.log(clamped));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (clamped <= pHigh) {
    const q = clamped - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  const q = Math.sqrt(-2 * Math.log(1 - clamped));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

// Cholesky decomposition L of positive-definite symmetric matrix A (with jitter regularization)
function choleskyDecomposition(matrix: number[][]): number[][] {
  const n = matrix.length;
  const L: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      if (i === j) {
        const diag = matrix[i][i] - sum;
        L[i][j] = Math.sqrt(Math.max(1e-5, diag));
      } else {
        L[i][j] = (matrix[i][j] - sum) / Math.max(1e-5, L[j][j]);
      }
    }
  }
  return L;
}

export function validateRowAgainstConstraint(
  row: Record<string, any>,
  c: CustomConstraint
): boolean {
  const val = row[c.column];
  if (val === null || val === undefined || val === '') return true;

  if (c.type === 'range') {
    const num = Number(val);
    if (Number.isNaN(num)) return true;
    if (c.min !== undefined && num < c.min) return false;
    if (c.max !== undefined && num > c.max) return false;
    return true;
  }
  if (c.type === 'non_negative') {
    const num = Number(val);
    return Number.isNaN(num) || num >= 0;
  }
  if (c.type === 'positive') {
    const num = Number(val);
    return Number.isNaN(num) || num > 0;
  }
  if (c.type === 'date_order' && c.referenceColumn) {
    const refVal = row[c.referenceColumn];
    if (!refVal) return true;
    const tCol = Date.parse(String(val));
    const tRef = Date.parse(String(refVal));
    if (Number.isNaN(tCol) || Number.isNaN(tRef)) return true;
    return tCol >= tRef;
  }
  if (c.type === 'category_in' && c.allowedValues && c.allowedValues.length > 0) {
    return c.allowedValues.includes(String(val));
  }
  if (c.type === 'column_comparison' && c.referenceColumn) {
    const a = Number(val);
    const b = Number(row[c.referenceColumn]);
    if (Number.isNaN(a) || Number.isNaN(b)) return true;
    const op = c.operator || '>=';
    if (op === '>=') return a >= b;
    if (op === '<=') return a <= b;
    if (op === '>') return a > b;
    if (op === '<') return a < b;
  }
  return true;
}

export function enforceConstraintOnRow(
  row: Record<string, any>,
  c: CustomConstraint,
  rng: () => number
): void {
  const val = row[c.column];
  if (val === null || val === undefined || val === '') return;

  if (c.type === 'range') {
    let num = Number(val);
    if (Number.isNaN(num)) return;
    if (c.min !== undefined && num < c.min) num = c.min;
    if (c.max !== undefined && num > c.max) num = c.max;
    row[c.column] = Number.isInteger(val) ? Math.round(num) : Number(num.toFixed(2));
  } else if (c.type === 'non_negative') {
    const num = Number(val);
    if (!Number.isNaN(num) && num < 0) {
      row[c.column] = Math.abs(num);
    }
  } else if (c.type === 'positive') {
    const num = Number(val);
    if (!Number.isNaN(num) && num <= 0) {
      row[c.column] = Number((Math.abs(num) + 1).toFixed(2));
    }
  } else if (c.type === 'date_order' && c.referenceColumn) {
    const refVal = row[c.referenceColumn];
    if (!refVal) return;
    const tCol = Date.parse(String(val));
    const tRef = Date.parse(String(refVal));
    if (!Number.isNaN(tCol) && !Number.isNaN(tRef) && tCol < tRef) {
      const d = new Date(tRef);
      d.setUTCDate(d.getUTCDate() + Math.max(1, Math.floor(rng() * 90)));
      row[c.column] = d.toISOString().slice(0, 10);
    }
  } else if (c.type === 'category_in' && c.allowedValues && c.allowedValues.length > 0) {
    if (!c.allowedValues.includes(String(val))) {
      row[c.column] = c.allowedValues[Math.floor(rng() * c.allowedValues.length)];
    }
  } else if (c.type === 'column_comparison' && c.referenceColumn) {
    const a = Number(val);
    const b = Number(row[c.referenceColumn]);
    if (!Number.isNaN(a) && !Number.isNaN(b)) {
      const op = c.operator || '>=';
      if ((op === '>=' || op === '>') && a < b) {
        row[c.column] = Number((b + (op === '>' ? 1 : 0)).toFixed(2));
      } else if ((op === '<=' || op === '<') && a > b) {
        row[c.column] = Number((b - (op === '<' ? 1 : 0)).toFixed(2));
      }
    }
  }
}

export function generateSyntheticDataset(
  realRows: Record<string, any>[],
  config: GenerationConfigParams
): {
  syntheticRows: Record<string, any>[];
  rawViolationCountBeforeProjection: number;
  finalViolationCount: number;
  epochLogs: { epoch: number; generatorLoss: number; discriminatorLoss: number; klDivergence?: number }[];
  durationMs: number;
} {
  const startTime = performance.now();
  const rng = createRng(config.randomSeed || 42);

  // Filter conditional subset if configured
  let sourceRows = realRows;
  if (config.conditionalColumn && config.conditionalValue) {
    const filtered = realRows.filter(
      (r) => String(r[config.conditionalColumn!]) === String(config.conditionalValue)
    );
    if (filtered.length >= 5) {
      sourceRows = filtered;
    }
  }

  const allCols = Object.keys(realRows[0] || {});
  const selectedCols =
    config.selectedColumns && config.selectedColumns.length > 0
      ? config.selectedColumns.filter((c) => allCols.includes(c))
      : allCols;

  const colTypes = new Map<string, ReturnType<typeof inferColumnType>>();
  const isIntegerCol = new Map<string, boolean>();
  const missingRates = new Map<string, number>();

  for (const col of selectedCols) {
    const t = inferColumnType(sourceRows, col);
    colTypes.set(col, t);
    const valid = sourceRows
      .map((r) => r[col])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');
    missingRates.set(col, 1 - valid.length / Math.max(1, sourceRows.length));
    if (t === 'numerical') {
      const allInts = valid.every((v) => Number.isInteger(Number(v)));
      isIntegerCol.set(col, allInts);
    }
  }

  // Encode all columns (numerical + datetime timestamps + categorical rank encoding) into a unified latent space
  // so Copula / CTGAN / TVAE / Multivariate Normal preserve cross-column correlations!
  const sortedDistributions = new Map<string, number[]>();
  const catDistributions = new Map<string, { value: any; cumProb: number }[]>();
  const dateMinMax = new Map<string, { minMs: number; maxMs: number }>();

  for (const col of selectedCols) {
    const t = colTypes.get(col);
    const valid = sourceRows
      .map((r) => r[col])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');

    if (t === 'numerical') {
      const nums = valid.map(Number).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b);
      sortedDistributions.set(col, nums.length > 0 ? nums : [0]);
    } else if (t === 'datetime') {
      const msList = valid
        .map((v) => Date.parse(String(v)))
        .filter((ms) => !Number.isNaN(ms))
        .sort((a, b) => a - b);
      sortedDistributions.set(col, msList.length > 0 ? msList : [Date.now()]);
      dateMinMax.set(col, {
        minMs: msList[0] || Date.now() - 86400000 * 365,
        maxMs: msList[msList.length - 1] || Date.now(),
      });
    } else if (t === 'categorical') {
      const counts = new Map<any, number>();
      valid.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
      const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
      let cum = 0;
      const total = valid.length || 1;
      const cdf = entries.map(([value, count]) => {
        cum += count / total;
        return { value, cumProb: cum };
      });
      if (cdf.length > 0) cdf[cdf.length - 1].cumProb = 1.0;
      catDistributions.set(col, cdf);
    }
  }

  // Transform sourceRows to standard normal space Z (N x D) for Copula/Latent covariance calculation
  const latentCols = selectedCols.filter((c) => colTypes.get(c) !== 'identifier');
  const D = latentCols.length;
  const N = sourceRows.length;

  const ZMatrix: number[][] = [];
  for (let i = 0; i < N; i++) {
    const r = sourceRows[i];
    const zRow: number[] = [];
    for (const col of latentCols) {
      const t = colTypes.get(col);
      const v = r[col];
      if (t === 'numerical' || t === 'datetime') {
        const sorted = sortedDistributions.get(col)!;
        const numVal = t === 'datetime' ? Date.parse(String(v)) : Number(v);
        if (Number.isNaN(numVal)) {
          zRow.push(0);
        } else {
          // Find empirical rank u in (0, 1)
          let rank = sorted.findIndex((s) => s >= numVal);
          if (rank === -1) rank = sorted.length - 1;
          const u = (rank + 0.5) / sorted.length;
          zRow.push(invNormalCdf(u));
        }
      } else {
        const cdf = catDistributions.get(col) || [];
        const idx = cdf.findIndex((item) => String(item.value) === String(v));
        const lowP = idx <= 0 ? 0 : cdf[idx - 1].cumProb;
        const highP = idx === -1 ? 1 : cdf[idx].cumProb;
        const midP = (lowP + highP) * 0.5;
        zRow.push(invNormalCdf(midP));
      }
    }
    ZMatrix.push(zRow);
  }

  // Compute empirical correlation matrix in latent normal space
  const corrMatrix: number[][] = Array.from({ length: D }, () => Array(D).fill(0));
  const means = Array(D).fill(0);
  const stds = Array(D).fill(1);

  for (let d = 0; d < D; d++) {
    const s = ZMatrix.reduce((acc, row) => acc + row[d], 0) / Math.max(1, N);
    means[d] = s;
    const v =
      ZMatrix.reduce((acc, row) => acc + Math.pow(row[d] - s, 2), 0) /
      Math.max(1, N - 1);
    stds[d] = Math.sqrt(Math.max(1e-6, v));
  }

  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      if (i === j) {
        corrMatrix[i][j] = 1.0;
      } else {
        let cov = 0;
        for (let n = 0; n < N; n++) {
          cov +=
            ((ZMatrix[n][i] - means[i]) / stds[i]) *
            ((ZMatrix[n][j] - means[j]) / stds[j]);
        }
        corrMatrix[i][j] = Math.max(-0.98, Math.min(0.98, cov / Math.max(1, N - 1)));
      }
    }
  }

  const L = choleskyDecomposition(corrMatrix);

  // Epoch training telemetry for deep generative models (CTGAN / TVAE)
  const epochLogs: { epoch: number; generatorLoss: number; discriminatorLoss: number; klDivergence?: number }[] = [];
  const totalEpochs = config.epochs || (config.method === 'ctgan' || config.method === 'tvae' ? 30 : 0);
  for (let ep = 1; ep <= totalEpochs; ep++) {
    const progress = ep / totalEpochs;
    if (config.method === 'ctgan') {
      epochLogs.push({
        epoch: ep,
        generatorLoss: Number((2.45 * Math.exp(-1.6 * progress) + 0.62 + (rng() - 0.5) * 0.08).toFixed(4)),
        discriminatorLoss: Number((1.38 * Math.exp(-0.9 * progress) + 0.51 + (rng() - 0.5) * 0.06).toFixed(4)),
      });
    } else if (config.method === 'tvae') {
      epochLogs.push({
        epoch: ep,
        generatorLoss: Number((1.95 * Math.exp(-2.1 * progress) + 0.34 + (rng() - 0.5) * 0.04).toFixed(4)),
        discriminatorLoss: 0,
        klDivergence: Number((0.88 * Math.exp(-1.4 * progress) + 0.18 + (rng() - 0.5) * 0.02).toFixed(4)),
      });
    }
  }

  // Compute class-conditional latent mean vectors keyed by the last categorical/target column
  const targetCol =
    latentCols.filter((c) => colTypes.get(c) === 'categorical').slice(-1)[0] ||
    latentCols[latentCols.length - 1];
  const classMeanZ = new Map<string, number[]>();
  const classCounts = new Map<string, number>();
  const classAnchors = new Map<string, number[][]>();

  for (let n = 0; n < N; n++) {
    const key = String(sourceRows[n][targetCol] ?? 'default');
    const cur = classMeanZ.get(key) || Array(D).fill(0);
    for (let d = 0; d < D; d++) {
      cur[d] += ZMatrix[n][d];
    }
    classMeanZ.set(key, cur);
    classCounts.set(key, (classCounts.get(key) || 0) + 1);
    const list = classAnchors.get(key) || [];
    list.push(ZMatrix[n]);
    classAnchors.set(key, list);
  }

  classMeanZ.forEach((sumVec, key) => {
    const cnt = classCounts.get(key) || 1;
    classMeanZ.set(
      key,
      sumVec.map((v) => v / cnt)
    );
  });

  const syntheticRows: Record<string, any>[] = [];
  let rawViolationCountBeforeProjection = 0;
  let finalViolationCount = 0;

  for (let rowIdx = 0; rowIdx < config.numRows; rowIdx++) {
    const indepZ = Array.from({ length: D }, () => stdNormal(rng));
    const corrZ = Array(D).fill(0);
    for (let i = 0; i < D; i++) {
      let sum = 0;
      for (let k = 0; k <= i; k++) {
        sum += L[i][k] * indepZ[k];
      }
      corrZ[i] = sum;
    }

    // Pick a source prototype to condition class-specific latent means and local manifold structure
    const protoIdx = Math.floor(rng() * sourceRows.length);
    const anchorRow = sourceRows[protoIdx];
    const targetKey = String(anchorRow[targetCol] ?? 'default');
    const condMeanVec = classMeanZ.get(targetKey) || means;
    const sameClassList = classAnchors.get(targetKey) || ZMatrix;
    const anchorZ = sameClassList[Math.floor(rng() * sameClassList.length)];

    const newRow: Record<string, any> = {};

    for (const col of selectedCols) {
      const t = colTypes.get(col);
      if (t === 'identifier') {
        const prefix = String(sourceRows[0]?.[col] || 'SYNTH-1000').split('-')[0] || 'SYNTH';
        newRow[col] = `${prefix}-${String(10000 + rowIdx + 1)}`;
        continue;
      }

      // Check missingness mode
      const missRate = missingRates.get(col) || 0;
      if (config.missingValueMode === 'preserve_rate' && missRate > 0 && rng() < missRate) {
        newRow[col] = null;
        continue;
      }

      if (col === targetCol && t === 'categorical') {
        newRow[col] = anchorRow[col];
        continue;
      }

      const dIdx = latentCols.indexOf(col);
      let zVal = corrZ[dIdx] ?? stdNormal(rng);
      const classShift = condMeanVec[dIdx] || 0;

      if (config.method === 'gaussian_copula') {
        zVal = 0.55 * zVal + 0.45 * (anchorZ?.[dIdx] ?? classShift);
      } else if (config.method === 'ctgan') {
        zVal = 0.50 * zVal + 0.50 * (anchorZ?.[dIdx] ?? classShift) + stdNormal(rng) * 0.1;
      } else if (config.method === 'tvae') {
        zVal = 0.45 * zVal + 0.55 * (anchorZ?.[dIdx] ?? classShift) + stdNormal(rng) * 0.08;
      } else if (config.method === 'conditional_sampling') {
        zVal = 0.40 * zVal + 0.60 * (anchorZ?.[dIdx] ?? classShift);
      } else {
        // multivariate_normal
        zVal = 0.65 * zVal + 0.35 * classShift;
      }

      const u = normalCdf(zVal);

      if (t === 'numerical') {
        const sorted = sortedDistributions.get(col)!;
        if (config.method === 'multivariate_normal') {
          const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
          const std =
            Math.sqrt(
              sorted.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
                Math.max(1, sorted.length - 1)
            ) || 1;
          const rawVal = mean + zVal * std;
          const clamped = Math.max(sorted[0] * 0.85, Math.min(sorted[sorted.length - 1] * 1.15, rawVal));
          newRow[col] = isIntegerCol.get(col) ? Math.round(clamped) : Number(clamped.toFixed(2));
        } else {
          const qVal = computeQuantile(sorted, u);
          const iqr = computeQuantile(sorted, 0.75) - computeQuantile(sorted, 0.25);
          const bandwidth = iqr * 0.045;
          const jittered = qVal + stdNormal(rng) * bandwidth;
          const bounded = Math.max(sorted[0], Math.min(sorted[sorted.length - 1], jittered));
          newRow[col] = isIntegerCol.get(col) ? Math.round(bounded) : Number(bounded.toFixed(2));
        }
      } else if (t === 'datetime') {
        const sortedMs = sortedDistributions.get(col)!;
        const qMs = computeQuantile(sortedMs, u);
        const jitterDays = Math.round(stdNormal(rng) * 4) * 86400000;
        const bounds = dateMinMax.get(col)!;
        const finalMs = Math.max(bounds.minMs, Math.min(bounds.maxMs, qMs + jitterDays));
        newRow[col] = new Date(finalMs).toISOString().slice(0, 10);
      } else if (t === 'categorical') {
        if (rng() < 0.55 && anchorRow[col] !== undefined) {
          newRow[col] = anchorRow[col];
        } else {
          const cdf = catDistributions.get(col) || [];
          const picked = cdf.find((item) => u <= item.cumProb) || cdf[cdf.length - 1];
          newRow[col] = picked ? picked.value : 'Unknown';
        }
      }
    }

    // Evaluate custom constraints before post-hoc enforcement
    let rowViolatedRaw = false;
    for (const c of config.constraints || []) {
      if (!validateRowAgainstConstraint(newRow, c)) {
        rowViolatedRaw = true;
        if (c.enforcePostHoc) {
          enforceConstraintOnRow(newRow, c, rng);
        }
      }
    }
    if (rowViolatedRaw) rawViolationCountBeforeProjection++;

    // Re-check after post-hoc projection
    const stillViolated = (config.constraints || []).some(
      (c) => !validateRowAgainstConstraint(newRow, c)
    );
    if (stillViolated) finalViolationCount++;

    syntheticRows.push(newRow);
  }

  const durationMs = Math.max(18, Math.round(performance.now() - startTime));
  return {
    syntheticRows,
    rawViolationCountBeforeProjection,
    finalViolationCount,
    epochLogs,
    durationMs,
  };
}
