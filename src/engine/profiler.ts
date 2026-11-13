import {
  ColumnDataType,
  ColumnProfile,
  ClassImbalanceInfo,
  FeatureCorrelation,
  ProfilingReportData,
  TransformationActionType,
  TransformationStep,
} from '../types/synthforge';

function isNumericValue(v: any): boolean {
  if (v === null || v === undefined || v === '' || typeof v === 'boolean') return false;
  if (typeof v === 'number') return !Number.isNaN(v) && Number.isFinite(v);
  if (typeof v === 'string') {
    const trimmed = v.trim();
    if (trimmed === '') return false;
    const n = Number(trimmed);
    return !Number.isNaN(n) && Number.isFinite(n);
  }
  return false;
}

function isDateString(v: any): boolean {
  if (typeof v !== 'string') return false;
  const trimmed = v.trim();
  // Match YYYY-MM-DD or ISO timestamps
  if (!/^\d{4}-\d{2}-\d{2}/.test(trimmed)) return false;
  const d = Date.parse(trimmed);
  return !Number.isNaN(d);
}

export function inferColumnType(rows: Record<string, any>[], col: string): ColumnDataType {
  const nonNull = rows
    .map((r) => r[col])
    .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');

  if (nonNull.length === 0) return 'categorical';

  const lowerName = col.toLowerCase();
  if (
    lowerName.endsWith('_id') ||
    lowerName === 'id' ||
    lowerName === 'customer_id' ||
    lowerName === 'transaction_id' ||
    lowerName === 'application_id' ||
    lowerName === 'batch_id'
  ) {
    const uniqueSet = new Set(nonNull.map(String));
    if (uniqueSet.size / nonNull.length > 0.9) {
      return 'identifier';
    }
  }

  const dateCount = nonNull.filter(isDateString).length;
  if (dateCount / nonNull.length >= 0.8) {
    return 'datetime';
  }

  const numCount = nonNull.filter(isNumericValue).length;
  if (numCount / nonNull.length >= 0.85) {
    const uniqueNums = new Set(nonNull.map((v) => Number(v)));
    // Binary 0/1 flags or very low cardinality targets can still be treated as numerical or categorical;
    // keep binary 0/1 with categorical-like names or <= 2 unique values if string, else numerical
    if (uniqueNums.size <= 2 && (lowerName.startsWith('is_') || lowerName.includes('churn') || lowerName.includes('defect') || lowerName.includes('risk'))) {
      return 'categorical';
    }
    return 'numerical';
  }

  return 'categorical';
}

export function computeQuantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

export function profileDataset(
  datasetId: string,
  rows: Record<string, any>[],
  targetColumn?: string
): ProfilingReportData {
  if (!rows || rows.length === 0) {
    return {
      datasetId,
      rowCount: 0,
      columnCount: 0,
      numericalCount: 0,
      categoricalCount: 0,
      datetimeCount: 0,
      totalMissingCount: 0,
      missingRate: 0,
      duplicateRowCount: 0,
      duplicateRate: 0,
      totalOutlierCount: 0,
      outlierRate: 0,
      qualityScore: 0,
      columns: [],
      correlations: [],
      warnings: [],
    };
  }

  const colNames = Object.keys(rows[0]);
  const rowCount = rows.length;
  const columnCount = colNames.length;

  let totalMissingCount = 0;
  let totalOutlierCount = 0;
  let totalNumericCells = 0;

  // Duplicate detection
  const seenRowHashes = new Set<string>();
  let duplicateRowCount = 0;
  for (const r of rows) {
    const key = colNames.map((c) => String(r[c] ?? 'NULL')).join('|');
    if (seenRowHashes.has(key)) {
      duplicateRowCount++;
    } else {
      seenRowHashes.add(key);
    }
  }

  const columns: ColumnProfile[] = [];
  const warnings: ProfilingReportData['warnings'] = [];

  for (const col of colNames) {
    const type = inferColumnType(rows, col);
    let missingCount = 0;
    const rawValues: any[] = [];

    for (const r of rows) {
      const v = r[col];
      if (v === null || v === undefined || String(v).trim() === '' || Number.isNaN(v)) {
        missingCount++;
      } else {
        rawValues.push(v);
      }
    }

    totalMissingCount += missingCount;
    const missingRate = rowCount > 0 ? missingCount / rowCount : 0;
    const uniqueValues = new Set(rawValues.map(String));
    const uniqueCount = uniqueValues.size;
    const cardinalityRate = rawValues.length > 0 ? uniqueCount / rawValues.length : 0;

    if (missingRate > 0.08) {
      warnings.push({
        severity: missingRate > 0.2 ? 'high' : 'medium',
        column: col,
        message: `Column "${col}" has ${(missingRate * 100).toFixed(1)}% missing values (${missingCount} rows).`,
        remediation: `Apply median/mode imputation in Data Preparation or enable Missing-Value Imputation in the Generator.`,
      });
    }

    if (type === 'numerical') {
      const nums = rawValues.map((v) => Number(v)).filter((n) => !Number.isNaN(n) && Number.isFinite(n));
      totalNumericCells += nums.length;
      const sorted = [...nums].sort((a, b) => a - b);
      const n = sorted.length;
      const mean = n > 0 ? sorted.reduce((acc, v) => acc + v, 0) / n : 0;
      const variance =
        n > 1 ? sorted.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (n - 1) : 0;
      const std = Math.sqrt(variance);
      const min = n > 0 ? sorted[0] : 0;
      const max = n > 0 ? sorted[n - 1] : 0;
      const q25 = computeQuantile(sorted, 0.25);
      const median = computeQuantile(sorted, 0.5);
      const q75 = computeQuantile(sorted, 0.75);
      const iqr = q75 - q25;
      const lowerFence = q25 - 1.5 * iqr;
      const upperFence = q75 + 1.5 * iqr;

      let outlierCount = 0;
      if (iqr > 0) {
        outlierCount = sorted.filter((v) => v < lowerFence || v > upperFence).length;
      } else if (std > 0) {
        outlierCount = sorted.filter((v) => Math.abs((v - mean) / std) > 3).length;
      }
      totalOutlierCount += outlierCount;
      const outlierRate = n > 0 ? outlierCount / n : 0;

      // Skewness
      const skewness =
        n > 2 && std > 1e-9
          ? sorted.reduce((acc, v) => acc + Math.pow((v - mean) / std, 3), 0) / n
          : 0;

      if (outlierRate > 0.05) {
        warnings.push({
          severity: outlierRate > 0.12 ? 'high' : 'medium',
          column: col,
          message: `Column "${col}" contains ${outlierCount} IQR outliers (${(outlierRate * 100).toFixed(1)}%) with skewness ${skewness.toFixed(2)}.`,
          remediation: `Consider IQR outlier clipping or use Gaussian Copula / TVAE to model heavy tails safely.`,
        });
      }

      // Build 8-bin histogram
      const binCount = 8;
      const span = max - min || 1;
      const binWidth = span / binCount;
      const histogram = Array.from({ length: binCount }, (_, idx) => {
        const bMin = min + idx * binWidth;
        const bMax = idx === binCount - 1 ? max + 1e-9 : min + (idx + 1) * binWidth;
        const cnt = sorted.filter((v) => v >= bMin && (idx === binCount - 1 ? v <= bMax : v < bMax)).length;
        return {
          bin: `${bMin.toFixed(1)}–${(min + (idx + 1) * binWidth).toFixed(1)}`,
          count: cnt,
          minVal: Number(bMin.toFixed(3)),
          maxVal: Number((min + (idx + 1) * binWidth).toFixed(3)),
        };
      });

      columns.push({
        name: col,
        type,
        missingCount,
        missingRate: Number(missingRate.toFixed(4)),
        uniqueCount,
        cardinalityRate: Number(cardinalityRate.toFixed(4)),
        mean: Number(mean.toFixed(3)),
        std: Number(std.toFixed(3)),
        min: Number(min.toFixed(3)),
        q25: Number(q25.toFixed(3)),
        median: Number(median.toFixed(3)),
        q75: Number(q75.toFixed(3)),
        max: Number(max.toFixed(3)),
        skewness: Number(skewness.toFixed(3)),
        outlierCount,
        outlierRate: Number(outlierRate.toFixed(4)),
        histogram,
      });
    } else {
      const counts = new Map<string, number>();
      for (const v of rawValues) {
        const s = String(v);
        counts.set(s, (counts.get(s) || 0) + 1);
      }
      const topCategories = Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([value, count]) => ({
          value,
          count,
          frequency: rawValues.length > 0 ? Number((count / rawValues.length).toFixed(4)) : 0,
        }));

      let minDate: string | undefined;
      let maxDate: string | undefined;
      if (type === 'datetime' && rawValues.length > 0) {
        const sortedDates = rawValues.map(String).sort();
        minDate = sortedDates[0];
        maxDate = sortedDates[sortedDates.length - 1];
      }

      columns.push({
        name: col,
        type,
        missingCount,
        missingRate: Number(missingRate.toFixed(4)),
        uniqueCount,
        cardinalityRate: Number(cardinalityRate.toFixed(4)),
        topCategories,
        minDate,
        maxDate,
      });
    }
  }

  // Compute pairwise Pearson correlations for numerical columns
  const numericCols = columns.filter((c) => c.type === 'numerical').map((c) => c.name);
  const correlations: FeatureCorrelation[] = [];

  for (let i = 0; i < numericCols.length; i++) {
    for (let j = 0; j < numericCols.length; j++) {
      const colA = numericCols[i];
      const colB = numericCols[j];
      if (i === j) {
        correlations.push({ colA, colB, correlation: 1 });
        continue;
      }
      const pairs: [number, number][] = [];
      for (const r of rows) {
        if (isNumericValue(r[colA]) && isNumericValue(r[colB])) {
          pairs.push([Number(r[colA]), Number(r[colB])]);
        }
      }
      if (pairs.length < 3) {
        correlations.push({ colA, colB, correlation: 0 });
        continue;
      }
      const meanA = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
      const meanB = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
      let num = 0;
      let denA = 0;
      let denB = 0;
      for (const [a, b] of pairs) {
        const da = a - meanA;
        const db = b - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
      }
      const corr = denA > 0 && denB > 0 ? num / Math.sqrt(denA * denB) : 0;
      correlations.push({
        colA,
        colB,
        correlation: Number(Math.max(-1, Math.min(1, corr)).toFixed(3)),
      });
    }
  }

  // Class imbalance on targetColumn (or last categorical column)
  const resolvedTarget =
    targetColumn && colNames.includes(targetColumn)
      ? targetColumn
      : columns.filter((c) => c.type === 'categorical').slice(-1)[0]?.name || colNames[colNames.length - 1];

  let classImbalance: ClassImbalanceInfo | undefined;
  const targetProfile = columns.find((c) => c.name === resolvedTarget);
  if (targetProfile && targetProfile.topCategories && targetProfile.topCategories.length >= 2) {
    const classes = targetProfile.topCategories.map((c) => ({
      label: c.value,
      count: c.count,
      share: c.frequency,
    }));
    const maxShare = Math.max(...classes.map((c) => c.share));
    const minShare = Math.max(0.001, Math.min(...classes.map((c) => c.share)));
    const imbalanceRatio = Number((maxShare / minShare).toFixed(2));
    const isImbalanced = imbalanceRatio >= 2.2;
    const entropy = Number(
      (-classes.reduce((acc, c) => acc + (c.share > 0 ? c.share * Math.log2(c.share) : 0), 0)).toFixed(3)
    );

    classImbalance = {
      targetColumn: resolvedTarget,
      classes,
      imbalanceRatio,
      minorityShare: Number(minShare.toFixed(4)),
      isImbalanced,
      entropy,
    };

    if (isImbalanced) {
      warnings.push({
        severity: imbalanceRatio > 4 ? 'high' : 'medium',
        column: resolvedTarget,
        message: `Target "${resolvedTarget}" exhibits class imbalance (ratio ${imbalanceRatio}:1, minority class ${(minShare * 100).toFixed(1)}%).`,
        remediation: `Use Conditional Sampling or CTGAN conditional vector oversampling to balance minority class representation.`,
      });
    }
  }

  if (duplicateRowCount > 0) {
    warnings.push({
      severity: duplicateRowCount / rowCount > 0.05 ? 'high' : 'info',
      message: `Detected ${duplicateRowCount} duplicate records (${((duplicateRowCount / rowCount) * 100).toFixed(1)}% of dataset).`,
      remediation: `Deduplicate records in the Data Preparation tab to avoid over-weighting repeated observations.`,
    });
  }

  const totalCells = rowCount * columnCount;
  const missingRate = totalCells > 0 ? totalMissingCount / totalCells : 0;
  const duplicateRate = rowCount > 0 ? duplicateRowCount / rowCount : 0;
  const outlierRate = totalNumericCells > 0 ? totalOutlierCount / totalNumericCells : 0;

  const rawQuality =
    100 -
    missingRate * 140 -
    duplicateRate * 120 -
    outlierRate * 85 -
    (classImbalance?.isImbalanced ? Math.min(8, (classImbalance.imbalanceRatio - 2) * 1.5) : 0);
  const qualityScore = Number(Math.max(40, Math.min(99.5, rawQuality)).toFixed(1));

  return {
    datasetId,
    rowCount,
    columnCount,
    numericalCount: columns.filter((c) => c.type === 'numerical').length,
    categoricalCount: columns.filter((c) => c.type === 'categorical').length,
    datetimeCount: columns.filter((c) => c.type === 'datetime').length,
    totalMissingCount,
    missingRate: Number(missingRate.toFixed(4)),
    duplicateRowCount,
    duplicateRate: Number(duplicateRate.toFixed(4)),
    totalOutlierCount,
    outlierRate: Number(outlierRate.toFixed(4)),
    qualityScore,
    columns,
    correlations,
    classImbalance,
    warnings,
  };
}

// Data Preparation & Curation Engine (supports preview & apply)
export function applyDataTransformation(
  rows: Record<string, any>[],
  action: TransformationActionType,
  column?: string,
  strategy?: string
): { transformedRows: Record<string, any>[]; step: TransformationStep } {
  const cloned = rows.map((r) => ({ ...r }));
  const rowsBefore = cloned.length;
  let result = cloned;
  let description = '';

  if (action === 'drop_duplicates') {
    const seen = new Set<string>();
    const cols = Object.keys(cloned[0] || {});
    result = cloned.filter((r) => {
      const k = cols.map((c) => String(r[c] ?? 'NULL')).join('|');
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    description = `Removed ${rowsBefore - result.length} exact duplicate records.`;
  } else if (action === 'impute_missing' && column) {
    const colType = inferColumnType(cloned, column);
    const valid = cloned
      .map((r) => r[column])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== '');
    let fillVal: any = '';
    if (colType === 'numerical') {
      const nums = valid.map(Number).sort((a, b) => a - b);
      if (strategy === 'mean') {
        fillVal = Number((nums.reduce((a, b) => a + b, 0) / (nums.length || 1)).toFixed(2));
      } else {
        fillVal = Number(computeQuantile(nums, 0.5).toFixed(2));
      }
    } else {
      const freq = new Map<string, number>();
      valid.forEach((v) => freq.set(String(v), (freq.get(String(v)) || 0) + 1));
      fillVal = Array.from(freq.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';
    }
    let imputedCount = 0;
    result = cloned.map((r) => {
      if (r[column] === null || r[column] === undefined || String(r[column]).trim() === '') {
        imputedCount++;
        return { ...r, [column]: fillVal };
      }
      return r;
    });
    description = `Imputed ${imputedCount} missing values in "${column}" using ${strategy || 'median/mode'} (${fillVal}).`;
  } else if (action === 'clip_outliers_iqr' && column) {
    const nums = cloned
      .map((r) => Number(r[column]))
      .filter((n) => !Number.isNaN(n))
      .sort((a, b) => a - b);
    const q25 = computeQuantile(nums, 0.25);
    const q75 = computeQuantile(nums, 0.75);
    const iqr = q75 - q25;
    const low = Number((q25 - 1.5 * iqr).toFixed(2));
    const high = Number((q75 + 1.5 * iqr).toFixed(2));
    let clipped = 0;
    result = cloned.map((r) => {
      const v = Number(r[column]);
      if (Number.isNaN(v)) return r;
      if (v < low) {
        clipped++;
        return { ...r, [column]: low };
      }
      if (v > high) {
        clipped++;
        return { ...r, [column]: high };
      }
      return r;
    });
    description = `Clipped ${clipped} outliers in "${column}" to IQR bounds [${low}, ${high}].`;
  } else if (action === 'normalize_zscore' && column) {
    const nums = cloned.map((r) => Number(r[column])).filter((n) => !Number.isNaN(n));
    const mean = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
    const std = Math.sqrt(
      nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, nums.length - 1)
    ) || 1;
    result = cloned.map((r) => {
      const v = Number(r[column]);
      if (Number.isNaN(v)) return r;
      return { ...r, [column]: Number(((v - mean) / std).toFixed(4)) };
    });
    description = `Standardized "${column}" via Z-score (μ=${mean.toFixed(2)}, σ=${std.toFixed(2)}).`;
  } else if (action === 'normalize_minmax' && column) {
    const nums = cloned.map((r) => Number(r[column])).filter((n) => !Number.isNaN(n));
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const span = max - min || 1;
    result = cloned.map((r) => {
      const v = Number(r[column]);
      if (Number.isNaN(v)) return r;
      return { ...r, [column]: Number(((v - min) / span).toFixed(4)) };
    });
    description = `Scaled "${column}" to [0, 1] Min-Max range (min=${min.toFixed(2)}, max=${max.toFixed(2)}).`;
  } else if (action === 'encode_categorical' && column) {
    const uniqueVals = Array.from(
      new Set(cloned.map((r) => String(r[column] ?? 'Missing')))
    ).sort();
    const mapping = new Map<string, number>();
    uniqueVals.forEach((v, i) => mapping.set(v, i));
    result = cloned.map((r) => ({
      ...r,
      [column]: mapping.get(String(r[column] ?? 'Missing')) ?? 0,
    }));
    description = `Ordinal-encoded "${column}" across ${uniqueVals.length} categories (${uniqueVals.slice(0, 4).join(', ')}).`;
  } else if (action === 'drop_column' && column) {
    result = cloned.map((r) => {
      const copy = { ...r };
      delete copy[column];
      return copy;
    });
    description = `Dropped feature column "${column}" via feature selection.`;
  } else if (action === 'balance_classes_oversample' && column) {
    const groups = new Map<string, Record<string, any>[]>();
    for (const r of cloned) {
      const k = String(r[column] ?? 'Unknown');
      const arr = groups.get(k) || [];
      arr.push(r);
      groups.set(k, arr);
    }
    const maxCount = Math.max(...Array.from(groups.values()).map((g) => g.length));
    const balanced: Record<string, any>[] = [...cloned];
    groups.forEach((groupRows) => {
      if (groupRows.length > 0 && groupRows.length < maxCount) {
        const needed = Math.min(maxCount - groupRows.length, Math.floor(groupRows.length * 1.5));
        for (let i = 0; i < needed; i++) {
          const baseRow = { ...groupRows[i % groupRows.length] };
          balanced.push(baseRow);
        }
      }
    });
    result = balanced;
    description = `Balanced minority classes on "${column}" (+${result.length - rowsBefore} minority records).`;
  }

  return {
    transformedRows: result,
    step: {
      id: `step_${Date.now()}`,
      action,
      column,
      strategy,
      description,
      rowsBefore,
      rowsAfter: result.length,
      appliedAt: new Date().toISOString(),
    },
  };
}
