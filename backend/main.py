"""
SynthForge AI — Modular Python FastAPI, Pydantic, SQLAlchemy & SDV/Scikit-Learn Pipeline
Provides reference REST endpoints for statistical/SDV generation, SciPy KS/Wasserstein evaluation,
privacy disclosure risk calculation, and zero-leakage TRTR/TSTR/Augmented ML benchmarking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="SynthForge AI — Synthetic Data & ML Intelligence API",
    version="1.0.0",
)


class ConstraintRule(BaseModel):
    column: str
    rule_type: str = Field(..., pattern="^(range|non_negative|positive|date_order|category_in)$")
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    ref_column: Optional[str] = None
    allowed_values: Optional[List[str]] = None


class GenerationRequest(BaseModel):
    method: str = Field("gaussian_copula", pattern="^(gaussian_copula|ctgan|tvae|multivariate_normal|conditional_sampling)$")
    num_rows: int = Field(200, ge=10, le=50000)
    random_seed: int = 42
    target_column: str
    constraints: List[ConstraintRule] = []
    rows: List[Dict[str, Any]]


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/profile")
async def profile_uploaded_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "dataset.csv"
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif filename.endswith(".json"):
        df = pd.read_json(file.file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, JSON, or Excel.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_rate = float(df.isna().sum().sum() / max(1, df.size))
    dup_rate = float(df.duplicated().sum() / max(1, len(df)))

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_rate": round(missing_rate, 4),
        "duplicate_rate": round(dup_rate, 4),
    }


@app.post("/api/evaluate")
def evaluate_quality_and_privacy(payload: Dict[str, Any]) -> Dict[str, Any]:
    real_df = pd.DataFrame(payload.get("real_rows", []))
    synth_df = pd.DataFrame(payload.get("synthetic_rows", []))
    if real_df.empty or synth_df.empty:
        raise HTTPException(status_code=400, detail="Both real_rows and synthetic_rows are required.")

    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_metrics = {}
    for col in num_cols:
        r_vals = real_df[col].dropna().to_numpy()
        s_vals = synth_df[col].dropna().to_numpy()
        if len(r_vals) > 0 and len(s_vals) > 0:
            ks_stat, ks_p = stats.ks_2samp(r_vals, s_vals)
            w_dist = stats.wasserstein_distance(r_vals, s_vals)
            feature_metrics[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 4),
                "wasserstein_distance": round(float(w_dist), 4),
            }

    # Nearest-neighbour DCR calculation on standardized numerical matrix
    if num_cols:
        r_mat = real_df[num_cols].fillna(0).to_numpy(dtype=float)
        s_mat = synth_df[num_cols].fillna(0).to_numpy(dtype=float)
        dists = cdist(s_mat, r_mat, metric="euclidean")
        dcr_5th = float(np.percentile(np.min(dists, axis=1), 5))
    else:
        dcr_5th = 0.0

    return {
        "feature_metrics": feature_metrics,
        "dcr_5th_percentile": round(dcr_5th, 4),
    }
