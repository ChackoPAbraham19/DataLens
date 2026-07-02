import json
import math
import traceback
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from services.ingestor import ingest, IngestorError
from services.visualizer import build_charts
from services.insight import generate_insight
from utils.file_handler import TEMP_DIR


# ---------------------------------------------------------------------------
# Helpers — JSON serialization
# ---------------------------------------------------------------------------

def _safe(val: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if np.isnan(val) else float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    return val


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively walk a data structure replacing float('nan'), float('inf'),
    and float('-inf') with None (JSON null).

    json.dumps with default= only calls the default handler for types it
    cannot serialize.  Plain Python float('nan') is *natively* serializable
    (it emits the non-standard token ``NaN``), so the `_safe` default
    handler never saw it.  Pre-processing with this function guarantees
    standards-compliant JSON that any parser can read.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Status writers
# ---------------------------------------------------------------------------

def _write_status(job_dir: Path, status: str, progress: int, step: str) -> None:
    (job_dir / "status.json").write_text(
        json.dumps({"status": status, "progress": progress, "step": step}),
        encoding="utf-8",
    )


def _write_result(job_dir: Path, result: dict) -> None:
    sanitized = _sanitize_for_json(result)
    (job_dir / "result.json").write_text(
        json.dumps(sanitized, default=_safe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_status(job_dir, "done", 100, "Complete")


def _write_error(job_dir: Path, message: str, detail: Optional[str] = None) -> None:
    (job_dir / "error.json").write_text(
        json.dumps({"message": message, "detail": detail}),
        encoding="utf-8",
    )
    _write_status(job_dir, "failed", 0, "Failed")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame, filename: str, metadata: dict) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3)

    return {
        "filename":        filename,
        "source":          metadata.get("source", "upload"),
        "dataset":         metadata.get("dataset"),
        "rows":            df.shape[0],
        "columns":         df.shape[1],
        "total_cells":     total_cells,
        "missing_cells":   missing_cells,
        "missing_pct":     round(missing_cells / total_cells * 100, 2) if total_cells else 0,
        "duplicate_rows":  duplicate_rows,
        "duplicate_pct":   round(duplicate_rows / df.shape[0] * 100, 2) if df.shape[0] else 0,
        "memory_usage_mb": memory_mb,
        "profiled_at":     datetime.now(timezone.utc).isoformat(),
    }


def _dtype_summary(df: pd.DataFrame) -> dict:
    """Categorize columns by inferred type."""
    numeric_cols    = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols       = df.select_dtypes(include=["bool"]).columns.tolist()

    columns_detail = {}
    for col in df.columns:
        columns_detail[col] = {
            "dtype":          str(df[col].dtype),
            "inferred_type":  (
                "numeric"    if col in numeric_cols    else
                "datetime"   if col in datetime_cols   else
                "boolean"    if col in bool_cols       else
                "categorical"
            ),
            "missing":        int(df[col].isna().sum()),
            "missing_pct":    round(df[col].isna().mean() * 100, 2),
            "unique":         int(df[col].nunique()),
            "unique_pct":     round(df[col].nunique() / len(df) * 100, 2) if len(df) else 0,
        }

    return {
        "numeric_count":     len(numeric_cols),
        "datetime_count":    len(datetime_cols),
        "categorical_count": len(categorical_cols),
        "boolean_count":     len(bool_cols),
        "numeric_cols":      numeric_cols,
        "datetime_cols":     datetime_cols,
        "categorical_cols":  categorical_cols,
        "boolean_cols":      bool_cols,
        "columns_detail":    columns_detail,
    }


def _missing_analysis(df: pd.DataFrame) -> dict:
    """Per-column missing value stats + heatmap data."""
    missing_per_col = df.isna().sum()
    missing_pct_per_col = (df.isna().mean() * 100).round(2)

    # Heatmap data: sample up to 100 rows for visualization
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    heatmap_matrix = sample_df.isna().astype(int).values.tolist()

    # Missing pattern: which combos of columns tend to be missing together
    missing_cols = missing_per_col[missing_per_col > 0].index.tolist()
    patterns = []
    if missing_cols and len(missing_cols) <= 20:
        pattern_df = df[missing_cols].isna().astype(str).agg("".join, axis=1)
        top_patterns = pattern_df.value_counts().head(5)
        for pattern_str, count in top_patterns.items():
            patterns.append({
                "pattern": {col: (pattern_str[i] == "1") for i, col in enumerate(missing_cols)},
                "count": int(count),
            })

    return {
        "per_column": {
            col: {
                "missing":     int(missing_per_col[col]),
                "missing_pct": float(missing_pct_per_col[col]),
            }
            for col in df.columns
        },
        "heatmap": {
            "columns":   df.columns.tolist(),
            "row_sample": sample_size,
            "matrix":    heatmap_matrix,
        },
        "patterns": patterns,
    }


def _numeric_stats(df: pd.DataFrame) -> dict:
    """Descriptive stats + skew + kurtosis for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}

    result = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue

        q1  = float(series.quantile(0.25))
        q3  = float(series.quantile(0.75))
        iqr = q3 - q1

        result[col] = {
            "count":    int(series.count()),
            "mean":     round(float(series.mean()), 4),
            "std":      round(float(series.std()), 4),
            "min":      float(series.min()),
            "q1":       q1,
            "median":   float(series.median()),
            "q3":       q3,
            "max":      float(series.max()),
            "iqr":      round(iqr, 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
            "cv":       round(float(series.std() / series.mean()), 4) if series.mean() != 0 else None,
        }

    return result


def _outlier_analysis(df: pd.DataFrame) -> dict:
    """
    IQR + Z-score outlier detection for numeric columns.
    Returns outlier counts and sample outlier row indices.
    """
    numeric_df = df.select_dtypes(include=["number"])
    result = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 10:
            continue

        # IQR method
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (series < lower) | (series > upper)
        iqr_outliers = int(iqr_mask.sum())

        # Z-score method (|z| > 3)
        z_scores = np.abs(scipy_stats.zscore(series))
        zscore_outliers = int((z_scores > 3).sum())

        # Boxplot data for frontend
        boxplot_data = {
            "min":    float(series.min()),
            "q1":     float(q1),
            "median": float(series.median()),
            "q3":     float(q3),
            "max":    float(series.max()),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outlier_values": series[iqr_mask].head(20).tolist(),
        }

        result[col] = {
            "iqr_outliers":    iqr_outliers,
            "iqr_pct":         round(iqr_outliers / len(series) * 100, 2),
            "zscore_outliers": zscore_outliers,
            "zscore_pct":      round(zscore_outliers / len(series) * 100, 2),
            "boxplot":         boxplot_data,
        }

    return result


def _correlation_analysis(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return {"available": False, "reason": "Need at least 2 numeric columns."}

    # Cap columns to avoid giant matrices
    max_cols = 20
    if numeric_df.shape[1] > max_cols:
        # Pick top N by variance
        variances = numeric_df.var().sort_values(ascending=False)
        numeric_df = numeric_df[variances.head(max_cols).index]

    corr = numeric_df.corr(method="pearson").round(4)
    cols = corr.columns.tolist()

    # Strong correlations (|r| >= 0.7, excluding self)
    strong_pairs = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= 0.7:
                strong_pairs.append({
                    "col1": c1, "col2": c2,
                    "r": round(float(r), 4),
                    "strength": "strong positive" if r >= 0.7 else "strong negative",
                })

    return {
        "available": True,
        "columns": cols,
        "matrix": corr.values.tolist(),
        "strong_pairs": strong_pairs,
    }


def _categorical_analysis(df: pd.DataFrame) -> dict:
    """Value counts + cardinality info for categorical columns."""
    cat_df = df.select_dtypes(include=["object", "category"])
    result = {}

    for col in cat_df.columns:
        series = df[col].dropna()
        vc = series.value_counts()
        total = len(series)

        result[col] = {
            "unique":       int(series.nunique()),
            "top_values":   [
                {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
                for k, v in vc.head(10).items()
            ],
            "cardinality":  (
                "high"   if series.nunique() / total > 0.5 else
                "medium" if series.nunique() / total > 0.1 else
                "low"
            ),
        }

    return result


def _distribution_data(df: pd.DataFrame) -> dict:
    """
    Histogram bucket data for numeric cols, value counts for categoricals.
    This feeds directly into the frontend chart components.
    """
    result = {}

    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        counts, bin_edges = np.histogram(series, bins=min(30, len(series.unique())))
        result[col] = {
            "type": "histogram",
            "bins": [
                {
                    "range_start": round(float(bin_edges[i]), 4),
                    "range_end":   round(float(bin_edges[i+1]), 4),
                    "count":       int(counts[i]),
                }
                for i in range(len(counts))
            ],
        }

    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts().head(15)
        result[col] = {
            "type":   "bar",
            "values": [{"label": str(k), "count": int(v)} for k, v in vc.items()],
        }

    return result


# ---------------------------------------------------------------------------
# Main entry point (called by BackgroundTasks)
# ---------------------------------------------------------------------------

def run_profile(
    job_id: str,
    csv_path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Full profiling pipeline. Writes status updates throughout.
    Called as a FastAPI BackgroundTask — must never raise.
    """
    metadata = metadata or {}
    job_dir = TEMP_DIR / job_id

    try:
        _write_status(job_dir, "running", 5, "Loading dataset...")

        # 1. Ingest
        df, filename = ingest(csv_path)
        _write_status(job_dir, "running", 20, "Computing overview...")

        # 2. Overview
        overview = _overview(df, filename, metadata)
        _write_status(job_dir, "running", 30, "Analyzing data types...")

        # 3. Dtype summary
        dtype_info = _dtype_summary(df)
        _write_status(job_dir, "running", 40, "Analyzing missing values...")

        # 4. Missing analysis
        missing = _missing_analysis(df)
        _write_status(job_dir, "running", 50, "Computing statistics...")

        # 5. Numeric stats
        num_stats = _numeric_stats(df)
        _write_status(job_dir, "running", 60, "Detecting outliers...")

        # 6. Outlier analysis
        outliers = _outlier_analysis(df)
        _write_status(job_dir, "running", 70, "Computing correlations...")

        # 7. Correlation
        correlation = _correlation_analysis(df)
        _write_status(job_dir, "running", 75, "Analyzing categorical columns...")

        # 8. Categorical analysis
        categorical = _categorical_analysis(df)
        _write_status(job_dir, "running", 80, "Building chart data...")

        # 9. Distribution data for charts
        distributions = _distribution_data(df)
        _write_status(job_dir, "running", 85, "Building charts...")

        # 10. Chart payloads (visualizer)
        charts = build_charts(
            distributions=distributions,
            missing=missing,
            correlation=correlation,
            outliers=outliers,
        )
        _write_status(job_dir, "running", 90, "Generating insights...")

        # 11. Insights (rule-based + Claude AI)
        profile_summary = {
            "overview":     overview,
            "dtype_info":   dtype_info,
            "missing":      missing,
            "numeric_stats": num_stats,
            "outliers":     outliers,
            "correlation":  correlation,
            "categorical":  categorical,
        }
        insights = generate_insight(profile_summary)
        _write_status(job_dir, "running", 98, "Finalizing...")

        # 12. Write result
        result = {
            "overview":      overview,
            "dtype_info":    dtype_info,
            "missing":       missing,
            "numeric_stats": num_stats,
            "outliers":      outliers,
            "correlation":   correlation,
            "categorical":   categorical,
            "distributions": distributions,
            "charts":        charts,
            "insights":      insights,
        }
        _write_result(job_dir, result)

    except IngestorError as e:
        _write_error(job_dir, str(e))

    except Exception as e:
        _write_error(
            job_dir,
            message="Profiling failed due to an unexpected error.",
            detail=traceback.format_exc(),
        )