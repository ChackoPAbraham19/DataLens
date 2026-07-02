import os
import json
from typing import Optional

import httpx


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Rule-based insight
# ---------------------------------------------------------------------------

def _rule_based_insight(profile: dict) -> dict:
    overview = profile.get("overview", {})
    missing = profile.get("missing", {})
    outliers = profile.get("outliers", {})
    correlation = profile.get("correlation", {})
    dtype_info = profile.get("dtype_info", {})

    flags: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    # Missing values
    missing_pct = overview.get("missing_pct", 0)
    if missing_pct > 30:
        warnings.append(f"High missing data: {missing_pct:.1f}% of all cells are null.")
        suggestions.append("Consider imputation strategies or dropping columns with >50% missing.")
    elif missing_pct > 10:
        flags.append(f"Moderate missing data: {missing_pct:.1f}% of cells are null.")
        suggestions.append("Investigate missing patterns — may indicate data collection issues.")

    # Per-column missing
    per_col = missing.get("per_column", {})
    high_missing_cols = [c for c, v in per_col.items() if v["missing_pct"] > 50]
    if high_missing_cols:
        warnings.append(f"{len(high_missing_cols)} column(s) have >50% missing: {', '.join(high_missing_cols[:5])}.")
        suggestions.append("Consider dropping columns with >50% missing unless domain knowledge justifies them.")

    # Duplicates
    dup_pct = overview.get("duplicate_pct", 0)
    if dup_pct > 5:
        warnings.append(f"Dataset has {overview.get('duplicate_rows', 0)} duplicate rows ({dup_pct:.1f}%).")
        suggestions.append("Deduplicate before training models to avoid data leakage.")
    elif dup_pct > 0:
        flags.append(f"Minor duplication: {overview.get('duplicate_rows', 0)} duplicate rows ({dup_pct:.1f}%).")

    # Outliers
    high_outlier_cols = [
        c for c, v in outliers.items() if v.get("iqr_pct", 0) > 10
    ]
    if high_outlier_cols:
        warnings.append(f"High outlier rate in: {', '.join(high_outlier_cols[:5])}.")
        suggestions.append("Investigate outliers — may need capping, transformation, or removal depending on domain.")

    # Correlations
    strong_pairs = correlation.get("strong_pairs", [])
    if strong_pairs:
        pair_strs = [f"{p['col1']} ↔ {p['col2']} (r={p['r']})" for p in strong_pairs[:3]]
        flags.append(f"Strong correlations found: {'; '.join(pair_strs)}.")
        suggestions.append("Highly correlated features may cause multicollinearity in linear models — consider dropping one.")

    # High cardinality
    cat_info = profile.get("categorical", {})
    high_card_cols = [c for c, v in cat_info.items() if v.get("cardinality") == "high"]
    if high_card_cols:
        flags.append(f"High cardinality categoricals: {', '.join(high_card_cols[:5])}.")
        suggestions.append("High cardinality columns need target encoding or embedding — avoid one-hot.")

    # Dataset size
    rows = overview.get("rows", 0)
    cols = overview.get("columns", 0)
    if rows < 500:
        warnings.append(f"Small dataset: only {rows} rows. ML models may underfit.")
        suggestions.append("Consider data augmentation or simpler models for small datasets.")
    if cols > 100:
        flags.append(f"Wide dataset: {cols} columns. Dimensionality reduction may help.")
        suggestions.append("Consider PCA or feature selection before modeling.")

    # Skewness
    num_stats = profile.get("numeric_stats", {})
    skewed_cols = [c for c, v in num_stats.items() if abs(v.get("skewness", 0)) > 2]
    if skewed_cols:
        flags.append(f"Highly skewed columns: {', '.join(skewed_cols[:5])}.")
        suggestions.append("Log or Box-Cox transform skewed features for linear/distance-based models.")

    quality_score = _compute_quality_score(overview, outliers, per_col, dup_pct)

    return {
        "quality_score": quality_score,
        "quality_label": _quality_label(quality_score),
        "flags": flags,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def _compute_quality_score(overview: dict, outliers: dict, per_col: dict, dup_pct: float) -> int:
    score = 100

    # Missing
    missing_pct = overview.get("missing_pct", 0)
    score -= min(30, missing_pct * 1.5)

    # Duplicates
    score -= min(15, dup_pct * 2)

    # Outlier-heavy columns
    heavy_outlier_count = sum(1 for v in outliers.values() if v.get("iqr_pct", 0) > 10)
    score -= min(20, heavy_outlier_count * 5)

    # High missing columns
    high_missing = sum(1 for v in per_col.values() if v["missing_pct"] > 50)
    score -= min(20, high_missing * 5)

    return max(0, int(score))


def _quality_label(score: int) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Fair"
    if score >= 30: return "Poor"
    return "Critical"


# ---------------------------------------------------------------------------
# Claude AI insight
# ---------------------------------------------------------------------------

def _build_prompt(profile: dict) -> str:
    overview = profile.get("overview", {})
    dtype_info = profile.get("dtype_info", {})
    missing = profile.get("missing", {})
    outliers = profile.get("outliers", {})
    correlation = profile.get("correlation", {})
    num_stats = profile.get("numeric_stats", {})

    strong_pairs = correlation.get("strong_pairs", [])
    top_outlier_cols = sorted(
        [(c, v["iqr_pct"]) for c, v in outliers.items()],
        key=lambda x: x[1], reverse=True
    )[:5]

    skewed = [
        (c, v["skewness"]) for c, v in num_stats.items()
        if abs(v.get("skewness", 0)) > 1
    ]

    prompt = f"""You are a senior data scientist reviewing an automated dataset quality report.
Provide a concise, expert-level insight summary for this dataset.

## Dataset Overview
- File: {overview.get('filename', 'unknown')}
- Rows: {overview.get('rows')} | Columns: {overview.get('columns')}
- Missing: {overview.get('missing_pct')}% of cells
- Duplicates: {overview.get('duplicate_rows')} rows ({overview.get('duplicate_pct')}%)
- Memory: {overview.get('memory_usage_mb')} MB

## Column Types
- Numeric: {dtype_info.get('numeric_count')} columns
- Categorical: {dtype_info.get('categorical_count')} columns
- Datetime: {dtype_info.get('datetime_count')} columns

## Top Missing Columns
{json.dumps({k: v['missing_pct'] for k, v in list(missing.get('per_column', {}).items())[:8]}, indent=2)}

## Top Outlier Columns (by IQR %)
{json.dumps([{"col": c, "iqr_outlier_pct": p} for c, p in top_outlier_cols], indent=2)}

## Strong Correlations (|r| >= 0.7)
{json.dumps(strong_pairs[:5], indent=2)}

## Skewed Columns (|skewness| > 1)
{json.dumps([{"col": c, "skewness": round(s, 2)} for c, s in skewed[:5]], indent=2)}

---
Respond in JSON only with this exact structure:
{{
  "summary": "2-3 sentence plain-English summary of overall data quality",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "ml_readiness": "Ready | Needs Cleaning | Not Ready",
  "ml_readiness_reason": "1 sentence explaining the ML readiness verdict",
  "top_actions": ["action 1", "action 2", "action 3"],
  "domain_guess": "Your best guess at what domain/use-case this dataset is for"
}}"""
    return prompt


def _claude_insight_sync(profile: dict) -> Optional[dict]:
    if not ANTHROPIC_API_KEY:
        return None

    prompt = _build_prompt(profile)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["content"][0]["text"].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            return json.loads(raw_text.strip())
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insight(profile: dict) -> dict:
    rule_based = _rule_based_insight(profile)
    ai_insight = _claude_insight_sync(profile)

    return {
        "rule_based": rule_based,
        "ai": ai_insight,
        "ai_available": ai_insight is not None,
    }