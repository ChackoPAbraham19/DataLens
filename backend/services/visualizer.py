from typing import Any


def build_charts(
    distributions: dict,
    missing: dict,
    correlation: dict,
    outliers: dict,
) -> dict:
    return {
        "distributions": _distribution_charts(distributions),
        "missing_heatmap": _missing_heatmap_chart(missing),
        "correlation_matrix": _correlation_chart(correlation),
        "outliers": _outlier_charts(outliers),
    }


def _distribution_charts(distributions: dict) -> list[dict]:
    charts = []
    for col, data in distributions.items():
        if data["type"] == "histogram":
            charts.append({
                "column": col,
                "type": "histogram",
                "data": [
                    {
                        "name": f"{b['range_start']}–{b['range_end']}",
                        "count": b["count"],
                        "range_start": b["range_start"],
                        "range_end": b["range_end"],
                    }
                    for b in data["bins"]
                ],
            })
        elif data["type"] == "bar":
            charts.append({
                "column": col,
                "type": "bar",
                "data": [
                    {"name": v["label"], "count": v["count"]}
                    for v in data["values"]
                ],
            })
    return charts


def _missing_heatmap_chart(missing: dict) -> dict:
    heatmap = missing.get("heatmap", {})
    columns = heatmap.get("columns", [])
    matrix = heatmap.get("matrix", [])

    # Recharts-friendly: list of row objects
    rows = []
    for i, row in enumerate(matrix):
        row_obj: dict[str, Any] = {"row": i}
        for j, col in enumerate(columns):
            row_obj[col] = row[j] if j < len(row) else 0
        rows.append(row_obj)

    per_column = missing.get("per_column", {})
    bar_data = [
        {
            "column": col,
            "missing_pct": per_column[col]["missing_pct"],
            "missing": per_column[col]["missing"],
        }
        for col in columns
        if col in per_column
    ]

    return {
        "heatmap_rows": rows,
        "columns": columns,
        "bar_data": bar_data,
    }


def _correlation_chart(correlation: dict) -> dict:
    if not correlation.get("available"):
        return {"available": False, "reason": correlation.get("reason", "")}

    cols = correlation["columns"]
    matrix = correlation["matrix"]

    # Recharts cell format: [{x_col, y_col, value}]
    cells = []
    for i, row_col in enumerate(cols):
        for j, col_col in enumerate(cols):
            cells.append({
                "x": col_col,
                "y": row_col,
                "value": round(matrix[i][j], 4),
            })

    return {
        "available": True,
        "columns": cols,
        "cells": cells,
        "strong_pairs": correlation.get("strong_pairs", []),
    }


def _outlier_charts(outliers: dict) -> list[dict]:
    charts = []
    for col, data in outliers.items():
        bp = data.get("boxplot", {})
        charts.append({
            "column": col,
            "iqr_outliers": data["iqr_outliers"],
            "iqr_pct": data["iqr_pct"],
            "zscore_outliers": data["zscore_outliers"],
            "zscore_pct": data["zscore_pct"],
            "boxplot": {
                "min":          bp.get("min"),
                "q1":           bp.get("q1"),
                "median":       bp.get("median"),
                "q3":           bp.get("q3"),
                "max":          bp.get("max"),
                "lower_fence":  bp.get("lower_fence"),
                "upper_fence":  bp.get("upper_fence"),
                "outlier_values": bp.get("outlier_values", []),
            },
        })
    return charts