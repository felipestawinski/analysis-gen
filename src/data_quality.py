"""
Data health check and cleaning utilities for the analysis-gen service.
"""

from __future__ import annotations

import pandas as pd
from normalization import _is_text_dtype


def perform_data_health_check(df: pd.DataFrame) -> str:
    """Analyze DataFrame health and return a plain-text summary."""
    lines: list[str] = []
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    lines.append("=== DATA HEALTH CHECK REPORT ===\n")
    lines.append(f"Rows: {total_rows}")
    lines.append(f"Columns: {total_cols}")
    lines.append(f"Total cells: {total_cells}\n")

    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells else 0

    lines.append("--- Missing Values ---")
    lines.append(f"Total missing cells: {total_missing} ({missing_pct:.2f}%)")

    cols_with_missing = missing[missing > 0]
    if cols_with_missing.empty:
        lines.append("No columns with missing values.\n")
    else:
        lines.append(f"Columns affected: {len(cols_with_missing)}/{total_cols}")
        for col, count in cols_with_missing.items():
            pct = count / total_rows * 100
            lines.append(f"  • {col}: {count} missing ({pct:.1f}%)")
        lines.append("")

    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / total_rows * 100) if total_rows else 0
    lines.append("--- Duplicate Rows ---")
    lines.append(f"Duplicate rows: {dup_count} ({dup_pct:.2f}%)\n")

    lines.append("--- Column Data Types ---")
    for col in df.columns:
        lines.append(f"  • {col}: {df[col].dtype}")
    lines.append("")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        lines.append("--- Numeric Column Statistics ---")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                lines.append(f"  {col}: all values are missing")
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            lines.append(f"  {col}:")
            lines.append(f"    min={series.min()}, max={series.max()}, "
                         f"mean={series.mean():.2f}, median={series.median():.2f}")
            lines.append(f"    outliers (IQR method): {outlier_count}")
        lines.append("")

    issues = 0
    if missing_pct > 5:
        issues += 1
    if dup_pct > 5:
        issues += 1
    if any(cols_with_missing.get(c, 0) / total_rows > 0.3 for c in df.columns):
        issues += 1

    if issues == 0:
        verdict = "GOOD – The dataset looks healthy."
    elif issues == 1:
        verdict = "FAIR – Minor quality issues detected."
    else:
        verdict = "POOR – Significant quality issues found. Review the details above."

    lines.append(f"Overall health verdict: {verdict}")
    return "\n".join(lines)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a DataFrame: drop dupes, drop all-null cols, fill nulls, strip strings."""
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])

    for col in df.columns:
        if _is_text_dtype(df[col]):
            df[col] = df[col].str.strip()

    return df
