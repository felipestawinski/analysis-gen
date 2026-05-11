from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

SUPPORTED_FILE_TYPES = {"csv", "xlsx", "json"}


def detect_file_type(
    explicit_type: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    if explicit_type:
        normalized = explicit_type.strip().lower().lstrip(".")
        if normalized in SUPPORTED_FILE_TYPES:
            return normalized

    if filename:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix in SUPPORTED_FILE_TYPES:
            return suffix

    if content_type:
        content_type = content_type.lower()
        if "json" in content_type:
            return "json"
        if "sheet" in content_type or "excel" in content_type:
            return "xlsx"
        if "csv" in content_type or "text/plain" in content_type:
            return "csv"

    return "csv"


def _is_text_dtype(col_or_dtype) -> bool:
    """Return True for both legacy object dtype AND pandas 3.x StringDtype."""
    return (
        pd.api.types.is_object_dtype(col_or_dtype)
        or pd.api.types.is_string_dtype(col_or_dtype)
    )


def _try_convert_comma_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Detect object/string columns with comma-decimal numbers (e.g. ``'1234,56'``
    for 1234.56, common in Brazilian/European CSVs) and convert to float64.

    Only converts when >50 % of non-empty values match the comma-decimal
    pattern.  Values without a comma are parsed with dot-as-decimal to
    avoid mangling values like ``'4.0'``.
    """
    text_cols = [c for c in df.columns if _is_text_dtype(df[c])]
    for col in text_cols:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        non_empty = non_null[non_null.astype(str).str.strip() != ""]
        if len(non_empty) < 3:
            continue

        sample = non_empty.head(50).astype(str).str.strip().str.strip('"')
        comma_match = sample.str.match(r'^-?\d[\d.]*,\d+$')
        if comma_match.mean() < 0.5:
            continue

        try:
            raw = df[col].astype(str).str.strip().str.strip('"')
            has_comma = raw.str.contains(",", na=False, regex=False)

            converted = pd.Series(index=df.index, dtype=float)

            # Comma-decimal values: dot is thousands separator → remove,
            # then comma → dot.
            if has_comma.any():
                c = (
                    raw[has_comma]
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                converted.loc[has_comma] = pd.to_numeric(c, errors="coerce")

            # Non-comma values: parse normally (dot = decimal)
            if (~has_comma).any():
                converted.loc[~has_comma] = pd.to_numeric(
                    raw[~has_comma], errors="coerce"
                )

            # Only apply if we preserved a reasonable fraction of data
            if converted.notna().sum() >= len(non_empty) * 0.4:
                df[col] = converted
        except Exception:
            pass
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        str(column).strip() if str(column).strip() else f"column_{index + 1}"
        for index, column in enumerate(normalized.columns)
    ]
    normalized = normalized.dropna(axis=1, how="all")

    # Convert comma-decimal columns (e.g. "1234,56" → 1234.56, common in
    # Brazilian/European CSV exports) to float64 BEFORE the string-fill
    # step so they are correctly typed as numeric.
    normalized = _try_convert_comma_decimals(normalized)

    # Only fill NaN with empty string for object/string columns.
    # Using a blanket fillna("") converts numeric columns (float64) to
    # object dtype whenever they contain even a single NaN, which breaks
    # downstream column-type detection (e.g. _select_relevant_columns).
    for column in normalized.columns:
        if _is_text_dtype(normalized[column]):
            normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    return normalized


def dataframe_from_bytes(file_bytes: bytes, file_type: str) -> pd.DataFrame:
    normalized_type = detect_file_type(explicit_type=file_type)

    if normalized_type == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
    elif normalized_type == "xlsx":
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
    elif normalized_type == "json":
        parsed = json.loads(file_bytes.decode("utf-8", errors="replace"))
        if isinstance(parsed, list):
            df = pd.json_normalize(parsed)
        elif isinstance(parsed, dict):
            list_key = next((key for key, value in parsed.items() if isinstance(value, list)), None)
            if list_key:
                df = pd.json_normalize(parsed[list_key])
            else:
                df = pd.json_normalize([parsed])
        else:
            raise ValueError("Unsupported JSON structure for tabular analysis.")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return normalize_dataframe(df)


def dataframe_preview(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 12) -> Tuple[list[str], list[list[str]]]:
    import math

    safe_rows = max(1, min(max_rows, 200))
    safe_cols = max(1, min(max_cols, 100))

    limited = df.iloc[:safe_rows, :safe_cols]
    headers = [str(column) for column in limited.columns]

    def _safe_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return ""
        s = str(val)
        return "" if s == "nan" else s

    rows = [
        [_safe_str(cell) for cell in row]
        for row in limited.values.tolist()
    ]
    return headers, rows

