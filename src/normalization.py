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


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        str(column).strip() if str(column).strip() else f"column_{index + 1}"
        for index, column in enumerate(normalized.columns)
    ]
    normalized = normalized.dropna(axis=1, how="all")
    normalized = normalized.fillna("")

    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(str).str.strip()

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
    safe_rows = max(1, min(max_rows, 100))
    safe_cols = max(1, min(max_cols, 50))

    limited = df.iloc[:safe_rows, :safe_cols].astype(str)
    headers = [str(column) for column in limited.columns]
    rows = limited.values.tolist()
    return headers, rows
