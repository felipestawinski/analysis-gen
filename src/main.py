from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
import asyncio
import csv
import io
import matplotlib.pyplot as plt
import pandas as pd
import base64
import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import requests
from typing import List, Optional
import seaborn as sns
import time
import threading
import contextvars
from collections import OrderedDict
from normalization import detect_file_type, dataframe_from_bytes, dataframe_preview

# Load environment variables BEFORE imports that read env vars at module level
load_dotenv()

from rag import rag_store

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Per-request token tracking via contextvars (async-safe)
_request_tokens = contextvars.ContextVar('_request_tokens', default=0)

def _track_tokens(response):
    """Extract total_tokens from an OpenAI response and accumulate."""
    try:
        if response:
            total = 0
            if hasattr(response, 'usage') and response.usage:
                total = getattr(response.usage, 'total_tokens', 0)
            elif isinstance(response, dict) and 'usage' in response:
                total = response['usage'].get('total_tokens', 0)
                
            print(f"DEBUG_TOKENS: Extracted {total} tokens from response")
            _request_tokens.set(_request_tokens.get(0) + total)
    except Exception as e:
        print(f"DEBUG_TOKENS_ERROR: Exception extracting tokens: {e}")
    return response

class URLRequest(BaseModel):
    fileAddresses: List[HttpUrl]
    fileTypes: Optional[List[str]] = None
    prompt: str
    generateChart: bool = False
    chartRecommendation: bool = False
    chatId: Optional[str] = None
    forceRefresh: bool = False
    model: str = "gpt-5-mini"


class PreviewRequest(BaseModel):
    fileAddress: HttpUrl
    fileType: Optional[str] = None
    maxRows: int = 20
    maxCols: int = 12
    forceRefresh: bool = False


class PreloadRequest(BaseModel):
    fileAddress: str
    fileType: Optional[str] = None


CSV_CACHE_TTL_SECONDS = int(os.getenv("CSV_CACHE_TTL_SECONDS", "1800"))
CSV_CACHE_MAX_BYTES = int(os.getenv("CSV_CACHE_MAX_BYTES", str(200 * 1024 * 1024)))
CSV_CACHE_MAX_ENTRIES = int(os.getenv("CSV_CACHE_MAX_ENTRIES", "128"))
MAX_FILES_PER_ANALYSIS_REQUEST = int(os.getenv("MAX_FILES_PER_ANALYSIS_REQUEST", "4"))
MAX_PARALLEL_DOWNLOADS = int(os.getenv("MAX_PARALLEL_DOWNLOADS", "4"))
LARGE_DATASET_ROW_THRESHOLD = int(os.getenv("LARGE_DATASET_ROW_THRESHOLD", "120000"))
LARGE_DATASET_CELL_THRESHOLD = int(os.getenv("LARGE_DATASET_CELL_THRESHOLD", "1500000"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "120000"))
LLM_COMPACT_CONTEXT_CHARS = int(os.getenv("LLM_COMPACT_CONTEXT_CHARS", "80000"))

csv_cache: OrderedDict[str, dict] = OrderedDict()
csv_cache_total_bytes = 0
csv_cache_lock = threading.RLock()

# --- DataFrame cache (parsed DataFrames, keyed by fileAddress) ---
df_cache: OrderedDict[str, dict] = OrderedDict()
df_cache_lock = threading.RLock()


def _evict_expired_entries_locked(now_ts: float):
    global csv_cache_total_bytes
    keys_to_remove = []
    for key, entry in csv_cache.items():
        if now_ts - entry["cached_at"] > CSV_CACHE_TTL_SECONDS:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        removed = csv_cache.pop(key, None)
        if removed:
            csv_cache_total_bytes -= removed["size_bytes"]
            print(f"CSV cache expired: {key}")


def _evict_to_limits_locked():
    global csv_cache_total_bytes
    while (
        len(csv_cache) > CSV_CACHE_MAX_ENTRIES
        or csv_cache_total_bytes > CSV_CACHE_MAX_BYTES
    ):
        evicted_key, evicted_entry = csv_cache.popitem(last=False)
        csv_cache_total_bytes -= evicted_entry["size_bytes"]
        print(f"CSV cache evicted (LRU): {evicted_key}")


def get_cached_csv_text(cache_key: str) -> Optional[str]:
    now_ts = time.time()
    with csv_cache_lock:
        _evict_expired_entries_locked(now_ts)
        entry = csv_cache.get(cache_key)
        if not entry:
            return None

        csv_cache.move_to_end(cache_key)
        print(f"CSV cache hit: {cache_key}")
        return entry["csv_text"]


def set_cached_csv_text(cache_key: str, csv_text: str):
    global csv_cache_total_bytes
    text_size = len(csv_text.encode("utf-8"))
    if text_size > CSV_CACHE_MAX_BYTES:
        print(f"CSV cache skip (entry too large): {cache_key}")
        return

    now_ts = time.time()
    with csv_cache_lock:
        existing = csv_cache.pop(cache_key, None)
        if existing:
            csv_cache_total_bytes -= existing["size_bytes"]

        csv_cache[cache_key] = {
            "csv_text": csv_text,
            "cached_at": now_ts,
            "size_bytes": text_size,
        }
        print(f"CSV cached: {cache_key} ({text_size} bytes)")
        csv_cache_total_bytes += text_size
        csv_cache.move_to_end(cache_key)
        _evict_expired_entries_locked(now_ts)
        _evict_to_limits_locked()


async def get_or_download_csv_text(client_http: httpx.AsyncClient, file_address: str, force_refresh: bool = False) -> str:
    cache_key = file_address.strip()
    if not force_refresh:
        cached_csv = get_cached_csv_text(cache_key)
        if cached_csv is not None:
            return cached_csv
        print(f"CSV cache miss: {cache_key}")
        print(f"Downloading CSV for first-time use: {cache_key}")
    else:
        print(f"CSV cache bypassed (forceRefresh=true): {cache_key}")
        print(f"Downloading CSV with forced refresh: {cache_key}")

    response = await client_http.get(cache_key)
    response.raise_for_status()
    csv_text = response.text
    set_cached_csv_text(cache_key, csv_text)
    return csv_text


async def get_or_download_file_bytes(
    client_http: httpx.AsyncClient,
    file_address: str,
    file_type: str,
    force_refresh: bool = False,
) -> bytes:
    if file_type == "csv":
        csv_text = await get_or_download_csv_text(client_http, file_address, force_refresh)
        return csv_text.encode("utf-8")

    response = await client_http.get(file_address)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# DataFrame cache helpers
# ---------------------------------------------------------------------------

def _get_cached_dataframe(cache_key: str) -> Optional[pd.DataFrame]:
    """Return a copy of the cached DataFrame, or None on miss/expiry."""
    now_ts = time.time()
    with df_cache_lock:
        # Evict expired entries first
        expired = [
            k for k, v in df_cache.items()
            if now_ts - v["cached_at"] > CSV_CACHE_TTL_SECONDS
        ]
        for k in expired:
            df_cache.pop(k, None)
            print(f"DF cache expired: {k}")

        entry = df_cache.get(cache_key)
        if entry is None:
            return None

        df_cache.move_to_end(cache_key)
        print(f"DF cache hit: {cache_key}")
        return entry["dataframe"].copy()


def _set_cached_dataframe(cache_key: str, df: pd.DataFrame) -> None:
    """Store a copy of *df* in df_cache with TTL and max-entries eviction."""
    now_ts = time.time()
    with df_cache_lock:
        # Replace existing entry if present
        df_cache.pop(cache_key, None)

        df_cache[cache_key] = {
            "dataframe": df.copy(),
            "cached_at": now_ts,
        }
        df_cache.move_to_end(cache_key)

        # Evict LRU entries beyond the max-entries limit
        while len(df_cache) > CSV_CACHE_MAX_ENTRIES:
            evicted_key, _ = df_cache.popitem(last=False)
            print(f"DF cache evicted (LRU): {evicted_key}")

        print(f"DF cached: {cache_key} (shape={df.shape})")


async def get_or_download_dataframe(
    client_http: httpx.AsyncClient,
    file_address: str,
    file_type: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame for *file_address*, checking df_cache first.
    Falls back to downloading + parsing when the cache misses.
    """
    cache_key = file_address.strip()
    if not force_refresh:
        cached_df = _get_cached_dataframe(cache_key)
        if cached_df is not None:
            return cached_df
        print(f"DF cache miss: {cache_key}")

    file_bytes = await get_or_download_file_bytes(
        client_http, file_address, file_type, force_refresh
    )
    df = dataframe_from_bytes(file_bytes, file_type)
    _set_cached_dataframe(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# /preload-file endpoint
# ---------------------------------------------------------------------------

@app.post("/preload-file")
async def preload_file(request: PreloadRequest):
    """
    Download and parse a file into a DataFrame, then store it in df_cache.
    Called fire-and-forget by the Go upload handler after a file is stored.
    """
    try:
        file_address = request.fileAddress.strip()
        file_type = detect_file_type(
            explicit_type=request.fileType,
            filename=file_address,
        )

        async with httpx.AsyncClient() as client_http:
            df = await get_or_download_dataframe(
                client_http,
                file_address,
                file_type,
                force_refresh=False,
            )

        rows, cols = df.shape
        print(f"Preload complete: {file_address} -> shape=({rows}, {cols})")

        # Pre-index for RAG so the first chat query has zero indexing latency
        try:
            if rag_store.enabled:
                rag_store.index_dataframe(file_address, df)
        except Exception as rag_err:
            print(f"[RAG] WARNING: preload indexing failed (non-fatal): {rag_err}")

        return {"status": "ok", "shape": [rows, cols]}

    except Exception as e:
        print(f"Preload failed for {request.fileAddress}: {e}")
        raise HTTPException(status_code=500, detail=f"Preload failed: {str(e)}")


def perform_data_health_check(df: pd.DataFrame) -> str:
    """
    Analyze the health of a DataFrame and return a plain-text summary
    covering missing values, duplicates, data types, outliers, and basic stats.
    """
    lines: list[str] = []
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    # --- General overview ---
    lines.append("=== DATA HEALTH CHECK REPORT ===\n")
    lines.append(f"Rows: {total_rows}")
    lines.append(f"Columns: {total_cols}")
    lines.append(f"Total cells: {total_cells}\n")

    # --- Missing / null values ---
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

    # --- Duplicate rows ---
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / total_rows * 100) if total_rows else 0
    lines.append("--- Duplicate Rows ---")
    lines.append(f"Duplicate rows: {dup_count} ({dup_pct:.2f}%)\n")

    # --- Data types ---
    lines.append("--- Column Data Types ---")
    for col in df.columns:
        lines.append(f"  • {col}: {df[col].dtype}")
    lines.append("")

    # --- Numeric column stats & outlier detection ---
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

    # --- Overall health score (simple heuristic) ---
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


@app.post("/data-health-check", response_class=PlainTextResponse)
async def data_health_check(file: UploadFile = File(...)):
    """
    Receive a CSV file, analyse its data quality, and return a plain-text report.
    Called by the Go backend (sendToDataHealthCheck in upload-ipfs.go).
    """
    try:
        contents = await file.read()
        file_type = detect_file_type(filename=file.filename, content_type=file.content_type)
        df = dataframe_from_bytes(contents, file_type)

        report = perform_data_health_check(df)
        return report

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid tabular file.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file could not be decoded as UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing file: {str(e)}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by:
    1. Dropping fully-duplicate rows
    2. Dropping columns that are 100% null
    3. Filling numeric nulls with column median
    4. Filling categorical nulls with column mode
    5. Stripping leading/trailing whitespace from string columns
    """
    # 1. Drop duplicate rows
    df = df.drop_duplicates()

    # 2. Drop columns that are entirely null
    df = df.dropna(axis=1, how="all")

    # 3 & 4. Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])

    # 5. Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    return df


@app.post("/data-health-check-clean")
async def data_health_check_clean(file: UploadFile = File(...)):
    """
    Receive a CSV file, clean it (drop duplicates, fill nulls, etc.),
    and return the cleaned CSV as a downloadable file.
    Called by the Go backend when the user chooses to upload the cleaned file.
    """
    try:
        contents = await file.read()
        file_type = detect_file_type(filename=file.filename, content_type=file.content_type)
        df = dataframe_from_bytes(contents, file_type)

        cleaned_df = clean_dataframe(df)

        # Convert cleaned DataFrame back to CSV
        output = io.StringIO()
        cleaned_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"},
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid tabular file.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file could not be decoded as UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning file: {str(e)}")


@app.post("/preview-gen")
async def preview_gen(request: PreviewRequest):
    try:
        file_type = detect_file_type(explicit_type=request.fileType, filename=str(request.fileAddress))
        async with httpx.AsyncClient() as client_http:
            file_bytes = await get_or_download_file_bytes(
                client_http,
                str(request.fileAddress),
                file_type,
                request.forceRefresh,
            )

        df = dataframe_from_bytes(file_bytes, file_type)
        headers, rows = dataframe_preview(df, request.maxRows, request.maxCols)

        return {
            "headers": headers,
            "rows": rows,
            "fileType": file_type,
        }
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


def detect_language(text: str) -> str:
    """
    Simple language detection based on common Portuguese words
    """
    portuguese_indicators = [
        'qual', 'quem', 'onde', 'quando', 'como', 'por que', 'porque',
        'jogador', 'melhor', 'maior', 'menor', 'média', 'total',
        'gols', 'pontos', 'time', 'equipe', 'artilheiro', 'temporada'
    ]
    
    text_lower = text.lower()
    portuguese_count = sum(1 for word in portuguese_indicators if word in text_lower)
    
    return 'pt' if portuguese_count > 0 else 'en'

def is_visualization_request(prompt: str) -> bool:
    """
    Detect if the user is asking for a visualization/chart/graph
    """
    visualization_keywords = [
        # English
        'visualize', 'visualization', 'chart', 'graph', 'plot', 'show me',
        'display', 'draw', 'create a chart', 'create a graph', 'bar chart',
        'line chart', 'pie chart', 'scatter plot', 'histogram', 'heatmap',
        # Portuguese
        'visualizar', 'visualização', 'gráfico', 'plotar', 'mostre',
        'exibir', 'desenhar', 'criar gráfico', 'criar um gráfico',
        'gráfico de barras', 'gráfico de linhas', 'gráfico de pizza',
        'dispersão', 'histograma', 'mapa de calor',
        # Spanish
        'visualizar', 'visualización', 'gráfica', 'mostrar', 'dibujar'
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in visualization_keywords)


def _normalize_error_text(error: Exception) -> str:
    return str(error).lower()


def is_context_limit_error(error: Exception) -> bool:
    error_text = _normalize_error_text(error)
    return (
        "context_length_exceeded" in error_text
        or "input tokens exceed" in error_text
        or "maximum context length" in error_text
    )


def should_retry_fallback_model(error: Exception) -> bool:
    if is_context_limit_error(error):
        return False

    error_text = _normalize_error_text(error)
    no_retry_indicators = [
        "invalid_request_error",
        "does not exist",
        "unsupported",
        "invalid model",
        "model_not_found",
    ]
    return not any(token in error_text for token in no_retry_indicators)


def is_large_dataset(df: pd.DataFrame) -> bool:
    rows, cols = df.shape
    return rows >= LARGE_DATASET_ROW_THRESHOLD or (rows * cols) >= LARGE_DATASET_CELL_THRESHOLD


CHART_MAX_COLUMNS = int(os.getenv("CHART_MAX_COLUMNS", "30"))
CHART_MAX_ROWS = int(os.getenv("CHART_MAX_ROWS", "500000"))


def _reduce_df_for_chart(df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
    """
    Reduce a wide DataFrame to only the columns relevant to the chart request.

    Many sports/analytics files have 600+ columns but only ~3 000 rows.
    Passing ALL columns into the chart pipeline triggers the large-dataset
    guard (cell count) even though the row count is perfectly manageable.

    This function selects up to CHART_MAX_COLUMNS columns using the
    existing prompt-aware column ranker (_select_relevant_columns), so
    the chart code generator and executor only see what they need.
    """
    if df.shape[1] <= CHART_MAX_COLUMNS:
        return df

    selected = _select_relevant_columns(df, user_prompt, CHART_MAX_COLUMNS)
    reduced = df[selected].copy()
    print(
        f"[chart-reduce] Reduced DataFrame from {df.shape[1]} cols → "
        f"{reduced.shape[1]} cols for chart generation "
        f"(rows={reduced.shape[0]}, cells={reduced.shape[0] * reduced.shape[1]})"
    )
    return reduced


def _coerce_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


# Portuguese/English stop words excluded from token matching so common
# words like "com", "mais", "the" don't cause false-positive column hits.
_COL_MATCH_STOP = frozenset({
    "que", "qual", "quais", "com", "mais", "dos", "das", "por", "para",
    "uma", "uns", "the", "and", "for", "are", "was", "how", "what",
    "which", "were", "has", "como", "foram", "nos", "nas", "ele", "ela",
    "isso", "este", "esta", "esses", "esse", "essa", "seu", "sua",
    "cada", "entre", "sobre", "desde", "mesmo", "quando", "onde",
})


def _select_relevant_columns(df: pd.DataFrame, user_prompt: str, max_columns: int) -> list[str]:
    """Return up to *max_columns* column names, prioritised by relevance
    to *user_prompt*.

    Improvements over the previous substring-only approach:

    * **Token scoring** — each column is scored by how many of its name
      tokens appear in the prompt (after stop-word removal).  A full
      substring match (e.g. ``"passes precisos"`` found verbatim in the
      prompt) gets the highest score.
    * **Deterministic tie-breaking** — columns with equal scores keep
      their original file order, so results are reproducible.
    * **Budget-fill** — remaining slots are filled with numeric columns
      first (most analytically useful), then categorical, then the rest.
    """
    all_columns = list(df.columns)
    if len(all_columns) <= max_columns:
        return all_columns

    prompt_lower = user_prompt.lower()
    prompt_tokens = {
        w for w in re.findall(r'\b\w{3,}\b', prompt_lower)
        if w not in _COL_MATCH_STOP
    }

    # --- score every column ------------------------------------------------
    scored: list[tuple[str, int, int]] = []   # (col, score, orig_index)
    for idx, col in enumerate(all_columns):
        col_lower = str(col).lower()
        score = 0

        # Full column name is a substring of the prompt (strongest signal)
        if col_lower in prompt_lower:
            score += 1000

        # Token-level overlap
        col_tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        if col_tokens and prompt_tokens:
            overlap = col_tokens & prompt_tokens
            if overlap:
                # Coverage: fraction of the column name matched
                score += int(len(overlap) / len(col_tokens) * 200)
                # Bonus per matching token
                score += len(overlap) * 30

        scored.append((col, score, idx))

    # Prompt-matched columns first (score desc, then original order)
    matched = sorted(
        [(c, s, i) for c, s, i in scored if s > 0],
        key=lambda x: (-x[1], x[2]),
    )

    ranked: list[str] = []
    seen: set[str] = set()
    for col, _s, _i in matched:
        ranked.append(col)
        seen.add(col)

    # Fill remaining budget: numeric → categorical → everything else
    numeric_columns = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [
        c for c in all_columns
        if pd.api.types.is_object_dtype(df[c])
        or str(df[c].dtype) in ("string", "category")
    ]

    for col in numeric_columns + categorical_columns + all_columns:
        if col not in seen:
            ranked.append(col)
            seen.add(col)
        if len(ranked) >= max_columns:
            break

    return ranked[:max_columns]


def _build_sample_rows(df: pd.DataFrame, selected_columns: list[str], sample_rows: int) -> list[dict]:
    if df.empty or sample_rows <= 0:
        return []

    sample_rows = min(sample_rows, len(df))
    head_count = max(1, sample_rows // 2)
    tail_count = sample_rows - head_count

    head_df = df[selected_columns].head(head_count)
    tail_df = df[selected_columns].tail(tail_count) if tail_count > 0 else pd.DataFrame(columns=selected_columns)
    sample_df = pd.concat([head_df, tail_df]).drop_duplicates().head(sample_rows)

    records = sample_df.to_dict(orient="records")
    return [{key: _coerce_json_value(value) for key, value in row.items()} for row in records]


def _extract_prompt_columns(df: pd.DataFrame, user_prompt: str) -> list[str]:
    """Return DataFrame columns whose name appears in *user_prompt*.

    Uses substring + token-overlap matching.  Returns at most 5 columns,
    ordered by match quality (strongest first).
    """
    prompt_lower = user_prompt.lower()
    prompt_tokens = set(re.findall(r'\b\w{3,}\b', prompt_lower))

    candidates: list[tuple[str, int]] = []
    for col in df.columns:
        col_lower = str(col).lower()
        score = 0

        # Full name in prompt
        if col_lower in prompt_lower:
            score += 100

        # Token overlap ≥ 50 % of column tokens
        col_tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        if col_tokens and prompt_tokens:
            overlap = col_tokens & prompt_tokens
            if overlap and len(overlap) / len(col_tokens) >= 0.5:
                score += 50

        if score > 0:
            candidates.append((col, score))

    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:5]]


def _build_supplementary_context(
    df: pd.DataFrame,
    user_prompt: str,
    existing_context: str,
    max_rows: int = 30,
    max_chars: int = 8000,
) -> str:
    """Extract targeted rows for columns mentioned in *user_prompt*.

    When a file has 700+ columns, both RAG chunks and the statistical
    summary may exclude the columns the user is asking about.  This
    function provides a small, targeted CSV extract containing only the
    identifying columns (Period Name, Date, …) plus the queried columns
    with non-null values so the LLM can cite exact figures.

    Returns an empty string if no supplementary data is needed.
    """
    target_cols = _extract_prompt_columns(df, user_prompt)
    if not target_cols:
        return ""

    # Only supplement columns absent from the existing RAG context
    existing_lower = (existing_context or "").lower()
    missing_cols = [c for c in target_cols if str(c).lower() not in existing_lower]
    if not missing_cols:
        return ""

    # Identifying columns for row-level context
    id_candidates = [
        "Period Name", "Date", "Player Name",
        "ADVERS\u00c1RIO", "CAMPEONATO", "EVENTO",
    ]
    id_cols = [c for c in id_candidates if c in df.columns]
    extract_cols = list(dict.fromkeys(id_cols + missing_cols))  # dedupe, keep order

    working = df[extract_cols].copy()

    # Keep only rows where at least one target column has data
    mask = pd.Series(False, index=df.index)
    for col in missing_cols:
        if col not in working.columns:
            continue
        if pd.api.types.is_numeric_dtype(working[col]):
            mask |= working[col].notna()
        else:
            mask |= working[col].notna() & (working[col].astype(str).str.strip() != "")

    filtered = working[mask].head(max_rows)
    if filtered.empty:
        return ""

    buf = io.StringIO()
    filtered.to_csv(buf, index=False)
    text = buf.getvalue()
    return text[:max_chars] if len(text) > max_chars else text


def _build_dataframe_context(
    df: pd.DataFrame,
    user_prompt: str,
    max_columns: int,
    max_numeric_stats: int,
    max_categorical_stats: int,
) -> dict:
    selected_columns = _select_relevant_columns(df, user_prompt, max_columns)
    working_df = df[selected_columns]

    missing_by_column = (
        (working_df.isna().sum() / max(len(working_df), 1) * 100)
        .sort_values(ascending=False)
        .head(12)
        .round(2)
        .to_dict()
    )

    numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(working_df[col])]
    categorical_columns = [
        col for col in selected_columns if pd.api.types.is_object_dtype(working_df[col]) or str(working_df[col].dtype) in ("string", "category")
    ]

    numeric_summary = {}
    for col in numeric_columns[:max_numeric_stats]:
        series = working_df[col].dropna()
        if series.empty:
            continue
        numeric_summary[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()) if series.count() > 1 else 0.0, 4),
            "min": round(float(series.min()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

    categorical_summary = {}
    for col in categorical_columns[:max_categorical_stats]:
        series = working_df[col].dropna().astype(str)
        if series.empty:
            continue
        categorical_summary[col] = {
            "unique_values": int(series.nunique()),
            "top_values": series.value_counts().head(3).to_dict(),
        }

    schema_preview = {}
    for col in selected_columns:
        dtype_str = str(working_df[col].dtype)
        is_numeric = col in numeric_columns
        col_meta: dict = {
            "dtype": dtype_str,
            "kind": "numeric" if is_numeric else "categorical",
        }
        if is_numeric and col in numeric_summary:
            stats = numeric_summary[col]
            col_meta["min"] = stats["min"]
            col_meta["max"] = stats["max"]
            col_meta["mean"] = stats["mean"]
            col_meta["median"] = stats["median"]
        elif col in categorical_summary:
            cat = categorical_summary[col]
            col_meta["unique_count"] = cat["unique_values"]
            col_meta["top_3"] = list(cat["top_values"].keys())[:3]
        schema_preview[str(col)] = col_meta

    context = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "selected_columns": [str(col) for col in selected_columns],
        "omitted_columns_count": max(0, int(df.shape[1]) - len(selected_columns)),
        "missing_percentage_top_columns": {str(col): value for col, value in missing_by_column.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "schema_preview": schema_preview,
    }
    return context


def build_llm_dataset_message(df: pd.DataFrame, user_prompt: str, large_dataset_mode: bool = False, retrieved_context: str = "") -> str:
    context_profiles = [
        {"max_columns": 70, "max_numeric_stats": 30, "max_categorical_stats": 20, "max_chars": LLM_MAX_CONTEXT_CHARS},
        {"max_columns": 45, "max_numeric_stats": 20, "max_categorical_stats": 14, "max_chars": LLM_COMPACT_CONTEXT_CHARS},
        {"max_columns": 25, "max_numeric_stats": 12, "max_categorical_stats": 8, "max_chars": 50000},
        {"max_columns": 15, "max_numeric_stats": 8, "max_categorical_stats": 5, "max_chars": 30000},
    ]

    if large_dataset_mode:
        context_profiles = context_profiles[1:]

    # --- Single-pass: try the largest applicable profile first (call #1) ---
    profile = context_profiles[0]
    context_payload = _build_dataframe_context(
        df,
        user_prompt,
        max_columns=profile["max_columns"],
        max_numeric_stats=profile["max_numeric_stats"],
        max_categorical_stats=profile["max_categorical_stats"],
    )

    def _build_message(payload: dict) -> str:
        base = (
            "Dataset Information (JSON):\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        )
        if retrieved_context:
            base += (
                "Relevant Data Rows (retrieved via semantic search — use these "
                "to answer row-level questions with precision):\n"
                f"{retrieved_context}\n\n"
            )
        base += (
            f"User Question: {user_prompt}\n\n"
            "Answer with the same language used by the user. "
            "Provide concise conclusions with specific values from the provided context. "
            "When relevant data rows are provided, prefer citing them over the statistical summary."
        )
        return base

    message = _build_message(context_payload)

    if len(message) <= profile["max_chars"]:
        print(
            "LLM context profile selected:",
            {
                "max_columns": profile["max_columns"],
                "message_chars": len(message),
            },
        )
        return message

    # --- Result is too large: scale down proportionally and recompute (call #2) ---
    ratio = profile["max_chars"] / max(len(message), 1)
    fallback_profile = {
        "max_columns": max(5, int(profile["max_columns"] * ratio)),
        "max_numeric_stats": max(3, int(profile["max_numeric_stats"] * ratio)),
        "max_categorical_stats": max(2, int(profile["max_categorical_stats"] * ratio)),
        "max_chars": profile["max_chars"],
    }

    context_payload = _build_dataframe_context(
        df,
        user_prompt,
        max_columns=fallback_profile["max_columns"],
        max_numeric_stats=fallback_profile["max_numeric_stats"],
        max_categorical_stats=fallback_profile["max_categorical_stats"],
    )
    message = _build_message(context_payload)

    print(
        "LLM context profile selected:",
        {
            "max_columns": fallback_profile["max_columns"],
            "message_chars": len(message),
        },
    )
    return message


def token_limit_notice(user_prompt: str) -> str:
    lang = detect_language(user_prompt)
    if lang == 'pt':
        return "⚠️ O limite de tokens foi excedido para este modelo. A resposta abaixo foi gerada em modo compacto para continuar a análise."
    return "⚠️ The token limit was exceeded for this model. The response below was generated in compact fallback mode so analysis could continue."

def analyze_dataframe_with_openai(df: pd.DataFrame, user_prompt: str, model: str = "gpt-5-mini", stream: bool = False, retrieved_context: str = ""):
    """
    Send dataframe info and user prompt to OpenAI for analysis with fallback options.
    When stream=True, returns a generator that yields text chunks instead of a full string.
    Token tracking is skipped in streaming mode (usage is not available per-chunk).
    """
    system_prompt = """\
You are a senior KPI data analyst. Your audience is business stakeholders, not \
engineers. You specialize in identifying trends, anomalies, and key performance \
indicators from tabular datasets.

---

Examples of ideal responses:

User question: "What are the main KPIs in this dataset?"
Response:
## Key KPIs
- **Revenue**: avg R$42,300/month, peak in March (R$68,100)
- **Churn Rate**: 3.2% average, spiked to 8.1% in Q2
- **Active Users**: 12,400 avg, +4% month-over-month growth
## Summary
The dataset tracks financial and engagement metrics. Revenue and churn are inversely \
correlated. Investigate the Q2 churn spike.

---

User question: "Are there data quality issues?"
Response:
## Data Quality Report
- **Missing values**: `email` column — 12.4% missing (action: impute or drop)
- **Duplicates**: 38 duplicate rows found (3.1% of total)
- **Outliers**: `revenue` has values above R$500k — likely data entry errors
## Verdict: FAIR — minor issues, review before using for reporting.

---

Goal: Answer the user's question using only the provided dataset context.
- Always use markdown with headers and bullet points.
- Always cite specific numbers, column names, and percentages from the data.
- Never fabricate values not present in the context.
- Respond in the same language as the user's question.
- Keep the response under 350 words unless the user explicitly asks for a detailed report.\
"""

    large_dataset_mode = is_large_dataset(df)
    user_message = build_llm_dataset_message(df, user_prompt, large_dataset_mode, retrieved_context=retrieved_context)
    print(
        "OpenAI request context:",
        {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "large_dataset_mode": large_dataset_mode,
            "message_chars": len(user_message),
            "message_est_tokens": len(user_message) // 4,
            "model": model,
            "stream": stream,
        },
    )

    # ── Streaming path ────────────────────────────────────────────────────────
    if stream:
        if os.getenv("OPENAI_API_KEY"):
            def _stream_generator():
                try:
                    llm_started_at = time.perf_counter()
                    stream_response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        max_completion_tokens=16000,
                        stream=True,
                    )
                    print("using model (streaming)", model)
                    first_chunk = True
                    total_yielded = 0
                    chunk_count = 0
                    for chunk in stream_response:
                        if first_chunk:
                            print(f"OpenAI first-chunk latency: {time.perf_counter() - llm_started_at:.2f}s")
                            first_chunk = False
                        text = chunk.choices[0].delta.content or ""
                        if text:
                            total_yielded += len(text)
                            chunk_count += 1
                        yield text
                    print(f"OpenAI stream complete: {chunk_count} chunks, {total_yielded} chars yielded, {time.perf_counter() - llm_started_at:.2f}s total")
                except Exception as e:
                    print(f"OpenAI streaming failed with model '{model}': {str(e)}")
                    # Fall back to non-streaming text and yield it as a single chunk
                    fallback = analyze_dataframe_fallback(df, user_prompt)
                    yield fallback
            return _stream_generator()
        else:
            # No API key — return a one-shot generator from the fallback
            def _fallback_gen():
                yield analyze_dataframe_fallback(df, user_prompt)
            return _fallback_gen()

    # ── Non-streaming path (unchanged) ────────────────────────────────────────
    token_limit_exceeded = False
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm_started_at = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=16000
            )
            _track_tokens(response)
            print("using model", model)
            print(f"OpenAI latency: {time.perf_counter() - llm_started_at:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI failed with model '{model}': {str(e)}")
            token_limit_exceeded = is_context_limit_error(e)
            # Fall through to alternatives

    # Try Hugging Face as backup
    if os.getenv("HUGGINGFACE_API_KEY"):
        hf_result = analyze_dataframe_with_huggingface(df, user_prompt)
        if not hf_result.startswith("Error") and not hf_result.startswith("Hugging Face API error"):
            if token_limit_exceeded:
                return f"{token_limit_notice(user_prompt)}\n\n{hf_result}"
            return hf_result

    # Use fallback analysis
    fallback_result = analyze_dataframe_fallback(df, user_prompt)
    if token_limit_exceeded:
        return f"{token_limit_notice(user_prompt)}\n\n{fallback_result}"
    return fallback_result

def analyze_dataframe_with_huggingface(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Use Hugging Face's free API as an alternative to OpenAI
    """
    # Create a summary of the dataframe
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "sample_data": df.head(3).to_dict(),  # Reduced sample to save tokens
    }
    
    # Detect language and adjust prompt
    lang = detect_language(user_prompt)
    
    if lang == 'pt':
        prompt = f"""
        Analise estes dados CSV e responda à pergunta:
        
        Informações dos dados: {df.shape[0]} linhas, {df.shape[1]} colunas
        Colunas: {list(df.columns)}
        Amostra: {df_info['sample_data']}
        
        Pergunta: {user_prompt}
        
        Forneça uma análise clara com descobertas específicas em português.
        """
    else:
        prompt = f"""
        Analyze this CSV data and answer the question:
        
        Data info: {df.shape[0]} rows, {df.shape[1]} columns
        Columns: {list(df.columns)}
        Sample: {df_info['sample_data']}
        
        Question: {user_prompt}
        
        Provide a clear analysis with specific findings.
        """
    
    try:
        # Using Hugging Face's free inference API
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'] if result else "No response generated"
        else:
            return f"Hugging Face API error: {response.status_code}"
            
    except Exception as e:
        return f"Error with Hugging Face API: {str(e)}"

def analyze_dataframe_fallback(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Fallback analysis without external APIs - purely based on pandas operations
    """
    lang = detect_language(user_prompt)
    analysis = []
    
    # Basic info in detected language
    if lang == 'pt':
        analysis.append(f"O conjunto de dados contém {len(df)} linhas e {len(df.columns)} colunas.")
        analysis.append(f"Colunas: {', '.join(df.columns)}")
    else:
        analysis.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        analysis.append(f"Columns: {', '.join(df.columns)}")
    
    # Look for common patterns in the prompt
    prompt_lower = user_prompt.lower()
    
    # Portuguese and English keywords for "top/highest"
    top_keywords = ['top', 'highest', 'best', 'most', 'maior', 'melhor', 'artilheiro', 'líder']
    avg_keywords = ['average', 'mean', 'média', 'meio']
    total_keywords = ['total', 'sum', 'soma', 'somatório']
    
    if any(word in prompt_lower for word in top_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in ['score', 'goal', 'point', 'value', 'gol', 'ponto', 'valor']):
                    top_value = df[col].max()
                    top_index = df[col].idxmax()
                    
                    if lang == 'pt':
                        analysis.append(f"Maior {col}: {top_value}")
                    else:
                        analysis.append(f"Highest {col}: {top_value}")
                    
                    # Try to find name column
                    name_cols = [c for c in df.columns if any(word in c.lower() for word in ['name', 'player', 'team', 'nome', 'jogador', 'time'])]
                    if name_cols:
                        top_name = df.loc[top_index, name_cols[0]]
                        if lang == 'pt':
                            analysis.append(f"Melhor desempenho: {top_name} com {top_value} {col}")
                        else:
                            analysis.append(f"Top performer: {top_name} with {top_value} {col}")
                    break
    
    elif any(word in prompt_lower for word in avg_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            avg_val = df[col].mean()
            if lang == 'pt':
                analysis.append(f"Média de {col}: {avg_val:.2f}")
            else:
                analysis.append(f"Average {col}: {avg_val:.2f}")
    
    elif any(word in prompt_lower for word in total_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            total_val = df[col].sum()
            if lang == 'pt':
                analysis.append(f"Total de {col}: {total_val}")
            else:
                analysis.append(f"Total {col}: {total_val}")
    
    # Add basic statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        if lang == 'pt':
            analysis.append("\nEstatísticas Básicas:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                analysis.append(f"{col}: mín={df[col].min()}, máx={df[col].max()}, média={df[col].mean():.2f}")
        else:
            analysis.append("\nBasic Statistics:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                analysis.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
    
    if lang == 'pt':
        return "\n".join(analysis) if analysis else "Não foi possível analisar os dados para esta pergunta específica."
    else:
        return "\n".join(analysis) if analysis else "Unable to analyze the data for this specific question."

def generate_visualization_code_with_ai(df: pd.DataFrame, user_prompt: str, model: str = "gpt-5-mini") -> str:
    """
    Use AI to generate Python code for visualization based on user prompt
    Returns only the Python code as a string
    """
    # --- Two-tier schema strategy -------------------------------------------
    # Tier 1 (all columns): just names — lets the AI know every column exists
    #   and can write df['any_column'] in code, even if not in Tier 2.
    # Tier 2 (relevant columns): full detail (dtype + sample values) only for
    #   the top-N columns ranked by _select_relevant_columns(), which scores
    #   columns against the user prompt using keyword matching.
    # This preserves response quality for whatever the user asks while keeping
    # token usage bounded regardless of file width.
    # -------------------------------------------------------------------------
    MAX_DETAIL_COLS = 50  # columns with full detail sent to the model
    MAX_SAMPLE_ROWS = 3   # sample rows shown for the relevant columns only

    all_col_names = list(df.columns)                       # Tier 1 — names only
    relevant_cols = _select_relevant_columns(              # Tier 2 — full detail
        df, user_prompt, max_columns=MAX_DETAIL_COLS
    )
    relevant_dtypes = {col: str(df.dtypes[col]) for col in relevant_cols}
    relevant_sample = df[relevant_cols].head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    relevant_numeric = [
        col for col in relevant_cols if pd.api.types.is_numeric_dtype(df[col])
    ]
    relevant_categorical = [
        col for col in relevant_cols
        if pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) in ("string", "category")
    ]
    
    system_prompt = """You are a Python data visualization expert. Generate ONLY executable Python code to create a visualization based on the user's request.

IMPORTANT RULES:
1. Return ONLY Python code, no explanations, no markdown, no comments except necessary ones
2. The DataFrame is already loaded as 'df' - do NOT reload it
3. Use matplotlib and/or seaborn for visualizations
4. The code must save the figure to 'output_chart.png' using plt.savefig('output_chart.png', dpi=150, bbox_inches='tight')
5. Always include plt.close() at the end
6. Use proper labels, titles, and formatting
7. Handle any potential errors (missing data, etc.)
8. Make the visualization clear and professional
9. If the user's language is Portuguese, use Portuguese labels; if English, use English labels
10. The user message includes a "Data Quality Notes" section — these are HINTS about known issues.
    Apply dtype-conversion hints (comma-decimal, timedelta) unconditionally.
    Apply row-filtering hints (AGGREGATION-LEVEL) ONLY when the column flagged is NOT a
    metric/count column that the user explicitly asked to compare or visualise.

ENVIRONMENT CONSTRAINTS (violations will crash at runtime):
- pandas >= 3.0, Python 3.12+
- DO NOT use infer_datetime_format parameter — it was removed in pandas 2.0
- DO NOT use DataFrame.append() or Series.append() — use pd.concat() instead
- DO NOT use squeeze parameter in read_csv — it was removed in pandas 2.0
- DO NOT re-import pandas, matplotlib, seaborn, or numpy.
  They are pre-loaded as: pd, plt, sns, np. Use these names directly.
- When formatting dates on a pandas Series, use .dt.strftime() NOT .strftime().
  Series does not have strftime() directly; the datetime accessor (.dt) is required.

CRITICAL — never destroy data the user asked to see:
- If the user's request mentions a column by name (e.g. 'Red Cards', 'Yellow Cards', 'Goals'),
  do NOT filter that column to a single value (e.g. == 0) under any circumstance.
  Such columns are measurement/count columns, not row-type flags.
- An AGGREGATION-LEVEL hint should be skipped entirely when the flagged column is
  one of the columns the user wants to compare or plot.

Example output format:
plt.figure(figsize=(12, 7))
# Your visualization code here
plt.title('Chart Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.savefig('output_chart.png', dpi=150, bbox_inches='tight')
plt.close()
"""
    
    # Dynamically inspect the DataFrame for data-quality issues in this specific file
    quality_notes = _inspect_df_quality(df)

    user_message = f"""
# Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns

# ALL column names (the full df has all of these — you may reference any of them):
# {all_col_names}

# Detailed schema for the {len(relevant_cols)} columns most relevant to the request
# (dtype + sample values). Use these to understand formats and data types:
# - Relevant dtypes:     {relevant_dtypes}
# - Relevant numeric:    {relevant_numeric}
# - Relevant categorical:{relevant_categorical}
# - Sample data ({MAX_SAMPLE_ROWS} rows, relevant cols only): {relevant_sample}

# Data Quality Notes (apply these fixes in your code before plotting):
{quality_notes}

User Request: {user_prompt}

Generate Python code to create the visualization. Remember: the DataFrame is already available as 'df'.
"""
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=3500
            )
            _track_tokens(response)
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            print(f"[chart-gen] finish_reason={finish_reason!r}  raw_len={len(raw)} chars")

            if finish_reason == "length":
                # Response was cut off before the closing ``` — code is incomplete
                print("[chart-gen] WARNING: response truncated (finish_reason=length). "
                      "Raising max_completion_tokens or simplifying the prompt may help.")
                return None, ("Não foi possível gerar o gráfico: o código gerado foi cortado antes de terminar. "
                              "Tente reformular a pergunta de forma mais simples.")

            # Strip markdown code fences if present
            code = raw
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            code = code.strip()
            if not code:
                print(f"[chart-gen] ERROR: code extraction produced empty string. Raw response:\n{raw[:500]}")
                return None, "Não foi possível gerar o gráfico: a resposta do modelo não continha código válido."

            return code, None
        except Exception as e:
            msg = str(e)
            print(f"[chart-gen] ERROR calling OpenAI: {msg}")
            return None, f"Erro ao chamar a API de geração de gráfico: {msg}"
    else:
        print("[chart-gen] No OpenAI API key found")
        return None, "Chave da API OpenAI não configurada. Não é possível gerar o gráfico."

def _inspect_df_quality(df: pd.DataFrame) -> str:
    """
    Dynamically inspect a DataFrame and return a plain-text block describing
    data-quality issues that the AI should fix before plotting.

    Detection is generic — no hardcoded column names — so it works for any CSV:

    1. Comma-decimal columns: object columns whose values look like numbers but
       use a comma as the decimal separator (e.g. "121,49").
    2. Zero-sentinel columns: numeric columns where a large share of values is
       exactly 0 but the non-zero values are clearly much higher, suggesting 0
       means "not recorded" rather than a true measurement of zero.
    3. Aggregation-level columns: integer columns with low cardinality that
       contain 0 alongside other small integers — typical of a "period" or
       "level" column where 0 flags summary/aggregate rows.
    4. Timedelta columns: columns with dtype timedelta64 — these must NOT be
       passed to pd.to_datetime(); use .dt.total_seconds() instead.

    Performance: uses vectorised .str.match() and caps inspection to the first
    MAX_COLS_TO_INSPECT columns of each category to stay fast on wide files.
    """
    notes = []
    MAX_COLS = 120  # cap per category to keep latency low on wide DataFrames

    # ── 0. Timedelta columns ───────────────────────────────────────────────────
    timedelta_cols = [
        col for col in df.columns
        if pd.api.types.is_timedelta64_dtype(df[col])
    ]
    if timedelta_cols:
        notes.append(
            f"# TIMEDELTA COLUMNS: the following columns have dtype timedelta64 "
            f"(they store durations like HH:MM:SS, not timestamps).\n"
            f"#   Columns: {timedelta_cols}\n"
            f"#   NEVER call pd.to_datetime() on them — it will raise a TypeError.\n"
            f"#   To get a numeric value use: df[col].dt.total_seconds()"
        )

    # ── 1. Comma-decimal string columns ───────────────────────────────────────
    obj_cols = list(df.select_dtypes(include='object').columns)[:MAX_COLS]
    comma_cols = []
    _comma_re = r'^-?\d+,\d+$'
    for col in obj_cols:
        sample = df[col].dropna().astype(str).str.strip().head(50)
        if sample.empty:
            continue
        # Vectorised: no per-row Python lambda
        if sample.str.match(_comma_re).sum() >= max(1, len(sample) * 0.3):
            comma_cols.append(col)

    if comma_cols:
        notes.append(
            f"# COMMA-DECIMAL COLUMNS: the following columns store numbers with a comma "
            f"as the decimal separator (e.g. '121,49'). Convert them before plotting:\n"
            f"#   Columns: {comma_cols}\n"
            f"#   Fix: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')"
        )

    # ── 2. Zero-sentinel columns ───────────────────────────────────────────────
    num_cols = [
        col for col in df.select_dtypes(include='number').columns
        if not pd.api.types.is_timedelta64_dtype(df[col])
    ][:MAX_COLS]
    sentinel_cols = []
    for col in num_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) < 5:
            continue
        zero_mask = series == 0
        zero_ratio = zero_mask.sum() / len(series)
        non_zero = series[~zero_mask]
        if zero_ratio >= 0.15 and len(non_zero) > 0 and non_zero.median() > 10:
            sentinel_cols.append((col, round(zero_ratio * 100, 1)))

    if sentinel_cols:
        col_list = [c for c, _ in sentinel_cols]
        pct_list  = [f"{c}: {p}% zeros" for c, p in sentinel_cols]
        notes.append(
            f"# ZERO-SENTINEL COLUMNS: the following numeric columns have a high proportion "
            f"of exact zeros that likely mean 'not recorded' (not a true 0 measurement).\n"
            f"#   {'; '.join(pct_list)}\n"
            f"#   Fix: replace zeros with NaN before plotting:\n"
            f"#   for col in {col_list}:\n"
            f"#       df[col] = df[col].where(df[col] > 0, other=float('nan'))"
        )

    # ── 3. Aggregation-level columns (potential period / row-type flag) ────────
    #
    # IMPORTANT: metric/count columns (goals, cards, assists, shots, …) have the
    # same numeric pattern as period/level flags (low cardinality, contains 0,
    # small positive integers) but must NEVER be treated as row-type selectors.
    # We skip any column whose name contains a known metric keyword.
    _METRIC_KEYWORDS = {
        # cards / discipline
        'card', 'cards', 'yellow', 'red', 'booking', 'caution', 'foul', 'fouls',
        # goals / scoring
        'goal', 'goals', 'gol', 'gols', 'score', 'scored', 'assist', 'assists',
        # shots
        'shot', 'shots', 'attempt', 'attempts', 'chute', 'chutes',
        # general counts / KPIs
        'count', 'total', 'sum', 'num', 'number', 'qty', 'quantity',
        'point', 'points', 'ponto', 'pontos',
        'win', 'wins', 'loss', 'losses', 'draw', 'draws',
        'pass', 'passes', 'tackle', 'tackles', 'interception', 'interceptions',
        'save', 'saves', 'clean', 'concede', 'conceded',
        'km', 'distance', 'minute', 'minutes', 'min',
        'rating', 'rate', 'ratio', 'pct', 'percent',
    }

    def _is_metric_col(col_name: str) -> bool:
        """Return True if the column name looks like a measurement/count column."""
        name_lower = str(col_name).lower()
        # Split on common non-alpha separators so 'yellow_cards' → ['yellow', 'cards']
        import re as _re
        tokens = set(_re.split(r'[\s_\-/]+', name_lower))
        return bool(tokens & _METRIC_KEYWORDS)

    agg_cols = []
    for col in num_cols:
        # Skip columns that are clearly value/count metrics
        if _is_metric_col(col):
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        unique_vals = series.unique()
        if len(unique_vals) > 10:
            continue
        unique_sorted = sorted(unique_vals)
        if (
            0 in unique_sorted
            and all(v >= 0 and v == int(v) for v in unique_sorted)
            and max(unique_sorted) <= 20
            and (series == 0).sum() > 0
            and (series != 0).sum() > 0
        ):
            agg_cols.append((col, [int(v) for v in unique_sorted]))

    if agg_cols:
        for col, vals in agg_cols:
            notes.append(
                f"# AGGREGATION-LEVEL COLUMN: '{col}' contains values {vals}. "
                f"Value 0 typically marks full-aggregate/summary rows, while other values "
                f"mark sub-period rows (e.g. halves, quarters).\n"
                f"# If you are plotting one data point per event/match, filter to the "
                f"aggregate rows first to avoid double-counting:\n"
                f"#   df = df[pd.to_numeric(df['{col}'], errors='coerce') == 0].copy()\n"
                f"# NOTE: skip this filter if '{col}' is itself a metric the user wants to plot."
            )

    if not notes:
        return "# No significant data-quality issues detected — df appears clean."

    return "\n".join(notes)


def _preprocess_df_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform generic, safe pre-processing on a DataFrame copy before passing it
    to AI-generated chart code.

    Fixes applied unconditionally (safe for any dataset):
      1. timedelta64 columns → float64 (total seconds).
         Prevents the "dtype timedelta64[us] cannot be converted to datetime64[ns]"
         TypeError when AI-generated code calls pd.to_datetime() on duration cols.
      2. Object columns that look like comma-decimal numbers are converted to
         float (comma-as-decimal is always a parsing artifact in numeric context).

    Uses vectorised .str.match() and caps inspection to MAX_COLS columns to stay
    fast on wide DataFrames (600+ columns).
    """
    df = df.copy()

    MAX_COLS = 120
    _comma_re = r'^-?\d+,\d+$'

    # 1. Convert timedelta64 columns to total seconds (float)
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()

    # 2. Convert comma-decimal string columns to float (vectorised)
    obj_cols = list(df.select_dtypes(include='object').columns)[:MAX_COLS]
    for col in obj_cols:
        sample = df[col].dropna().astype(str).str.strip().head(50)
        if sample.empty:
            continue
        if sample.str.match(_comma_re).sum() >= max(1, len(sample) * 0.3):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(',', '.', regex=False),
                errors='coerce'
            )

    return df


# ---------------------------------------------------------------------------
# Static lint pass for AI-generated code
# ---------------------------------------------------------------------------

_RE_INFER_DATETIME_FMT = re.compile(
    r',\s*infer_datetime_format\s*=\s*(?:True|False)', re.IGNORECASE
)
_RE_SQUEEZE_KWARG = re.compile(
    r',\s*squeeze\s*=\s*(?:True|False)', re.IGNORECASE
)
_RE_DF_APPEND = re.compile(
    r'(\w+)\s*=\s*\1\.append\((.+?)\)',
)
_RE_IMPORT_PANDAS = re.compile(
    r'^\s*(?:import\s+pandas(?:\s+as\s+\w+)?|from\s+pandas\s+import\s+.+)\s*$',
    re.MULTILINE,
)
_RE_IMPORT_MATPLOTLIB = re.compile(
    r'^\s*(?:import\s+matplotlib(?:\.pyplot)?(?:\s+as\s+\w+)?|from\s+matplotlib(?:\.pyplot)?\s+import\s+.+)\s*$',
    re.MULTILINE,
)
_RE_IMPORT_SEABORN = re.compile(
    r'^\s*(?:import\s+seaborn(?:\s+as\s+\w+)?|from\s+seaborn\s+import\s+.+)\s*$',
    re.MULTILINE,
)
_RE_IMPORT_NUMPY = re.compile(
    r'^\s*(?:import\s+numpy(?:\s+as\s+\w+)?|from\s+numpy\s+import\s+.+)\s*$',
    re.MULTILINE,
)
# Fix .strftime() → .dt.strftime() on Series (LLMs frequently forget .dt)
_RE_SERIES_STRFTIME = re.compile(
    r'(?<!\.dt)\.strftime\(',
)


def _lint_generated_code(code: str) -> str:
    """
    Apply regex-based fixes to AI-generated code before execution.
    Catches known deprecated/removed APIs and redundant imports that would
    bypass the shimmed exec environment.
    """
    original = code

    # 1. Strip infer_datetime_format=... from to_datetime() calls
    code = _RE_INFER_DATETIME_FMT.sub('', code)

    # 2. Strip squeeze=... from read_csv() calls
    code = _RE_SQUEEZE_KWARG.sub('', code)

    # 3. Replace df = df.append(x) → df = pd.concat([df, x])
    code = _RE_DF_APPEND.sub(r'\1 = pd.concat([\1, \2])', code)

    # 4. Remove redundant imports (pandas, matplotlib, seaborn, numpy)
    #    These are already pre-loaded as pd, plt, sns, np in the exec env.
    code = _RE_IMPORT_PANDAS.sub('# (import removed — pd already available)', code)
    code = _RE_IMPORT_MATPLOTLIB.sub('# (import removed — plt already available)', code)
    code = _RE_IMPORT_SEABORN.sub('# (import removed — sns already available)', code)
    code = _RE_IMPORT_NUMPY.sub('# (import removed — np already available)', code)

    # 5. Fix .strftime() → .dt.strftime() on pandas Series
    code = _RE_SERIES_STRFTIME.sub('.dt.strftime(', code)

    if code != original:
        print("[chart-lint] Pre-exec lint applied fixes to generated code")

    return code


def execute_visualization_code(df: pd.DataFrame, code: str, output_path: str = "output_chart.png") -> tuple[bool, str | None]:
    """
    Execute the AI-generated visualization code safely.
    Pre-processes df (timedelta→float, comma-decimal fix) before exec().
    Monkey-patches the real pandas module during execution so that all import
    paths (import pandas, from pandas import ..., etc.) use the compat shims.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    # ── Pandas compatibility shims (module-level monkey-patch) ────────────────
    # AI models are trained on older pandas patterns and may generate calls with
    # deprecated/removed kwargs.  We monkey-patch the real pd module so that
    # ALL import paths are covered — not just the 'pd' alias in exec_globals.
    #
    # Removed in pandas 2.0+:
    #   pd.to_datetime(infer_datetime_format=...)  → just drop the arg
    #   pd.read_csv(squeeze=...)                   → just drop the arg
    #   DataFrame.append / Series.append           → caught by lint, but
    #       also patched here as a last resort.
    # -------------------------------------------------------------------------
    _REMOVED_TO_DATETIME_KWARGS = {"infer_datetime_format"}
    _real_to_datetime = pd.to_datetime
    _real_read_csv = pd.read_csv

    def _compat_to_datetime(*args, **kwargs):
        for k in _REMOVED_TO_DATETIME_KWARGS:
            kwargs.pop(k, None)
        return _real_to_datetime(*args, **kwargs)

    def _compat_read_csv(*args, **kwargs):
        kwargs.pop("squeeze", None)
        return _real_read_csv(*args, **kwargs)

    # Apply lint fixes to the code text first
    code = _lint_generated_code(code)

    # Monkey-patch the REAL pandas module so every import path is covered
    pd.to_datetime = _compat_to_datetime
    pd.read_csv = _compat_read_csv
    # ── end shim setup ────────────────────────────────────────────────────────

    try:
        clean_df = _preprocess_df_for_chart(df)

        exec_globals = {
            'df': clean_df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': __import__('numpy'),
            'io': io,
            'base64': base64
        }

        exec(code, exec_globals)

        if os.path.exists(output_path):
            return True, None
        else:
            msg = f"Código executado mas '{output_path}' não foi criado. Verifique se plt.savefig() está no código."
            print(f"[chart-exec] {msg}")
            return False, msg

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {str(e)}"
        print(f"[chart-exec] ERROR executing generated code: {msg}\n{tb}")
        return False, f"Erro ao executar o gráfico: {msg}"

    finally:
        # ── Restore original pandas functions ──────────────────────────────
        pd.to_datetime = _real_to_datetime
        pd.read_csv = _real_read_csv


def generate_chart_from_prompt(df: pd.DataFrame, prompt: str, analysis_response: str = "", model: str = "gpt-5-mini") -> tuple[str | None, str | None, str | None]:
    """
    Generate a chart based on the prompt using AI-generated code.
    Returns (chart_base64, None, viz_code) on success, or (None, error_message, None) on failure.

    Implements a single auto-retry: if the first attempt fails at execution,
    the error is fed back to the LLM so it can self-correct.
    """
    MAX_ATTEMPTS = 2
    output_path = "output_chart.png"
    last_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # ── Code generation ────────────────────────────────────────────────
        if attempt == 1:
            print("Attempting AI-generated visualization...")
            viz_code, gen_error = generate_visualization_code_with_ai(df, prompt, model)
        else:
            # Retry: append the previous error so the LLM can self-correct
            retry_prompt = (
                f"{prompt}\n\n"
                f"[RETRY] The previous code you generated failed with this error:\n"
                f"{last_error}\n"
                f"Fix the code so it does not produce this error. "
                f"Remember: pandas >= 3.0, do NOT use deprecated APIs."
            )
            print(f"[chart-gen] Retry attempt {attempt}: feeding error back to LLM")
            viz_code, gen_error = generate_visualization_code_with_ai(df, retry_prompt, model)

        if gen_error:
            print(f"[chart-gen] Code generation failed (attempt {attempt}): {gen_error}")
            last_error = gen_error
            continue

        print(f"Generated visualization code (attempt {attempt}):")
        print(viz_code)
        print("-" * 50)

        # ── Execution ──────────────────────────────────────────────────────
        if os.path.exists(output_path):
            os.remove(output_path)

        success, exec_error = execute_visualization_code(df, viz_code, output_path)
        if success:
            try:
                with open(output_path, 'rb') as img_file:
                    chart_base64 = base64.b64encode(img_file.read()).decode()
                os.remove(output_path)
                if attempt > 1:
                    print(f"AI-generated visualization succeeded on retry (attempt {attempt})!")
                else:
                    print("AI-generated visualization successful!")
                return chart_base64, None, viz_code
            except Exception as e:
                msg = f"Erro ao ler o gráfico gerado: {str(e)}"
                print(f"[chart-gen] {msg}")
                return None, msg, None

        # Execution failed — store the error for the retry prompt
        last_error = exec_error or "Erro desconhecido ao executar o código de visualização."
        print(f"[chart-gen] Execution failed (attempt {attempt}): {last_error}")

    # All attempts exhausted
    return None, last_error or "Erro desconhecido ao executar o código de visualização.", None

def generate_chart_fallback(df: pd.DataFrame, prompt: str, analysis_response: str = "") -> str:
    """
    Fallback chart generation using rule-based approach
    This is the original generate_chart_from_prompt logic
    """
    try:
        # Set style for better looking charts
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        prompt_lower = prompt.lower()
        lang = detect_language(prompt)
        
        # Detect chart type requested
        chart_type = None
        if any(word in prompt_lower for word in ['bar chart', 'bar graph', 'barra', 'barras']):
            chart_type = 'bar'
        elif any(word in prompt_lower for word in ['line chart', 'line graph', 'linha', 'linhas', 'trend', 'tendência']):
            chart_type = 'line'
        elif any(word in prompt_lower for word in ['pie chart', 'pie graph', 'pizza', 'torta']):
            chart_type = 'pie'
        elif any(word in prompt_lower for word in ['scatter', 'dispersão', 'scatter plot']):
            chart_type = 'scatter'
        elif any(word in prompt_lower for word in ['histogram', 'histograma', 'distribution', 'distribuição']):
            chart_type = 'histogram'
        elif any(word in prompt_lower for word in ['heatmap', 'heat map', 'mapa de calor', 'correlação', 'correlation']):
            chart_type = 'heatmap'
        elif any(word in prompt_lower for word in ['box plot', 'boxplot', 'box', 'caixa']):
            chart_type = 'box'
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
        
        # Generate chart based on type or infer from data
        if chart_type == 'heatmap' and len(numeric_cols) > 1:
            plt.close()
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            title = 'Correlation Heatmap' if lang == 'en' else 'Mapa de Correlação'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
        elif chart_type == 'pie':
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                data = df[col].value_counts().head(10)
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.pie(df[col].head(10), labels=df.index[:10], autopct='%1.1f%%')
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
                
        elif chart_type == 'histogram':
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency' if lang == 'en' else 'Frequência', fontsize=12)
                title = f'Distribution of {col}' if lang == 'en' else f'Distribuição de {col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'box':
            if len(numeric_cols) > 0:
                df[numeric_cols[:5]].boxplot(ax=ax)
                ax.set_ylabel('Value' if lang == 'en' else 'Valor', fontsize=12)
                ax.set_title('Box Plot', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'line':
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                df[col].plot(kind='line', ax=ax, linewidth=2, marker='o')
                ax.set_ylabel(col, fontsize=12)
                ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
                title = f'{col} Trend' if lang == 'en' else f'Tendência de {col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif any(word in prompt_lower for word in ['top', 'scorer', 'highest', 'best', 'most', 'maior', 'melhor', 'artilheiro']):
            if len(numeric_cols) > 0:
                target_col = None
                for col in df.columns:
                    if any(word in col.lower() for word in ['goal', 'score', 'point', 'gol', 'ponto', 'valor', 'value']):
                        target_col = col
                        break
                
                if not target_col:
                    target_col = numeric_cols[0]
                
                name_cols = [c for c in df.columns if any(word in c.lower() for word in ['name', 'player', 'team', 'nome', 'jogador', 'time', 'equipe'])]
                
                if name_cols:
                    top_data = df.nlargest(15, target_col)
                    bars = ax.barh(range(len(top_data)), top_data[target_col])
                    ax.set_yticks(range(len(top_data)))
                    ax.set_yticklabels(top_data[name_cols[0]], fontsize=10)
                    ax.set_xlabel(target_col, fontsize=12)
                    title = f'Top 15 by {target_col}' if lang == 'en' else f'Top 15 por {target_col}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    
                    for i, (idx, bar) in enumerate(zip(top_data.index, bars)):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.1f}', ha='left', va='center', fontsize=9)
                else:
                    top_data = df.nlargest(15, target_col)
                    ax.bar(range(len(top_data)), top_data[target_col])
                    ax.set_xlabel('Rank' if lang == 'en' else 'Posição', fontsize=12)
                    ax.set_ylabel(target_col, fontsize=12)
                    title = f'Top 15 {target_col}' if lang == 'en' else f'Top 15 {target_col}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    
        else:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].mean().nlargest(15)
                grouped.plot(kind='barh', ax=ax)
                ax.set_xlabel(num_col, fontsize=12)
                ax.set_ylabel(cat_col, fontsize=12)
                title = f'Average {num_col} by {cat_col}' if lang == 'en' else f'Média de {num_col} por {cat_col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]
                df[col].head(20).plot(kind='bar', ax=ax)
                ax.set_ylabel(col, fontsize=12)
                ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
                ax.set_title(f'{col} Values', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        chart_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close('all')
        
        return chart_base64
        
    except Exception as e:
        print(f"Error generating fallback chart: {str(e)}")
        plt.close('all')
        return None

def recommend_chart_type(df: pd.DataFrame) -> str:
    """
    Analyze the DataFrame structure and recommend the best chart types.
    Works without any external API — uses data shape heuristics.
    """
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'string', 'category']).columns)
    total_rows = len(df)
    num_numeric = len(numeric_cols)
    num_categorical = len(categorical_cols)

    recommendations = []
    recommendations.append("## 📊 Recomendação de Gráficos\n")
    recommendations.append(f"**Resumo dos dados:** {total_rows} linhas, {num_numeric} colunas numéricas, {num_categorical} colunas categóricas.\n")
    recommendations.append(f"**Colunas numéricas:** {', '.join(numeric_cols) if numeric_cols else 'Nenhuma'}")
    recommendations.append(f"**Colunas categóricas:** {', '.join(categorical_cols) if categorical_cols else 'Nenhuma'}\n")
    recommendations.append("### Gráficos recomendados:\n")

    rank = 1

    # Bar chart: categorical + numeric
    if num_categorical > 0 and num_numeric > 0:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        recommendations.append(f"**{rank}. Gráfico de Barras** 📊")
        recommendations.append(f"   - Ideal para comparar `{num}` entre diferentes `{cat}`.")
        recommendations.append(f"   - Exemplo: \"Gere um gráfico de barras de {num} por {cat}\"\n")
        rank += 1

    # Scatter plot: 2+ numeric
    if num_numeric >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        recommendations.append(f"**{rank}. Gráfico de Dispersão (Scatter Plot)** 🔵")
        recommendations.append(f"   - Ideal para ver a correlação entre `{x}` e `{y}`.")
        recommendations.append(f"   - Exemplo: \"Gere um scatter plot de {y} vs {x}\"\n")
        rank += 1

    # Histogram: numeric data
    if num_numeric > 0:
        col = numeric_cols[0]
        recommendations.append(f"**{rank}. Histograma** 📈")
        recommendations.append(f"   - Ideal para ver a distribuição de `{col}`.")
        recommendations.append(f"   - Exemplo: \"Gere um histograma de {col}\"\n")
        rank += 1

    # Heatmap: 3+ numeric (correlation)
    if num_numeric >= 3:
        recommendations.append(f"**{rank}. Mapa de Calor (Heatmap)** 🟥")
        recommendations.append(f"   - Ideal para ver a correlação entre todas as variáveis numéricas.")
        recommendations.append(f"   - Exemplo: \"Gere um mapa de calor de correlação\"\n")
        rank += 1

    # Pie chart: categorical with few categories
    if num_categorical > 0:
        cat = categorical_cols[0]
        unique_count = df[cat].nunique()
        if unique_count <= 10:
            recommendations.append(f"**{rank}. Gráfico de Pizza** 🥧")
            recommendations.append(f"   - Ideal para ver a proporção de `{cat}` ({unique_count} categorias).")
            recommendations.append(f"   - Exemplo: \"Gere um gráfico de pizza de {cat}\"\n")
            rank += 1

    # Line chart: if data looks sequential
    if num_numeric > 0 and total_rows > 5:
        col = numeric_cols[0]
        recommendations.append(f"**{rank}. Gráfico de Linhas** 📉")
        recommendations.append(f"   - Ideal para ver tendências de `{col}` ao longo do tempo.")
        recommendations.append(f"   - Exemplo: \"Gere um gráfico de linhas de {col}\"\n")
        rank += 1

    # Box plot: numeric data
    if num_numeric > 0:
        recommendations.append(f"**{rank}. Box Plot** 📦")
        recommendations.append(f"   - Ideal para ver a distribuição e outliers das colunas numéricas.")
        recommendations.append(f"   - Exemplo: \"Gere um box plot das colunas numéricas\"\n")
        rank += 1

    recommendations.append("---")
    recommendations.append("💡 **Dica:** Use o botão **Gerar Gráfico** para criar qualquer uma dessas visualizações. Basta digitar o que deseja e clicar no botão verde.")

    return "\n".join(recommendations)


def recommend_chart_type_with_ai(df: pd.DataFrame, model: str = "gpt-5-mini") -> str:
    """
    Use OpenAI to provide intelligent chart recommendations. Falls back to heuristic version.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return recommend_chart_type(df)

    df_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'string', 'category']).columns),
        "sample_data": df.head(3).to_dict(),
        "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else None
    }

    system_prompt = """You are a data visualization expert. The user wants to know which chart types are best suited to visualize their dataset.

IMPORTANT RULES:
1. Always respond in Portuguese (pt-BR).
2. Recommend 3-5 chart types ranked from most to least suitable.
3. For each chart, explain WHY it's suitable for this specific data.
4. Reference actual column names from the dataset in your examples.
5. End with a tip telling the user to use the "Gerar Gráfico" button (green button) to create the visualization.
6. Format the response nicely with markdown, emojis, and clear structure.
7. DO NOT generate any code or chart — only recommend and explain."""

    user_message = f"""Analyze this dataset and recommend the best chart types to visualize it:

- Shape: {df_summary['shape']} (rows, columns)
- Columns: {df_summary['columns']}
- Data types: {df_summary['dtypes']}
- Numeric columns: {df_summary['numeric_columns']}
- Categorical columns: {df_summary['categorical_columns']}
- Sample data: {df_summary['sample_data']}
- Basic statistics: {df_summary['basic_stats']}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=1500
        )
        _track_tokens(response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI chart recommendation failed: {str(e)}")
        return recommend_chart_type(df)


@app.post("/analysis-gen")
async def download_csv(request: URLRequest):
    # Reset per-request token counter
    _request_tokens.set(0)
    try:
        started_at = time.perf_counter()
        print(f"User prompt: {request.prompt}")
        print(f"Number of files to process: {len(request.fileAddresses)}")
        print(f"Generate chart requested: {request.generateChart}")
        print(f"Chart recommendation requested: {request.chartRecommendation}")
        print(f"Chat ID: {request.chatId}")
        print(f"Force refresh: {request.forceRefresh}")

        if len(request.fileAddresses) > MAX_FILES_PER_ANALYSIS_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Máximo de {MAX_FILES_PER_ANALYSIS_REQUEST} arquivos por análise. Reduza a seleção e tente novamente.",
            )

        resolved_model = request.model.strip() if request.model and request.model.strip() else "gpt-5-mini"
        request.model = resolved_model

        async def load_single_dataframe(
            client_http: httpx.AsyncClient,
            idx: int,
            file_address: HttpUrl,
            semaphore: asyncio.Semaphore,
        ):
            async with semaphore:
                incoming_file_type = (
                    request.fileTypes[idx]
                    if request.fileTypes and idx < len(request.fileTypes)
                    else None
                )
                resolved_file_type = detect_file_type(
                    explicit_type=incoming_file_type,
                    filename=str(file_address),
                )

                print(f"Processing file {idx + 1}/{len(request.fileAddresses)}: {file_address}")

                # --- df_cache fast-path ---
                if not request.forceRefresh:
                    cached_df = _get_cached_dataframe(str(file_address).strip())
                    if cached_df is not None:
                        print(f"DF cache hit for file {idx + 1}, skipping download+parse.")
                        return idx, cached_df, resolved_file_type

                # --- Fallback: download + parse ---
                df_local = await get_or_download_dataframe(
                    client_http,
                    str(file_address),
                    resolved_file_type,
                    request.forceRefresh,
                )
                print(f"Loaded DataFrame {idx + 1} with shape: {df_local.shape}")
                print(f"Columns: {list(df_local.columns)}")
                return idx, df_local, resolved_file_type
        
        # Download and parse all tabular files
        download_started_at = time.perf_counter()
        async with httpx.AsyncClient() as client_http:
            semaphore = asyncio.Semaphore(max(1, MAX_PARALLEL_DOWNLOADS))
            tasks = [
                load_single_dataframe(client_http, idx, file_address, semaphore)
                for idx, file_address in enumerate(request.fileAddresses)
            ]
            loaded_results = await asyncio.gather(*tasks)

        loaded_results.sort(key=lambda item: item[0])
        dataframes = [item[1] for item in loaded_results]
        processed_file_types = [item[2] for item in loaded_results]
        print(f"Download + parse stage latency: {time.perf_counter() - download_started_at:.2f}s")
        
        # Combine all dataframes into one
        combine_started_at = time.perf_counter()
        if len(dataframes) == 1:
            combined_df = dataframes[0]
        else:
            # Try to concatenate vertically if columns match, otherwise merge horizontally
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                print(f"Combined {len(dataframes)} dataframes vertically")
            except Exception as e:
                print(f"Could not concatenate vertically: {e}. Attempting horizontal merge...")
                combined_df = dataframes[0]
                for df in dataframes[1:]:
                    combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how='outer')
                print(f"Combined {len(dataframes)} dataframes horizontally")
        print(f"Combine stage latency: {time.perf_counter() - combine_started_at:.2f}s")
        
        print(f"Final combined DataFrame shape: {combined_df.shape}")
        print(f"Final columns: {list(combined_df.columns)}")
        large_data_mode = is_large_dataset(combined_df)
        print(f"Large dataset mode: {large_data_mode}")

        # ── RAG: index the combined DataFrame and retrieve relevant rows ──
        # Wrapped in try/except so RAG failures never block the analysis.
        rag_file_key = "|".join(sorted(str(a) for a in request.fileAddresses))
        retrieved_context = ""
        try:
            if rag_store.enabled:
                rag_started_at = time.perf_counter()
                if request.forceRefresh:
                    rag_store.invalidate(rag_file_key)
                rag_store.index_dataframe(rag_file_key, combined_df)
                print(f"RAG index stage latency: {time.perf_counter() - rag_started_at:.2f}s")

            if rag_store.enabled and not request.chartRecommendation:
                rag_retrieve_started_at = time.perf_counter()
                top_k = 20 if request.generateChart else 30
                retrieved_context = rag_store.retrieve(rag_file_key, request.prompt, top_k=top_k)
                print(f"RAG retrieval latency: {time.perf_counter() - rag_retrieve_started_at:.2f}s")
        except Exception as rag_err:
            print(f"[RAG] WARNING: RAG failed, continuing without retrieval context: {rag_err}")
            retrieved_context = ""

        # ── Supplementary context for query-specific columns ──────────
        # Wide files (700+ cols) may have prompt-relevant columns missing
        # from both RAG chunks and the statistical summary.  Extract them
        # directly from the DataFrame so the LLM can cite exact values.
        try:
            supplementary = _build_supplementary_context(
                combined_df, request.prompt, retrieved_context,
            )
            if supplementary:
                sep = "\n\n--- Targeted data for columns mentioned in your query ---\n"
                retrieved_context = (
                    (retrieved_context + sep + supplementary)
                    if retrieved_context
                    else supplementary
                )
                print(f"[SUP] Supplementary context: {len(supplementary)} chars")
        except Exception as sup_err:
            print(f"[SUP] WARNING: supplementary extraction failed: {sup_err}")

        # Route to chart recommendation or regular analysis
        analysis_started_at = time.perf_counter()
        if request.chartRecommendation:
            print("Generating chart recommendation...")
            analysis_text = recommend_chart_type_with_ai(combined_df, request.model)
        elif request.generateChart:
            # Skip text analysis entirely when only a chart is requested.
            # generate_chart_from_prompt will receive an empty fallback hint.
            analysis_text = ""
        else:
            # ── TEXT-ONLY PATH: stream the response directly to the client ──
            print("Generating streaming text analysis...")
            text_generator = analyze_dataframe_with_openai(
                combined_df, request.prompt, request.model, stream=True,
                retrieved_context=retrieved_context,
            )
            print(f"Analysis stage latency (stream started): {time.perf_counter() - analysis_started_at:.2f}s")
            return StreamingResponse(text_generator, media_type="text/plain")

        print(f"Analysis stage latency: {time.perf_counter() - analysis_started_at:.2f}s")

        # Only generate chart when explicitly requested by the user via the "Gerar Gráfico" button
        chart_base64 = None
        chart_code = None
        response_message = None
        if request.generateChart:
            chart_started_at = time.perf_counter()

            # Smart reduction: narrow wide DataFrames to only chart-relevant columns
            # so that files with many columns but few rows (e.g. 3 000 rows × 759 cols)
            # are not incorrectly deferred.
            chart_df = _reduce_df_for_chart(combined_df, request.prompt)

            if chart_df.shape[0] > CHART_MAX_ROWS:
                response_message = (
                    "O dataset possui muitas linhas para gerar o gráfico. "
                    "Refine o filtro para reduzir os dados."
                )
                print(f"Chart generation deferred: {chart_df.shape[0]} rows exceeds CHART_MAX_ROWS={CHART_MAX_ROWS}.")
            else:
                print(f"Generating chart (explicit request) with shape {chart_df.shape}...")
                chart_base64, chart_error, chart_code = generate_chart_from_prompt(
                    chart_df, request.prompt, "", request.model
                )
                if chart_error:
                    response_message = chart_error

            print(f"Chart stage latency: {time.perf_counter() - chart_started_at:.2f}s")

        # Create data summary
        data_summary = {
            "rows": len(combined_df),
            "columns": len(combined_df.columns),
            "column_names": list(combined_df.columns),
            "numeric_columns": list(combined_df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(combined_df.select_dtypes(include=['object']).columns),
            "files_processed": len(request.fileAddresses),
            "file_types": processed_file_types,
            "visualization_generated": chart_base64 is not None,
            "chart_deferred": bool(request.generateChart and chart_base64 is None and response_message is not None),
            "large_dataset_mode": large_data_mode,
        }
        print(f"Total /analysis-gen latency: {time.perf_counter() - started_at:.2f}s")

        # Collect total tokens consumed across all OpenAI calls in this request
        total_tokens_used = _request_tokens.get(0)
        print(f"DEBUG_TOKENS: Total tokens consumed in this request: {total_tokens_used}")

        # Return JSON response (chart / recommendation paths)
        return {
            "text_response": analysis_text,
            "chart_base64": chart_base64,
            "chart_code": chart_code,
            "data_summary": data_summary,
            "message": response_message,
            "tokens_used": total_tokens_used,
        }
        
    except httpx.RequestError as e:
        print('Error downloading file')
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)