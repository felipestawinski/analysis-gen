"""
Context construction for the analysis-gen service.

Column selection, role detection, supplementary context building,
and LLM dataset message assembly.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Optional

import pandas as pd

from normalization import _is_text_dtype

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LARGE_DATASET_ROW_THRESHOLD = int(os.getenv("LARGE_DATASET_ROW_THRESHOLD", "120000"))
LARGE_DATASET_CELL_THRESHOLD = int(os.getenv("LARGE_DATASET_CELL_THRESHOLD", "1500000"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "120000"))
LLM_COMPACT_CONTEXT_CHARS = int(os.getenv("LLM_COMPACT_CONTEXT_CHARS", "80000"))
CHART_MAX_COLUMNS = int(os.getenv("CHART_MAX_COLUMNS", "30"))
CHART_MAX_ROWS = int(os.getenv("CHART_MAX_ROWS", "500000"))

# ---------------------------------------------------------------------------
# Language / request helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    portuguese_indicators = [
        'qual', 'quem', 'onde', 'quando', 'como', 'por que', 'porque',
        'jogador', 'melhor', 'maior', 'menor', 'média', 'total',
        'gols', 'pontos', 'time', 'equipe', 'artilheiro', 'temporada'
    ]
    text_lower = text.lower()
    portuguese_count = sum(1 for word in portuguese_indicators if word in text_lower)
    return 'pt' if portuguese_count > 0 else 'en'


def is_visualization_request(prompt: str) -> bool:
    visualization_keywords = [
        'visualize', 'visualization', 'chart', 'graph', 'plot', 'show me',
        'display', 'draw', 'create a chart', 'create a graph', 'bar chart',
        'line chart', 'pie chart', 'scatter plot', 'histogram', 'heatmap',
        'visualizar', 'visualização', 'gráfico', 'plotar', 'mostre',
        'exibir', 'desenhar', 'criar gráfico', 'criar um gráfico',
        'gráfico de barras', 'gráfico de linhas', 'gráfico de pizza',
        'dispersão', 'histograma', 'mapa de calor',
        'visualizar', 'visualización', 'gráfica', 'mostrar', 'dibujar'
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in visualization_keywords)


def is_large_dataset(df: pd.DataFrame) -> bool:
    rows, cols = df.shape
    return rows >= LARGE_DATASET_ROW_THRESHOLD or (rows * cols) >= LARGE_DATASET_CELL_THRESHOLD


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

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
        "invalid_request_error", "does not exist",
        "unsupported", "invalid model", "model_not_found",
    ]
    return not any(token in error_text for token in no_retry_indicators)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Column matching constants
# ---------------------------------------------------------------------------

_COL_MATCH_STOP = frozenset({
    "que", "qual", "quais", "com", "mais", "dos", "das", "por", "para",
    "uma", "uns", "the", "and", "for", "are", "was", "how", "what",
    "which", "were", "has", "como", "foram", "nos", "nas", "ele", "ela",
    "isso", "este", "esta", "esses", "esse", "essa", "seu", "sua",
    "cada", "entre", "sobre", "desde", "mesmo", "quando", "onde",
})

_CONCEPT_SYNONYMS: dict[str, set[str]] = {
    'game':     {'jogo', 'match', 'period', 'event', 'partida', 'evento'},
    'player':   {'jogador', 'atleta', 'name', 'player'},
    'heart':    {'heart', 'cardíaco', 'cardíaca', 'bpm'},
    'speed':    {'velocity', 'velocidade', 'sprint'},
    'distance': {'odometer', 'distância', 'dist'},
    'team':     {'time', 'equipe', 'club', 'clube'},
    'match':    {'jogo', 'game', 'partida', 'period', 'event', 'evento'},
    'rate':     {'rate', 'taxa', 'freq'},
    'average':  {'avg', 'mean', 'média'},
    'jogo':     {'game', 'match', 'partida', 'period', 'event'},
    'jogador':  {'player', 'atleta', 'name'},
    'vitória':  {'resultado', 'result', 'victory', 'win'},
    'vitórias': {'resultado', 'result', 'victories', 'wins'},
    'vitoria':  {'resultado', 'result', 'victory', 'win'},
    'vitorias': {'resultado', 'result', 'victories', 'wins'},
    'derrota':  {'resultado', 'result', 'defeat', 'loss'},
    'derrotas': {'resultado', 'result', 'defeats', 'losses'},
    'victory':  {'resultado', 'result', 'vitória', 'win'},
    'defeat':   {'resultado', 'result', 'derrota', 'loss'},
    'resultado':{'result', 'vitória', 'derrota', 'victory', 'defeat', 'win', 'loss', 'empate', 'draw'},
    'result':   {'resultado', 'vitória', 'derrota', 'victory', 'defeat', 'win', 'loss'},
    'win':      {'vitória', 'resultado', 'victory', 'result'},
    'loss':     {'derrota', 'resultado', 'defeat', 'result'},
}


# ---------------------------------------------------------------------------
# Column role detection
# ---------------------------------------------------------------------------

_IDENTITY_KEYWORDS = frozenset({
    'name', 'date', 'mando', 'resultado', 'atividade', 'microciclo',
    'campeonato', 'adversário', 'adversario', 'semana',
})

_METRIC_INDICATORS = frozenset({
    'band', 'dist', 'dur', 'eff', 'load', 'rate', 'heart', 'velocity',
    'acceleration', 'power', 'running', 'recovery', 'ima', 'tackles',
    'metabolic', 'rhie', 'count', 'exertion', 'odometer', 'peak',
    'unix', 'avg', 'min', 'max', 'per', 'tot', 'free', 'field',
    'bench', 'impacts', 'deliveries', 'possessions', 'equivalent',
})

_IDENTITY_EXACT_PATTERNS = {
    'player name', 'period name', 'period number', 'date', 'position',
    'start time', 'end time', 'duration',
}


def _detect_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify every column into identity, metric, or dimension roles."""
    identity: list[str] = []
    metric: list[str] = []
    dimension: list[str] = []

    for col in df.columns:
        col_lower = str(col).lower().strip()
        tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        has_metric_indicator = bool(tokens & _METRIC_INDICATORS)

        if col_lower in _IDENTITY_EXACT_PATTERNS:
            identity.append(col)
            continue
        if (tokens & _IDENTITY_KEYWORDS) and not has_metric_indicator:
            identity.append(col)
            continue
        if not is_numeric and not has_metric_indicator:
            nunique = df[col].nunique()
            if nunique < max(10, int(len(df) * 0.3)):
                identity.append(col)
            else:
                dimension.append(col)
            continue
        metric.append(col)

    return {"identity": identity, "metric": metric, "dimension": dimension}


# ---------------------------------------------------------------------------
# Column relevance scoring
# ---------------------------------------------------------------------------

def _select_relevant_columns(df: pd.DataFrame, user_prompt: str, max_columns: int) -> list[str]:
    """Return up to *max_columns* column names, prioritised by relevance to *user_prompt*."""
    all_columns = list(df.columns)
    if len(all_columns) <= max_columns:
        return all_columns

    prompt_lower = user_prompt.lower()
    prompt_tokens = {
        w for w in re.findall(r'\b\w{3,}\b', prompt_lower)
        if w not in _COL_MATCH_STOP
    }

    expanded_tokens = set(prompt_tokens)
    for token in prompt_tokens:
        syns = _CONCEPT_SYNONYMS.get(token)
        if syns:
            expanded_tokens |= syns

    scored: list[tuple[str, int, int]] = []
    for idx, col in enumerate(all_columns):
        col_lower = str(col).lower()
        score = 0
        if col_lower in prompt_lower:
            score += 1000
        col_tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        if col_tokens and prompt_tokens:
            overlap = col_tokens & prompt_tokens
            if overlap:
                score += int(len(overlap) / len(col_tokens) * 200)
                score += len(overlap) * 30
        if col_tokens and expanded_tokens and score == 0:
            syn_overlap = col_tokens & expanded_tokens
            if syn_overlap:
                score += int(len(syn_overlap) / len(col_tokens) * 150)
                score += len(syn_overlap) * 20
        scored.append((col, score, idx))

    matched = sorted(
        [(c, s, i) for c, s, i in scored if s > 0],
        key=lambda x: (-x[1], x[2]),
    )

    ranked: list[str] = []
    seen: set[str] = set()
    roles = _detect_column_roles(df)
    for col in roles["identity"]:
        if col in all_columns and col not in seen:
            ranked.append(col)
            seen.add(col)
    for col, _s, _i in matched:
        if col not in seen:
            ranked.append(col)
            seen.add(col)

    numeric_columns = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [c for c in all_columns if _is_text_dtype(df[c])]
    for col in numeric_columns + categorical_columns + all_columns:
        if col not in seen:
            ranked.append(col)
            seen.add(col)
        if len(ranked) >= max_columns:
            break

    return ranked[:max_columns]


# ---------------------------------------------------------------------------
# Sample rows / supplementary context
# ---------------------------------------------------------------------------

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
    """Return DataFrame columns whose name appears in *user_prompt*."""
    prompt_lower = user_prompt.lower()
    prompt_tokens = set(re.findall(r'\b\w{3,}\b', prompt_lower))
    expanded_tokens = set(prompt_tokens)
    for token in prompt_tokens:
        syns = _CONCEPT_SYNONYMS.get(token)
        if syns:
            expanded_tokens |= syns

    identity_col_set = set(_detect_column_roles(df)["identity"])
    candidates: list[tuple[str, int]] = []
    for col in df.columns:
        col_lower = str(col).lower()
        score = 0
        if col_lower in prompt_lower:
            score += 100
        col_tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        if col_tokens and prompt_tokens:
            overlap = col_tokens & prompt_tokens
            if overlap and len(overlap) / len(col_tokens) >= 0.5:
                score += 50
        if score == 0 and col_tokens and expanded_tokens:
            syn_overlap = col_tokens & expanded_tokens
            if syn_overlap and len(syn_overlap) / len(col_tokens) >= 0.5:
                if col in identity_col_set:
                    score += 40
                else:
                    score += 15
        if score > 0:
            candidates.append((col, score))

    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:10]]


_ENTITY_VALUE_RE = re.compile(
    r'\b(?:player|jogador|atleta|team|time|equipe)\s+(\d+)\b',
    re.IGNORECASE,
)


def _build_supplementary_context(
    df: pd.DataFrame,
    user_prompt: str,
    existing_context: str,
    max_rows: int = 30,
    max_chars: int = 8000,
) -> str:
    """Extract targeted rows for columns mentioned in *user_prompt*."""
    id_candidates = [
        "Period Name", "Date", "Player Name",
        "ADVERSÁRIO", "CAMPEONATO", "EVENTO",
    ]
    id_cols: list[str] = []
    for c in id_candidates:
        if c in df.columns:
            id_cols.append(c)
        elif f" {c}" in df.columns:
            id_cols.append(f" {c}")

    entity_values = [int(v) for v in _ENTITY_VALUE_RE.findall(user_prompt)]
    if entity_values:
        identity_cols = _detect_column_roles(df)["identity"]
        entity_col = None
        for ic in identity_cols:
            if ic in df.columns and pd.api.types.is_numeric_dtype(df[ic]):
                col_vals = set(df[ic].dropna().astype(int).unique())
                if all(v in col_vals for v in entity_values):
                    entity_col = ic
                    break
        if entity_col:
            extract_cols = list(dict.fromkeys(id_cols + [entity_col]))
            working = df[extract_cols].copy()
            per_entity = max(1, max_rows // len(entity_values))
            parts = []
            for ev in entity_values:
                entity_rows = working[working[entity_col] == ev].head(per_entity)
                parts.append(entity_rows)
            filtered = pd.concat(parts).head(max_rows)
            print(f"[SUP] Entity-aware filter: col='{entity_col}', values={entity_values}, rows={len(filtered)}")
            if not filtered.empty:
                buf = io.StringIO()
                filtered.to_csv(buf, index=False)
                text = buf.getvalue()
                return text[:max_chars] if len(text) > max_chars else text

    target_cols = _extract_prompt_columns(df, user_prompt)
    if not target_cols:
        return ""
    existing_lower = (existing_context or "").lower()
    missing_cols = [c for c in target_cols if str(c).lower() not in existing_lower]
    if not missing_cols:
        return ""

    extract_cols = list(dict.fromkeys(id_cols + missing_cols))
    working = df[extract_cols].copy()
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


# ---------------------------------------------------------------------------
# DataFrame context builder
# ---------------------------------------------------------------------------

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
    categorical_columns = [col for col in selected_columns if _is_text_dtype(working_df[col])]

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

    identity_col_set = set(_detect_column_roles(df)["identity"])

    schema_preview = {}
    for col in selected_columns:
        dtype_str = str(working_df[col].dtype)
        is_numeric = col in numeric_columns
        is_identity = col in identity_col_set

        if is_numeric and is_identity:
            sample_vals = working_df[col].dropna().unique()
            col_meta: dict = {
                "dtype": dtype_str,
                "kind": "identity",
                "role_hint": "This column contains entity identifiers used for grouping/filtering, not measurements.",
                "unique_count": int(working_df[col].nunique()),
                "sample_values": sorted([int(v) if float(v) == int(v) else float(v) for v in sample_vals[:20]]),
            }
        elif is_numeric:
            col_meta = {"dtype": dtype_str, "kind": "numeric"}
            if col in numeric_summary:
                stats = numeric_summary[col]
                col_meta["min"] = stats["min"]
                col_meta["max"] = stats["max"]
                col_meta["mean"] = stats["mean"]
                col_meta["median"] = stats["median"]
        else:
            col_meta = {"dtype": dtype_str, "kind": "categorical"}
            if col in categorical_summary:
                cat = categorical_summary[col]
                col_meta["unique_count"] = cat["unique_values"]
                col_meta["top_3"] = list(cat["top_values"].keys())[:3]
        schema_preview[str(col)] = col_meta

    prompt_columns = _extract_prompt_columns(df, user_prompt)
    prompt_value_counts: dict[str, dict] = {}
    for col in prompt_columns:
        if col in df.columns and _is_text_dtype(df[col]):
            series = df[col].dropna().astype(str)
            if not series.empty:
                vc = series.value_counts().to_dict()
                prompt_value_counts[str(col)] = {
                    "total_non_null": int(series.count()),
                    "unique_values": int(series.nunique()),
                    "value_counts": {str(k): int(v) for k, v in vc.items()},
                }
                if col not in categorical_summary:
                    categorical_summary[col] = {
                        "unique_values": int(series.nunique()),
                        "top_values": series.value_counts().head(5).to_dict(),
                    }

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
    if prompt_value_counts:
        context["prompt_column_value_counts"] = prompt_value_counts
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

    profile = context_profiles[0]
    context_payload = _build_dataframe_context(
        df, user_prompt,
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
            "For counting or aggregation questions, use the 'prompt_column_value_counts' "
            "section (computed over ALL rows) as the authoritative source. "
            "For row-level detail questions, prefer citing relevant data rows."
        )
        return base

    message = _build_message(context_payload)

    if len(message) <= profile["max_chars"]:
        print("LLM context profile selected:", {"max_columns": profile["max_columns"], "message_chars": len(message)})
        return message

    ratio = profile["max_chars"] / max(len(message), 1)
    fallback_profile = {
        "max_columns": max(5, int(profile["max_columns"] * ratio)),
        "max_numeric_stats": max(3, int(profile["max_numeric_stats"] * ratio)),
        "max_categorical_stats": max(2, int(profile["max_categorical_stats"] * ratio)),
        "max_chars": profile["max_chars"],
    }

    context_payload = _build_dataframe_context(
        df, user_prompt,
        max_columns=fallback_profile["max_columns"],
        max_numeric_stats=fallback_profile["max_numeric_stats"],
        max_categorical_stats=fallback_profile["max_categorical_stats"],
    )
    message = _build_message(context_payload)
    print("LLM context profile selected:", {"max_columns": fallback_profile["max_columns"], "message_chars": len(message)})
    return message


# ---------------------------------------------------------------------------
# Chart DataFrame reduction
# ---------------------------------------------------------------------------

def _reduce_df_for_chart(df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
    """Reduce a wide DataFrame to only the columns relevant to the chart request."""
    if df.shape[1] <= CHART_MAX_COLUMNS:
        return df
    selected = _select_relevant_columns(df, user_prompt, CHART_MAX_COLUMNS)
    roles = _detect_column_roles(df)
    for col in roles["identity"]:
        if col not in selected and col in df.columns:
            selected.append(col)
    reduced = df[selected].copy()
    print(
        f"[chart-reduce] Reduced DataFrame from {df.shape[1]} cols → "
        f"{reduced.shape[1]} cols for chart generation "
        f"(rows={reduced.shape[0]}, cells={reduced.shape[0] * reduced.shape[1]}, "
        f"identity_cols={len(roles['identity'])})"
    )
    return reduced
