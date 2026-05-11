"""
LLM text analysis for the analysis-gen service.

Contains the OpenAI chat completion calls (streaming + non-streaming)
and the rule-based fallback analyser.
"""

from __future__ import annotations

import os
import time
import contextvars
from typing import List, Optional

import pandas as pd
from openai import OpenAI

from context import (
    detect_language, is_large_dataset, is_context_limit_error,
    build_llm_dataset_message,
)
from prompts import TEXT_ANALYSIS_SYSTEM_PROMPT, token_limit_notice

# ---------------------------------------------------------------------------
# OpenAI client + token tracking
# ---------------------------------------------------------------------------

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _client = OpenAI(api_key=api_key)
    return _client

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


def reset_request_tokens():
    _request_tokens.set(0)


def get_request_tokens() -> int:
    return _request_tokens.get(0)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_dataframe_with_openai(
    df: pd.DataFrame,
    user_prompt: str,
    model: str = "gpt-4o",
    stream: bool = False,
    retrieved_context: str = "",
    chat_history: Optional[List[dict]] = None,
):
    """
    Send dataframe info and user prompt to OpenAI for analysis with fallback options.
    When stream=True, returns a generator that yields text chunks instead of a full string.
    """
    system_prompt = TEXT_ANALYSIS_SYSTEM_PROMPT

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

    def _build_llm_messages():
        msgs = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-5:]:
                role = msg.get("role", "user") if isinstance(msg, dict) else msg.role
                content = msg.get("content", "") if isinstance(msg, dict) else msg.content
                if role in ("user", "assistant") and content.strip():
                    msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_message})
        print(f"OpenAI messages count: {len(msgs)} (history: {len(msgs) - 2})")
        return msgs

    # ── Streaming path ────────────────────────────────────────────────────────
    if stream:
        if os.getenv("OPENAI_API_KEY"):
            def _stream_generator():
                try:
                    llm_started_at = time.perf_counter()
                    llm_messages = _build_llm_messages()
                    stream_response = _get_client().chat.completions.create(
                        model=model,
                        messages=llm_messages,
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
                    fallback = analyze_dataframe_fallback(df, user_prompt)
                    yield fallback
            return _stream_generator()
        else:
            def _fallback_gen():
                yield analyze_dataframe_fallback(df, user_prompt)
            return _fallback_gen()

    # ── Non-streaming path ────────────────────────────────────────────────────
    token_limit_exceeded = False
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm_started_at = time.perf_counter()
            llm_messages = _build_llm_messages()
            response = _get_client().chat.completions.create(
                model=model,
                messages=llm_messages,
                max_completion_tokens=16000,
            )
            _track_tokens(response)
            print("using model", model)
            print(f"OpenAI latency: {time.perf_counter() - llm_started_at:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI failed with model '{model}': {str(e)}")
            token_limit_exceeded = is_context_limit_error(e)

    fallback_result = analyze_dataframe_fallback(df, user_prompt)
    if token_limit_exceeded:
        return f"{token_limit_notice(user_prompt, detect_language)}\n\n{fallback_result}"
    return fallback_result


# ---------------------------------------------------------------------------
# Fallback (no API key / API failure)
# ---------------------------------------------------------------------------

def analyze_dataframe_fallback(df: pd.DataFrame, user_prompt: str) -> str:
    """Fallback analysis without external APIs — purely based on pandas operations."""
    lang = detect_language(user_prompt)
    analysis = []

    if lang == 'pt':
        analysis.append(f"O conjunto de dados contém {len(df)} linhas e {len(df.columns)} colunas.")
        analysis.append(f"Colunas: {', '.join(df.columns)}")
    else:
        analysis.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        analysis.append(f"Columns: {', '.join(df.columns)}")

    prompt_lower = user_prompt.lower()
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

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        if lang == 'pt':
            analysis.append("\nEstatísticas Básicas:")
            for col in numeric_cols[:3]:
                analysis.append(f"{col}: mín={df[col].min()}, máx={df[col].max()}, média={df[col].mean():.2f}")
        else:
            analysis.append("\nBasic Statistics:")
            for col in numeric_cols[:3]:
                analysis.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")

    if lang == 'pt':
        return "\n".join(analysis) if analysis else "Não foi possível analisar os dados para esta pergunta específica."
    else:
        return "\n".join(analysis) if analysis else "Unable to analyze the data for this specific question."
