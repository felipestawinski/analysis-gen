"""
Caching and download helpers for the analysis-gen service.

Manages an in-memory LRU cache for raw CSV text and parsed DataFrames,
with TTL-based expiration and byte-budget eviction.
"""

from __future__ import annotations

import os
import time
import threading
from collections import OrderedDict
from typing import Optional

import httpx
import pandas as pd

from normalization import detect_file_type, dataframe_from_bytes

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

CSV_CACHE_TTL_SECONDS = int(os.getenv("CSV_CACHE_TTL_SECONDS", "1800"))
CSV_CACHE_MAX_BYTES = int(os.getenv("CSV_CACHE_MAX_BYTES", str(200 * 1024 * 1024)))
CSV_CACHE_MAX_ENTRIES = int(os.getenv("CSV_CACHE_MAX_ENTRIES", "128"))
MAX_FILES_PER_ANALYSIS_REQUEST = int(os.getenv("MAX_FILES_PER_ANALYSIS_REQUEST", "4"))
MAX_PARALLEL_DOWNLOADS = int(os.getenv("MAX_PARALLEL_DOWNLOADS", "4"))

# ---------------------------------------------------------------------------
# CSV text cache
# ---------------------------------------------------------------------------

csv_cache: OrderedDict[str, dict] = OrderedDict()
csv_cache_total_bytes = 0
csv_cache_lock = threading.RLock()


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
# DataFrame cache
# ---------------------------------------------------------------------------

df_cache: OrderedDict[str, dict] = OrderedDict()
df_cache_lock = threading.RLock()


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
