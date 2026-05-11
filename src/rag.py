"""
RAG (Retrieval-Augmented Generation) module for the analysis-gen service.

Provides a ``RAGStore`` that indexes pandas DataFrames into ChromaDB and
retrieves the most relevant row-chunks for a given user query.  The
retrieved rows are then injected into the LLM context alongside the
existing schema summary, allowing the model to answer row-level questions
with precision instead of relying solely on aggregated statistics.

Design decisions
~~~~~~~~~~~~~~~~
* **ChromaDB PersistentClient** — survives restarts, no external DB needed.
* **OpenAI text-embedding-3-small** — cheap ($0.02/1M tokens), fast, and
  produces high-quality embeddings for tabular text.
* **Adaptive chunk sizing** — rows per chunk is calculated dynamically
  based on the file's width so each chunk stays under the 8 192-token
  embedding limit.
* **top_k = 30** default — enough for the LLM to ground its answer without
  blowing up the context window.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports — chromadb is only needed when RAG is enabled.  If the
# package is missing the module still loads but ``RAGStore`` will log a
# warning and become a no-op so the rest of the service keeps working.
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.utils import embedding_functions

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
_DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "15"))

# Maximum number of rows to index.  Prevents excessive embedding API calls
# for very large files (100k+ rows).  Rows beyond this limit are simply
# not indexed — the schema summary still covers them.
_MAX_INDEXABLE_ROWS = int(os.getenv("RAG_MAX_INDEXABLE_ROWS", "20000"))

# text-embedding-3-small accepts up to 8 192 tokens.
# CSV/tabular data tokenizes at ~2–3 chars per token (not 4), so we use
# a conservative ceiling to guarantee we never exceed the limit.
_MAX_EMBEDDING_CHARS = 15_000  # ~5 000–7 500 tokens — safe margin

# Maximum columns to include in RAG chunks.  Wide files (700+ cols) are
# trimmed so that even a single-row chunk fits within the token limit.
_MAX_RAG_COLUMNS = int(os.getenv("RAG_MAX_COLUMNS", "80"))

# Maximum characters in the combined retrieved context injected into the
# LLM prompt.  Keeps RAG context from dominating the prompt on large files.
_MAX_RETRIEVED_CHARS = int(os.getenv("RAG_MAX_RETRIEVED_CHARS", "30000"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_key_to_collection_name(file_key: str) -> str:
    """Convert an arbitrary file key (URL, path, …) to a valid ChromaDB
    collection name.  ChromaDB collection names must be 3-63 chars, start
    and end with an alphanumeric, and contain only ``[a-zA-Z0-9_-]``.
    We use a truncated SHA-256 hash prefixed with ``rag_``.
    """
    digest = hashlib.sha256(file_key.encode("utf-8")).hexdigest()[:48]
    return f"rag_{digest}"


def _select_columns_for_rag(df: pd.DataFrame, max_columns: int = _MAX_RAG_COLUMNS) -> pd.DataFrame:
    """Return a column-limited copy of *df* suitable for embedding.

    For files with more columns than *max_columns*, we keep:
    1. **Identity columns first** — grouping / filtering columns (Player Name,
       Date, RESULTADO FINAL, ADVERSÁRIO, etc.) are always included because
       they carry the most semantic meaning for search queries.
    2. Remaining categorical/text columns (up to half the remaining budget).
    3. Fill the rest with numeric columns.

    This means a 700-column file is narrowed to ~80 columns, keeping the
    CSV text small enough for embedding while preserving search quality.
    """
    if len(df.columns) <= max_columns:
        return df

    # Import identity detection from context module
    try:
        from context import _detect_column_roles
        roles = _detect_column_roles(df)
        identity_cols = [c for c in roles["identity"] if c in df.columns]
    except Exception:
        identity_cols = []

    selected: list[str] = []
    seen: set[str] = set()

    # 1. Force identity columns
    for col in identity_cols:
        if col not in seen:
            selected.append(col)
            seen.add(col)

    remaining_budget = max_columns - len(selected)

    # Split remaining columns into categorical and numeric
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in seen]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in seen]

    # Allocate: up to half the remaining budget for categorical, rest for numeric
    cat_budget = min(len(cat_cols), remaining_budget // 2)
    num_budget = remaining_budget - cat_budget

    for col in cat_cols[:cat_budget]:
        selected.append(col)
    for col in num_cols[:num_budget]:
        selected.append(col)

    print(f"[RAG] Wide file ({len(df.columns)} cols) -> trimmed to {len(selected)} cols "
          f"for embedding (identity={len(identity_cols)}, cat={cat_budget}, num={num_budget})")
    return df[selected[:max_columns]]


def _dataframe_chunk_to_text(df_chunk: pd.DataFrame) -> str:
    """Serialize a DataFrame chunk to text for embedding.

    Produces two sections:
    1. A short natural-language **summary line** listing unique values in
       key columns (dates, names, categories).  This dramatically improves
       embedding similarity for exact-value queries like "1/20/2024".
    2. The raw CSV data so the LLM can cite specific numbers.

    For numeric identity columns (e.g. ``Player Name`` with int IDs like
    1, 2, 3), the summary labels them as "Player 1, Player 2, …" so that
    semantic search can match queries like "player 1 and player 2".
    """
    # Keywords that signal a column is an identity/entity column
    _IDENTITY_KWS = {'name', 'player', 'jogador', 'atleta', 'team', 'time',
                     'equipe', 'position', 'date', 'period', 'evento',
                     'adversário', 'adversario', 'campeonato'}

    # --- Build summary of key column values for search quality -----------
    summary_parts: list[str] = []
    for col in df_chunk.columns:
        col_lower = str(col).lower().strip()
        col_tokens = set(re.findall(r'\b\w{3,}\b', col_lower))
        is_key = bool(col_tokens & _IDENTITY_KWS) or any(
            kw in col_lower for kw in ('date', 'name', 'player', 'period')
        )
        if is_key:
            is_numeric = pd.api.types.is_numeric_dtype(df_chunk[col])
            uniques = df_chunk[col].dropna().unique()
            if len(uniques) > 0 and len(uniques) <= 20:
                if is_numeric:
                    # Label numeric IDs with the column concept so
                    # embeddings capture "Player 1" not just "1".
                    # e.g. "Player Name" → prefix "Player"
                    prefix_word = col_lower.split()[0].title() if ' ' in col_lower else col_lower.title()
                    labeled = [f"{prefix_word} {int(v)}" for v in sorted(uniques)]
                    summary_parts.append(f"{col}: {', '.join(labeled[:10])}")
                else:
                    str_uniques = [str(v) for v in uniques[:10]]
                    summary_parts.append(f"{col}: {', '.join(str_uniques)}")

    summary = ""
    if summary_parts:
        summary = "Key values in this chunk — " + " | ".join(summary_parts) + "\n"

    # --- CSV body --------------------------------------------------------
    buf = io.StringIO()
    df_chunk.to_csv(buf, index=False)

    return summary + buf.getvalue()


def _estimate_rows_per_chunk(df: pd.DataFrame, max_chars: int = _MAX_EMBEDDING_CHARS) -> int:
    """Estimate how many rows fit in a single embedding chunk without
    exceeding *max_chars* characters.

    Samples a few rows, measures the average CSV-serialized size, then
    calculates how many rows fit (including the header).  Always returns
    at least 1.
    """
    if df.empty:
        return 50  # arbitrary default for empty frames

    # Sample up to 10 rows to get a representative byte estimate
    sample_size = min(10, len(df))
    sample_df = df.head(sample_size)
    sample_text = _dataframe_chunk_to_text(sample_df)

    # header_size = first line + newline
    first_newline = sample_text.find("\n")
    header_size = first_newline + 1 if first_newline >= 0 else len(sample_text)

    # Average bytes per data row (total - header) / sample_size
    data_size = len(sample_text) - header_size
    avg_row_size = max(data_size / max(sample_size, 1), 1)

    # How many rows fit in (max_chars - header)?
    available = max_chars - header_size
    rows_per_chunk = max(1, int(available / avg_row_size))

    return rows_per_chunk


# ---------------------------------------------------------------------------
# RAGStore
# ---------------------------------------------------------------------------

class RAGStore:
    """Manages ChromaDB collections for per-file DataFrame indexing and
    retrieval.

    Usage::

        store = RAGStore()
        store.index_dataframe("https://…/file.csv", df)
        rows_text = store.retrieve("https://…/file.csv", "Who scored the most?")
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        chunk_size: int | None = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._fixed_chunk_size = chunk_size  # None = adaptive (recommended)
        self.top_k = top_k
        self._enabled = False

        if not _CHROMA_AVAILABLE:
            print("[RAG] WARNING: chromadb is not installed — RAG is disabled.")
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[RAG] WARNING: OPENAI_API_KEY not set — RAG is disabled.")
            return

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=embedding_model,
            )
            self._enabled = True
            chunk_desc = f"fixed={chunk_size}" if chunk_size else "adaptive"
            print(f"[RAG] Initialized — persist_dir={persist_dir}, "
                  f"model={embedding_model}, chunk_size={chunk_desc}")
        except Exception as exc:
            print(f"[RAG] WARNING: Failed to initialize ChromaDB: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_indexed(self, file_key: str) -> bool:
        """Return ``True`` if *file_key* already has a ChromaDB collection
        with at least one document."""
        if not self._enabled:
            return False
        col_name = _file_key_to_collection_name(file_key)
        try:
            collection = self._client.get_collection(
                name=col_name,
                embedding_function=self._embedding_fn,
            )
            return collection.count() > 0
        except Exception:
            return False

    def index_dataframe(
        self,
        file_key: str,
        df: pd.DataFrame,
        *,
        force: bool = False,
    ) -> None:
        """Chunk *df* into groups of rows (sized to stay under the
        embedding token limit), embed each chunk, and store in a ChromaDB
        collection keyed by *file_key*.

        If the file is already indexed and *force* is ``False``, this is a
        no-op (fast path).
        """
        if not self._enabled:
            return

        if not force and self.is_indexed(file_key):
            print(f"[RAG] Already indexed: {file_key[:80]}")
            return

        col_name = _file_key_to_collection_name(file_key)

        # Delete existing collection if re-indexing
        try:
            self._client.delete_collection(name=col_name)
        except Exception:
            pass  # collection didn't exist — that's fine

        collection = self._client.get_or_create_collection(
            name=col_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Cap the number of rows to avoid excessive embedding costs
        indexable_df = df.head(_MAX_INDEXABLE_ROWS)
        # Limit columns for wide files so chunks fit the token budget
        indexable_df = _select_columns_for_rag(indexable_df)
        total_rows = len(indexable_df)
        if total_rows == 0:
            print(f"[RAG] Empty DataFrame — skipping indexing for {file_key[:80]}")
            return

        # Determine chunk size: fixed override or adaptive
        if self._fixed_chunk_size:
            rows_per_chunk = self._fixed_chunk_size
        else:
            rows_per_chunk = _estimate_rows_per_chunk(indexable_df)

        started_at = time.perf_counter()

        # Build chunks
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        chunk_idx = 0
        for start_row in range(0, total_rows, rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, total_rows)
            chunk_df = indexable_df.iloc[start_row:end_row]
            chunk_text = _dataframe_chunk_to_text(chunk_df)

            # Hard safety net: if the chunk still exceeds the limit
            # (e.g. very long cell values), truncate to fit.
            if len(chunk_text) > _MAX_EMBEDDING_CHARS:
                chunk_text = chunk_text[:_MAX_EMBEDDING_CHARS]

            documents.append(chunk_text)
            ids.append(f"chunk_{chunk_idx}")
            metadatas.append({
                "chunk_index": chunk_idx,
                "row_start": start_row,
                "row_end": end_row,
            })
            chunk_idx += 1

        # Add in batches — total tokens per API call must stay under
        # OpenAI's 300 000-token-per-request limit.  CSV/tabular data
        # tokenizes at ~2 chars/token, so we estimate conservatively and
        # keep each batch well under the ceiling.
        _MAX_BATCH_TOKENS = 100_000
        batch_docs: list[str] = []
        batch_ids: list[str] = []
        batch_metas: list[dict] = []
        batch_tokens = 0

        def _flush_batch():
            nonlocal batch_docs, batch_ids, batch_metas, batch_tokens
            if not batch_docs:
                return
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
            )
            batch_docs, batch_ids, batch_metas = [], [], []
            batch_tokens = 0

        for doc, doc_id, meta in zip(documents, ids, metadatas):
            est_tokens = len(doc) // 2  # conservative: ~2 chars/token for CSV
            # If adding this doc would exceed the limit, flush first
            if batch_tokens + est_tokens > _MAX_BATCH_TOKENS and batch_docs:
                _flush_batch()
            batch_docs.append(doc)
            batch_ids.append(doc_id)
            batch_metas.append(meta)
            batch_tokens += est_tokens

        _flush_batch()  # remaining items

        elapsed = time.perf_counter() - started_at
        print(
            f"[RAG] Indexed {file_key[:80]} — "
            f"{len(documents)} chunks ({rows_per_chunk} rows/chunk), "
            f"{total_rows} rows, {elapsed:.2f}s"
        )

    def retrieve(
        self,
        file_key: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """Embed *query*, search the file's collection, and return the
        top-K most relevant row-chunks as CSV text.

        Uses **hybrid retrieval**: extracts date patterns and specific
        values from the query, then filters chunks that contain those
        values before applying semantic ranking.  Falls back to pure
        semantic search if no specific values are detected.

        Returns an empty string if RAG is disabled or the file is not
        indexed — the caller can safely concatenate it to the prompt.
        """
        if not self._enabled:
            return ""

        k = top_k if top_k is not None else self.top_k
        col_name = _file_key_to_collection_name(file_key)

        try:
            collection = self._client.get_collection(
                name=col_name,
                embedding_function=self._embedding_fn,
            )
        except Exception:
            print(f"[RAG] Collection not found for {file_key[:80]} — skipping retrieval")
            return ""

        # Don't request more results than the collection contains
        available = collection.count()
        if available == 0:
            return ""

        # --- Hybrid search: extract specific values from the query --------
        # Dates like 1/20/2024, 2024-01-20, 20/01/2024
        date_patterns = re.findall(
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            query
        )

        # Entity+number references like "player 1", "jogador 2", "atleta 10"
        _ENTITY_PATTERN = re.compile(
            r'\b(player|jogador|atleta|team|time|equipe)\s+(\d+)\b',
            re.IGNORECASE,
        )
        entity_refs = _ENTITY_PATTERN.findall(query)

        started_at = time.perf_counter()
        results = None

        # Strategy 1: if query contains dates, try filtering first
        if date_patterns:
            for date_str in date_patterns:
                try:
                    filtered = collection.query(
                        query_texts=[query],
                        n_results=min(k, available),
                        where_document={"$contains": date_str},
                    )
                    if filtered and filtered.get("documents") and filtered["documents"][0]:
                        results = filtered
                        print(f"[RAG] Hybrid search: found {len(results['documents'][0])} chunks containing '{date_str}'")
                        break
                except Exception:
                    pass  # filter not supported or no matches — fall through

        # Strategy 1.5: entity+number filtering — e.g. "player 1", "player 2"
        # The chunk summary labels numeric IDs as "Player 1", "Player 2", etc.
        # so we search for those labeled strings to find the right chunks.
        if entity_refs and (results is None or not results.get("documents") or not results["documents"][0]):
            # Collect chunks that match ANY of the entity references
            all_entity_docs: list[str] = []
            seen_ids: set[str] = set()
            for entity_word, entity_num in entity_refs:
                # The chunk text uses title-case prefix: "Player 1", "Jogador 2"
                search_label = f"{entity_word.title()} {entity_num}"
                try:
                    filtered = collection.query(
                        query_texts=[query],
                        n_results=min(k, available),
                        where_document={"$contains": search_label},
                    )
                    if filtered and filtered.get("documents") and filtered["documents"][0]:
                        for doc_id, doc in zip(filtered["ids"][0], filtered["documents"][0]):
                            if doc_id not in seen_ids:
                                all_entity_docs.append(doc)
                                seen_ids.add(doc_id)
                        print(f"[RAG] Entity search: found chunks containing '{search_label}'")
                except Exception:
                    pass

            if all_entity_docs:
                # Build a synthetic results dict compatible with downstream processing
                results = {"documents": [all_entity_docs[:k]], "ids": [list(seen_ids)[:k]]}
                print(f"[RAG] Entity search total: {len(all_entity_docs)} unique chunks for {len(entity_refs)} entities")

        # Strategy 2: fallback to pure semantic search
        if results is None or not results.get("documents") or not results["documents"][0]:
            effective_k = min(k, available)
            results = collection.query(
                query_texts=[query],
                n_results=effective_k,
            )
            print(f"[RAG] Semantic search: {len(results['documents'][0]) if results.get('documents') else 0} chunks")

        elapsed = time.perf_counter() - started_at

        if not results or not results.get("documents"):
            return ""

        retrieved_docs = results["documents"][0]  # list of chunk texts
        print(
            f"[RAG] Retrieved {len(retrieved_docs)} chunks for "
            f"{file_key[:80]} in {elapsed:.2f}s"
        )

        # Combine chunks, removing duplicate CSV headers (only keep the
        # first header line since all chunks share the same columns).
        # Also enforce a hard character cap so RAG context never dominates
        # the LLM prompt.
        combined_lines: list[str] = []
        header_added = False
        total_chars = 0
        for doc in retrieved_docs:
            lines = doc.strip().split("\n")
            if not lines:
                continue
            if not header_added:
                combined_lines.extend(lines)
                total_chars += sum(len(l) + 1 for l in lines)
                header_added = True
            else:
                # Skip the header line (first line of each chunk)
                for line in lines[1:]:
                    if total_chars + len(line) + 1 > _MAX_RETRIEVED_CHARS:
                        break
                    combined_lines.append(line)
                    total_chars += len(line) + 1
                if total_chars >= _MAX_RETRIEVED_CHARS:
                    break

        result = "\n".join(combined_lines)
        print(f"[RAG] Retrieved context size: {len(result)} chars")
        return result

    def invalidate(self, file_key: str) -> None:
        """Delete the ChromaDB collection for *file_key*."""
        if not self._enabled:
            return
        col_name = _file_key_to_collection_name(file_key)
        try:
            self._client.delete_collection(name=col_name)
            print(f"[RAG] Invalidated collection for {file_key[:80]}")
        except Exception:
            pass  # collection didn't exist


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

rag_store = RAGStore()

