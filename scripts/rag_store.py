"""
RAG vector store for votez-activity-analyzer.

Responsibilities:
- Embed session chunks using a local multilingual sentence-transformers model.
- Persist one FAISS index per session under state/vectorstore/.
- Retrieve the top-k most relevant chunks for a given intervention, using a
  hybrid strategy:
    1. Always include the session_notes chunk (session framing).
    2. Always include up to NEIGHBOR_COUNT speech chunks immediately before
       and after the intervention's speech index (local debate context).
    3. Fill remaining slots with similarity-retrieved chunks (semantic context).

The embedding model is loaded once and reused across calls within a process.
The FAISS index for each session is rebuilt if the session has been reprocessed
(detected via a hash of the chunk texts stored in the metadata file).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Suppress tokenizers parallelism warning from transformers
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Lazy-loaded globals — populated on first call to avoid import-time cost.
_MODEL = None
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

VECTORSTORE_DIR = Path("state/vectorstore")

# Number of immediate-neighbor speech chunks to always include on each side.
# 3 before + 3 after = 6 neighbor slots, leaving room for similarity fill.
NEIGHBOR_COUNT = 3

# Default number of chunks to return per retrieval call.
# 1 session_notes + 6 neighbors + 3 similarity = 10 total.
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class RetrievedChunk(NamedTuple):
    chunk_id: str
    session_id: str
    chunk_type: str
    source_speech_index: int | None
    text: str
    score: float
    reason: str  # "session_notes" | "neighbor" | "similarity"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, dim)."""
    model = _get_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _index_path(session_id: str) -> Path:
    return VECTORSTORE_DIR / f"{session_id}.index"


def _meta_path(session_id: str) -> Path:
    return VECTORSTORE_DIR / f"{session_id}.meta.json"


def _chunks_hash(chunks: list[dict]) -> str:
    """Deterministic hash of chunk texts — used to detect stale indexes."""
    combined = "".join(c["chunk_id"] + c["text"] for c in chunks)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _load_meta(session_id: str) -> dict | None:
    p = _meta_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(session_id: str, meta: dict) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    _meta_path(session_id).write_text(
        json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

def build_session_index(session_id: str, chunks: list[dict], force: bool = False) -> None:
    """
    Embed chunks and write a FAISS index + metadata for the session.

    Skipped if the index already exists and the chunk content has not changed
    (unless force=True).

    chunks: list of dicts with keys chunk_id, session_id, chunk_type,
            source_speech_index, text.
    """
    import faiss

    if not chunks:
        return

    current_hash = _chunks_hash(chunks)
    if not force:
        meta = _load_meta(session_id)
        if meta and meta.get("chunks_hash") == current_hash and _index_path(session_id).exists():
            return  # Index is up to date.

    texts = [c["text"] for c in chunks]
    embeddings = _embed(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalised vecs = cosine similarity
    index.add(embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_index_path(session_id)))

    meta = {
        "session_id": session_id,
        "model": _MODEL_NAME,
        "chunks_hash": current_hash,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "session_id": c["session_id"],
                "chunk_type": c["chunk_type"],
                "source_speech_index": c.get("source_speech_index"),
                "text": c["text"],
            }
            for c in chunks
        ],
    }
    _save_meta(session_id, meta)


def _load_index(session_id: str):
    import faiss
    p = _index_path(session_id)
    if not p.exists():
        raise FileNotFoundError(f"No FAISS index for session {session_id}. Run build_session_index first.")
    return faiss.read_index(str(p))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(
    session_id: str,
    intervention_text: str,
    intervention_speech_index: int | None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Return up to top_k chunks for the given intervention.

    Strategy (in priority order):
      1. session_notes chunk — always included (provides session framing).
      2. Neighboring speech chunks — NEIGHBOR_COUNT before and after the
         intervention's speech index (provides local debate context).
      3. Similarity-retrieved chunks — fill remaining slots semantically.

    All chunks are from the same session (v0 rule).
    Results are deduplicated and ordered: session_notes first, then neighbors
    in order, then similarity results by descending score.
    """
    meta = _load_meta(session_id)
    if not meta or not meta.get("chunks"):
        return []

    chunks_meta: list[dict] = meta["chunks"]
    chunk_by_id = {c["chunk_id"]: c for c in chunks_meta}

    selected_ids: list[str] = []
    reasons: dict[str, str] = {}

    # 1. session_notes — always first.
    notes_chunks = [c for c in chunks_meta if c["chunk_type"] == "session_notes"]
    for c in notes_chunks:
        if c["chunk_id"] not in reasons:
            selected_ids.append(c["chunk_id"])
            reasons[c["chunk_id"]] = "session_notes"

    # 2. Neighbors of the intervention's speech index.
    if intervention_speech_index is not None:
        speech_chunks = [
            c for c in chunks_meta
            if c["chunk_type"] == "speech" and c["source_speech_index"] is not None
        ]
        # Sort by source_speech_index to make neighbor search reliable.
        speech_chunks_sorted = sorted(speech_chunks, key=lambda c: c["source_speech_index"])
        speech_indices = [c["source_speech_index"] for c in speech_chunks_sorted]

        # Find position of the intervention's speech in the sorted list.
        # Use closest index if exact match not found (intervention text may be
        # too short to have been indexed as a chunk).
        try:
            pos = speech_indices.index(intervention_speech_index)
        except ValueError:
            # Find nearest indexed speech index.
            if speech_indices:
                diffs = [abs(i - intervention_speech_index) for i in speech_indices]
                pos = diffs.index(min(diffs))
            else:
                pos = -1

        if pos >= 0:
            start = max(0, pos - NEIGHBOR_COUNT)
            end = min(len(speech_chunks_sorted), pos + NEIGHBOR_COUNT + 1)
            for c in speech_chunks_sorted[start:end]:
                if c["chunk_id"] not in reasons:
                    selected_ids.append(c["chunk_id"])
                    reasons[c["chunk_id"]] = "neighbor"

    # 3. Similarity search to fill remaining slots.
    remaining = top_k - len(selected_ids)
    similarity_scores: dict[str, float] = {}

    if remaining > 0 and chunks_meta:
        try:
            index = _load_index(session_id)
            query_vec = _embed([intervention_text])
            # Retrieve more candidates than needed to filter out already-selected.
            k = min(len(chunks_meta), remaining + len(selected_ids) + 5)
            scores_arr, idx_arr = index.search(query_vec, k)
            for score, idx in zip(scores_arr[0], idx_arr[0]):
                if idx < 0 or idx >= len(chunks_meta):
                    continue
                cid = chunks_meta[idx]["chunk_id"]
                if cid not in reasons:
                    similarity_scores[cid] = float(score)
                    reasons[cid] = "similarity"
                    selected_ids.append(cid)
                    if len(selected_ids) >= top_k:
                        break
        except (FileNotFoundError, Exception):
            pass  # Degrade gracefully if index unavailable.

    # Build result list in priority order.
    result: list[RetrievedChunk] = []
    seen: set[str] = set()
    for cid in selected_ids:
        if cid in seen:
            continue
        seen.add(cid)
        c = chunk_by_id.get(cid)
        if c is None:
            continue
        result.append(
            RetrievedChunk(
                chunk_id=cid,
                session_id=session_id,
                chunk_type=c["chunk_type"],
                source_speech_index=c.get("source_speech_index"),
                text=c["text"],
                score=similarity_scores.get(cid, 1.0),  # pinned chunks get score 1.0
                reason=reasons[cid],
            )
        )

    return result[:top_k]
