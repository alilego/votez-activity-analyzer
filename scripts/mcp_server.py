"""
MCP-style tool server for votez-activity-analyzer.

Implements the tool contract defined in mcp-tools.md.

Design principles (from the contract):
- The LLM/agent may NOT write to storage directly.
- All mutations go through validated tool calls here.
- Every tool returns {"ok": true, ...} or {"ok": false, "error": {...}}.
- store_intervention_analysis is idempotent and auditable (overwrites with log).

Usage:
    from mcp_server import MCPServer
    server = MCPServer(db_path=Path("state/state.sqlite"), run_id="run_xyz")

    result = server.call("get_run_config", {})
    result = server.call("get_intervention", {"intervention_id": "iv:abc:5"})
    result = server.call("retrieve_context", {"intervention_id": "iv:abc:5"})
    result = server.call("store_intervention_analysis", {
        "intervention_id": "iv:abc:5",
        "constructiveness_label": "constructive",
        "topics": ["proces legislativ"],
        "confidence": 0.85,
        "evidence_chunk_ids": ["ch:8846:3", "ch:8846:5"],
    })
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent))

import rag_store

# ---------------------------------------------------------------------------
# Config constants (matches get_run_config output and classification-rubric.md)
# ---------------------------------------------------------------------------

CONTRACT_VERSION = "v0"
MAX_TOPICS_PER_INTERVENTION = 5
MAX_TOPIC_LENGTH = 64
CONSTRUCTIVENESS_LABELS = {"constructive", "neutral", "non_constructive"}
RAG_TOP_K = rag_store.DEFAULT_TOP_K
RAG_MIN_SCORE = 0.0
ANALYSIS_SOURCE = "llm_agent_v1"
ANALYSIS_VERSION = "llm_v1"


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _ok(**kwargs) -> dict:
    return {"ok": True, **kwargs}


def _err(code: str, message: str, **details) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# MCPServer
# ---------------------------------------------------------------------------

class MCPServer:
    """
    Local MCP tool server.

    All tool calls are dispatched through self.call(tool_name, input_dict).
    The server holds a single SQLite connection for the lifetime of a run.
    """

    def __init__(self, db_path: Path, run_id: str) -> None:
        self._db_path = db_path
        self._run_id = run_id
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

        # Registry of all tools.
        self._tools: dict[str, Any] = {
            # Read tools
            "get_run_config": self._get_run_config,
            "get_session": self._get_session,
            "get_intervention": self._get_intervention,
            "get_member": self._get_member,
            # RAG tools
            "retrieve_context": self._retrieve_context,
            "get_chunk": self._get_chunk,
            # Write tools
            "store_intervention_analysis": self._store_intervention_analysis,
            "append_unmatched_speaker": self._append_unmatched_speaker,
            "write_run_summary": self._write_run_summary,
        }

    def call(self, tool_name: str, params: dict) -> dict:
        """Dispatch a tool call by name. Always returns a JSON-serialisable dict."""
        fn = self._tools.get(tool_name)
        if fn is None:
            return _err("UNSUPPORTED", f"Unknown tool: {tool_name}")
        try:
            return fn(params)
        except Exception as exc:  # noqa: BLE001
            return _err("INTERNAL_ERROR", str(exc))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -----------------------------------------------------------------------
    # 2) Read tools
    # -----------------------------------------------------------------------

    def _get_run_config(self, _params: dict) -> dict:
        return _ok(
            config={
                "contract_version": CONTRACT_VERSION,
                "max_topics_per_intervention": MAX_TOPICS_PER_INTERVENTION,
                "max_topic_length": MAX_TOPIC_LENGTH,
                "constructiveness_labels": sorted(CONSTRUCTIVENESS_LABELS),
                "rag": {
                    "top_k": RAG_TOP_K,
                    "min_score": RAG_MIN_SCORE,
                },
            }
        )

    def _get_session(self, params: dict) -> dict:
        session_id = params.get("session_id", "")
        if not session_id:
            return _err("VALIDATION_ERROR", "session_id is required")

        # Resolve stenogram_path from session_topics or session_chunks table.
        row = self._conn.execute(
            "SELECT stenogram_path FROM session_topics WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            row = self._conn.execute(
                "SELECT stenogram_path FROM session_chunks WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return _err("NOT_FOUND", f"Session not found: {session_id}")

        stenogram_path = Path(row["stenogram_path"])
        try:
            data = json.loads(stenogram_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _err("INTERNAL_ERROR", f"Could not read stenogram: {exc}")

        return _ok(
            session={
                "session_id": session_id,
                "session_date": str(data.get("stenograma_date", "")),
                "source_url": str(data.get("source_url", "")),
                "initial_notes": str(data.get("initial_notes", "")),
            }
        )

    def _get_intervention(self, params: dict) -> dict:
        intervention_id = params.get("intervention_id", "")
        if not intervention_id:
            return _err("VALIDATION_ERROR", "intervention_id is required")

        row = self._conn.execute(
            """
            SELECT intervention_id, session_id, session_date,
                   member_id, raw_speaker, normalized_speaker, text, speech_index
            FROM interventions_raw
            WHERE intervention_id = ?
            """,
            (intervention_id,),
        ).fetchone()
        if row is None:
            return _err("NOT_FOUND", f"Intervention not found: {intervention_id}")

        return _ok(
            intervention={
                "intervention_id": row["intervention_id"],
                "session_id": row["session_id"],
                "session_date": row["session_date"] or "",
                "member_id": row["member_id"],
                "raw_speaker": row["raw_speaker"],
                "normalized_speaker": row["normalized_speaker"],
                "text": row["text"],
                "speech_index": row["speech_index"],
            }
        )

    def _get_member(self, params: dict) -> dict:
        member_id = params.get("member_id", "")
        if not member_id:
            return _err("VALIDATION_ERROR", "member_id is required")

        row = self._conn.execute(
            "SELECT member_id, name, chamber, party_id FROM members WHERE member_id = ?",
            (member_id,),
        ).fetchone()
        if row is None:
            return _err("NOT_FOUND", f"Member not found: {member_id}")

        return _ok(
            member={
                "member_id": row["member_id"],
                "name": row["name"],
                "chamber": row["chamber"],
                "party_id": row["party_id"],
                "party_name": row["party_id"],  # v0: party_name == party_id slug
            }
        )

    # -----------------------------------------------------------------------
    # 3) RAG tools
    # -----------------------------------------------------------------------

    def _retrieve_context(self, params: dict) -> dict:
        intervention_id = params.get("intervention_id", "")
        if not intervention_id:
            return _err("VALIDATION_ERROR", "intervention_id is required")

        top_k = params.get("top_k", RAG_TOP_K)
        if not isinstance(top_k, int) or not (1 <= top_k <= 50):
            return _err("VALIDATION_ERROR", "top_k must be an integer between 1 and 50")

        # Fetch intervention to get session_id, speech_index, text.
        iv_result = self._get_intervention({"intervention_id": intervention_id})
        if not iv_result["ok"]:
            return iv_result
        iv = iv_result["intervention"]

        chunks = rag_store.retrieve_chunks(
            session_id=iv["session_id"],
            intervention_text=iv["text"],
            intervention_speech_index=iv["speech_index"],
            top_k=top_k,
        )

        # Map internal chunk_type to contract type names.
        type_map = {
            "session_notes": "session_notes",
            "speech": "debate_context",
        }

        # Sort: pinned chunks first (score=1.0), then by descending score,
        # tie-break by chunk_id ascending (determinism requirement).
        def _sort_key(c: rag_store.RetrievedChunk):
            pinned = 0 if c.reason in ("session_notes", "neighbor") else 1
            return (pinned, -c.score, c.chunk_id)

        sorted_chunks = sorted(chunks, key=_sort_key)

        return _ok(
            context=[
                {
                    "chunk_id": c.chunk_id,
                    "session_id": c.session_id,
                    "type": type_map.get(c.chunk_type, c.chunk_type),
                    "score": round(c.score, 6),
                    "text": c.text,
                }
                for c in sorted_chunks
            ]
        )

    def _get_chunk(self, params: dict) -> dict:
        chunk_id = params.get("chunk_id", "")
        if not chunk_id:
            return _err("VALIDATION_ERROR", "chunk_id is required")

        row = self._conn.execute(
            "SELECT chunk_id, session_id, chunk_type, text FROM session_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return _err("NOT_FOUND", f"Chunk not found: {chunk_id}")

        type_map = {"session_notes": "session_notes", "speech": "debate_context"}
        return _ok(
            chunk={
                "chunk_id": row["chunk_id"],
                "session_id": row["session_id"],
                "type": type_map.get(row["chunk_type"], row["chunk_type"]),
                "text": row["text"],
            }
        )

    # -----------------------------------------------------------------------
    # 4) Write tools
    # -----------------------------------------------------------------------

    def _store_intervention_analysis(self, params: dict) -> dict:
        intervention_id = params.get("intervention_id", "")
        label = params.get("constructiveness_label", "")
        topics_raw = params.get("topics", [])
        confidence = params.get("confidence")
        evidence_ids_raw = params.get("evidence_chunk_ids", [])

        # --- Validate: required fields ---
        if not intervention_id:
            return _err("VALIDATION_ERROR", "intervention_id is required")
        if label not in CONSTRUCTIVENESS_LABELS:
            return _err(
                "VALIDATION_ERROR",
                f"constructiveness_label must be one of: {sorted(CONSTRUCTIVENESS_LABELS)}",
                received=label,
            )

        # --- Validate: topics ---
        if not isinstance(topics_raw, list):
            return _err("VALIDATION_ERROR", "topics must be a list")
        topics: list[str] = []
        seen_topics: set[str] = set()
        for t in topics_raw:
            if not isinstance(t, str):
                return _err("VALIDATION_ERROR", "Each topic must be a string")
            t = t.strip()
            if not t:
                return _err("VALIDATION_ERROR", "Topics must be non-empty after trimming")
            if len(t) > MAX_TOPIC_LENGTH:
                return _err(
                    "VALIDATION_ERROR",
                    f"Topic exceeds max length {MAX_TOPIC_LENGTH}: {t!r}",
                )
            if t not in seen_topics:
                topics.append(t)
                seen_topics.add(t)
        if len(topics) > MAX_TOPICS_PER_INTERVENTION:
            return _err(
                "VALIDATION_ERROR",
                f"Too many topics: {len(topics)} > {MAX_TOPICS_PER_INTERVENTION}",
            )

        # --- Validate: confidence ---
        if confidence is None:
            return _err("VALIDATION_ERROR", "confidence is required")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return _err("VALIDATION_ERROR", "confidence must be a number")
        if not (0.0 <= confidence <= 1.0):
            return _err("VALIDATION_ERROR", "confidence must be between 0.0 and 1.0")

        # --- Validate: intervention exists and get session_id ---
        iv_row = self._conn.execute(
            "SELECT session_id FROM interventions_raw WHERE intervention_id = ?",
            (intervention_id,),
        ).fetchone()
        if iv_row is None:
            return _err("NOT_FOUND", f"Intervention not found: {intervention_id}")
        session_id = iv_row["session_id"]

        # --- Validate: evidence_chunk_ids ---
        if not isinstance(evidence_ids_raw, list):
            return _err("VALIDATION_ERROR", "evidence_chunk_ids must be a list")
        evidence_ids: list[str] = list(dict.fromkeys(evidence_ids_raw))  # deduplicate, preserve order
        for cid in evidence_ids:
            if not isinstance(cid, str) or not cid:
                return _err("VALIDATION_ERROR", "Each evidence_chunk_id must be a non-empty string")
            chunk_row = self._conn.execute(
                "SELECT session_id FROM session_chunks WHERE chunk_id = ?", (cid,)
            ).fetchone()
            if chunk_row is None:
                return _err(
                    "VALIDATION_ERROR",
                    f"evidence_chunk_id does not exist: {cid}",
                )
            if chunk_row["session_id"] != session_id:
                return _err(
                    "VALIDATION_ERROR",
                    f"evidence_chunk_id {cid!r} belongs to session {chunk_row['session_id']!r}, "
                    f"expected {session_id!r}",
                )

        # --- Idempotency: check if identical payload already stored ---
        existing = self._conn.execute(
            """
            SELECT relevance_label, topics_json, confidence, evidence_chunk_ids_json
            FROM intervention_analysis
            WHERE intervention_id = ?
            """,
            (intervention_id,),
        ).fetchone()

        new_topics_json = json.dumps(topics, ensure_ascii=True)
        new_evidence_json = json.dumps(evidence_ids, ensure_ascii=True)

        if existing is not None:
            same = (
                existing["relevance_label"] == label
                and existing["topics_json"] == new_topics_json
                and abs((existing["confidence"] or 0.0) - confidence) < 1e-9
                and existing["evidence_chunk_ids_json"] == new_evidence_json
            )
            if same:
                # Exact match — idempotent no-op.
                return _ok(
                    stored={
                        "intervention_id": intervention_id,
                        "constructiveness_label": label,
                        "topics": topics,
                        "confidence": confidence,
                        "evidence_chunk_ids": evidence_ids,
                    }
                )

        # --- Write (overwrite with audit) ---
        now = _utc_now()
        self._conn.execute(
            """
            INSERT INTO intervention_analysis (
                intervention_id, run_id, relevance_label, relevance_source,
                topics_json, confidence, evidence_chunk_ids_json,
                analysis_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(intervention_id) DO UPDATE SET
                run_id                 = excluded.run_id,
                relevance_label        = excluded.relevance_label,
                relevance_source       = excluded.relevance_source,
                topics_json            = excluded.topics_json,
                confidence             = excluded.confidence,
                evidence_chunk_ids_json = excluded.evidence_chunk_ids_json,
                analysis_version       = excluded.analysis_version,
                updated_at             = excluded.updated_at
            """,
            (
                intervention_id,
                self._run_id,
                label,
                ANALYSIS_SOURCE,
                new_topics_json,
                confidence,
                new_evidence_json,
                ANALYSIS_VERSION,
                now,
            ),
        )
        self._conn.commit()

        return _ok(
            stored={
                "intervention_id": intervention_id,
                "constructiveness_label": label,
                "topics": topics,
                "confidence": confidence,
                "evidence_chunk_ids": evidence_ids,
            }
        )

    def _append_unmatched_speaker(self, params: dict) -> dict:
        session_id = params.get("session_id", "")
        raw_speaker = params.get("raw_speaker", "")
        normalized_speaker = params.get("normalized_speaker", "")

        if not session_id:
            return _err("VALIDATION_ERROR", "session_id is required")
        if not raw_speaker:
            return _err("VALIDATION_ERROR", "raw_speaker is required")
        if not normalized_speaker:
            return _err("VALIDATION_ERROR", "normalized_speaker is required")

        # Resolve a stenogram_path for this session (best-effort).
        row = self._conn.execute(
            "SELECT stenogram_path FROM session_topics WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        stenogram_path = row["stenogram_path"] if row else ""

        self._conn.execute(
            """
            INSERT INTO unmatched_speakers
                (run_id, session_id, stenogram_path, raw_speaker, normalized_speaker, occurrences)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(run_id, session_id, normalized_speaker) DO UPDATE SET
                raw_speaker     = excluded.raw_speaker,
                occurrences     = unmatched_speakers.occurrences + 1,
                updated_at      = CURRENT_TIMESTAMP
            """,
            (self._run_id, session_id, stenogram_path, raw_speaker, normalized_speaker),
        )
        self._conn.commit()
        return _ok(stored=True)

    def _write_run_summary(self, params: dict) -> dict:
        run_id = params.get("run_id", "")
        started_at = params.get("started_at", "")
        finished_at = params.get("finished_at", "")
        stats = params.get("stats", {})

        if not run_id:
            return _err("VALIDATION_ERROR", "run_id is required")
        if not isinstance(stats, dict):
            return _err("VALIDATION_ERROR", "stats must be an object")

        sessions_processed = int(stats.get("sessions_processed", 0))
        interventions_total = int(stats.get("interventions_total", 0))
        interventions_classified = int(stats.get("interventions_classified", 0))
        unmatched_speakers = int(stats.get("unmatched_speakers", 0))

        self._conn.execute(
            """
            UPDATE runs SET
                started_at               = COALESCE(?, started_at),
                finished_at              = ?,
                status                   = 'completed',
                sessions_processed       = ?,
                interventions_total      = ?,
                interventions_classified = ?,
                unmatched_speakers       = ?
            WHERE run_id = ?
            """,
            (
                started_at or None,
                finished_at or _utc_now(),
                sessions_processed,
                interventions_total,
                interventions_classified,
                unmatched_speakers,
                run_id,
            ),
        )
        self._conn.commit()
        return _ok(stored=True)
