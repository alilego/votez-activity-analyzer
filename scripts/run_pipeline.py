#!/usr/bin/env python3
"""
Minimal orchestrator for incremental runs.

Flow:
1) Select new/changed stenograms
2) Execute analyzer command  (speaker resolution, chunking, RAG index, baseline labels)
3) Optionally run LLM agent  (LLM classification via MCP, when --analyzer-mode=llm)
4) Export frontend artifacts
5) Mark files as processed only if all steps succeed
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress urllib3's LibreSSL warning on macOS — the system Python ships with
# LibreSSL instead of OpenSSL; the warning is cosmetic and not actionable.
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

import shutil

from init_db import init_db
from mark_processed_stenograms import mark_candidates
from prompt_logger import GENERATED_PROMPTS_DIR
from select_stenograms import DEFAULT_DB_PATH, DEFAULT_INPUT_DIR, StenogramCandidate, select_candidates


def _default_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _insert_running_run(conn: sqlite3.Connection, run_id: str) -> None:
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO runs (
            run_id, started_at, status,
            sessions_processed, interventions_total,
            interventions_classified, unmatched_speakers
        )
        VALUES (?, ?, 'running', 0, 0, 0, 0)
        ON CONFLICT(run_id) DO UPDATE SET
            started_at = excluded.started_at,
            status = 'running'
        """,
        (run_id, now),
    )


def _finish_run(conn: sqlite3.Connection, run_id: str, status: str, sessions_processed: int) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?,
            finished_at = ?,
            sessions_processed = ?
        WHERE run_id = ?
        """,
        (status, _utc_now_iso(), sessions_processed, run_id),
    )


def _has_pending_llm_work(db_path: Path, model: str, reprocess_topics: bool) -> bool:
    """Return True if there are sessions or interventions still waiting for LLM processing."""
    model_source = f"llm_v1:{model}"
    skip_condition = "st.topics_source LIKE 'llm_v1:%'" if not reprocess_topics else f"st.topics_source = '{model_source}'"
    with sqlite3.connect(db_path) as conn:
        pending_topics = conn.execute(
            f"""
            SELECT COUNT(DISTINCT sc.session_id)
            FROM session_chunks sc
            WHERE NOT EXISTS (
                SELECT 1 FROM session_topics st
                WHERE st.session_id = sc.session_id
                  AND {skip_condition}
            )
            """
        ).fetchone()[0]
        pending_interventions = conn.execute(
            """
            SELECT COUNT(*)
            FROM interventions_raw iv
            WHERE NOT EXISTS (
                SELECT 1 FROM intervention_analysis ia
                WHERE ia.intervention_id = iv.intervention_id
                  AND ia.relevance_source = 'llm_agent_v1'
            )
            """
        ).fetchone()[0]
    return (pending_topics + pending_interventions) > 0


def _write_candidate_file(run_id: str, candidates: list[str]) -> Path:
    run_inputs_dir = Path("state/run_inputs")
    run_inputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_inputs_dir / f"{run_id}_stenograms.json"
    out_path.write_text(json.dumps({"run_id": run_id, "files": candidates}, indent=2), encoding="utf-8")
    return out_path


def _check_ollama_model(model: str) -> None:
    """Warn if the requested Ollama model doesn't exist, with setup instructions."""
    import urllib.request, urllib.error, json as _json
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            tags = _json.loads(resp.read())
        available = [m["name"] for m in tags.get("models", [])]
        # Normalise: ollama may return "llama3.1:8b-8k" or "llama3.1:8b-8k:latest"
        available_base = [n.split(":")[0] + ":" + n.split(":")[1] if n.count(":") >= 1 else n for n in available]
        if model not in available and model not in available_base:
            print(f"\n  WARNING: model '{model}' not found in Ollama.")
            if model == "llama3.1:8b-8k":
                print("  Run these commands once to create it:")
                print("    ollama pull llama3.1:8b")
                print("    ollama create llama3.1:8b-8k -f - << 'EOF'")
                print("    FROM llama3.1:8b")
                print("    PARAMETER num_ctx 8192")
                print("    EOF")
            else:
                print(f"  Run: ollama pull {model}")
            print()
    except urllib.error.URLError:
        pass  # Ollama not running yet — llm_session_topics.py will catch this.


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incremental analyzer pipeline.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Path to stenogram directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--run-id",
        default=_default_run_id(),
        help="Run identifier used in state metadata",
    )
    parser.add_argument(
        "--analyzer-cmd",
        default="",
        help=(
            "Analyzer command to execute. If omitted, defaults to "
            "'python scripts/analyze_interventions.py'. "
            "The command receives env vars: VOTEZ_RUN_ID and VOTEZ_STENOGRAM_LIST_PATH."
        ),
    )
    parser.add_argument(
        "--analyzer-mode",
        choices=["baseline", "llm"],
        default="baseline",
        help=(
            "baseline (default): keyword/RAG baseline only. "
            "llm: run baseline first, then classify via LLM agent."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider for llm mode (default: ollama).",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Model name for LLM mode (default: llama3.1:8b-8k for ollama, gpt-4o-mini for openai).",
    )
    parser.add_argument(
        "--llm-sessions-limit",
        type=int,
        default=0,
        help="Extract topics for at most N sessions in LLM mode (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--llm-speech-limit",
        type=int,
        default=0,
        help="Classify at most N interventions in LLM mode (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--reprocess-session-topics",
        action="store_true",
        help=(
            "In llm mode: re-extract session topics even for sessions already processed "
            "by any LLM. By default, sessions with any llm_v1:* topics_source are skipped."
        ),
    )
    parser.add_argument(
        "--stenogram",
        default="",
        help=(
            "Path to a single stenogram file to (re)process. "
            "Bypasses the file-selection step and forces the full LLM pipeline "
            "for that file only. Implies --reprocess-session-topics for that session."
        ),
    )
    parser.add_argument(
        "--build-prompts",
        action="store_true",
        help=(
            "Write LLM prompt files for ALL sessions to state/generated_prompts/ without "
            "calling any LLM.  The directory is wiped and fully refreshed on every run.  "
            "After reviewing the prompts, place model responses with matching names in "
            "state/external_prompts_output/ and re-run with --ingest-external-outputs."
        ),
    )
    parser.add_argument(
        "--ingest-external-outputs",
        action="store_true",
        help=(
            "Read external model responses from state/external_prompts_output/ (files "
            "named identically to the corresponding generated_prompts/ files) and store "
            "them to the DB, then export.  No LLM calls are made."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected files and exit.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip output export step (not recommended).",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    input_dir = Path(args.input_dir)
    repo_root = Path.cwd()
    run_id = args.run_id

    print(f"\n{'='*60}")
    print(f"Pipeline start  run_id={run_id}")
    print(f"  DB:        {db_path}")
    print(f"  Input dir: {input_dir}")
    print(f"  Mode:      {args.analyzer_mode}")
    if args.build_prompts:
        print(f"  [BUILD-PROMPTS] Writing prompt files only — no LLM calls.")
    elif args.ingest_external_outputs:
        print(f"  [INGEST-EXTERNAL] Reading responses from state/external_prompts_output/")
    if args.analyzer_mode == "llm" and not args.ingest_external_outputs:
        provider = args.llm_provider
        model_hint = args.llm_model or ("llama3.1:8b-8k" if provider == "ollama" else "gpt-4o-mini")
        print(f"  Provider:  {provider}  model={model_hint}")
        if args.build_prompts:
            print(f"  LLM steps: 3b) build topic prompts  3c) build intervention prompts")
        else:
            print(f"  LLM steps: 3b) session topics  3c) intervention classification")
        if provider == "ollama" and not args.build_prompts:
            _check_ollama_model(model_hint)
    print(f"{'='*60}\n")

    # Always bootstrap DB on pipeline start:
    # - creates file if missing
    # - creates schema if missing/uninitialized
    print("Step 1/4  Initializing database...")
    init_db(db_path)
    print(f"  DB ready: {db_path}")

    # --build-prompts: wipe state/generated_prompts/ so every run produces a
    # complete, fresh snapshot rather than mixing old and new prompt files.
    if args.build_prompts:
        if GENERATED_PROMPTS_DIR.exists():
            shutil.rmtree(GENERATED_PROMPTS_DIR)
            print(f"  Cleared {GENERATED_PROMPTS_DIR}/ (build-prompts refresh)")
        GENERATED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # --stenogram: bypass file selection, force a single file through the pipeline.
    forced_session_id: str | None = None
    if args.stenogram.strip():
        forced_path = str(Path(args.stenogram.strip()))
        print(f"\nStep 2/4  Forced stenogram: {forced_path}")
        # Resolve session_id for this stenogram from the DB.
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT DISTINCT session_id FROM session_chunks WHERE stenogram_path = ? LIMIT 1",
                (forced_path,),
            ).fetchone()
        if row is None:
            print(f"  ERROR: stenogram '{forced_path}' not found in DB. Run the baseline first.")
            return 1
        forced_session_id = row[0]
        print(f"  session_id={forced_session_id}")
        # Reset LLM analysis for this session so it is fully reprocessed.
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            n_ia = conn.execute(
                """DELETE FROM intervention_analysis
                   WHERE intervention_id IN (
                       SELECT intervention_id FROM interventions_raw WHERE session_id = ?
                   ) AND relevance_source = 'llm_agent_v1'""",
                (forced_session_id,),
            ).rowcount
            conn.execute(
                "UPDATE session_topics SET topics_source='keyword_baseline_v1', updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (forced_session_id,),
            )
            conn.commit()
        print(f"  Reset: {n_ia} intervention_analysis rows + session_topics for session {forced_session_id}")
        candidates = [StenogramCandidate(path=str(forced_path), content_hash="", file_mtime_ns=0, reason="forced")]
    else:
        print("\nStep 2/4  Selecting stenograms...")
        try:
            candidates = select_candidates(db_path=db_path, input_dir=input_dir, repo_root=repo_root)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

        if not candidates:
            print("  No new/changed stenograms found.")
            if args.analyzer_mode == "llm":
                model = args.llm_model.strip() or ("llama3.1:8b-8k" if args.llm_provider == "ollama" else "gpt-4o-mini")
                if _has_pending_llm_work(db_path, model, args.reprocess_session_topics):
                    print("  Pending LLM work detected — skipping baseline, proceeding to LLM steps.")
                else:
                    print("  No pending LLM work either. Nothing to do.")
                    return 0
            else:
                return 0

        print(f"  {len(candidates)} stenogram(s) selected:")
        for c in candidates:
            print(f"    {c.path}  [{c.reason}]")

    if args.dry_run:
        print("\nDry run: exiting without processing.")
        return 0

    candidate_paths = [c.path for c in candidates]
    candidate_file = _write_candidate_file(run_id, candidate_paths)

    env = os.environ.copy()
    env["VOTEZ_RUN_ID"] = run_id
    env["VOTEZ_STENOGRAM_LIST_PATH"] = str(candidate_file)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        _insert_running_run(conn, run_id)
        conn.commit()

    # Only run baseline when there are new stenograms — skip when resuming LLM-only work.
    if candidates:
        if args.analyzer_cmd.strip():
            proc = subprocess.run(args.analyzer_cmd, shell=True, env=env)
        else:
            default_cmd = [sys.executable, "scripts/analyze_interventions.py"]
            proc = subprocess.run(default_cmd, env=env)

        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            if proc.returncode != 0:
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
                print(f"\nBaseline analyzer failed (exit code {proc.returncode}). Nothing was marked processed.")
                return proc.returncode

            # Mark stenograms as processed right after baseline succeeds.
            # This ensures that if LLM steps are interrupted, re-running won't redo the baseline.
            # LLM steps have their own per-item skip logic (topics_source, relevance_source).
            marked = mark_candidates(conn, candidates, run_id)
            conn.commit()
            print(f"  Stenograms marked as processed: {marked}")

    # Optional LLM passes (run outside conn context to avoid lock contention).
    if args.analyzer_mode == "llm":

        # --ingest-external-outputs: skip LLM calls, read from external_prompts_output/.
        if args.ingest_external_outputs:
            print(f"\nStep 3b/4  Ingesting external session-topic outputs...")
            topics_cmd = [
                sys.executable, "scripts/llm_session_topics.py",
                "--provider", args.llm_provider,
                "--ingest-external-outputs",
            ]
            if args.llm_model.strip():
                topics_cmd += ["--model", args.llm_model.strip()]
            topics_proc = subprocess.run(topics_cmd, env=env)
            if topics_proc.returncode != 0:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _finish_run(conn, run_id, "failed", 0)
                    conn.commit()
                print(f"\nIngestion of session topics failed (exit code {topics_proc.returncode}).")
                return topics_proc.returncode

            print(f"\nStep 3c/4  Ingesting external intervention outputs...")
            llm_cmd = [
                sys.executable, "scripts/llm_agent.py",
                "--provider", args.llm_provider,
                "--ingest-external-outputs",
            ]
            if args.llm_model.strip():
                llm_cmd += ["--model", args.llm_model.strip()]
            llm_proc = subprocess.run(llm_cmd, env=env)
            if llm_proc.returncode != 0:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _finish_run(conn, run_id, "failed", 0)
                    conn.commit()
                print(f"\nIngestion of interventions failed (exit code {llm_proc.returncode}).")
                return llm_proc.returncode

        else:
            # Normal LLM flow (or --build-prompts flow).
            # Step 3b: LLM session topic extraction (runs before intervention classification
            # so that LLM-derived session topics are available as grounding context).
            limit_note = f" (limit={args.llm_sessions_limit})" if args.llm_sessions_limit > 0 else ""
            build_note = " [build-prompts only]" if args.build_prompts else ""
            print(f"\nStep 3b/4  Running LLM session topic extraction{limit_note}{build_note}...")
            topics_cmd = [
                sys.executable, "scripts/llm_session_topics.py",
                "--provider", args.llm_provider,
            ]
            if args.llm_model.strip():
                topics_cmd += ["--model", args.llm_model.strip()]
            if forced_session_id:
                topics_cmd += ["--session-id", forced_session_id, "--reprocess-session-topics"]
            elif args.reprocess_session_topics:
                topics_cmd += ["--reprocess-session-topics"]
            if args.llm_sessions_limit > 0:
                topics_cmd += ["--limit", str(args.llm_sessions_limit)]
            if args.build_prompts:
                topics_cmd += ["--build-prompts"]
            topics_proc = subprocess.run(topics_cmd, env=env)
            if topics_proc.returncode != 0:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _finish_run(conn, run_id, "failed", 0)
                    conn.commit()
                print(f"\nLLM session topic extraction failed (exit code {topics_proc.returncode}). Nothing was marked processed.")
                return topics_proc.returncode

            # Step 3c: LLM intervention classification.
            llm_cmd = [sys.executable, "scripts/llm_agent.py", "--provider", args.llm_provider]
            if args.llm_model.strip():
                llm_cmd += ["--model", args.llm_model.strip()]
            if forced_session_id:
                llm_cmd += ["--session-id", forced_session_id]
            if args.llm_speech_limit > 0:
                llm_cmd += ["--limit", str(args.llm_speech_limit)]
            if args.build_prompts:
                llm_cmd += ["--build-prompts"]
                print(f"\nStep 3c/4  Building intervention prompts (build-prompts mode)...")
            elif args.llm_speech_limit > 0:
                print(f"\nStep 3c/4  Running LLM intervention classification (limit={args.llm_speech_limit})...")
            else:
                print(f"\nStep 3c/4  Running LLM intervention classification (all interventions)...")
            llm_proc = subprocess.run(llm_cmd, env=env)
            if llm_proc.returncode != 0:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _finish_run(conn, run_id, "failed", 0)
                    conn.commit()
                print(f"\nLLM intervention classification failed (exit code {llm_proc.returncode}). Nothing was marked processed.")
                return llm_proc.returncode

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        # --build-prompts: skip export — nothing was stored to DB.
        skip_export = args.skip_export or args.build_prompts
        if not skip_export:
            print(f"\nStep 4/4  Exporting frontend JSON artifacts...")
            export_cmd = [sys.executable, "scripts/export_outputs.py", "--db-path", str(db_path)]
            export_proc = subprocess.run(export_cmd, env=env)
            if export_proc.returncode != 0:
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
                print(f"\nExport failed (exit code {export_proc.returncode}). Nothing was marked processed.")
                return export_proc.returncode
        elif args.build_prompts:
            print("\nStep 4/4  Export skipped (build-prompts mode — no DB changes).")
        else:
            print("\nStep 4/4  Export skipped (--skip-export).")

        _finish_run(conn, run_id, "completed", len(candidates))
        conn.commit()

    print(f"\n{'='*60}")
    print(f"Pipeline complete  run_id={run_id}")
    print(f"  Stenograms processed: {len(candidates)}")
    if args.build_prompts:
        print(f"  Prompts: {GENERATED_PROMPTS_DIR}/")
        print(f"  Next: copy model responses (same filename) to state/external_prompts_output/")
        print(f"        then run: python3 scripts/run_pipeline.py --analyzer-mode llm --ingest-external-outputs")
    else:
        print(f"  Outputs: {'skipped' if args.skip_export else 'outputs/'}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
