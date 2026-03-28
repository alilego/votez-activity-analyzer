#!/usr/bin/env python3
"""
Benchmark local Ollama models on the gold-standard sessions using isolated DB copies.

This script keeps the main DB untouched:
  1. Copy the source DB to state/model_benchmarks/<model>/state.sqlite
  2. Reset prior LLM intervention outputs for the gold sessions in the copy
  3. Re-run session topic extraction for each gold session
  4. Re-run intervention classification only for the gold speeches
  4. Run evaluate_accuracy.py --json and persist the report per model

Usage:
  python3 scripts/benchmark_local_models.py
  python3 scripts/benchmark_local_models.py --models qwen3:14b qwen2.5:14b-32k
  python3 scripts/benchmark_local_models.py --models qwen3:14b --only-hard --reuse-existing-topics
  python3 scripts/benchmark_local_models.py --skip-missing-sessions
  python3 scripts/benchmark_local_models.py --session-limit 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from model_profiles import STEP_3_1_CANDIDATE_MODELS, get_model_profile, normalize_model_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DB = ROOT / "state" / "state.sqlite"
DEFAULT_GOLD_PATH = ROOT / "tests" / "gold_standard.json"
DEFAULT_BENCHMARK_ROOT = ROOT / "state" / "model_benchmarks"
REQUIRED_SOURCE_TABLES = ("session_chunks", "interventions_raw")


def _safe_model_dirname(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in normalize_model_name(model))


def _load_gold_speeches(gold_path: Path) -> list[dict]:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    speeches = data.get("speeches", [])
    return speeches if isinstance(speeches, list) else []


def _select_gold_speeches(
    gold_path: Path,
    *,
    session_ids: list[str] | None = None,
    only_hard: bool = False,
) -> list[dict]:
    allowed_sessions = set(session_ids or [])
    speeches = _load_gold_speeches(gold_path)
    selected: list[dict] = []
    for item in speeches:
        if item.get("session_id") is None or item.get("speech_index") is None:
            continue
        session_id = str(item["session_id"])
        if allowed_sessions and session_id not in allowed_sessions:
            continue
        if only_hard and item.get("difficulty") not in ("medium", "hard"):
            continue
        selected.append(item)
    return selected


def _load_gold_session_ids(gold_path: Path, *, only_hard: bool = False) -> list[str]:
    speeches = _select_gold_speeches(gold_path, only_hard=only_hard)
    session_ids = {str(item["session_id"]) for item in speeches}
    return sorted(session_ids)


def _load_gold_targets(gold_path: Path, session_ids: list[str] | None = None, *, only_hard: bool = False) -> dict[str, list[int]]:
    speeches = _select_gold_speeches(gold_path, session_ids=session_ids, only_hard=only_hard)
    return _load_gold_targets_from_speeches(speeches)


def _load_gold_targets_from_speeches(gold_speeches: list[dict]) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for item in gold_speeches:
        session_id = str(item["session_id"])
        speech_index = int(item["speech_index"])
        bucket = targets.setdefault(session_id, [])
        if speech_index not in bucket:
            bucket.append(speech_index)
    for session_id in targets:
        targets[session_id].sort()
    return dict(sorted(targets.items()))


def _copy_db(source_db: Path, target_db: Path) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, target_db)


def _find_missing_session_topics(source_db: Path, session_ids: list[str]) -> list[str]:
    if not session_ids:
        return []

    placeholders = ",".join("?" for _ in session_ids)
    with sqlite3.connect(source_db) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT session_id
            FROM session_topics
            WHERE session_id IN ({placeholders})
            """,
            session_ids,
        ).fetchall()
    available = {str(row[0]) for row in rows}
    return [session_id for session_id in session_ids if session_id not in available]


def _find_missing_source_sessions(source_db: Path, session_ids: list[str]) -> dict[str, list[str]]:
    if not session_ids:
        return {}

    placeholders = ",".join("?" for _ in session_ids)
    missing: dict[str, list[str]] = {}
    with sqlite3.connect(source_db) as conn:
        for table in REQUIRED_SOURCE_TABLES:
            rows = conn.execute(
                f"""
                SELECT DISTINCT session_id
                FROM {table}
                WHERE session_id IN ({placeholders})
                """,
                session_ids,
            ).fetchall()
            available = {str(row[0]) for row in rows}
            absent = [session_id for session_id in session_ids if session_id not in available]
            if absent:
                missing[table] = absent
    return missing


def _union_missing_sessions(session_ids: list[str], missing_by_table: dict[str, list[str]]) -> list[str]:
    missing = {session_id for values in missing_by_table.values() for session_id in values}
    return [session_id for session_id in session_ids if session_id in missing]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_run_exists(db_path: Path, run_id: str) -> None:
    now = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
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
                finished_at = NULL,
                status = 'running'
            """,
            (run_id, now),
        )
        conn.commit()


def _finish_run(db_path: Path, run_id: str, status: str, sessions_processed: int) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?,
                finished_at = ?,
                sessions_processed = ?
            WHERE run_id = ?
            """,
            (status, _utc_now_iso(), int(sessions_processed), run_id),
        )
        conn.commit()


def _reset_llm_outputs(db_path: Path, session_ids: list[str], *, reset_session_topics: bool = True) -> dict[str, int]:
    placeholders = ",".join("?" for _ in session_ids)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        deleted = conn.execute(
            f"""
            DELETE FROM intervention_analysis
            WHERE relevance_source = 'llm_agent_v1'
              AND intervention_id IN (
                  SELECT intervention_id
                  FROM interventions_raw
                  WHERE session_id IN ({placeholders})
              )
            """,
            session_ids,
        ).rowcount
        reset_topics = 0
        if reset_session_topics:
            reset_topics = conn.execute(
                f"""
                UPDATE session_topics
                SET topics_source = 'keyword_baseline_v1',
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id IN ({placeholders})
                """,
                session_ids,
            ).rowcount
        conn.commit()
    return {"deleted_intervention_analysis": int(deleted), "reset_session_topics": int(reset_topics)}


def _lookup_gold_intervention_ids(db_path: Path, gold_targets: dict[str, list[int]]) -> list[str]:
    intervention_ids: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for session_id, speech_indexes in gold_targets.items():
            if not speech_indexes:
                continue
            placeholders = ",".join("?" for _ in speech_indexes)
            rows = conn.execute(
                f"""
                SELECT intervention_id, speech_index
                FROM interventions_raw
                WHERE session_id = ?
                  AND member_id IS NOT NULL
                  AND speech_index IN ({placeholders})
                ORDER BY speech_index
                """,
                [session_id, *speech_indexes],
            ).fetchall()
            found_by_index = {int(r["speech_index"]): r["intervention_id"] for r in rows}
            missing = [idx for idx in speech_indexes if idx not in found_by_index]
            if missing:
                raise RuntimeError(
                    f"Missing intervention_ids in DB for session {session_id}: speech_index {missing}"
                )
            intervention_ids.extend(found_by_index[idx] for idx in speech_indexes)
    return intervention_ids


def _run_command(cmd: list[str], *, cwd: Path) -> float:
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return elapsed


def _evaluate_model(db_path: Path, gold_path: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/evaluate_accuracy.py",
        "--db-path",
        str(db_path),
        "--gold-path",
        str(gold_path),
        "--json",
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr or result.stdout.strip() or "evaluate_accuracy.py failed")
    return json.loads(result.stdout)


def benchmark_model(
    *,
    model: str,
    source_db: Path,
    benchmark_root: Path,
    session_ids: list[str],
    gold_speeches: list[dict],
    pipeline_architecture: str,
    reuse_existing_topics: bool,
) -> dict:
    model = normalize_model_name(model)
    model_dir = benchmark_root / _safe_model_dirname(model)
    benchmark_db = model_dir / "state.sqlite"
    run_id = f"benchmark_{_safe_model_dirname(model)}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if benchmark_db.exists():
        benchmark_db.unlink()
    _copy_db(source_db, benchmark_db)
    _ensure_run_exists(benchmark_db, run_id)
    reset_stats = _reset_llm_outputs(benchmark_db, session_ids, reset_session_topics=not reuse_existing_topics)
    gold_targets = _load_gold_targets_from_speeches(gold_speeches)
    target_intervention_ids = _lookup_gold_intervention_ids(benchmark_db, gold_targets)
    intervention_ids_path = model_dir / "gold_intervention_ids.json"
    intervention_ids_path.write_text(
        json.dumps(target_intervention_ids, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    benchmark_gold_path = model_dir / "benchmark_gold.json"
    benchmark_gold_path.write_text(
        json.dumps({"speeches": gold_speeches}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timings = {"topics_seconds": 0.0, "classification_seconds": 0.0}

    try:
        if not reuse_existing_topics:
            for session_id in session_ids:
                timings["topics_seconds"] += _run_command(
                    [
                        sys.executable,
                        "scripts/llm_session_topics.py",
                        "--run-id",
                        run_id,
                        "--db-path",
                        str(benchmark_db),
                        "--provider",
                        "ollama",
                        "--model",
                        model,
                        "--session-id",
                        session_id,
                        "--reprocess-session-topics",
                    ],
                    cwd=ROOT,
                )
        timings["classification_seconds"] += _run_command(
            [
                sys.executable,
                "scripts/llm_agent.py",
                "--run-id",
                run_id,
                "--db-path",
                str(benchmark_db),
                "--provider",
                "ollama",
                "--model",
                model,
                "--intervention-ids-file",
                str(intervention_ids_path),
                "--pipeline-architecture",
                pipeline_architecture,
            ],
            cwd=ROOT,
        )

        evaluation = _evaluate_model(benchmark_db, benchmark_gold_path)
        _finish_run(benchmark_db, run_id, "completed", len(session_ids))
        profile = get_model_profile("ollama", model)
        summary = {
            "model": model,
            "pipeline_architecture": pipeline_architecture,
            "topic_mode": "reuse_existing" if reuse_existing_topics else "rerun_per_model",
            "benchmark_db_path": str(benchmark_db),
            "run_id": run_id,
            "session_ids": session_ids,
            "gold_target_speeches": sum(len(v) for v in gold_targets.values()),
            "reset_stats": reset_stats,
            "num_ctx": profile.num_ctx if profile else None,
            "estimated_vram_gb": profile.estimated_vram_gb if profile else None,
            "timings": {
                **timings,
                "total_seconds": round(timings["topics_seconds"] + timings["classification_seconds"], 2),
            },
            "evaluation": evaluation,
        }
        (model_dir / "benchmark_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    except Exception:
        _finish_run(benchmark_db, run_id, "failed", 0)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models on the gold-standard sessions.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB), help="Source SQLite DB to copy for each model.")
    parser.add_argument("--gold-path", default=str(DEFAULT_GOLD_PATH), help="Gold-standard JSON path.")
    parser.add_argument(
        "--benchmark-root",
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Directory where per-model DB copies and reports are written.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(STEP_3_1_CANDIDATE_MODELS),
        help="Ollama model names to benchmark.",
    )
    parser.add_argument(
        "--pipeline-architecture",
        choices=["three_layer", "one_pass"],
        default="three_layer",
        help="Classification pipeline architecture to use for all benchmarked models.",
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=0,
        help="Benchmark only the first N gold sessions (0 = all gold sessions).",
    )
    parser.add_argument(
        "--only-hard",
        action="store_true",
        help="Benchmark only medium+hard gold speeches. Useful for faster model screening.",
    )
    parser.add_argument(
        "--reuse-existing-topics",
        action="store_true",
        help=(
            "Skip session-topic re-extraction and reuse topics already present in the copied DB. "
            "Much faster, but only suitable for quick classification-focused screening."
        ),
    )
    parser.add_argument(
        "--skip-missing-sessions",
        action="store_true",
        help="Skip gold sessions absent from the source DB instead of failing the benchmark.",
    )
    args = parser.parse_args()

    source_db = Path(args.source_db)
    gold_path = Path(args.gold_path)
    benchmark_root = Path(args.benchmark_root)

    if not source_db.exists():
        print(f"ERROR: source DB not found: {source_db}")
        return 1
    if not gold_path.exists():
        print(f"ERROR: gold standard not found: {gold_path}")
        return 1

    session_ids = _load_gold_session_ids(gold_path, only_hard=args.only_hard)
    if args.session_limit > 0:
        session_ids = session_ids[: args.session_limit]
    gold_speeches = _select_gold_speeches(gold_path, session_ids=session_ids, only_hard=args.only_hard)

    missing_by_table = _find_missing_source_sessions(source_db, session_ids)
    if missing_by_table:
        missing_sessions = _union_missing_sessions(session_ids, missing_by_table)
        if args.skip_missing_sessions:
            session_ids = [session_id for session_id in session_ids if session_id not in set(missing_sessions)]
            gold_speeches = _select_gold_speeches(gold_path, session_ids=session_ids, only_hard=args.only_hard)
            print(
                f"Skipping {len(missing_sessions)} gold session(s) missing from the source DB; "
                f"benchmarking the remaining {len(session_ids)} session(s)."
            )
        else:
            print(
                "ERROR: source DB is missing gold sessions required for benchmarking. "
                "Re-import those sessions or rerun with --skip-missing-sessions."
            )
            for table in REQUIRED_SOURCE_TABLES:
                table_missing = missing_by_table.get(table, [])
                if table_missing:
                    print(f"  {table}: {', '.join(table_missing)}")
            return 1

    if not session_ids:
        print("ERROR: no gold sessions remain to benchmark after source DB filtering.")
        return 1

    if args.reuse_existing_topics:
        missing_topics = _find_missing_session_topics(source_db, session_ids)
        if missing_topics:
            print(
                "ERROR: --reuse-existing-topics requires existing session_topics rows for every selected session."
            )
            print(f"  Missing session_topics: {', '.join(missing_topics)}")
            return 1

    print(f"Benchmarking {len(args.models)} model(s) across {len(session_ids)} gold session(s).")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for raw_model in args.models:
        model = normalize_model_name(raw_model)
        print(f"\n=== Model: {model} ===")
        try:
            summary = benchmark_model(
                model=model,
                source_db=source_db,
                benchmark_root=benchmark_root,
                session_ids=session_ids,
                gold_speeches=gold_speeches,
                pipeline_architecture=args.pipeline_architecture,
                reuse_existing_topics=args.reuse_existing_topics,
            )
            cls = summary["evaluation"]["classification"]
            law = summary["evaluation"]["law_attribution"]
            print(
                f"accuracy={cls['accuracy']}%  law_exact={law['exact_match_pct']}%  "
                f"elapsed={summary['timings']['total_seconds']}s"
            )
            all_results.append(summary)
        except Exception as exc:
            failure = {"model": model, "status": "failed", "error": str(exc)}
            print(f"FAILED: {exc}")
            all_results.append(failure)

    summary_path = benchmark_root / "summary.json"
    summary_path.write_text(json.dumps({"results": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
