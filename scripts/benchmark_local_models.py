#!/usr/bin/env python3
"""
Benchmark LLM models on the gold-standard sessions using isolated DB copies.

This script keeps the main DB untouched:
  1. Copy the source DB to state/model_benchmarks/<model>/state.sqlite
  2. Reset prior LLM intervention outputs for the gold sessions in the copy
  3. Re-run session topic extraction for each gold session
  4. Re-run intervention classification only for the gold speeches
  4. Run evaluate_accuracy.py --json and persist the report per model

Usage:
  python3 scripts/benchmark_local_models.py
  python3 scripts/benchmark_local_models.py --models qwen3:14b qwen2.5:14b-32k
  python3 scripts/benchmark_local_models.py --provider openai --models gpt-5.4-mini gpt-4o-mini
  python3 scripts/benchmark_local_models.py --models ollama/qwen3:14b openai/gpt-5.4-mini
  python3 scripts/benchmark_local_models.py --models qwen3:14b --only-hard --reuse-existing-topics
  python3 scripts/benchmark_local_models.py --skip-missing-sessions
  python3 scripts/benchmark_local_models.py --benchmark-scope limited
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from model_profiles import (
    DEFAULT_PIPELINE_ARCHITECTURE_SELECTION,
    PIPELINE_ARCHITECTURE_CHOICES,
    STEP_3_1_CANDIDATE_MODELS,
    get_model_profile,
    normalize_model_name,
    resolve_pipeline_architecture,
)
from select_stenograms import DEFAULT_INPUT_DIR


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DB = ROOT / "state" / "state.sqlite"
DEFAULT_GOLD_PATH = ROOT / "tests" / "gold_standard.json"
DEFAULT_BENCHMARK_ROOT = ROOT / "state" / "model_benchmarks"
REQUIRED_SOURCE_TABLES = ("session_chunks", "interventions_raw")
SUPPORTED_PROVIDERS = ("ollama", "openai")
DEFAULT_PROVIDER = "ollama"
DEFAULT_LIMITED_BENCHMARK_SESSION_LIMIT = 3
DEFAULT_OPENAI_CANDIDATE_MODELS = ("gpt-5.4-mini", "gpt-4o-mini")


@dataclass(frozen=True)
class BenchmarkModelSpec:
    provider: str
    model: str

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"


def _safe_model_dirname(provider: str, model: str) -> str:
    label = f"{provider}__{normalize_model_name(model)}"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in label)


def _parse_benchmark_model_spec(raw_model: str, default_provider: str) -> BenchmarkModelSpec:
    candidate = (raw_model or "").strip()
    if not candidate:
        raise ValueError("Benchmark model name cannot be empty.")

    provider = (default_provider or DEFAULT_PROVIDER).strip().lower()
    model = candidate

    if "/" in candidate:
        maybe_provider, maybe_model = candidate.split("/", 1)
        maybe_provider = maybe_provider.strip().lower()
        maybe_model = maybe_model.strip()
        if maybe_provider in SUPPORTED_PROVIDERS and maybe_model:
            provider = maybe_provider
            model = maybe_model

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    normalized_model = normalize_model_name(model)
    if not normalized_model:
        raise ValueError(f"Benchmark model name cannot be empty: {raw_model!r}")

    return BenchmarkModelSpec(provider=provider, model=normalized_model)


def _default_models_for_provider(provider: str) -> list[str]:
    if provider.strip().lower() == "openai":
        return list(DEFAULT_OPENAI_CANDIDATE_MODELS)
    return list(STEP_3_1_CANDIDATE_MODELS)


def _apply_benchmark_scope_defaults(
    *,
    benchmark_scope: str,
    session_limit: int,
    only_hard: bool,
) -> tuple[int, bool]:
    if benchmark_scope != "limited":
        return session_limit, only_hard

    resolved_session_limit = session_limit or DEFAULT_LIMITED_BENCHMARK_SESSION_LIMIT
    return resolved_session_limit, True if not only_hard else only_hard


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


def _write_stenogram_list_file(path: Path, *, run_id: str, files: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "files": files}, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_input_session_file_map(input_dir: Path) -> dict[str, str]:
    if not input_dir.exists():
        return {}

    session_to_file: dict[str, str] = {}
    for file_path in sorted(input_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            continue
        rel_path = file_path.resolve().relative_to(ROOT.resolve()).as_posix()
        session_to_file.setdefault(session_id, rel_path)
    return session_to_file


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


def _summary_timestamp_for_existing_file(path: Path) -> str | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, timezone.utc).replace(microsecond=0).isoformat()


def _load_summary_runs(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    runs = payload.get("runs")
    if isinstance(runs, list):
        return [item for item in runs if isinstance(item, dict)]

    results = payload.get("results")
    if isinstance(results, list):
        return [
            {
                "run_started_at": _summary_timestamp_for_existing_file(summary_path),
                "results": results,
            }
        ]
    return []


def _append_summary_run(summary_path: Path, run_started_at: str, results: list[dict]) -> dict:
    runs = _load_summary_runs(summary_path)
    runs.append(
        {
            "run_started_at": run_started_at,
            "results": results,
        }
    )
    return {
        "results": results,
        "run_started_at": run_started_at,
        "runs": runs,
    }


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


def _prepare_missing_gold_sessions(
    *,
    source_db: Path,
    benchmark_root: Path,
    input_dir: Path,
    session_ids: list[str],
    run_started_at: str,
) -> tuple[Path, list[str], list[str]]:
    missing_by_table = _find_missing_source_sessions(source_db, session_ids)
    missing_sessions = _union_missing_sessions(session_ids, missing_by_table)
    if not missing_sessions:
        return source_db, [], []

    session_to_file = _load_input_session_file_map(input_dir)
    preparable_paths: list[str] = []
    prepared_session_ids: list[str] = []
    still_missing_session_ids: list[str] = []
    for session_id in missing_sessions:
        rel_path = session_to_file.get(session_id)
        if rel_path:
            prepared_session_ids.append(session_id)
            preparable_paths.append(rel_path)
        else:
            still_missing_session_ids.append(session_id)

    if not preparable_paths:
        return source_db, [], still_missing_session_ids

    prepared_root = benchmark_root / "_prepared_source"
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_db = prepared_root / "state.sqlite"
    stenogram_list_path = prepared_root / "stenograms.json"
    prepare_run_id = f"benchmark_prepare_{run_started_at.replace(':', '').replace('+', '_')}"

    if prepared_db.exists():
        prepared_db.unlink()
    _copy_db(source_db, prepared_db)
    _write_stenogram_list_file(stenogram_list_path, run_id=prepare_run_id, files=preparable_paths)
    _run_command(
        [
            sys.executable,
            "scripts/analyze_interventions.py",
            "--run-id",
            prepare_run_id,
            "--stenogram-list-path",
            str(stenogram_list_path),
            "--db-path",
            str(prepared_db),
        ],
        cwd=ROOT,
    )
    return prepared_db, prepared_session_ids, still_missing_session_ids


def benchmark_model(
    *,
    provider: str,
    model: str,
    source_db: Path,
    benchmark_root: Path,
    session_ids: list[str],
    gold_speeches: list[dict],
    pipeline_architecture: str,
    reuse_existing_topics: bool,
) -> dict:
    provider = provider.strip().lower()
    model = normalize_model_name(model)
    resolved_pipeline_architecture = resolve_pipeline_architecture(provider, model, pipeline_architecture)
    model_dir = benchmark_root / _safe_model_dirname(provider, model)
    benchmark_db = model_dir / "state.sqlite"
    run_id = f"benchmark_{_safe_model_dirname(provider, model)}"
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
                        provider,
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
                provider,
                "--model",
                model,
                "--intervention-ids-file",
                str(intervention_ids_path),
                "--pipeline-architecture",
                resolved_pipeline_architecture,
            ],
            cwd=ROOT,
        )

        evaluation = _evaluate_model(benchmark_db, benchmark_gold_path)
        _finish_run(benchmark_db, run_id, "completed", len(session_ids))
        profile = get_model_profile(provider, model)
        summary = {
            "provider": provider,
            "model": model,
            "pipeline_architecture_requested": pipeline_architecture,
            "pipeline_architecture": resolved_pipeline_architecture,
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
    parser = argparse.ArgumentParser(description="Benchmark OpenAI and Ollama models on the gold-standard sessions.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB), help="Source SQLite DB to copy for each model.")
    parser.add_argument("--gold-path", default=str(DEFAULT_GOLD_PATH), help="Gold-standard JSON path.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory of stenogram JSON files used to auto-import missing gold sessions when available.",
    )
    parser.add_argument(
        "--benchmark-root",
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Directory where per-model DB copies and reports are written.",
    )
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=DEFAULT_PROVIDER,
        help=(
            "Default provider for unprefixed --models entries. Use provider-qualified model names "
            "such as openai/gpt-5.4-mini or ollama/qwen3:14b to mix providers in one run."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model names to benchmark. If omitted, uses the default candidate list for --provider. "
            "You can mix providers via openai/<model> and ollama/<model>."
        ),
    )
    parser.add_argument(
        "--pipeline-architecture",
        choices=list(PIPELINE_ARCHITECTURE_CHOICES),
        default=DEFAULT_PIPELINE_ARCHITECTURE_SELECTION,
        help="Classification pipeline architecture to use for all benchmarked models ('auto' resolves per model).",
    )
    parser.add_argument(
        "--benchmark-scope",
        choices=["full", "limited"],
        default="full",
        help=(
            "full: evaluate all selected gold sessions. "
            "limited: cheaper smoke benchmark that defaults to the first 3 gold sessions "
            "and medium+hard gold speeches only."
        ),
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
    input_dir = Path(args.input_dir)
    benchmark_root = Path(args.benchmark_root)
    run_started_at = _utc_now_iso()
    raw_models = args.models if args.models is not None else _default_models_for_provider(args.provider)
    try:
        model_specs = [_parse_benchmark_model_spec(raw_model, args.provider) for raw_model in raw_models]
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    session_limit, only_hard = _apply_benchmark_scope_defaults(
        benchmark_scope=args.benchmark_scope,
        session_limit=args.session_limit,
        only_hard=args.only_hard,
    )

    if not source_db.exists():
        print(f"ERROR: source DB not found: {source_db}")
        return 1
    if not gold_path.exists():
        print(f"ERROR: gold standard not found: {gold_path}")
        return 1

    session_ids = _load_gold_session_ids(gold_path, only_hard=only_hard)
    if session_limit > 0:
        session_ids = session_ids[:session_limit]
    gold_speeches = _select_gold_speeches(gold_path, session_ids=session_ids, only_hard=only_hard)

    prepared_source_db, auto_prepared_session_ids, still_missing_input_sessions = _prepare_missing_gold_sessions(
        source_db=source_db,
        benchmark_root=benchmark_root,
        input_dir=input_dir,
        session_ids=session_ids,
        run_started_at=run_started_at,
    )
    if auto_prepared_session_ids:
        print(
            f"Auto-imported {len(auto_prepared_session_ids)} gold session(s) into a prepared benchmark source DB: "
            f"{', '.join(auto_prepared_session_ids)}"
        )

    missing_by_table = _find_missing_source_sessions(prepared_source_db, session_ids)
    if missing_by_table:
        missing_sessions = _union_missing_sessions(session_ids, missing_by_table)
        if args.skip_missing_sessions:
            session_ids = [session_id for session_id in session_ids if session_id not in set(missing_sessions)]
            gold_speeches = _select_gold_speeches(gold_path, session_ids=session_ids, only_hard=only_hard)
            print(
                f"Skipping {len(missing_sessions)} gold session(s) missing from the source DB; "
                f"benchmarking the remaining {len(session_ids)} session(s)."
            )
        else:
            print(
                "ERROR: source DB is missing gold sessions required for benchmarking. "
                "Re-import those sessions or rerun with --skip-missing-sessions."
            )
            if still_missing_input_sessions:
                print(
                    "  Missing from input dir as well: "
                    f"{', '.join(still_missing_input_sessions)}"
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

    print(f"Benchmarking {len(model_specs)} model(s) across {len(session_ids)} gold session(s).")
    if args.benchmark_scope == "limited":
        print(
            "Using limited benchmark scope: "
            f"session_limit={session_limit}, medium+hard gold speeches only."
        )
    benchmark_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for model_spec in model_specs:
        print(f"\n=== Model: {model_spec.display_name} ===")
        try:
            summary = benchmark_model(
                provider=model_spec.provider,
                model=model_spec.model,
                source_db=prepared_source_db,
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
            failure = {"provider": model_spec.provider, "model": model_spec.model, "status": "failed", "error": str(exc)}
            print(f"FAILED: {exc}")
            all_results.append(failure)

    summary_path = benchmark_root / "summary.json"
    summary_payload = _append_summary_run(summary_path, run_started_at, all_results)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
