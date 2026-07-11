#!/usr/bin/env python3
"""
End-to-end update: scrape → sync → pipeline → exports → crawler → deploy to frontend.

Combines votez-scraper (stenogram extraction from cdep.ro) with the full analysis
pipeline into a single orchestrated run.  Each step is incremental — only new or
changed data is processed.  The final step copies all JSON outputs to the frontend
data directory.

Usage:
    python3 scripts/full_update.py
    python3 scripts/full_update.py --llm-provider openai --llm-model gpt-5-nano
    python3 scripts/full_update.py --only-step 7        # re-export JSON outputs only
    python3 scripts/full_update.py --only-step 8        # deploy to frontend only
    python3 scripts/full_update.py --skip-scrape
    python3 scripts/full_update.py --skip-crawler

Step 7 writes ``outputs/laws_votes/laws_votes.json`` (laws + adoption vote aggregates)
in addition to the usual ``export_outputs`` / activity snapshots.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SCRAPER_DIR = REPO_ROOT.parent / "votez-scraper"
DEFAULT_FRONTEND_DIR = REPO_ROOT.parent / "votez-frontend"
DEFAULT_INPUT_DIR = REPO_ROOT / "input" / "stenograme"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_DB_PATH = REPO_ROOT / "state" / "state.sqlite"

# outputs/ subdirectories to sync into votez-frontend/data/activity_analizer/
FRONTEND_DATA_SUBDIRS = [
    "members",
    "parties",
    "topics",
    "session_topics",
    "productivity",
    "activity",
    "laws_votes",
]

# outputs/ files to sync directly into votez-frontend/data/activity_analizer/
FRONTEND_DATA_FILES = [
    "adopted_law_reader_summaries.json",
    "analysis_intervals.json",
]

# outputs/ files to sync directly into votez-frontend/data/
FRONTEND_ROOT_DATA_FILES = [
    "evolutia_partidelor_camera_deputatilor.json",
    "evolutia_partidelor_senat.json",
]

# Scraper registry files to sync into votez-frontend/lib/
SCRAPER_LIB_FILES = [
    "toti_deputatii.json",
    "toti_senatorii.json",
    "deputati_dupa_partid.json",
    "deputati_dupa_circumscriptie.json",
    "senatori_dupa_partid.json",
    "senatori_dupa_circumscriptie.json",
]


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── Step 1: Scrape ──────────────────────────────────────────

def run_scraper(scraper_dir: Path, year: int | None, month: int | None) -> bool:
    """Run votez-scraper: deputies/senators registry + stenograms."""
    main_script = scraper_dir / "main.py"
    steno_script = scraper_dir / "main_stenograme.py"
    if not steno_script.exists():
        print(f"  ERROR: scraper not found at {scraper_dir}")
        print(f"  Expected: {steno_script}")
        print(f"  Clone it with: git clone <votez-scraper-repo> {scraper_dir}")
        return False

    print(f"  Scraper dir: {scraper_dir}")
    ok = True

    # 1a. Deputies & senators registry (main.py --scrape)
    if main_script.exists():
        cmd = [sys.executable, str(main_script), "--scrape"]
        print(f"  [1a] Deputies/senators: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(scraper_dir))
        if proc.returncode != 0:
            print("  WARNING: deputies/senators scrape failed. Continuing with existing registry data.")
            ok = False
    else:
        print(f"  [1a] Skipped — {main_script} not found")

    # 1b. Stenograms (main_stenograme.py --scrape)
    cmd = [sys.executable, str(steno_script), "--scrape"]
    if year is not None:
        cmd += ["--year", str(year)]
    if month is not None:
        cmd += ["--month", str(month)]
    print(f"  [1b] Stenograms:        {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(scraper_dir))
    if proc.returncode != 0:
        ok = False

    return ok


# ── Step 2: Sync ────────────────────────────────────────────

def sync_stenograms(scraper_dir: Path, input_dir: Path) -> int:
    """Copy new/changed stenogram files from scraper output to analyzer input.

    Returns the number of files copied.
    """
    source_dir = scraper_dir / "output" / "stenograme"
    if not source_dir.exists():
        print(f"  WARNING: scraper output dir does not exist: {source_dir}")
        return 0

    input_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for src_file in sorted(source_dir.glob("stenograma_*.json")):
        dst_file = input_dir / src_file.name
        if dst_file.exists() and filecmp.cmp(src_file, dst_file, shallow=False):
            continue
        shutil.copy2(src_file, dst_file)
        copied += 1

    return copied


# ── Step 3–5: Pipeline, exports, crawler ────────────────────

def run_pipeline(args: argparse.Namespace) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / "run_pipeline.py")]
    cmd += ["--analyzer-mode", args.analyzer_mode]
    if args.llm_provider:
        cmd += ["--llm-provider", args.llm_provider]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    if args.dry_run:
        cmd += ["--dry-run"]
    proc = subprocess.run(cmd)
    return proc.returncode == 0


def run_productivity_export() -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / "export_effectiveness.py")]
    proc = subprocess.run(cmd)
    return proc.returncode == 0


def run_crawler(update_existing: bool, hydrate_law_initiators: bool) -> bool:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "crawl_deputy_activity.py"),
        "--export-activity",
    ]
    if update_existing:
        cmd.append("--update-existing")
    if hydrate_law_initiators:
        cmd.append("--hydrate-law-initiators")
    proc = subprocess.run(cmd)
    return proc.returncode == 0


# ── Step 6: Adopted-law PDF/text enrichment ─────────────────

def run_adopted_law_enrichment(args: argparse.Namespace) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / "hydrate_adopted_laws.py")]
    if args.force_adopted_law_extract:
        cmd.append("--force-extract")
    if args.adopted_law_limit is not None:
        cmd += ["--limit", str(args.adopted_law_limit)]
    if args.adopted_law_extract_only:
        cmd.append("--extract-only")
    proc = subprocess.run(cmd)
    hydrate_ok = proc.returncode == 0

    if args.skip_adopted_law_analysis:
        print("  Adopted-law impact analysis skipped (--skip-adopted-law-analysis).")
        return hydrate_ok

    # Always run analysis regardless of hydration exit code — partial failures
    # (e.g. a 404 PDF) must not block analysis of laws that were already hydrated.
    # analyze_adopted_laws.py skips rows that have no extracted text or are
    # already analyzed, so running it when there is nothing to do is harmless.
    analysis_cmd = [sys.executable, str(SCRIPT_DIR / "analyze_adopted_laws.py")]
    provider = args.adopted_law_llm_provider
    model = args.adopted_law_llm_model
    if provider:
        analysis_cmd += ["--provider", provider]
    if model:
        analysis_cmd += ["--model", model]
    if args.force_adopted_law_analysis:
        analysis_cmd.append("--force")
    if args.adopted_law_limit is not None:
        analysis_cmd += ["--limit", str(args.adopted_law_limit)]
    proc = subprocess.run(analysis_cmd)
    analysis_ok = proc.returncode == 0

    return hydrate_ok and analysis_ok


# ── Step 7: Export JSON outputs from DB ─────────────────────

def run_export_outputs() -> bool:
    """Re-export all analysis outputs from DB to outputs/ (no reprocessing)."""
    ok = True
    # Members, parties, topics, session_topics
    proc = subprocess.run([sys.executable, str(SCRIPT_DIR / "export_outputs.py")])
    if proc.returncode != 0:
        ok = False
    # Activity snapshots (deputy + party activity JSONs)
    proc = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "crawl_deputy_activity.py"),
        "--only-export-activity",
    ])
    if proc.returncode != 0:
        ok = False
    # Laws + aggregated adoption votes (outputs/laws_votes/laws_votes.json)
    proc = subprocess.run([sys.executable, str(SCRIPT_DIR / "export_laws_votes.py")])
    if proc.returncode != 0:
        ok = False
    # Frontend aggregate files imported outside the standard analyzer subdirs.
    proc = subprocess.run([sys.executable, str(SCRIPT_DIR / "export_frontend_auxiliary.py")])
    if proc.returncode != 0:
        ok = False
    return ok


# ── Step 8: Deploy to frontend ──────────────────────────────

def _copy_if_changed(src: Path, dst: Path) -> bool:
    """Copy src to dst if dst is missing or differs. Returns True if copied."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return False
    shutil.copy2(src, dst)
    return True


def deploy_analyzer_outputs(outputs_dir: Path, frontend_data_dir: Path) -> int:
    """Copy analysis JSON outputs to votez-frontend/data/activity_analizer/.

    Returns the number of files copied.
    """
    if not outputs_dir.exists():
        print(f"  WARNING: outputs dir does not exist: {outputs_dir}")
        return 0

    copied = 0
    for subdir in FRONTEND_DATA_SUBDIRS:
        src_dir = outputs_dir / subdir
        if not src_dir.exists():
            continue
        for src_file in sorted(src_dir.rglob("*.json")):
            rel_path = src_file.relative_to(outputs_dir)
            dst_file = frontend_data_dir / rel_path
            if _copy_if_changed(src_file, dst_file):
                copied += 1
    return copied


def deploy_analyzer_output_files(outputs_dir: Path, frontend_data_dir: Path) -> int:
    """Copy top-level analyzer JSON outputs to votez-frontend/data/activity_analizer/."""
    copied = 0
    for filename in FRONTEND_DATA_FILES:
        src_file = outputs_dir / filename
        if not src_file.exists():
            continue
        if _copy_if_changed(src_file, frontend_data_dir / filename):
            copied += 1
    return copied


def deploy_frontend_root_data_files(outputs_dir: Path, frontend_root_data_dir: Path) -> int:
    """Copy top-level frontend JSON outputs to votez-frontend/data/."""
    copied = 0
    for filename in FRONTEND_ROOT_DATA_FILES:
        src_file = outputs_dir / filename
        if not src_file.exists():
            continue
        if _copy_if_changed(src_file, frontend_root_data_dir / filename):
            copied += 1
    return copied


def deploy_db(db_path: Path, frontend_data_dir: Path) -> bool:
    """Copy state.sqlite to votez-frontend/data/state.sqlite.

    Returns True if the file was copied (i.e. it changed or was absent).
    """
    if not db_path.exists():
        print(f"  WARNING: DB not found at {db_path}, skipping.")
        return False
    dst = frontend_data_dir / db_path.name
    return _copy_if_changed(db_path, dst)


def deploy_scraper_lib_files(scraper_dir: Path, frontend_lib_dir: Path) -> int:
    """Copy scraper registry files to votez-frontend/lib/.

    Syncs toti_deputatii.json, deputati_dupa_partid.json, etc.
    Returns the number of files copied.
    """
    scraper_output = scraper_dir / "output"
    if not scraper_output.exists():
        print(f"  WARNING: scraper output dir does not exist: {scraper_output}")
        return 0

    copied = 0
    for filename in SCRAPER_LIB_FILES:
        src_file = scraper_output / filename
        if not src_file.exists():
            continue
        dst_file = frontend_lib_dir / filename
        if _copy_if_changed(src_file, dst_file):
            copied += 1
    return copied


# ── Main ────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end update: scrape → sync → analyze → export → crawl → deploy to frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    scrape_group = parser.add_argument_group("scraper options")
    scrape_group.add_argument(
        "--scraper-dir",
        default=str(DEFAULT_SCRAPER_DIR),
        help=f"Path to votez-scraper project (default: {DEFAULT_SCRAPER_DIR})",
    )
    scrape_group.add_argument(
        "--scrape-year", type=int, default=None,
        help="Restrict scrape to a specific year (e.g. 2026)",
    )
    scrape_group.add_argument(
        "--scrape-month", type=int, default=None,
        help="Restrict scrape to a specific month (1-12, requires --scrape-year)",
    )

    pipeline_group = parser.add_argument_group("pipeline options")
    pipeline_group.add_argument(
        "--analyzer-mode", choices=["baseline", "llm"], default="llm",
        help="baseline: keyword only. llm: baseline + LLM classification (default: llm).",
    )
    pipeline_group.add_argument(
        "--llm-provider", choices=["openai", "ollama"], default=None,
        help="LLM provider (default: ollama).",
    )
    pipeline_group.add_argument(
        "--llm-model", default=None,
        help="Model name for LLM mode.",
    )

    crawler_group = parser.add_argument_group("crawler options")
    crawler_group.add_argument(
        "--update-existing-crawler", action="store_true",
        help="Pass --update-existing to the deputy activity crawler.",
    )
    crawler_group.add_argument(
        "--hydrate-law-initiators", action="store_true",
        help=(
            "After crawling, fetch each law's Expunerea de motive PDF, OCR it with "
            "Tesseract, and mark initiating deputies in dep_act_member_laws. "
            "Slow — skipped by default."
        ),
    )

    adopted_law_group = parser.add_argument_group("adopted law enrichment options")
    adopted_law_group.add_argument(
        "--force-adopted-law-extract",
        action="store_true",
        help="Re-extract adopted-law PDF text even when dep_act_laws already has adopted_law_text_json.",
    )
    adopted_law_group.add_argument(
        "--adopted-law-limit",
        type=int,
        default=None,
        help="Process at most N adopted laws in the enrichment step.",
    )
    adopted_law_group.add_argument(
        "--adopted-law-extract-only",
        action="store_true",
        help="Do not download adopted-law PDFs; only extract from cached outputs/pdfs/adopted_laws PDFs.",
    )
    adopted_law_group.add_argument(
        "--skip-adopted-law-analysis",
        action="store_true",
        help="Hydrate adopted-law PDFs/text, but skip citizen-facing law impact analysis.",
    )
    adopted_law_group.add_argument(
        "--force-adopted-law-analysis",
        action="store_true",
        help="Re-run citizen-facing adopted-law analysis even when analysis JSON already exists.",
    )
    adopted_law_group.add_argument(
        "--adopted-law-llm-provider",
        choices=["openai", "ollama"],
        default=None,
        help="LLM provider for adopted-law analysis (default: openai).",
    )
    adopted_law_group.add_argument(
        "--adopted-law-llm-model",
        default=None,
        help="LLM model for adopted-law analysis (default: gpt-5-mini for OpenAI).",
    )

    frontend_group = parser.add_argument_group("frontend deploy options")
    frontend_group.add_argument(
        "--frontend-dir",
        default=str(DEFAULT_FRONTEND_DIR),
        help=f"Path to votez-frontend project root (default: {DEFAULT_FRONTEND_DIR})",
    )

    skip_group = parser.add_argument_group("skip steps")
    skip_group.add_argument("--skip-scrape", action="store_true", help="Skip the scraping step.")
    skip_group.add_argument("--skip-sync", action="store_true", help="Skip the file sync step.")
    skip_group.add_argument("--skip-pipeline", action="store_true", help="Skip the analysis pipeline.")
    skip_group.add_argument("--skip-productivity", action="store_true", help="Skip the productivity export.")
    skip_group.add_argument("--skip-crawler", action="store_true", help="Skip the deputy activity crawler.")
    skip_group.add_argument("--skip-adopted-law-enrichment", action="store_true", help="Skip adopted-law PDF/text enrichment.")
    skip_group.add_argument("--skip-export", action="store_true", help="Skip exporting JSON outputs from DB.")
    skip_group.add_argument("--skip-deploy", action="store_true", help="Skip deploying outputs to the frontend.")
    skip_group.add_argument(
        "--only-step", type=int, choices=range(1, 9), metavar="{1..8}",
        help=(
            "Run only this step and skip all others. "
            "1=scrape  2=sync  3=pipeline  4=productivity  5=crawler  6=adopted-law-enrichment  7=export  8=deploy"
        ),
    )

    parser.add_argument("--dry-run", action="store_true", help="Dry-run the pipeline (no DB writes).")
    args = parser.parse_args()

    if args.scrape_month is not None and args.scrape_year is None:
        parser.error("--scrape-month requires --scrape-year")

    if args.only_step is not None:
        args.skip_scrape       = args.only_step != 1
        args.skip_sync         = args.only_step != 2
        args.skip_pipeline     = args.only_step != 3
        args.skip_productivity = args.only_step != 4
        args.skip_crawler      = args.only_step != 5
        args.skip_adopted_law_enrichment = args.only_step != 6
        args.skip_export       = args.only_step != 7
        args.skip_deploy       = args.only_step != 8

    scraper_dir = Path(args.scraper_dir)
    frontend_dir = Path(args.frontend_dir)
    frontend_data_dir = frontend_dir / "data" / "activity_analizer"
    frontend_lib_dir = frontend_dir / "lib"
    steps_total = 8
    failed = False

    # ── Step 1: Scrape ──────────────────────────────────────
    if not args.skip_scrape:
        _header(f"Step 1/{steps_total}  Scraping from cdep.ro (deputies/senators + stenograms)")
        if not run_scraper(scraper_dir, args.scrape_year, args.scrape_month):
            print("\n  Scraping failed. Continuing with existing data.\n")
            failed = True
    else:
        print(f"\nStep 1/{steps_total}  Scraping — skipped (--skip-scrape)")

    # ── Step 2: Sync ────────────────────────────────────────
    if not args.skip_sync:
        _header(f"Step 2/{steps_total}  Syncing scraper outputs to input/")
        input_root = DEFAULT_INPUT_DIR.parent

        # 2a. Registry files (toti_deputatii.json, toti_senatorii.json)
        registry_copied = 0
        for name in ("toti_deputatii.json", "toti_senatorii.json"):
            src = scraper_dir / "output" / name
            if src.exists() and _copy_if_changed(src, input_root / name):
                registry_copied += 1
        print(f"  Registry files → {input_root}: {registry_copied} copied")

        # 2b. Stenogram files
        copied = sync_stenograms(scraper_dir, DEFAULT_INPUT_DIR)
        print(f"  Stenograms    → {DEFAULT_INPUT_DIR}: {copied} copied")
    else:
        print(f"\nStep 2/{steps_total}  Sync — skipped (--skip-sync)")

    # ── Step 3: Pipeline ────────────────────────────────────
    if not args.skip_pipeline:
        _header(f"Step 3/{steps_total}  Running analysis pipeline")
        if not run_pipeline(args):
            print("\n  Pipeline failed.")
            return 1
    else:
        print(f"\nStep 3/{steps_total}  Pipeline — skipped (--skip-pipeline)")

    # ── Step 4: Productivity export ─────────────────────────
    if not args.skip_productivity:
        _header(f"Step 4/{steps_total}  Exporting productivity metrics")
        if not run_productivity_export():
            print("\n  Productivity export failed.")
            failed = True
    else:
        print(f"\nStep 4/{steps_total}  Productivity — skipped (--skip-productivity)")

    # ── Step 5: Deputy activity crawler ─────────────────────
    if not args.skip_crawler:
        hydrate_note = " + OCR law initiators" if args.hydrate_law_initiators else ""
        _header(f"Step 5/{steps_total}  Crawling deputy activity{hydrate_note} + exporting snapshots")
        if not run_crawler(args.update_existing_crawler, args.hydrate_law_initiators):
            print("\n  Crawler failed.")
            failed = True
    else:
        print(f"\nStep 5/{steps_total}  Crawler — skipped (--skip-crawler)")

    # ── Step 6: Adopted-law PDF/text enrichment ─────────────
    if not args.skip_adopted_law_enrichment and not args.dry_run:
        _header(f"Step 6/{steps_total}  Hydrating adopted-law PDFs and text")
        if not run_adopted_law_enrichment(args):
            print("\n  Adopted-law enrichment failed.")
            failed = True
    elif args.dry_run:
        print(f"\nStep 6/{steps_total}  Adopted-law enrichment — skipped (dry-run)")
    else:
        print(f"\nStep 6/{steps_total}  Adopted-law enrichment — skipped (--skip-adopted-law-enrichment)")

    # ── Step 7: Export JSON outputs from DB ─────────────────
    if not args.skip_export and not args.dry_run:
        _header(f"Step 7/{steps_total}  Exporting JSON outputs from DB")
        if not run_export_outputs():
            print("\n  Export failed.")
            failed = True
    elif args.dry_run:
        print(f"\nStep 7/{steps_total}  Export — skipped (dry-run)")
    else:
        print(f"\nStep 7/{steps_total}  Export — skipped (--skip-export)")

    # ── Step 8: Deploy to frontend ──────────────────────────
    if not args.skip_deploy and not args.dry_run:
        _header(f"Step 8/{steps_total}  Deploying to frontend")

        print(f"  Analyzer outputs → {frontend_data_dir}")
        data_deployed = deploy_analyzer_outputs(DEFAULT_OUTPUTS_DIR, frontend_data_dir)
        print(f"    Files copied (new/changed): {data_deployed}")

        print(f"  Analyzer aggregate files → {frontend_data_dir}")
        aggregate_deployed = deploy_analyzer_output_files(DEFAULT_OUTPUTS_DIR, frontend_data_dir)
        print(f"    Files copied (new/changed): {aggregate_deployed}")

        print(f"  Frontend root data files → {frontend_dir / 'data'}")
        root_data_deployed = deploy_frontend_root_data_files(DEFAULT_OUTPUTS_DIR, frontend_dir / "data")
        print(f"    Files copied (new/changed): {root_data_deployed}")

        print(f"  Database          → {frontend_data_dir / 'state.sqlite'}")
        db_copied = deploy_db(DEFAULT_DB_PATH, frontend_data_dir)
        print(f"    Copied: {db_copied}")

        print(f"  Scraper registry  → {frontend_lib_dir}")
        lib_deployed = deploy_scraper_lib_files(scraper_dir, frontend_lib_dir)
        print(f"    Files copied (new/changed): {lib_deployed}")
    elif args.dry_run:
        print(f"\nStep 8/{steps_total}  Deploy — skipped (dry-run)")
    else:
        print(f"\nStep 8/{steps_total}  Deploy — skipped (--skip-deploy)")

    # ── Summary ─────────────────────────────────────────────
    _header("Done")
    if failed:
        print("  Completed with errors (see above).")
        return 1
    print("  All steps completed successfully.")
    print(f"  Outputs:  outputs/")
    print(f"  Database: state/state.sqlite")
    if not args.skip_deploy and not args.dry_run:
        print(f"  Frontend: {frontend_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
