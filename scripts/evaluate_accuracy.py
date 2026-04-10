#!/usr/bin/env python3
"""
Evaluation harness for measuring classification and attribution accuracy
against the human-reviewed gold standard.

Usage:
  python3 scripts/evaluate_accuracy.py
  python3 scripts/evaluate_accuracy.py --db-path state/state.sqlite --gold-path tests/gold_standard.json
  python3 scripts/evaluate_accuracy.py --verbose          # print every mismatch
  python3 scripts/evaluate_accuracy.py --only-hard        # evaluate only hard/medium cases
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["constructive", "neutral", "non_constructive"]
DEFAULT_DB_PATH = Path("state/state.sqlite")
DEFAULT_GOLD_PATH = Path("tests/gold_standard.json")
DISPLAY_PREVIEW_CHARS = 120


def _norm(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(ch)
    ).strip()


def _norm_topic(topic: str) -> str:
    key = _norm(topic)
    key = re.sub(r"[^a-z0-9\s]+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def _norm_law_id(law_id: str) -> str:
    key = _norm(law_id)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _display_preview(text: str, max_chars: int = DISPLAY_PREVIEW_CHARS) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


# ── Metrics ──────────────────────────────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def topic_overlap(expected: list[str], predicted: list[str]) -> tuple[int, int, int]:
    exp_set = {_norm_topic(t) for t in expected if t}
    pred_set = {_norm_topic(t) for t in predicted if t}
    if not exp_set and not pred_set:
        return 0, 0, 0
    tp = len(exp_set & pred_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    return tp, fp, fn


def law_id_match(expected: list[str], predicted_topics: list[dict], session_topics: list[dict]) -> dict:
    if not expected:
        return {"expected": 0, "exact_match": 0, "partial_match": 0, "missed": 0}

    pred_law_ids: set[str] = set()
    for t in predicted_topics:
        if isinstance(t, dict) and t.get("law_id"):
            pred_law_ids.add(_norm_law_id(t["law_id"]))
    for t in session_topics:
        if isinstance(t, dict) and t.get("law_id"):
            pred_law_ids.add(_norm_law_id(t["law_id"]))

    exact = 0
    partial = 0
    missed = 0
    for exp_law in expected:
        exp_key = _norm_law_id(exp_law)
        if exp_key in pred_law_ids:
            exact += 1
        elif any(exp_key in p or p in exp_key for p in pred_law_ids if len(p) >= 6):
            partial += 1
        else:
            nums = re.findall(r"\d+/\d{4}", exp_key)
            if nums and any(n in p for p in pred_law_ids for n in nums):
                partial += 1
            else:
                missed += 1

    return {
        "expected": len(expected),
        "exact_match": exact,
        "partial_match": partial,
        "missed": missed,
    }


# ── Data loading ─────────────────────────────────────────────────────────

def load_gold(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["speeches"]


def load_predictions(db_path: Path, gold_speeches: list[dict]) -> dict[tuple, dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    predictions: dict[tuple, dict] = {}
    for gs in gold_speeches:
        sid = str(gs["session_id"])
        idx = int(gs["speech_index"])
        row = conn.execute(
            """
            SELECT ir.intervention_id, ia.relevance_label, ia.topics_json,
                   ia.confidence, ia.reasoning, ia.layer_a_json
            FROM interventions_raw ir
            JOIN intervention_analysis ia ON ia.intervention_id = ir.intervention_id
            WHERE ir.session_id = ? AND ir.speech_index = ? AND ir.member_id IS NOT NULL
            """,
            (sid, idx),
        ).fetchone()
        if row:
            topics_raw = row["topics_json"]
            topics = json.loads(topics_raw) if topics_raw else []
            predictions[(sid, idx)] = {
                "intervention_id": row["intervention_id"],
                "label": row["relevance_label"],
                "topics": topics if isinstance(topics, list) else [],
                "confidence": row["confidence"] or 0.0,
                "reasoning": row["reasoning"] or "",
                "layer_a": json.loads(row["layer_a_json"]) if row["layer_a_json"] else {},
            }

    session_topics: dict[str, list[dict]] = {}
    for sid in {str(gs["session_id"]) for gs in gold_speeches}:
        row = conn.execute(
            "SELECT topics_json FROM session_topics WHERE session_id = ?", (sid,)
        ).fetchone()
        if row and row["topics_json"]:
            st = json.loads(row["topics_json"])
            session_topics[sid] = st if isinstance(st, list) else []

    conn.close()
    return predictions, session_topics


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(
    gold_speeches: list[dict],
    predictions: dict[tuple, dict],
    session_topics: dict[str, list[dict]],
    verbose: bool = False,
    only_hard: bool = False,
) -> dict:
    confusion = defaultdict(lambda: defaultdict(int))
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()

    topic_tp_total = 0
    topic_fp_total = 0
    topic_fn_total = 0

    law_stats = {"expected": 0, "exact_match": 0, "partial_match": 0, "missed": 0}

    correct = 0
    total = 0
    skipped = 0
    mismatches: list[dict] = []

    conf_buckets: dict[str, list[bool]] = defaultdict(list)

    for gs in gold_speeches:
        if only_hard and gs.get("difficulty") not in ("hard", "medium"):
            continue

        sid = str(gs["session_id"])
        idx = int(gs["speech_index"])
        key = (sid, idx)
        pred = predictions.get(key)

        if not pred:
            skipped += 1
            continue

        expected_label = gs["expected_label"]
        predicted_label = pred["label"]
        total += 1

        confusion[expected_label][predicted_label] += 1

        is_correct = expected_label == predicted_label
        if is_correct:
            correct += 1
            per_label_tp[expected_label] += 1
        else:
            per_label_fn[expected_label] += 1
            per_label_fp[predicted_label] += 1
            mismatches.append({
                "id": gs["id"],
                "session_id": sid,
                "speech_index": idx,
                "speaker": gs["raw_speaker"],
                "expected": expected_label,
                "predicted": predicted_label,
                "confidence": pred["confidence"],
                "difficulty": gs.get("difficulty", ""),
                "text_preview": gs["text"],
                "gold_notes": gs.get("labeling_notes", ""),
                "pred_reasoning": pred["reasoning"],
            })

        conf = pred["confidence"]
        if conf >= 0.8:
            bucket = "0.80-1.00"
        elif conf >= 0.65:
            bucket = "0.65-0.79"
        elif conf >= 0.5:
            bucket = "0.50-0.64"
        else:
            bucket = "0.00-0.49"
        conf_buckets[bucket].append(is_correct)

        expected_topics = gs.get("expected_topics", [])
        predicted_topics = pred["topics"]
        pred_topic_labels = []
        for t in predicted_topics:
            if isinstance(t, str):
                pred_topic_labels.append(t)
            elif isinstance(t, dict):
                pred_topic_labels.append(t.get("label", ""))

        tp, fp, fn = topic_overlap(expected_topics, pred_topic_labels)
        topic_tp_total += tp
        topic_fp_total += fp
        topic_fn_total += fn

        expected_laws = gs.get("expected_law_ids", [])
        st = session_topics.get(sid, [])
        law_result = law_id_match(expected_laws, predicted_topics, st)
        for k in law_stats:
            law_stats[k] += law_result[k]

    # Build report
    report: dict = {}

    report["coverage"] = {
        "gold_total": len(gold_speeches),
        "evaluated": total,
        "skipped_no_prediction": skipped,
        "coverage_pct": round(total / len(gold_speeches) * 100, 1) if gold_speeches else 0,
    }

    overall_acc = correct / total if total > 0 else 0.0
    report["classification"] = {
        "accuracy": round(overall_acc * 100, 2),
        "correct": correct,
        "total": total,
        "errors": total - correct,
    }

    per_label_metrics = {}
    for label in LABELS:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        per_label_metrics[label] = {
            "precision": round(p * 100, 1),
            "recall": round(r * 100, 1),
            "f1": round(f1 * 100, 1),
            "support": tp + fn,
        }
    report["per_label"] = per_label_metrics

    matrix = {}
    for exp in LABELS:
        row = {}
        for pred_label in LABELS:
            row[pred_label] = confusion[exp][pred_label]
        matrix[exp] = row
    report["confusion_matrix"] = matrix

    topic_p, topic_r, topic_f1 = precision_recall_f1(topic_tp_total, topic_fp_total, topic_fn_total)
    report["topic_attribution"] = {
        "precision": round(topic_p * 100, 1),
        "recall": round(topic_r * 100, 1),
        "f1": round(topic_f1 * 100, 1),
    }

    if law_stats["expected"] > 0:
        exact_rate = (law_stats["exact_match"] / law_stats["expected"]) * 100
        partial_rate = ((law_stats["exact_match"] + law_stats["partial_match"]) / law_stats["expected"]) * 100
    else:
        exact_rate = 0.0
        partial_rate = 0.0
    report["law_attribution"] = {
        **law_stats,
        "exact_match_pct": round(exact_rate, 1),
        "exact_or_partial_pct": round(partial_rate, 1),
    }

    conf_calibration = {}
    for bucket in ["0.80-1.00", "0.65-0.79", "0.50-0.64", "0.00-0.49"]:
        items = conf_buckets.get(bucket, [])
        if items:
            acc = sum(items) / len(items) * 100
            conf_calibration[bucket] = {"count": len(items), "accuracy": round(acc, 1)}
        else:
            conf_calibration[bucket] = {"count": 0, "accuracy": 0.0}
    report["confidence_calibration"] = conf_calibration

    report["mismatches"] = mismatches

    return report


# ── Display ──────────────────────────────────────────────────────────────

def print_report(report: dict, verbose: bool = False) -> None:
    cov = report["coverage"]
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"\n  Coverage: {cov['evaluated']}/{cov['gold_total']} speeches evaluated ({cov['coverage_pct']}%)")
    if cov["skipped_no_prediction"] > 0:
        print(f"  Skipped (no prediction in DB): {cov['skipped_no_prediction']}")

    cls = report["classification"]
    print(f"\n── Classification Accuracy ──────────────────────────────────")
    print(f"  Overall: {cls['accuracy']}% ({cls['correct']}/{cls['total']})")
    print(f"  Errors:  {cls['errors']}")

    print(f"\n── Per-Label Metrics ────────────────────────────────────────")
    print(f"  {'Label':<20s} {'Precision':>9s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print(f"  {'─'*56}")
    for label in LABELS:
        m = report["per_label"][label]
        print(f"  {label:<20s} {m['precision']:>8.1f}% {m['recall']:>7.1f}% {m['f1']:>7.1f}% {m['support']:>7d}")

    print(f"\n── Confusion Matrix ────────────────────────────────────────")
    header = "expected \\ predicted"
    print(f"  {header:<20s}", end="")
    for label in LABELS:
        print(f" {label[:12]:>12s}", end="")
    print()
    print(f"  {'─'*56}")
    matrix = report["confusion_matrix"]
    for exp in LABELS:
        print(f"  {exp:<20s}", end="")
        for pred_label in LABELS:
            val = matrix[exp][pred_label]
            marker = "" if exp == pred_label else " ←" if val > 0 else ""
            print(f" {val:>11d}{marker}", end="" if not marker else "")
            if marker:
                pass
            else:
                print(end="")
        print()

    topic = report["topic_attribution"]
    print(f"\n── Topic Attribution ───────────────────────────────────────")
    print(f"  Precision: {topic['precision']}%  Recall: {topic['recall']}%  F1: {topic['f1']}%")

    law = report["law_attribution"]
    print(f"\n── Law/Amendment Attribution ───────────────────────────────")
    print(f"  Expected law references: {law['expected']}")
    print(f"  Exact match:    {law['exact_match']} ({law['exact_match_pct']}%)")
    print(f"  Partial match:  {law['partial_match']} (cumulative: {law['exact_or_partial_pct']}%)")
    print(f"  Missed:         {law['missed']}")

    cal = report["confidence_calibration"]
    print(f"\n── Confidence Calibration ──────────────────────────────────")
    print(f"  {'Bucket':<12s} {'Count':>6s} {'Accuracy':>10s}")
    print(f"  {'─'*30}")
    for bucket in ["0.80-1.00", "0.65-0.79", "0.50-0.64", "0.00-0.49"]:
        c = cal[bucket]
        print(f"  {bucket:<12s} {c['count']:>6d} {c['accuracy']:>9.1f}%")

    mismatches = report["mismatches"]
    if mismatches:
        print(f"\n── Mismatches ({len(mismatches)}) ──────────────────────────────────────")
        for m in mismatches:
            print(f"\n  id={m['id']} [{m['difficulty']}] {m['speaker'][:35]}")
            print(f"    expected={m['expected']}  predicted={m['predicted']}  conf={m['confidence']:.2f}")
            print(f"    text: {_display_preview(m['text_preview'])}")
            if verbose:
                if m["gold_notes"]:
                    print(f"    gold notes: {_display_preview(m['gold_notes'], 200)}")
                if m["pred_reasoning"]:
                    print(f"    pred reasoning: {_display_preview(m['pred_reasoning'])}")

    print(f"\n{'='*70}")

    target_cls = 98.0
    target_law = 95.0
    cls_gap = target_cls - cls["accuracy"]
    law_gap = target_law - law["exact_or_partial_pct"]
    print(f"  Target classification accuracy: {target_cls}%  →  gap: {cls_gap:+.1f}pp")
    print(f"  Target law attribution:         {target_law}%  →  gap: {law_gap:+.1f}pp")
    print(f"{'='*70}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline accuracy against gold standard.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--gold-path", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed mismatch info")
    parser.add_argument("--only-hard", action="store_true", help="Evaluate only hard+medium cases")
    parser.add_argument("--json", action="store_true", help="Output raw JSON report")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    gold_path = Path(args.gold_path)

    if not gold_path.exists():
        print(f"ERROR: Gold standard not found: {gold_path}")
        return 1
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1

    gold_speeches = load_gold(gold_path)
    predictions, session_topics = load_predictions(db_path, gold_speeches)

    report = evaluate(
        gold_speeches=gold_speeches,
        predictions=predictions,
        session_topics=session_topics,
        verbose=args.verbose,
        only_hard=args.only_hard,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
