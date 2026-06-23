import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequential-edit-dir",
        default=os.path.join("my_exp_2", "outputs", "sequential_edit", "qwen25_3b_seq"),
    )
    parser.add_argument("--baseline-dir", default=os.path.join("my_exp_2", "outputs", "baseline"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join("my_exp_2", "outputs", "metrics", "sequential", "qwen25_3b_seq"),
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_sequential_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in summary.get("method_summaries", []):
        rows.append(
            {
                "method": row.get("method"),
                "successful_steps": row.get("successful_steps"),
                "total_steps": row.get("total_steps"),
                "success_rate": row.get("success_rate"),
                "current_reliability": row.get("mean_current_reliability"),
                "current_generalization": row.get("mean_current_generalization"),
                "retention": row.get("mean_retention"),
                "global_locality_generation": row.get("mean_global_locality"),
                "domain_score_generation": row.get("mean_domain_score"),
                "sequential_quality": row.get("mean_sequential_quality"),
                "time_sec": row.get("mean_time_sec"),
                "peak_gpu_gb": row.get("mean_peak_gpu_gb"),
                "target_old_sources": json.dumps(row.get("target_old_sources", {}), ensure_ascii=False),
            }
        )
    return rows


def step_metric(payload: Dict[str, Any], key: str) -> Optional[float]:
    return (payload.get("sequential_metrics") or {}).get(key)


def collect_step_rows(sequential_edit_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(sequential_edit_dir, "*", "step_*.json"))):
        payload = load_json(path)
        target_old = payload.get("target_old_resolution") or {}
        global_eval = payload.get("global_eval") or {}
        rows.append(
            {
                "method": payload.get("method"),
                "fact_id": payload.get("fact_id"),
                "step_index": payload.get("step_index"),
                "status": payload.get("status"),
                "time_sec": payload.get("time_sec"),
                "peak_gpu_gb": ((payload.get("gpu") or {}).get("peak_allocated_gb")),
                "target_old_source": target_old.get("target_old_source"),
                "target_old_is_valid": target_old.get("target_old_is_valid"),
                "target_old_is_stable": target_old.get("target_old_is_stable"),
                "accepted_quality_score": target_old.get("accepted_quality_score"),
                "current_reliability": step_metric(payload, "current_reliability"),
                "current_generalization": step_metric(payload, "current_generalization"),
                "current_reverse": step_metric(payload, "current_reverse"),
                "current_neighbor": step_metric(payload, "current_neighbor"),
                "current_fact_locality": step_metric(payload, "current_fact_locality"),
                "retention": step_metric(payload, "retention"),
                "retained_generalization": step_metric(payload, "retained_generalization"),
                "global_locality_generation": step_metric(payload, "global_locality_generation"),
                "domain_score_generation": step_metric(payload, "domain_score_generation"),
                "sequential_quality": step_metric(payload, "sequential_quality"),
                "general_questions_total": ((global_eval.get("general_questions") or {}).get("total")),
                "general_questions_accuracy": ((global_eval.get("general_questions") or {}).get("accuracy")),
                "domain_questions_total": ((global_eval.get("domain_questions") or {}).get("total")),
                "domain_questions_accuracy": ((global_eval.get("domain_questions") or {}).get("accuracy")),
                "path": path,
            }
        )
    return rows


def collect_failed_steps(sequential_edit_dir: str) -> List[Dict[str, Any]]:
    failed = []
    for path in sorted(glob.glob(os.path.join(sequential_edit_dir, "*", "step_*.json"))):
        payload = load_json(path)
        if payload.get("status") != "success":
            failed.append(
                {
                    "method": payload.get("method"),
                    "fact_id": payload.get("fact_id"),
                    "step_index": payload.get("step_index"),
                    "error_type": (payload.get("error") or {}).get("type"),
                    "error_message": (payload.get("error") or {}).get("message"),
                    "path": path,
                }
            )
    return failed


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    baseline_summary_path = os.path.join(args.baseline_dir, "summary.json")
    sequential_summary_path = os.path.join(args.sequential_edit_dir, "summary.json")
    baseline_summary = load_json(baseline_summary_path) if os.path.exists(baseline_summary_path) else None
    sequential_summary = load_json(sequential_summary_path) if os.path.exists(sequential_summary_path) else None

    payload = {
        "baseline": baseline_summary,
        "sequential_edit": sequential_summary,
        "failed_steps": collect_failed_steps(args.sequential_edit_dir) if os.path.exists(args.sequential_edit_dir) else [],
    }
    write_json(os.path.join(args.output_dir, "metrics_summary.json"), payload)

    if sequential_summary:
        rows = flatten_sequential_summary(sequential_summary)
        write_csv(os.path.join(args.output_dir, "sequential_summary.csv"), rows)
        write_json(os.path.join(args.output_dir, "sequential_summary.json"), rows)
        step_rows = collect_step_rows(args.sequential_edit_dir)
        write_csv(os.path.join(args.output_dir, "step_metrics.csv"), step_rows)
        write_json(os.path.join(args.output_dir, "step_metrics.json"), step_rows)

    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "has_baseline": baseline_summary is not None,
                "has_sequential_edit": sequential_summary is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
