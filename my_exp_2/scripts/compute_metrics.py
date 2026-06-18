import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-edit-dir", default=os.path.join("my_exp_2", "outputs", "single_edit", "qwen25_3b_lora_rome_memit"))
    parser.add_argument("--baseline-dir", default=os.path.join("my_exp_2", "outputs", "baseline"))
    parser.add_argument("--output-dir", default=os.path.join("my_exp_2", "outputs", "metrics"))
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


def flatten_single_edit_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in summary.get("method_summaries", []):
        rows.append(
            {
                "method": row.get("method"),
                "successful_cases": row.get("successful_cases"),
                "total_cases": row.get("total_cases"),
                "success_rate": row.get("success_rate"),
                "reliability": row.get("mean_reliability"),
                "generalization": row.get("mean_generalization"),
                "reverse": row.get("mean_reverse"),
                "neighbor": row.get("mean_neighbor"),
                "fact_locality": row.get("mean_fact_locality"),
                "global_locality": row.get("mean_global_locality"),
                "domain_score": row.get("mean_domain_score"),
                "edit_quality": row.get("mean_edit_quality"),
                "time_sec": row.get("mean_time_sec"),
                "peak_gpu_gb": row.get("mean_peak_gpu_gb"),
                "target_old_sources": json.dumps(row.get("target_old_sources", {}), ensure_ascii=False),
            }
        )
    return rows


def collect_failed_cases(single_edit_dir: str) -> List[Dict[str, Any]]:
    failed = []
    for path in sorted(glob.glob(os.path.join(single_edit_dir, "*", "case_*.json"))):
        payload = load_json(path)
        if payload.get("status") != "success":
            failed.append(
                {
                    "method": payload.get("method"),
                    "fact_id": payload.get("fact_id"),
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
    single_summary_path = os.path.join(args.single_edit_dir, "summary.json")
    baseline_summary = load_json(baseline_summary_path) if os.path.exists(baseline_summary_path) else None
    single_summary = load_json(single_summary_path) if os.path.exists(single_summary_path) else None

    payload = {
        "baseline": baseline_summary,
        "single_edit": single_summary,
        "failed_cases": collect_failed_cases(args.single_edit_dir) if os.path.exists(args.single_edit_dir) else [],
    }
    write_json(os.path.join(args.output_dir, "metrics_summary.json"), payload)

    if single_summary:
        rows = flatten_single_edit_summary(single_summary)
        write_csv(os.path.join(args.output_dir, "single_edit_summary.csv"), rows)
        write_json(os.path.join(args.output_dir, "single_edit_summary.json"), rows)

    print(json.dumps({"output_dir": args.output_dir, "has_baseline": baseline_summary is not None, "has_single_edit": single_summary is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
