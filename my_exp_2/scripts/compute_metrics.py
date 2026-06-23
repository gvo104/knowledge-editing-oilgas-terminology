import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

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


def metric(payload: Dict[str, Any], key: str) -> Optional[float]:
    return (payload.get("metrics") or {}).get(key)


def target_old_mode(payload: Dict[str, Any]) -> str:
    source = (payload.get("target_old_resolution") or {}).get("target_old_source")
    if source == "model_current_answer":
        return "replacement"
    if source == "fallback_to_target_new":
        return "insertion"
    return str(source or "unknown")


def collect_case_rows(single_edit_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(single_edit_dir, "*", "case_*.json"))):
        payload = load_json(path)
        fact = payload.get("fact") or {}
        target_old = payload.get("target_old_resolution") or {}
        gpu = payload.get("gpu") or {}
        rows.append(
            {
                "method": payload.get("method"),
                "fact_id": payload.get("fact_id"),
                "status": payload.get("status"),
                "subject": fact.get("subject"),
                "relation": fact.get("relation"),
                "object": fact.get("object"),
                "level": fact.get("level"),
                "subject_len": len(str(fact.get("subject") or "")),
                "target_len": len(str(fact.get("object") or "")),
                "target_old_source": target_old.get("target_old_source"),
                "edit_mode": target_old_mode(payload),
                "target_old_is_valid": target_old.get("target_old_is_valid"),
                "target_old_is_stable": target_old.get("target_old_is_stable"),
                "accepted_quality_score": target_old.get("accepted_quality_score"),
                "reliability": metric(payload, "reliability"),
                "generalization": metric(payload, "generalization"),
                "reverse": metric(payload, "reverse"),
                "neighbor": metric(payload, "neighbor"),
                "fact_locality": metric(payload, "fact_locality"),
                "global_locality": metric(payload, "global_locality"),
                "domain_score": metric(payload, "domain_score"),
                "edit_quality": metric(payload, "edit_quality"),
                "time_sec": payload.get("time_sec"),
                "peak_gpu_gb": gpu.get("peak_allocated_gb"),
                "error_type": (payload.get("error") or {}).get("type"),
                "path": path,
            }
        )
    return rows


def mean(values: List[Any]) -> Optional[float]:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 6)


def grouped_analysis(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    result = []
    for key, records in sorted(grouped.items()):
        item = {group_key: value for group_key, value in zip(group_keys, key)}
        item.update(
            {
                "count": len(records),
                "success_count": sum(1 for record in records if record.get("status") == "success"),
                "mean_reliability": mean([record.get("reliability") for record in records]),
                "mean_generalization": mean([record.get("generalization") for record in records]),
                "mean_fact_locality": mean([record.get("fact_locality") for record in records]),
                "mean_global_locality": mean([record.get("global_locality") for record in records]),
                "mean_domain_score": mean([record.get("domain_score") for record in records]),
                "mean_edit_quality": mean([record.get("edit_quality") for record in records]),
                "mean_time_sec": mean([record.get("time_sec") for record in records]),
            }
        )
        result.append(item)
    return result


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
        case_rows = collect_case_rows(args.single_edit_dir)
        write_csv(os.path.join(args.output_dir, "case_metrics.csv"), case_rows)
        write_json(os.path.join(args.output_dir, "case_metrics.json"), case_rows)
        write_csv(os.path.join(args.output_dir, "fact_metrics.csv"), case_rows)
        write_json(os.path.join(args.output_dir, "fact_metrics.json"), case_rows)
        write_csv(os.path.join(args.output_dir, "subject_relation_analysis.csv"), grouped_analysis(case_rows, ["method", "level", "relation"]))
        write_csv(os.path.join(args.output_dir, "replacement_vs_insertion.csv"), grouped_analysis(case_rows, ["method", "edit_mode"]))

    print(json.dumps({"output_dir": args.output_dir, "has_baseline": baseline_summary is not None, "has_single_edit": single_summary is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
