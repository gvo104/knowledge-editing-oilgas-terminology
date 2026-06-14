import argparse
import glob
import json
import os
from statistics import mean
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Benchmark result directory with summary.json and method subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path. Defaults to <result-dir>/report.md.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="How many highlighted cases to include when case ids are not specified.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Explicit case ids to include in the detailed section.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def safe_case_id(case_id: Any) -> int:
    return int(case_id)


def load_case_results(result_dir: str, methods: List[str]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    by_case: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for method in methods:
        method_dir = os.path.join(result_dir, method.lower())
        for path in sorted(glob.glob(os.path.join(method_dir, "case_*.json"))):
            payload = load_json(path)
            case_id = safe_case_id(payload["case_id"])
            by_case.setdefault(case_id, {})[method] = payload
    return by_case


def metric_value(case_payload: Dict[str, Any], key: str) -> Optional[float]:
    return case_payload.get("normalized_metrics", {}).get(key)


def case_prompt(case_payload: Dict[str, Any]) -> str:
    return case_payload.get("input_case", {}).get("prompt", "")


def case_subject(case_payload: Dict[str, Any]) -> str:
    return case_payload.get("input_case", {}).get("subject", "")


def case_ground_truth(case_payload: Dict[str, Any]) -> str:
    return case_payload.get("input_case", {}).get("ground_truth", "")


def case_target_new(case_payload: Dict[str, Any]) -> str:
    return case_payload.get("input_case", {}).get("target_new", "")


def choose_case_ids(
    cases: Dict[int, Dict[str, Dict[str, Any]]],
    methods: List[str],
    explicit_case_ids: Optional[List[str]],
    num_cases: int,
) -> List[int]:
    if explicit_case_ids:
        return [int(case_id) for case_id in explicit_case_ids]

    scored: List[tuple[float, int]] = []
    for case_id, method_map in cases.items():
        rewrite_values = [
            metric_value(method_map[method], "post_rewrite_acc")
            for method in methods
            if method in method_map
        ]
        locality_values = [
            metric_value(method_map[method], "post_locality_acc")
            for method in methods
            if method in method_map
        ]
        rewrite_values = [value for value in rewrite_values if value is not None]
        locality_values = [value for value in locality_values if value is not None]
        if not rewrite_values:
            continue
        rewrite_spread = max(rewrite_values) - min(rewrite_values)
        locality_spread = (max(locality_values) - min(locality_values)) if locality_values else 0.0
        score = rewrite_spread + 0.5 * locality_spread
        scored.append((score, case_id))

    scored.sort(reverse=True)
    selected = [case_id for _, case_id in scored[:num_cases]]
    return selected


def build_summary_table(method_summaries: List[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Success | Rewrite | Rephrase | Portability | Locality | Time (s) | Peak GPU (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_summaries:
        lines.append(
            "| {method} | {success} | {rewrite} | {rephrase} | {portability} | {locality} | {time_sec} | {gpu} |".format(
                method=row["method"],
                success=f'{row["successful_cases"]}/{row["total_cases"]}',
                rewrite=fmt(row.get("mean_post_rewrite_acc")),
                rephrase=fmt(row.get("mean_post_rephrase_acc")),
                portability=fmt(row.get("mean_post_portability_acc")),
                locality=fmt(row.get("mean_post_locality_acc")),
                time_sec=fmt(row.get("mean_time_sec")),
                gpu=fmt(row.get("mean_peak_gpu_gb")),
            )
        )
    return "\n".join(lines)


def best_method_line(method_summaries: List[Dict[str, Any]], metric_key: str, label: str) -> str:
    rows = [row for row in method_summaries if row.get(metric_key) is not None]
    if not rows:
        return f"- {label}: n/a"
    best = max(rows, key=lambda row: row[metric_key])
    return f"- {label}: `{best['method']}` ({fmt(best[metric_key])})"


def fastest_method_line(method_summaries: List[Dict[str, Any]]) -> str:
    rows = [row for row in method_summaries if row.get("mean_time_sec") is not None]
    if not rows:
        return "- Fastest method: n/a"
    best = min(rows, key=lambda row: row["mean_time_sec"])
    return f"- Fastest method: `{best['method']}` ({fmt(best['mean_time_sec'])} s)"


def build_case_section(case_id: int, method_map: Dict[str, Dict[str, Any]], methods: List[str]) -> str:
    reference_payload = next(iter(method_map.values()))
    lines = [
        f"## Case {case_id}",
        "",
        f"- Prompt: `{case_prompt(reference_payload)}`",
        f"- Subject: `{case_subject(reference_payload)}`",
        f"- Ground truth: `{case_ground_truth(reference_payload)}`",
        f"- Target new: `{case_target_new(reference_payload)}`",
        "",
        "| Method | Rewrite | Rephrase | Portability | Locality | Time (s) | Peak GPU (GB) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for method in methods:
        payload = method_map.get(method)
        if payload is None:
            lines.append(f"| {method} | - | - | - | - | - | - | missing |")
            continue
        nm = payload.get("normalized_metrics", {})
        gpu = payload.get("gpu", {}) or {}
        lines.append(
            "| {method} | {rewrite} | {rephrase} | {portability} | {locality} | {time_sec} | {gpu_peak} | {status} |".format(
                method=method,
                rewrite=fmt(nm.get("post_rewrite_acc")),
                rephrase=fmt(nm.get("post_rephrase_acc")),
                portability=fmt(nm.get("post_portability_acc")),
                locality=fmt(nm.get("post_locality_acc")),
                time_sec=fmt(payload.get("time_sec")),
                gpu_peak=fmt(gpu.get("peak_allocated_gb")),
                status=payload.get("status", "unknown"),
            )
        )

    observations = []
    rewrite_rank = []
    locality_rank = []
    for method in methods:
        payload = method_map.get(method)
        if not payload:
            continue
        rewrite = payload.get("normalized_metrics", {}).get("post_rewrite_acc")
        locality = payload.get("normalized_metrics", {}).get("post_locality_acc")
        if rewrite is not None:
            rewrite_rank.append((rewrite, method))
        if locality is not None:
            locality_rank.append((locality, method))

    if rewrite_rank:
        rewrite_rank.sort(reverse=True)
        observations.append(f"Best rewrite: `{rewrite_rank[0][1]}` ({fmt(rewrite_rank[0][0])})")
    if locality_rank:
        locality_rank.sort(reverse=True)
        observations.append(f"Best locality: `{locality_rank[0][1]}` ({fmt(locality_rank[0][0])})")

    if observations:
        lines.extend(["", "- " + "\n- ".join(observations)])

    lines.append("")
    return "\n".join(lines)


def build_report(result_dir: str, summary: Dict[str, Any], case_ids: List[int], cases: Dict[int, Dict[str, Dict[str, Any]]]) -> str:
    method_summaries = summary["method_summaries"]
    methods = summary["methods"]

    lines = [
        "# Benchmark Report",
        "",
        f"- Result dir: `{result_dir}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Cases loaded: `{summary['cases_loaded']}`",
        f"- Process model: `{summary.get('process_model', 'n/a')}`",
        "",
        "## Summary",
        "",
        build_summary_table(method_summaries),
        "",
        "## Quick Findings",
        "",
        best_method_line(method_summaries, "mean_post_rewrite_acc", "Best rewrite"),
        best_method_line(method_summaries, "mean_post_rephrase_acc", "Best rephrase"),
        best_method_line(method_summaries, "mean_post_portability_acc", "Best portability"),
        best_method_line(method_summaries, "mean_post_locality_acc", "Best locality"),
        fastest_method_line(method_summaries),
        "",
        "## Selected Cases",
        "",
    ]

    for case_id in case_ids:
        method_map = cases.get(case_id)
        if not method_map:
            continue
        lines.append(build_case_section(case_id, method_map, methods))

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_path = args.output or os.path.join(args.result_dir, "report.md")

    summary = load_json(os.path.join(args.result_dir, "summary.json"))
    methods = summary["methods"]
    cases = load_case_results(args.result_dir, methods)
    case_ids = choose_case_ids(cases, methods, args.case_ids, args.num_cases)

    report = build_report(args.result_dir, summary, case_ids, cases)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote markdown report to {output_path}")
    print(f"Included case ids: {case_ids}")


if __name__ == "__main__":
    main()
