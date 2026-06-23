import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-edit-dir", default=os.path.join("my_exp_2", "outputs", "single_edit", "qwen25_3b_lora_rome_memit"))
    parser.add_argument("--baseline-dir", default=os.path.join("my_exp_2", "outputs", "baseline"))
    parser.add_argument("--output", default=os.path.join("my_exp_2", "outputs", "metrics", "report.md"))
    parser.add_argument("--num-cases", type=int, default=5)
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


def load_optional(path: str) -> Optional[Dict[str, Any]]:
    return load_json(path) if os.path.exists(path) else None


def build_baseline_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Baseline", ""]
    if not summary:
        return "\n".join(lines + ["Baseline results were not found.", ""])
    lines.extend(
        [
            "| Dataset | Questions | Accuracy | Time (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in summary.get("datasets", {}).items():
        lines.append(f"| {name} | {row.get('total')} | {fmt(row.get('accuracy'))} | {fmt(row.get('time_sec'))} |")
    lines.append("")
    return "\n".join(lines)


def build_single_summary_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Single Edit Summary", ""]
    if not summary:
        return "\n".join(lines + ["Single-edit results were not found.", ""])
    lines.extend(
        [
            f"- Model: `{summary.get('model')}`",
            f"- Data dir: `{summary.get('data_dir')}`",
            f"- Eval scope: `{summary.get('eval_scope')}`",
            "",
            "| Method | Success | Reliability | Generalization | Reverse | Neighbor | Fact locality | Global locality | Domain | Edit quality | Time (s) | GPU (GB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("method_summaries", []):
        lines.append(
            "| {method} | {success}/{total} | {rel} | {gen} | {rev} | {nei} | {fact_loc} | {glob_loc} | {domain} | {quality} | {time} | {gpu} |".format(
                method=row.get("method"),
                success=row.get("successful_cases"),
                total=row.get("total_cases"),
                rel=fmt(row.get("mean_reliability")),
                gen=fmt(row.get("mean_generalization")),
                rev=fmt(row.get("mean_reverse")),
                nei=fmt(row.get("mean_neighbor")),
                fact_loc=fmt(row.get("mean_fact_locality")),
                glob_loc=fmt(row.get("mean_global_locality")),
                domain=fmt(row.get("mean_domain_score")),
                quality=fmt(row.get("mean_edit_quality")),
                time=fmt(row.get("mean_time_sec")),
                gpu=fmt(row.get("mean_peak_gpu_gb")),
            )
        )
    lines.extend(["", "### Target Old Resolution", ""])
    lines.extend(["| Method | Sources |", "|---|---|"])
    for row in summary.get("method_summaries", []):
        sources = row.get("target_old_sources") or {}
        rendered = ", ".join(f"{key}: {value}" for key, value in sorted(sources.items())) or "-"
        lines.append(f"| {row.get('method')} | {rendered} |")
    lines.append("")
    return "\n".join(lines)


def metric(payload: Dict[str, Any], key: str) -> Optional[float]:
    return (payload.get("metrics") or {}).get(key)


def load_case_results(single_edit_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_fact: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(single_edit_dir, "*", "case_*.json"))):
        payload = load_json(path)
        fact_id = str(payload.get("fact_id"))
        method = str(payload.get("method") or os.path.basename(os.path.dirname(path)))
        by_fact.setdefault(fact_id, {})[method] = payload
    return by_fact


def choose_cases(by_fact: Dict[str, Dict[str, Dict[str, Any]]], num_cases: int) -> List[str]:
    scored = []
    for fact_id, method_map in by_fact.items():
        values = [metric(payload, "edit_quality") for payload in method_map.values()]
        values = [value for value in values if value is not None]
        if values:
            scored.append((max(values) - min(values), fact_id))
    scored.sort(reverse=True)
    return [fact_id for _, fact_id in scored[:num_cases]]


def build_cases_section(single_edit_dir: str, methods: List[str], num_cases: int) -> str:
    by_fact = load_case_results(single_edit_dir)
    lines = ["## Selected Cases", ""]
    if not by_fact:
        return "\n".join(lines + ["No case results found.", ""])

    for fact_id in choose_cases(by_fact, num_cases):
        method_map = by_fact[fact_id]
        reference = next(iter(method_map.values()))
        edit_request = reference.get("input_edit_request") or {}
        lines.extend(
            [
                f"### {fact_id}",
                "",
                f"- Prompt: `{edit_request.get('prompt')}`",
                f"- Subject: `{edit_request.get('subject')}`",
                f"- Target new: `{edit_request.get('target_new')}`",
            "",
                f"- Target old source: `{(reference.get('target_old_resolution') or {}).get('target_old_source')}`",
                f"- Raw model answer: `{(reference.get('target_old_resolution') or {}).get('raw_model_answer')}`",
                "",
                "| Method | Status | Reliability | Generalization | Reverse | Global locality | Edit quality | Time (s) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            payload = method_map.get(method) or method_map.get(method.lower())
            if not payload:
                lines.append(f"| {method} | missing | - | - | - | - | - | - |")
                continue
            lines.append(
                "| {method} | {status} | {rel} | {gen} | {rev} | {glob} | {quality} | {time} |".format(
                    method=method,
                    status=payload.get("status"),
                    rel=fmt(metric(payload, "reliability")),
                    gen=fmt(metric(payload, "generalization")),
                    rev=fmt(metric(payload, "reverse")),
                    glob=fmt(metric(payload, "global_locality")),
                    quality=fmt(metric(payload, "edit_quality")),
                    time=fmt(payload.get("time_sec")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def build_errors_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Errors", ""]
    if not summary:
        return "\n".join(lines + ["No single-edit summary found.", ""])
    any_errors = False
    for row in summary.get("method_summaries", []):
        failed = row.get("failed_fact_ids") or []
        if failed:
            any_errors = True
            lines.append(f"- `{row.get('method')}` failed cases: {', '.join(failed)}")
    if not any_errors:
        lines.append("No failed cases in summary.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    single_summary = load_optional(os.path.join(args.single_edit_dir, "summary.json"))
    baseline_summary = load_optional(os.path.join(args.baseline_dir, "summary.json"))
    methods = single_summary.get("methods", []) if single_summary else []

    lines = [
        "# OilGas Knowledge Editing Report",
        "",
        build_baseline_section(baseline_summary),
        build_single_summary_section(single_summary),
        build_cases_section(args.single_edit_dir, methods, args.num_cases),
        build_errors_section(single_summary),
        "## Current Limitations",
        "",
        "- This MVP covers single-edit experiments only.",
        "- Post-edit scoring is based on EasyEdit internal metrics.",
        "- SFT and LocFT-BF are planned as follow-up stages.",
        "",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
