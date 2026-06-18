import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequential-edit-dir",
        default=os.path.join("my_exp_2", "outputs", "sequential_edit", "qwen25_3b_seq"),
    )
    parser.add_argument("--baseline-dir", default=os.path.join("my_exp_2", "outputs", "baseline"))
    parser.add_argument(
        "--output",
        default=os.path.join("my_exp_2", "outputs", "metrics", "sequential", "qwen25_3b_seq", "report.md"),
    )
    parser.add_argument("--num-steps", type=int, default=5)
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optional(path: str) -> Optional[Dict[str, Any]]:
    return load_json(path) if os.path.exists(path) else None


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_baseline_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Baseline", ""]
    if not summary:
        return "\n".join(lines + ["Baseline results were not found.", ""])
    lines.extend(["| Dataset | Questions | Accuracy | Time (s) |", "|---|---:|---:|---:|"])
    for name, row in summary.get("datasets", {}).items():
        lines.append(f"| {name} | {row.get('total')} | {fmt(row.get('accuracy'))} | {fmt(row.get('time_sec'))} |")
    lines.append("")
    return "\n".join(lines)


def build_sequential_summary_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Sequential Edit Summary", ""]
    if not summary:
        return "\n".join(lines + ["Sequential-edit results were not found.", ""])
    generation_eval = summary.get("generation_eval") or {}
    runtime_options = summary.get("runtime_options") or {}
    lines.extend(
        [
            f"- Model: `{summary.get('model')}`",
            f"- Data dir: `{summary.get('data_dir')}`",
            f"- Eval scope: `{summary.get('eval_scope')}`",
            f"- Retention fact limit: `{generation_eval.get('retention_fact_limit')}`",
            f"- General limit: `{generation_eval.get('general_limit')}`",
            f"- Domain limit: `{generation_eval.get('domain_limit')}`",
            f"- Allow CPU: `{runtime_options.get('allow_cpu')}`",
            f"- Save generation details: `{runtime_options.get('save_generation_details')}`",
            "",
            "| Method | Success | Current rel | Current gen | Retention | Global locality | Domain score | Sequential quality | Time (s) | GPU (GB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("method_summaries", []):
        lines.append(
            "| {method} | {success}/{total} | {rel} | {gen} | {ret} | {glob} | {domain} | {quality} | {time} | {gpu} |".format(
                method=row.get("method"),
                success=row.get("successful_steps"),
                total=row.get("total_steps"),
                rel=fmt(row.get("mean_current_reliability")),
                gen=fmt(row.get("mean_current_generalization")),
                ret=fmt(row.get("mean_retention")),
                glob=fmt(row.get("mean_global_locality")),
                domain=fmt(row.get("mean_domain_score")),
                quality=fmt(row.get("mean_sequential_quality")),
                time=fmt(row.get("mean_time_sec")),
                gpu=fmt(row.get("mean_peak_gpu_gb")),
            )
        )
    lines.extend(["", "### Target Old Resolution", "", "| Method | Sources |", "|---|---|"])
    for row in summary.get("method_summaries", []):
        sources = row.get("target_old_sources") or {}
        rendered = ", ".join(f"{key}: {value}" for key, value in sorted(sources.items())) or "-"
        lines.append(f"| {row.get('method')} | {rendered} |")
    lines.append("")
    return "\n".join(lines)


def load_step_results(sequential_edit_dir: str) -> Dict[Tuple[int, str], Dict[str, Dict[str, Any]]]:
    by_step: Dict[Tuple[int, str], Dict[str, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(sequential_edit_dir, "*", "step_*.json"))):
        payload = load_json(path)
        step_index = int(payload.get("step_index") or 0)
        fact_id = str(payload.get("fact_id"))
        method = str(payload.get("method") or os.path.basename(os.path.dirname(path)))
        by_step.setdefault((step_index, fact_id), {})[method] = payload
    return by_step


def choose_steps(by_step: Dict[Tuple[int, str], Dict[str, Dict[str, Any]]], num_steps: int) -> List[Tuple[int, str]]:
    scored = []
    for step_key, method_map in by_step.items():
        values = []
        for payload in method_map.values():
            metrics = payload.get("sequential_metrics") or {}
            value = metrics.get("sequential_quality")
            if value is not None:
                values.append(value)
        if values:
            scored.append((max(values) - min(values), step_key))
    scored.sort(reverse=True)
    return [step_key for _, step_key in scored[:num_steps]]


def seq_metric(payload: Dict[str, Any], key: str) -> Optional[float]:
    return (payload.get("sequential_metrics") or {}).get(key)


def build_steps_section(sequential_edit_dir: str, methods: List[str], num_steps: int) -> str:
    by_step = load_step_results(sequential_edit_dir)
    lines = ["## Selected Steps", ""]
    if not by_step:
        return "\n".join(lines + ["No step results found.", ""])

    for step_index, fact_id in choose_steps(by_step, num_steps):
        method_map = by_step[(step_index, fact_id)]
        reference = next(iter(method_map.values()))
        edit_request = reference.get("input_edit_request") or {}
        target_old = reference.get("target_old_resolution") or {}
        lines.extend(
            [
                f"### Step {step_index}: {fact_id}",
                "",
                f"- Prompt: `{edit_request.get('prompt')}`",
                f"- Subject: `{edit_request.get('subject')}`",
                f"- Target new: `{edit_request.get('target_new')}`",
                f"- Target old source: `{target_old.get('target_old_source')}`",
                f"- Raw model answer: `{target_old.get('raw_model_answer')}`",
                "",
                "| Method | Status | Current rel | Current gen | Retention | Global locality | Domain score | Sequential quality | Time (s) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            payload = method_map.get(method) or method_map.get(method.lower())
            if not payload:
                lines.append(f"| {method} | missing | - | - | - | - | - | - | - |")
                continue
            lines.append(
                "| {method} | {status} | {rel} | {gen} | {ret} | {glob} | {domain} | {quality} | {time} |".format(
                    method=method,
                    status=payload.get("status"),
                    rel=fmt(seq_metric(payload, "current_reliability")),
                    gen=fmt(seq_metric(payload, "current_generalization")),
                    ret=fmt(seq_metric(payload, "retention")),
                    glob=fmt(seq_metric(payload, "global_locality_generation")),
                    domain=fmt(seq_metric(payload, "domain_score_generation")),
                    quality=fmt(seq_metric(payload, "sequential_quality")),
                    time=fmt(payload.get("time_sec")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def build_errors_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Errors", ""]
    if not summary:
        return "\n".join(lines + ["No sequential summary found.", ""])
    any_errors = False
    for row in summary.get("method_summaries", []):
        failed = row.get("failed_fact_ids") or []
        if failed:
            any_errors = True
            lines.append(f"- `{row.get('method')}` failed steps: {', '.join(failed)}")
    if not any_errors:
        lines.append("No failed steps in summary.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sequential_summary = load_optional(os.path.join(args.sequential_edit_dir, "summary.json"))
    baseline_summary = load_optional(os.path.join(args.baseline_dir, "summary.json"))
    methods = sequential_summary.get("methods", []) if sequential_summary else []
    limitations = (sequential_summary or {}).get("limitations") or [
        "Sequential evaluation uses generation-based retention/domain/general checks.",
        "Current edit quality still comes from EasyEdit internal metrics for the current fact.",
        "This MVP covers sequential LoRA/ROME/MEMIT only.",
    ]

    lines = [
        "# OilGas Sequential Knowledge Editing Report",
        "",
        build_baseline_section(baseline_summary),
        build_sequential_summary_section(sequential_summary),
        build_steps_section(args.sequential_edit_dir, methods, args.num_steps),
        build_errors_section(sequential_summary),
        "## Current Limitations",
        "",
    ]
    lines.extend(f"- {line}" for line in limitations)
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
