import argparse
import glob
import json
import os
import tempfile
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


def shorten_text(value: Any, limit: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return "-"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_baseline_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Baseline", ""]
    if not summary:
        return "\n".join(lines + ["Baseline results were not found.", ""])
    lines.extend(["| Dataset | Questions | Accuracy | Time (s) |", "|---|---:|---:|---:|"])
    for name, row in summary.get("datasets", {}).items():
        lines.append(f"| {name} | {row.get('total')} | {fmt(row.get('accuracy'))} | {fmt(row.get('time_sec'))} |")
    lines.append("")
    return "\n".join(lines)


def build_methodology_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Evaluation Setup", ""]
    if not summary:
        return "\n".join(lines + ["Sequential-edit results were not found.", ""])
    generation_eval = summary.get("generation_eval") or {}
    runtime_options = summary.get("runtime_options") or {}
    lines.extend(
        [
            f"- Model: `{summary.get('model')}`",
            f"- Data dir: `{summary.get('data_dir')}`",
            f"- Eval scope: `{summary.get('eval_scope')}`",
            f"- GPU-only run: `{not runtime_options.get('allow_cpu', False)}`",
            f"- Retention fact limit: `{generation_eval.get('retention_fact_limit')}`",
            f"- General limit: `{generation_eval.get('general_limit')}`",
            f"- Domain limit: `{generation_eval.get('domain_limit')}`",
            "",
            "Этот sequential-отчет построен для ускоренного режима оценки.",
            "Retention считается не по всем ранее внесенным фактам, а по ограниченной подвыборке `retention_fact_limit`.",
            "Global locality и domain score также считаются не по полным наборам вопросов, а по фиксированным подвыборкам `general_limit` и `domain_limit`.",
            "Это сделано из-за ограничений по времени, VRAM и RAM: полный sequential benchmark на каждом шаге был бы существенно тяжелее.",
            "Поэтому графики и агрегаты корректно отражают динамику именно в этой sample-based конфигурации и подходят для сравнения методов между собой, но не являются полной оценкой сохранения всех прошлых фактов и всех контрольных вопросов.",
            "",
        ]
    )
    return "\n".join(lines)


def build_sequential_summary_section(summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## Sequential Edit Summary", ""]
    if not summary:
        return "\n".join(lines + ["Sequential-edit results were not found.", ""])
    lines.extend(
        [
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


def collect_step_rows(sequential_edit_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(sequential_edit_dir, "*", "step_*.json"))):
        payload = load_json(path)
        target_old = payload.get("target_old_resolution") or {}
        metrics = payload.get("sequential_metrics") or {}
        gpu = payload.get("gpu") or {}
        rows.append(
            {
                "method": str(payload.get("method")),
                "fact_id": str(payload.get("fact_id")),
                "step_index": int(payload.get("step_index") or 0),
                "status": str(payload.get("status")),
                "time_sec": payload.get("time_sec"),
                "peak_gpu_gb": gpu.get("peak_allocated_gb"),
                "target_old_source": target_old.get("target_old_source"),
                "current_reliability": metrics.get("current_reliability"),
                "current_generalization": metrics.get("current_generalization"),
                "retention": metrics.get("retention"),
                "global_locality_generation": metrics.get("global_locality_generation"),
                "domain_score_generation": metrics.get("domain_score_generation"),
                "sequential_quality": metrics.get("sequential_quality"),
            }
        )
    return rows


def build_analysis_section(summary: Optional[Dict[str, Any]], step_rows: List[Dict[str, Any]]) -> str:
    lines = ["## Analysis", ""]
    if not summary:
        return "\n".join(lines + ["Sequential-edit results were not found.", ""])

    rows = summary.get("method_summaries", [])
    if not rows:
        return "\n".join(lines + ["No method summaries found.", ""])

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in step_rows:
        by_method.setdefault(row["method"], []).append(row)

    best_quality = max(rows, key=lambda row: row.get("mean_sequential_quality") or float("-inf"))
    best_retention = max(rows, key=lambda row: row.get("mean_retention") or float("-inf"))
    best_locality = max(rows, key=lambda row: row.get("mean_global_locality") or float("-inf"))
    fastest = min(rows, key=lambda row: row.get("mean_time_sec") or float("inf"))

    lines.append(
        f"- Лучший итоговый метод по `sequential_quality`: `{best_quality.get('method')}` ({fmt(best_quality.get('mean_sequential_quality'))})."
    )
    lines.append(
        f"- Лучший метод по сохранению ранее внесенных фактов в текущей sample-based настройке: `{best_retention.get('method')}` ({fmt(best_retention.get('mean_retention'))})."
    )
    lines.append(
        f"- Лучший метод по глобальной локальности на контрольной подвыборке: `{best_locality.get('method')}` ({fmt(best_locality.get('mean_global_locality'))})."
    )
    lines.append(f"- Самый быстрый метод по среднему времени шага: `{fastest.get('method')}` ({fmt(fastest.get('mean_time_sec'))} s).")
    lines.append("")
    lines.append("| Method | First 5 quality | Last 5 quality | First 5 retention | Last 5 retention | Comment |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for row in rows:
        method = row.get("method")
        steps = sorted(by_method.get(method, []), key=lambda item: item["step_index"])
        first5 = steps[:5]
        last5 = steps[-5:]

        def avg(records: List[Dict[str, Any]], key: str) -> Optional[float]:
            values = [record.get(key) for record in records if record.get(key) is not None]
            if not values:
                return None
            return sum(values) / len(values)

        first_quality = avg(first5, "sequential_quality")
        last_quality = avg(last5, "sequential_quality")
        first_retention = avg(first5, "retention")
        last_retention = avg(last5, "retention")

        comment = "stable"
        if (last_quality or 0.0) == 0.0 and (first_quality or 0.0) > 0.0:
            comment = "degrades to zero on later steps"
        elif (last_quality or 0.0) < (first_quality or 0.0) * 0.6:
            comment = "noticeable quality decay"
        elif (last_retention or 0.0) < (first_retention or 0.0) * 0.7:
            comment = "retention drops over time"

        lines.append(
            f"| {method} | {fmt(first_quality)} | {fmt(last_quality)} | {fmt(first_retention)} | {fmt(last_retention)} | {comment} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_graphs_section(graph_files: Dict[str, str]) -> str:
    lines = ["## Graphs", ""]
    if not graph_files:
        return "\n".join(lines + ["Graphs were not generated.", ""])
    lines.append("Графики построены по step-level JSON и отражают динамику sequential-метрик по шагам.")
    lines.append("")
    for name, path in graph_files.items():
        lines.append(f"- [{name}]({os.path.basename(path)})")
    lines.append("")
    return "\n".join(lines)


def build_steps_section(sequential_edit_dir: str, methods: List[str], num_steps: int) -> str:
    by_step = load_step_results(sequential_edit_dir)
    lines = ["## Selected Steps", ""]
    if not by_step:
        return "\n".join(lines + ["No step results found.", ""])

    for step_index, fact_id in choose_steps(by_step, num_steps):
        method_map = by_step[(step_index, fact_id)]
        reference = next(iter(method_map.values()))
        edit_request = reference.get("input_edit_request") or {}
        lines.extend(
            [
                f"### Step {step_index}: {fact_id}",
                "",
                f"- Prompt: `{edit_request.get('prompt')}`",
                f"- Subject: `{edit_request.get('subject')}`",
                f"- Target new: `{edit_request.get('target_new')}`",
                "",
                "| Method | Status | Target old source | Accepted quality | Raw model answer | Current rel | Current gen | Retention | Global locality | Domain score | Sequential quality | Time (s) |",
                "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            payload = method_map.get(method) or method_map.get(method.lower())
            if not payload:
                lines.append(f"| {method} | missing | - | - | - | - | - | - | - | - | - | - |")
                continue
            target_old = payload.get("target_old_resolution") or {}
            lines.append(
                "| {method} | {status} | {source} | {accepted_quality} | {raw_answer} | {rel} | {gen} | {ret} | {glob} | {domain} | {quality} | {time} |".format(
                    method=method,
                    status=payload.get("status"),
                    source=target_old.get("target_old_source") or "-",
                    accepted_quality=fmt(target_old.get("accepted_quality_score")),
                    raw_answer=shorten_text(target_old.get("raw_model_answer")),
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


def generate_graphs(step_rows: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    if not step_rows:
        return {}

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-codex"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("current_reliability", "Current reliability"),
        ("current_generalization", "Current generalization"),
        ("retention", "Retention"),
        ("global_locality_generation", "Global locality"),
        ("domain_score_generation", "Domain score"),
        ("sequential_quality", "Sequential quality"),
        ("time_sec", "Time per step (s)"),
        ("peak_gpu_gb", "Peak GPU memory (GB)"),
    ]

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in step_rows:
        by_method.setdefault(row["method"], []).append(row)
    for records in by_method.values():
        records.sort(key=lambda item: item["step_index"])

    graph_files: Dict[str, str] = {}
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for method, records in sorted(by_method.items()):
            xs = [record["step_index"] for record in records]
            ys = [record.get(key) for record in records]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=method)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = f"{key}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        graph_files[key] = path
    return graph_files


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
    step_rows = collect_step_rows(args.sequential_edit_dir)
    output_dir = os.path.dirname(args.output)
    graph_files = generate_graphs(step_rows, output_dir)

    lines = [
        "# OilGas Sequential Knowledge Editing Report",
        "",
        build_baseline_section(baseline_summary),
        build_methodology_section(sequential_summary),
        build_sequential_summary_section(sequential_summary),
        build_analysis_section(sequential_summary, step_rows),
        build_graphs_section(graph_files),
        build_steps_section(args.sequential_edit_dir, methods, args.num_steps),
        build_errors_section(sequential_summary),
        "## Current Limitations",
        "",
    ]
    lines.extend(f"- {line}" for line in limitations)
    lines.append("")

    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
