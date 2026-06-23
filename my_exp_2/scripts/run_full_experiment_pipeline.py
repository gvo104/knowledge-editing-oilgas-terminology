import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import load_oilgas_dataset, write_json
from run_single_edit_experiment import METHOD_HPARAM_PATHS


STAGES = ("preflight", "baseline", "single", "sequential", "reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--methods", nargs="+", default=["LoRA", "ROME", "MEMIT", "WISE"], choices=sorted(METHOD_HPARAM_PATHS))
    parser.add_argument("--data-dir", default=os.path.join("my_exp_2", "data"))
    parser.add_argument("--model", default=os.path.join("my_exp", "models", "Qwen2.5-3B"))
    parser.add_argument("--output-root", default=os.path.join("my_exp_2", "outputs", "full_pipeline"))
    parser.add_argument("--max-facts", type=int, default=20)
    parser.add_argument("--eval-scope", choices=["fact-only", "fact-plus-general", "full"], default="full")
    parser.add_argument("--single-eval-mode", choices=["sample", "full"], default="full")
    parser.add_argument("--sequential-eval-mode", choices=["sample", "full"], default="full")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--target-old-max-new-tokens", type=int, default=32)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-smoke-preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dir(args: argparse.Namespace) -> str:
    return os.path.join(args.output_root, args.run_name)


def manifest_path(base_dir: str) -> str:
    return os.path.join(base_dir, "run_manifest.json")


def load_manifest(base_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    path = manifest_path(base_dir)
    if os.path.exists(path):
        return load_json(path)
    return {
        "run_name": args.run_name,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "root": os.path.abspath(base_dir),
        "config": {
            "methods": args.methods,
            "data_dir": os.path.abspath(args.data_dir),
            "model": os.path.abspath(args.model),
            "max_facts": args.max_facts,
            "eval_scope": args.eval_scope,
            "single_eval_mode": args.single_eval_mode,
            "sequential_eval_mode": args.sequential_eval_mode,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "target_old_max_new_tokens": args.target_old_max_new_tokens,
        },
        "stages": {},
    }


def save_manifest(base_dir: str, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    write_json(manifest_path(base_dir), manifest)


def stage_done(base_dir: str, stage: str) -> str:
    return os.path.join(base_dir, ".stage_done", f"{stage}.done")


def mark_stage_done(base_dir: str, stage: str) -> None:
    path = stage_done(base_dir, stage)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(now_iso() + "\n")


def is_stage_done(base_dir: str, stage: str) -> bool:
    return os.path.exists(stage_done(base_dir, stage))


def prepare_output_dir(args: argparse.Namespace) -> str:
    base_dir = run_dir(args)
    if args.overwrite and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    if os.path.exists(base_dir) and os.listdir(base_dir) and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {base_dir}. Use --resume to continue or --overwrite to replace it."
        )
    os.makedirs(base_dir, exist_ok=True)
    for name in ("preflight", "baseline", "single_edit", "sequential_edit", "metrics", "reports", "logs"):
        os.makedirs(os.path.join(base_dir, name), exist_ok=True)
    return base_dir


def should_skip_stage(base_dir: str, stage: str, args: argparse.Namespace) -> bool:
    return args.resume and is_stage_done(base_dir, stage)


def command_to_string(command: List[str]) -> str:
    return " ".join(command)


def run_command(command: List[str], cwd: str, log_path: str, stage: str, manifest: Dict[str, Any]) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {command_to_string(command)}\n\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        returncode = process.wait()
    manifest.setdefault("stages", {}).setdefault(stage, {})["last_command"] = command
    manifest["stages"][stage]["returncode"] = returncode
    manifest["stages"][stage]["time_sec"] = round(time.time() - started, 4)
    manifest["stages"][stage]["log"] = log_path
    return returncode


def require_success(returncode: int, stage: str) -> None:
    if returncode != 0:
        raise SystemExit(f"Stage failed: {stage}, returncode={returncode}")


def stop_requested(args: argparse.Namespace, stage: str) -> bool:
    return args.stop_after == stage


def run_python(script: str, *args: str) -> List[str]:
    return [sys.executable, os.path.join("my_exp_2", "scripts", script), *args]


def preflight(base_dir: str, args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    stage = "preflight"
    if args.skip_preflight:
        manifest["stages"][stage] = {"status": "skipped", "reason": "--skip-preflight"}
        return
    if should_skip_stage(base_dir, stage, args):
        manifest["stages"][stage] = {"status": "skipped", "reason": "resume marker exists"}
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    preflight_dir = os.path.join(base_dir, "preflight")
    logs_dir = os.path.join(base_dir, "logs")
    report: Dict[str, Any] = {
        "status": "ok",
        "started_at": now_iso(),
        "checks": {},
        "errors": [],
    }

    data = load_oilgas_dataset(args.data_dir)
    report["checks"]["data"] = {
        "triplets": len(data["triplets"]),
        "fact_questions": len(data["fact_questions"]),
        "domain_questions": len(data["domain_questions"]),
        "general_questions": len(data["general_questions"]),
        "sequential_order": len(data["sequential_order"]),
    }
    for method in args.methods:
        if method not in METHOD_HPARAM_PATHS or not os.path.exists(METHOD_HPARAM_PATHS[method]):
            report["errors"].append(f"Missing hparams for {method}")

    cuda_available = torch.cuda.is_available()
    report["checks"]["cuda"] = {
        "available": cuda_available,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            report["checks"]["cuda"]["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 4),
                }
            )
    elif not args.allow_cpu:
        report["errors"].append("CUDA is not available and --allow-cpu is not set")

    disk = shutil.disk_usage(base_dir)
    report["checks"]["disk"] = {
        "path": os.path.abspath(base_dir),
        "free_gb": round(disk.free / (1024**3), 4),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
        if cuda_available:
            model.to("cuda:0")
        model.eval()
        encoded = tokenizer("Проверка модели", return_tensors="pt")
        report["checks"]["model"] = {
            "loaded": True,
            "tokenizer_vocab_size": len(tokenizer),
            "probe_tokens": int(encoded["input_ids"].shape[-1]),
            "device": "cuda:0" if cuda_available else "cpu",
        }
        del model
        if cuda_available:
            torch.cuda.empty_cache()
    except Exception as exc:
        report["checks"]["model"] = {"loaded": False, "error": repr(exc)}
        report["errors"].append(f"Model load check failed: {exc}")

    validation_output = os.path.join(preflight_dir, "data_validation.json")
    rc = run_command(
        run_python("validate_data.py", "--data-dir", args.data_dir, "--output", validation_output),
        ROOT,
        os.path.join(logs_dir, "preflight_validate_data.log"),
        stage,
        manifest,
    )
    if rc != 0:
        report["errors"].append("validate_data.py failed")

    if not args.skip_smoke_preflight and not report["errors"]:
        smoke_single_dir = os.path.join(preflight_dir, "smoke_single_wise")
        rc = run_command(
            run_python(
                "run_single_edit_experiment.py",
                "--methods",
                "WISE",
                "--data-dir",
                args.data_dir,
                "--model",
                args.model,
                "--output-dir",
                smoke_single_dir,
                "--max-facts",
                "1",
                "--eval-scope",
                "fact-only",
                "--target-old-max-new-tokens",
                str(args.target_old_max_new_tokens),
            ),
            ROOT,
            os.path.join(logs_dir, "preflight_smoke_single_wise.log"),
            stage,
            manifest,
        )
        if rc != 0:
            report["errors"].append("WISE single-edit smoke failed")

        smoke_seq_dir = os.path.join(preflight_dir, "smoke_sequential")
        smoke_method = "WISE" if "WISE" in args.methods else args.methods[0]
        seq_command = run_python(
            "run_sequential_edit_experiment.py",
            "--methods",
            smoke_method,
            "--data-dir",
            args.data_dir,
            "--model",
            args.model,
            "--output-dir",
            smoke_seq_dir,
            "--max-steps",
            "1",
            "--eval-scope",
            "fact-only",
            "--target-old-max-new-tokens",
            str(args.target_old_max_new_tokens),
            "--generation-max-new-tokens",
            str(args.generation_max_new_tokens),
        )
        if args.allow_cpu:
            seq_command.append("--allow-cpu")
        rc = run_command(seq_command, ROOT, os.path.join(logs_dir, "preflight_smoke_sequential.log"), stage, manifest)
        if rc != 0:
            report["errors"].append(f"{smoke_method} sequential smoke failed")

    if report["errors"]:
        report["status"] = "error"
    report["time_sec"] = round(time.time() - started, 4)
    write_json(os.path.join(preflight_dir, "preflight_report.json"), report)
    write_preflight_md(os.path.join(preflight_dir, "preflight_report.md"), report)
    manifest["stages"][stage] = {
        "status": report["status"],
        "time_sec": report["time_sec"],
        "report": os.path.join(preflight_dir, "preflight_report.json"),
    }
    if report["status"] != "ok":
        raise SystemExit("Preflight failed. See preflight_report.json")
    mark_stage_done(base_dir, stage)


def write_preflight_md(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "# Preflight Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Time: `{report.get('time_sec')}` sec",
        "",
        "## CUDA",
        "",
    ]
    cuda = report.get("checks", {}).get("cuda", {})
    lines.append(f"- Available: `{cuda.get('available')}`")
    lines.append(f"- Torch CUDA: `{cuda.get('torch_cuda')}`")
    for device in cuda.get("devices", []):
        lines.append(f"- GPU {device.get('index')}: {device.get('name')} ({device.get('total_memory_gb')} GB)")
    lines.extend(["", "## Errors", ""])
    errors = report.get("errors") or []
    lines.extend([f"- {error}" for error in errors] or ["No errors."])
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_baseline(base_dir: str, args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    stage = "baseline"
    if args.skip_baseline:
        manifest["stages"][stage] = {"status": "skipped", "reason": "--skip-baseline"}
        return
    if should_skip_stage(base_dir, stage, args):
        manifest["stages"][stage] = {"status": "skipped", "reason": "resume marker exists"}
        return
    output_dir = os.path.join(base_dir, "baseline")
    rc = run_command(
        run_python(
            "run_baseline_eval.py",
            "--data-dir",
            args.data_dir,
            "--model",
            args.model,
            "--output-dir",
            output_dir,
            "--max-new-tokens",
            str(args.generation_max_new_tokens),
        ),
        ROOT,
        os.path.join(base_dir, "logs", "baseline.log"),
        stage,
        manifest,
    )
    require_success(rc, stage)
    manifest["stages"][stage]["status"] = "ok"
    mark_stage_done(base_dir, stage)


def run_single(base_dir: str, args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    stage = "single"
    if should_skip_stage(base_dir, stage, args):
        manifest["stages"][stage] = {"status": "skipped", "reason": "resume marker exists"}
        return
    output_dir = os.path.join(base_dir, "single_edit")
    command = run_python(
        "run_single_edit_experiment.py",
        "--methods",
        *args.methods,
        "--data-dir",
        args.data_dir,
        "--model",
        args.model,
        "--output-dir",
        output_dir,
        "--max-facts",
        str(args.max_facts),
        "--eval-scope",
        args.eval_scope,
        "--target-old-max-new-tokens",
        str(args.target_old_max_new_tokens),
    )
    rc = run_command(command, ROOT, os.path.join(base_dir, "logs", "single_edit.log"), stage, manifest)
    require_success(rc, stage)
    manifest["stages"][stage]["status"] = "ok"
    manifest["stages"][stage]["note"] = "single_eval_mode is recorded by pipeline; current single runner uses EasyEdit internal post-edit metrics."
    mark_stage_done(base_dir, stage)


def run_sequential(base_dir: str, args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    stage = "sequential"
    if should_skip_stage(base_dir, stage, args):
        manifest["stages"][stage] = {"status": "skipped", "reason": "resume marker exists"}
        return
    output_dir = os.path.join(base_dir, "sequential_edit")
    command = run_python(
        "run_sequential_edit_experiment.py",
        "--methods",
        *args.methods,
        "--data-dir",
        args.data_dir,
        "--model",
        args.model,
        "--output-dir",
        output_dir,
        "--max-steps",
        str(args.max_facts),
        "--eval-scope",
        args.eval_scope,
        "--eval-mode",
        args.sequential_eval_mode,
        "--generation-max-new-tokens",
        str(args.generation_max_new_tokens),
        "--target-old-max-new-tokens",
        str(args.target_old_max_new_tokens),
    )
    if args.allow_cpu:
        command.append("--allow-cpu")
    rc = run_command(command, ROOT, os.path.join(base_dir, "logs", "sequential_edit.log"), stage, manifest)
    require_success(rc, stage)
    manifest["stages"][stage]["status"] = "ok"
    mark_stage_done(base_dir, stage)


def run_reports(base_dir: str, args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    stage = "reports"
    if should_skip_stage(base_dir, stage, args):
        manifest["stages"][stage] = {"status": "skipped", "reason": "resume marker exists"}
        return

    baseline_dir = os.path.join(base_dir, "baseline")
    single_dir = os.path.join(base_dir, "single_edit")
    sequential_dir = os.path.join(base_dir, "sequential_edit")
    single_metrics_dir = os.path.join(base_dir, "metrics", "single")
    sequential_metrics_dir = os.path.join(base_dir, "metrics", "sequential")
    overall_metrics_dir = os.path.join(base_dir, "metrics", "overall")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(overall_metrics_dir, exist_ok=True)

    commands = [
        (
            "analyze_single",
            run_python(
                "compute_metrics.py",
                "--single-edit-dir",
                single_dir,
                "--baseline-dir",
                baseline_dir,
                "--output-dir",
                single_metrics_dir,
            ),
        ),
        (
            "analyze_sequential",
            run_python(
                "compute_sequential_metrics.py",
                "--sequential-edit-dir",
                sequential_dir,
                "--baseline-dir",
                baseline_dir,
                "--output-dir",
                sequential_metrics_dir,
            ),
        ),
        (
            "generate_single_report",
            run_python(
                "generate_report.py",
                "--single-edit-dir",
                single_dir,
                "--baseline-dir",
                baseline_dir,
                "--output",
                os.path.join(reports_dir, "single_edit_report.md"),
            ),
        ),
        (
            "generate_sequential_report",
            run_python(
                "generate_sequential_report.py",
                "--sequential-edit-dir",
                sequential_dir,
                "--baseline-dir",
                baseline_dir,
                "--output",
                os.path.join(reports_dir, "sequential_edit_report.md"),
            ),
        ),
    ]
    for name, command in commands:
        rc = run_command(command, ROOT, os.path.join(base_dir, "logs", f"{name}.log"), stage, manifest)
        require_success(rc, name)

    write_overall_summary(base_dir, args)
    manifest["stages"][stage] = {
        "status": "ok",
        "reports": {
            "single": os.path.join(reports_dir, "single_edit_report.md"),
            "sequential": os.path.join(reports_dir, "sequential_edit_report.md"),
            "overall": os.path.join(reports_dir, "overall_summary.md"),
        },
    }
    mark_stage_done(base_dir, stage)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def method_rows(summary_path: str, mode: str) -> List[Dict[str, Any]]:
    if not os.path.exists(summary_path):
        return []
    summary = load_json(summary_path)
    rows = []
    for row in summary.get("method_summaries", []):
        if mode == "single":
            rows.append(
                {
                    "method": row.get("method"),
                    "success": f"{row.get('successful_cases')}/{row.get('total_cases')}",
                    "quality": row.get("mean_edit_quality"),
                    "reliability": row.get("mean_reliability"),
                    "generalization": row.get("mean_generalization"),
                    "retention": None,
                    "global": row.get("mean_global_locality"),
                    "domain": row.get("mean_domain_score"),
                    "time": row.get("mean_time_sec"),
                    "gpu": row.get("mean_peak_gpu_gb"),
                }
            )
        else:
            rows.append(
                {
                    "method": row.get("method"),
                    "success": f"{row.get('successful_steps')}/{row.get('total_steps')}",
                    "quality": row.get("mean_sequential_quality"),
                    "reliability": row.get("mean_current_reliability"),
                    "generalization": row.get("mean_current_generalization"),
                    "retention": row.get("mean_retention"),
                    "global": row.get("mean_global_locality"),
                    "domain": row.get("mean_domain_score"),
                    "time": row.get("mean_time_sec"),
                    "gpu": row.get("mean_peak_gpu_gb"),
                }
            )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_overall_summary(base_dir: str, args: argparse.Namespace) -> None:
    reports_dir = os.path.join(base_dir, "reports")
    metrics_dir = os.path.join(base_dir, "metrics", "overall")
    single_rows = method_rows(os.path.join(base_dir, "single_edit", "summary.json"), "single")
    sequential_rows = method_rows(os.path.join(base_dir, "sequential_edit", "summary.json"), "sequential")

    comparison_rows = []
    by_seq = {row["method"]: row for row in sequential_rows}
    for single in single_rows:
        method = single["method"]
        seq = by_seq.get(method, {})
        comparison_rows.append(
            {
                "method": method,
                "single_quality": single.get("quality"),
                "sequential_quality": seq.get("quality"),
                "single_reliability": single.get("reliability"),
                "sequential_reliability": seq.get("reliability"),
                "sequential_retention": seq.get("retention"),
                "single_global": single.get("global"),
                "sequential_global": seq.get("global"),
                "single_domain": single.get("domain"),
                "sequential_domain": seq.get("domain"),
            }
        )
    write_csv(os.path.join(metrics_dir, "single_vs_sequential.csv"), comparison_rows)
    write_json(
        os.path.join(metrics_dir, "method_comparison.json"),
        {"single": single_rows, "sequential": sequential_rows, "single_vs_sequential": comparison_rows},
    )

    best_single = max(single_rows, key=lambda row: row.get("quality") or -1, default=None)
    best_seq = max(sequential_rows, key=lambda row: row.get("quality") or -1, default=None)
    hypotheses = [
        "Compare methods separately for single-edit and sequential-edit because the best single-edit method may degrade under accumulated edits.",
        "Analyze insertion vs replacement through target_old_source; frequent fallback_to_target_new indicates knowledge insertion dominates.",
        "Group facts by subject/relation/level to identify terminology types that are difficult to edit or retain.",
    ]
    write_json(os.path.join(metrics_dir, "research_hypotheses.json"), hypotheses)

    lines = [
        "# Overall Knowledge Editing Summary",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Methods: `{', '.join(args.methods)}`",
        f"- Facts: `{args.max_facts}`",
        f"- Eval scope: `{args.eval_scope}`",
        f"- Single eval mode: `{args.single_eval_mode}`",
        f"- Sequential eval mode: `{args.sequential_eval_mode}`",
        "",
        "## Main Results",
        "",
        f"- Best single-edit quality: `{best_single.get('method') if best_single else '-'}` ({fmt(best_single.get('quality') if best_single else None)}).",
        f"- Best sequential quality: `{best_seq.get('method') if best_seq else '-'}` ({fmt(best_seq.get('quality') if best_seq else None)}).",
        "",
        "| Method | Single quality | Sequential quality | Seq retention | Single global | Seq global | Single domain | Seq domain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {method} | {single_quality} | {sequential_quality} | {sequential_retention} | {single_global} | {sequential_global} | {single_domain} | {sequential_domain} |".format(
                method=row["method"],
                single_quality=fmt(row.get("single_quality")),
                sequential_quality=fmt(row.get("sequential_quality")),
                sequential_retention=fmt(row.get("sequential_retention")),
                single_global=fmt(row.get("single_global")),
                sequential_global=fmt(row.get("sequential_global")),
                single_domain=fmt(row.get("single_domain")),
                sequential_domain=fmt(row.get("sequential_domain")),
            )
        )
    lines.extend(["", "## Research Hypotheses", ""])
    lines.extend(f"- {item}" for item in hypotheses)
    lines.append("")

    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "overall_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    base_dir = prepare_output_dir(args)
    manifest = load_manifest(base_dir, args)
    save_manifest(base_dir, manifest)

    preflight(base_dir, args, manifest)
    save_manifest(base_dir, manifest)
    if stop_requested(args, "preflight"):
        return

    run_baseline(base_dir, args, manifest)
    save_manifest(base_dir, manifest)
    if stop_requested(args, "baseline"):
        return

    run_single(base_dir, args, manifest)
    save_manifest(base_dir, manifest)
    if stop_requested(args, "single"):
        return

    run_sequential(base_dir, args, manifest)
    save_manifest(base_dir, manifest)
    if stop_requested(args, "sequential"):
        return

    run_reports(base_dir, args, manifest)
    save_manifest(base_dir, manifest)
    print(json.dumps({"run_dir": base_dir, "manifest": manifest_path(base_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
