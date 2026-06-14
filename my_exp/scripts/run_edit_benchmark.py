import argparse
import contextlib
import csv
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from statistics import mean
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from my_exp.scripts.generate_benchmark_report import build_report, choose_case_ids, load_case_results

METHOD_HPARAM_PATHS = {
    "LoRA": os.path.join(ROOT, "my_exp", "hparams", "lora_qwen25_3b_smoke.yaml"),
    "MEMIT": os.path.join(ROOT, "my_exp", "hparams", "memit_qwen25_3b_smoke.yaml"),
    "ROME": os.path.join(ROOT, "my_exp", "hparams", "rome_qwen25_3b_smoke.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to edit benchmark JSONL.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["LoRA", "ROME", "MEMIT"],
        choices=sorted(METHOD_HPARAM_PATHS.keys()),
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "my_exp", "results", "qwen25-3b", "pubtator_gene_first"),
    )
    parser.add_argument(
        "--worker-method",
        choices=sorted(METHOD_HPARAM_PATHS.keys()),
        default=None,
        help="Internal mode: run exactly one method in an isolated process.",
    )
    return parser.parse_args()


def load_cases(dataset_path: str, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


def first_nonempty_string(values: List[Any]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_rephrase_prompt(case: Dict[str, Any]) -> Optional[str]:
    prompt = case.get("rephrase_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    prompts = case.get("rephrase_prompts") or []
    return first_nonempty_string(prompts)


def build_locality_inputs(case: Dict[str, Any]) -> Optional[Dict[str, Dict[str, List[Optional[str]]]]]:
    locality_inputs: Dict[str, Dict[str, List[Optional[str]]]] = {}

    locality_prompt = case.get("locality_prompt")
    locality_ground_truth = case.get("locality_ground_truth")
    if locality_prompt is not None and locality_ground_truth is not None:
        locality_inputs["neighbor"] = {
            "prompt": [locality_prompt],
            "ground_truth": [locality_ground_truth],
        }

    general_prompt = case.get("general_locality_prompt")
    general_ground_truth = case.get("general_locality_ground_truth")
    if general_prompt is not None and general_ground_truth is not None:
        locality_inputs["general"] = {
            "prompt": [general_prompt],
            "ground_truth": [general_ground_truth],
        }

    return locality_inputs or None


def build_portability_inputs(case: Dict[str, Any]) -> Optional[Dict[str, Dict[str, List[Optional[str]]]]]:
    prompt = case.get("portability_prompt")
    ground_truth = case.get("portability_ground_truth")
    if prompt is None or ground_truth is None:
        return None
    return {
        "portability": {
            "prompt": [prompt],
            "ground_truth": [ground_truth],
        }
    }


def case_to_edit_kwargs(case: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "prompts": case["prompt"],
        "ground_truth": case["ground_truth"],
        "target_new": case["target_new"],
        "subject": case["subject"],
        "keep_original_weight": True,
        "verbose": False,
    }

    rephrase_prompt = build_rephrase_prompt(case)
    if rephrase_prompt is not None:
        kwargs["rephrase_prompts"] = rephrase_prompt

    locality_inputs = build_locality_inputs(case)
    if locality_inputs is not None:
        kwargs["locality_inputs"] = locality_inputs

    portability_inputs = build_portability_inputs(case)
    if portability_inputs is not None:
        kwargs["portability_inputs"] = portability_inputs

    return kwargs


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(mean(clean), 6)


def metric_mean(value: Any) -> Optional[float]:
    if isinstance(value, list):
        scalars = [float(item) for item in value]
        if not scalars:
            return None
        return float(mean(scalars))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def extract_nested_accuracy(metrics: Dict[str, Any], phase: str, bucket: str) -> Optional[float]:
    bucket_value = metrics.get(phase, {}).get(bucket)
    if not isinstance(bucket_value, dict) or not bucket_value:
        return None

    values: List[float] = []
    for key, value in bucket_value.items():
        if key.endswith("_acc"):
            metric_value = metric_mean(value)
            if metric_value is not None:
                values.append(metric_value)
    if not values:
        return None
    return float(mean(values))


def normalize_metric_record(raw_metric: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "pre_rewrite_acc": metric_mean(raw_metric.get("pre", {}).get("rewrite_acc")),
        "post_rewrite_acc": metric_mean(raw_metric.get("post", {}).get("rewrite_acc")),
        "pre_rephrase_acc": metric_mean(raw_metric.get("pre", {}).get("rephrase_acc")),
        "post_rephrase_acc": metric_mean(raw_metric.get("post", {}).get("rephrase_acc")),
        "pre_portability_acc": extract_nested_accuracy(raw_metric, "pre", "portability"),
        "post_portability_acc": extract_nested_accuracy(raw_metric, "post", "portability"),
        "post_locality_acc": extract_nested_accuracy(raw_metric, "post", "locality"),
    }


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "total_cases",
        "successful_cases",
        "failed_cases",
        "success_rate",
        "mean_time_sec",
        "mean_peak_gpu_gb",
        "mean_pre_rewrite_acc",
        "mean_post_rewrite_acc",
        "mean_pre_rephrase_acc",
        "mean_post_rephrase_acc",
        "mean_pre_portability_acc",
        "mean_post_portability_acc",
        "mean_post_locality_acc",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_method(method_name: str, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [result for result in case_results if result["status"] == "success"]
    failed = [result for result in case_results if result["status"] != "success"]

    return {
        "method": method_name,
        "total_cases": len(case_results),
        "successful_cases": len(successful),
        "failed_cases": len(failed),
        "success_rate": round(len(successful) / len(case_results), 6) if case_results else None,
        "mean_time_sec": mean_or_none([result.get("time_sec") for result in successful]),
        "mean_peak_gpu_gb": mean_or_none(
            [(result.get("gpu", {}) or {}).get("peak_allocated_gb") for result in successful]
        ),
        "mean_pre_rewrite_acc": mean_or_none([result["normalized_metrics"].get("pre_rewrite_acc") for result in successful]),
        "mean_post_rewrite_acc": mean_or_none([result["normalized_metrics"].get("post_rewrite_acc") for result in successful]),
        "mean_pre_rephrase_acc": mean_or_none([result["normalized_metrics"].get("pre_rephrase_acc") for result in successful]),
        "mean_post_rephrase_acc": mean_or_none([result["normalized_metrics"].get("post_rephrase_acc") for result in successful]),
        "mean_pre_portability_acc": mean_or_none([result["normalized_metrics"].get("pre_portability_acc") for result in successful]),
        "mean_post_portability_acc": mean_or_none([result["normalized_metrics"].get("post_portability_acc") for result in successful]),
        "mean_post_locality_acc": mean_or_none([result["normalized_metrics"].get("post_locality_acc") for result in successful]),
        "failed_case_ids": [result["case_id"] for result in failed],
    }


def ensure_dataset_copy(dataset_path: str, output_dir: str, max_cases: Optional[int]) -> str:
    cases = load_cases(dataset_path, max_cases=max_cases)
    dataset_copy_path = os.path.join(output_dir, "benchmark_cases.jsonl")
    with open(dataset_copy_path, "w", encoding="utf-8") as dst:
        for case in cases:
            dst.write(json.dumps(case, ensure_ascii=False) + "\n")
    return dataset_copy_path


def worker_gpu_helpers():
    import torch

    def reset_gpu_peak_if_available() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def gpu_stats_if_available() -> Dict[str, Optional[float]]:
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "allocated_gb": None,
                "reserved_gb": None,
                "peak_allocated_gb": None,
            }
        torch.cuda.synchronize()
        return {
            "cuda_available": True,
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 4),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 4),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        }

    return torch, reset_gpu_peak_if_available, gpu_stats_if_available


def worker_method_components(method_name: str):
    from easyeditor import BaseEditor, LoRAHyperParams, MEMITHyperParams, ROMEHyperParams

    method_classes = {
        "LoRA": LoRAHyperParams,
        "MEMIT": MEMITHyperParams,
        "ROME": ROMEHyperParams,
    }
    return BaseEditor, method_classes[method_name], METHOD_HPARAM_PATHS[method_name]


def run_case(editor: Any, case: Dict[str, Any], gpu_stats_fn: Any, reset_gpu_peak_fn: Any) -> Dict[str, Any]:
    reset_gpu_peak_fn()
    start_time = time.time()
    edit_kwargs = case_to_edit_kwargs(case)

    metrics, _, _ = editor.edit(**edit_kwargs)

    elapsed = round(time.time() - start_time, 4)
    raw_metric = metrics[0] if metrics else {}

    return {
        "status": "success",
        "case_id": case["case_id"],
        "time_sec": elapsed,
        "gpu": gpu_stats_fn(),
        "input_case": case,
        "normalized_metrics": normalize_metric_record(raw_metric),
        "raw_metrics": metrics,
    }


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


@contextlib.contextmanager
def suppress_stdio():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def run_method_worker(method_name: str, dataset_path: str, max_cases: Optional[int], output_dir: str) -> Dict[str, Any]:
    torch, reset_gpu_peak_fn, gpu_stats_fn = worker_gpu_helpers()
    BaseEditor, hparams_cls, hparams_path = worker_method_components(method_name)

    cases = load_cases(dataset_path, max_cases=max_cases)
    method_dir = os.path.join(output_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)
    stdout_path = os.path.join(method_dir, "stdout.log")
    stderr_path = os.path.join(method_dir, "stderr.log")

    with open(stdout_path, "w", encoding="utf-8") as method_log, open(stderr_path, "w", encoding="utf-8") as error_log:
        hparams = hparams_cls.from_hparams(hparams_path)
        with suppress_stdio():
            editor = BaseEditor.from_hparams(hparams)

        case_results: List[Dict[str, Any]] = []
        total_cases = len(cases)
        method_start_time = time.time()
        method_log.write(f"[start] method={method_name} total_cases={total_cases}\n")
        method_log.flush()

        print(
            "__PROGRESS__ " + json.dumps({"event": "method_start", "method": method_name, "total_cases": total_cases}),
            flush=True,
        )

        for idx, case in enumerate(cases, start=1):
            try:
                with suppress_stdio():
                    result = run_case(editor, case, gpu_stats_fn, reset_gpu_peak_fn)
            except Exception as exc:
                result = {
                    "status": "error",
                    "case_id": case["case_id"],
                    "time_sec": None,
                    "gpu": gpu_stats_fn(),
                    "input_case": case,
                    "normalized_metrics": {},
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                error_log.write(result["error"]["traceback"] + "\n")
                error_log.flush()

            case_results.append(result)
            write_json(os.path.join(method_dir, f"case_{int(case['case_id']):04d}.json"), result)

            elapsed_total = time.time() - method_start_time
            avg_per_case = elapsed_total / idx if idx else None
            eta_seconds = avg_per_case * (total_cases - idx) if avg_per_case is not None else None
            nm = result.get("normalized_metrics", {})
            method_log.write(
                "[case] {idx}/{total} case_id={case_id} status={status} time={time_sec} "
                "rewrite={rewrite} rephrase={rephrase} portability={portability} locality={locality} "
                "elapsed={elapsed} eta={eta}\n".format(
                    idx=idx,
                    total=total_cases,
                    case_id=case["case_id"],
                    status=result["status"],
                    time_sec=format_eta(result.get("time_sec")),
                    rewrite=fmt(nm.get("post_rewrite_acc")),
                    rephrase=fmt(nm.get("post_rephrase_acc")),
                    portability=fmt(nm.get("post_portability_acc")),
                    locality=fmt(nm.get("post_locality_acc")),
                    elapsed=format_eta(elapsed_total),
                    eta=format_eta(eta_seconds),
                )
            )
            method_log.flush()

            print(
                "__PROGRESS__ "
                + json.dumps(
                    {
                        "event": "case_done",
                        "method": method_name,
                        "current": idx,
                        "total": total_cases,
                        "case_id": case["case_id"],
                        "status": result["status"],
                        "time_sec": result.get("time_sec"),
                    }
                ),
                flush=True,
            )

        summary = summarize_method(method_name, case_results)
        write_json(os.path.join(method_dir, "summary.json"), summary)
        method_log.write(
            "[done] method={method} success={success}/{total} mean_time={mean_time} "
            "rewrite={rewrite} rephrase={rephrase} portability={portability} locality={locality}\n".format(
                method=method_name,
                success=summary["successful_cases"],
                total=summary["total_cases"],
                mean_time=format_eta(summary.get("mean_time_sec")),
                rewrite=fmt(summary.get("mean_post_rewrite_acc")),
                rephrase=fmt(summary.get("mean_post_rephrase_acc")),
                portability=fmt(summary.get("mean_post_portability_acc")),
                locality=fmt(summary.get("mean_post_locality_acc")),
            )
        )
        method_log.flush()

        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(
            "__PROGRESS__ "
            + json.dumps(
                {
                    "event": "method_done",
                    "method": method_name,
                    "successful_cases": summary["successful_cases"],
                    "total_cases": summary["total_cases"],
                    "mean_time_sec": summary.get("mean_time_sec"),
                }
            ),
            flush=True,
        )

        return summary


def run_method_subprocess(method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    method_dir = os.path.join(args.output_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)
    stdout_path = os.path.join(method_dir, "stdout.log")
    stderr_path = os.path.join(method_dir, "stderr.log")

    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset",
        args.dataset,
        "--output-dir",
        args.output_dir,
        "--worker-method",
        method_name,
    ]
    if args.max_cases is not None:
        command.extend(["--max-cases", str(args.max_cases)])

    progress_events: List[Dict[str, Any]] = []
    with open(stderr_path, "a", encoding="utf-8") as stderr_f:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("__PROGRESS__ "):
                try:
                    progress_events.append(json.loads(line[len("__PROGRESS__ ") :]))
                except json.JSONDecodeError:
                    stderr_f.write(line)
                    stderr_f.flush()
            else:
                stderr_f.write(line)
                stderr_f.flush()

        completed_returncode = process.wait()

        if completed_returncode != 0:
            stderr_f.write(f"Subprocess exited with return code {completed_returncode}\n")

        class Completed:
            def __init__(self, returncode: int):
                self.returncode = returncode

        completed = Completed(completed_returncode)

    summary_path = os.path.join(method_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {
            "method": method_name,
            "total_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "success_rate": 0.0,
            "mean_time_sec": None,
            "mean_peak_gpu_gb": None,
            "mean_pre_rewrite_acc": None,
            "mean_post_rewrite_acc": None,
            "mean_pre_rephrase_acc": None,
            "mean_post_rephrase_acc": None,
            "mean_pre_portability_acc": None,
            "mean_post_portability_acc": None,
            "mean_post_locality_acc": None,
            "failed_case_ids": [],
            "process_error": {
                "returncode": completed.returncode,
                "stdout_log": stdout_path,
                "stderr_log": stderr_path,
            },
        }

    summary["returncode"] = completed.returncode
    summary["stdout_log"] = stdout_path
    summary["stderr_log"] = stderr_path
    summary["_progress_events"] = progress_events
    return summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.worker_method is not None:
        run_method_worker(args.worker_method, args.dataset, args.max_cases, args.output_dir)
        return

    dataset_copy_path = ensure_dataset_copy(args.dataset, args.output_dir, args.max_cases)

    summary_rows = []
    total_cases = len(load_cases(args.dataset, max_cases=args.max_cases))
    total_units = total_cases * len(args.methods)
    progress = tqdm(total=total_units, desc="Benchmark", unit="run")
    for method_name in args.methods:
        method_summary = run_method_subprocess(method_name, args)
        for event in method_summary.pop("_progress_events", []):
            if event.get("event") == "case_done":
                progress.update(1)
                progress.set_postfix_str(
                    f"{event.get('method')} case_id={event.get('case_id')} status={event.get('status')}"
                )
        summary_rows.append(method_summary)
    progress.close()

    overall_summary = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_copy": dataset_copy_path,
        "cases_requested": args.max_cases,
        "cases_loaded": len(load_cases(args.dataset, max_cases=args.max_cases)),
        "methods": args.methods,
        "process_model": "one-method-per-subprocess",
        "method_summaries": summary_rows,
    }

    write_json(os.path.join(args.output_dir, "summary.json"), overall_summary)
    write_summary_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows)

    try:
        cases = load_case_results(args.output_dir, args.methods)
        case_ids = choose_case_ids(cases, args.methods, explicit_case_ids=None, num_cases=5)
        report = build_report(args.output_dir, overall_summary, case_ids, cases)
        report_path = os.path.join(args.output_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as exc:
        error_path = os.path.join(args.output_dir, "report_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"[warn] report generation failed: {exc}", flush=True)

    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
