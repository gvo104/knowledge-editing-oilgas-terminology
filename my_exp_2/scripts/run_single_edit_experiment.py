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
SCRIPT_DIR = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import build_easyedit_kwargs, build_single_edit_case, load_oilgas_dataset, write_json
from eval_utils import harmonic_or_none, mean_or_none, normalize_easyedit_metric
from target_old_resolver import resolve_target_old


METHOD_HPARAM_PATHS = {
    "LoRA": os.path.join(ROOT, "my_exp", "hparams", "lora_qwen25_3b_smoke.yaml"),
    "MEMIT": os.path.join(ROOT, "my_exp", "hparams", "memit_qwen25_3b_smoke.yaml"),
    "ROME": os.path.join(ROOT, "my_exp", "hparams", "rome_qwen25_3b_smoke.yaml"),
}

METHODS_REQUIRING_TARGET_OLD = {"LoRA", "ROME", "MEMIT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["LoRA", "ROME", "MEMIT"], choices=sorted(METHOD_HPARAM_PATHS))
    parser.add_argument("--data-dir", default=os.path.join("my_exp_2", "data"))
    parser.add_argument("--model", default=os.path.join("my_exp", "models", "Qwen2.5-3B"))
    parser.add_argument("--output-dir", default=os.path.join("my_exp_2", "outputs", "single_edit", "qwen25_3b_lora_rome_memit"))
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--eval-scope", choices=["fact-only", "fact-plus-general", "full"], default="fact-plus-general")
    parser.add_argument("--general-limit", type=int, default=5)
    parser.add_argument("--domain-limit", type=int, default=5)
    parser.add_argument("--target-old-probes", type=int, default=2)
    parser.add_argument("--target-old-max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--disable-target-old-resolution",
        action="store_true",
        help="Use target_old from JSON/fallback only. Intended for debugging.",
    )
    parser.add_argument("--worker-method", choices=sorted(METHOD_HPARAM_PATHS), default=None)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return str(value)


@contextlib.contextmanager
def suppress_stdio():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def gpu_helpers():
    import torch

    def reset_peak() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def stats() -> Dict[str, Optional[float]]:
        if not torch.cuda.is_available():
            return {"cuda_available": False, "allocated_gb": None, "reserved_gb": None, "peak_allocated_gb": None}
        torch.cuda.synchronize()
        return {
            "cuda_available": True,
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
        }

    return torch, reset_peak, stats


def method_components(method_name: str):
    from easyeditor import BaseEditor, LoRAHyperParams, MEMITHyperParams, ROMEHyperParams

    classes = {"LoRA": LoRAHyperParams, "MEMIT": MEMITHyperParams, "ROME": ROMEHyperParams}
    return BaseEditor, classes[method_name], METHOD_HPARAM_PATHS[method_name]


def fact_ids_for_run(data: Dict[str, Any], max_facts: Optional[int]) -> List[str]:
    fact_ids = [str(item["fact_id"]) for item in data["triplets"]]
    return fact_ids[:max_facts] if max_facts is not None else fact_ids


def run_case(
    editor: Any,
    case: Dict[str, Any],
    method_name: str,
    eval_scope: str,
    general_limit: Optional[int],
    domain_limit: Optional[int],
    target_old_probes: int,
    target_old_max_new_tokens: int,
    disable_target_old_resolution: bool,
    gpu_stats_fn: Any,
    reset_gpu_peak_fn: Any,
) -> Dict[str, Any]:
    reset_gpu_peak_fn()
    started = time.time()
    if disable_target_old_resolution:
        target_old_resolution = {
            "resolved_target_old": case["ground_truth"],
            "target_old_source": case["ground_truth_source"],
            "target_old_is_valid": None,
            "target_old_is_stable": None,
            "raw_model_answer": None,
            "probe_results": [],
        }
    else:
        target_old_resolution = resolve_target_old(
            editor.model,
            editor.tok,
            case,
            method_requires_target_old=method_name in METHODS_REQUIRING_TARGET_OLD,
            device=editor.hparams.device,
            max_probes=target_old_probes,
            max_new_tokens=target_old_max_new_tokens,
        )
        if target_old_resolution["resolved_target_old"] is not None:
            case["ground_truth"] = target_old_resolution["resolved_target_old"]
            case["ground_truth_source"] = target_old_resolution["target_old_source"]

    edit_kwargs = build_easyedit_kwargs(case, eval_scope=eval_scope, general_limit=general_limit, domain_limit=domain_limit)
    metrics, _, _ = editor.edit(**edit_kwargs)
    elapsed = round(time.time() - started, 4)
    raw_metric = metrics[0] if metrics else {}
    normalized = normalize_easyedit_metric(raw_metric)
    normalized["easyedit_edit_quality"] = normalized.get("edit_quality")

    return {
        "status": "success",
        "fact_id": case["fact_id"],
        "method": editor.alg_name,
        "time_sec": elapsed,
        "gpu": gpu_stats_fn(),
        "input_edit_request": case["edit_request"],
        "fact": case["fact"],
        "grouped_eval_questions": {
            "direct": case["direct_questions"],
            "paraphrase": case["paraphrase_questions"],
            "reverse": case["reverse_questions"],
            "neighbor": case["neighbor_questions"],
            "locality": case["locality_questions"],
        },
        "eval_scope": eval_scope,
        "ground_truth_source": case["ground_truth_source"],
        "target_old_resolution": target_old_resolution,
        "metrics": normalized,
        "raw_easyedit_metrics": json_safe(metrics),
        "generation_eval": {"status": "not_run", "reason": "MVP uses EasyEdit internal metrics for post-edit scoring."},
    }


def summarize_method(method_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") != "success"]
    target_old_sources: Dict[str, int] = {}
    for result in successful:
        source = (result.get("target_old_resolution") or {}).get("target_old_source")
        if source:
            target_old_sources[source] = target_old_sources.get(source, 0) + 1
    return {
        "method": method_name,
        "total_cases": len(results),
        "successful_cases": len(successful),
        "failed_cases": len(failed),
        "success_rate": round(len(successful) / len(results), 6) if results else None,
        "mean_time_sec": mean_or_none(result.get("time_sec") for result in successful),
        "mean_peak_gpu_gb": mean_or_none((result.get("gpu", {}) or {}).get("peak_allocated_gb") for result in successful),
        "mean_reliability": mean_or_none(result["metrics"].get("reliability") for result in successful),
        "mean_generalization": mean_or_none(result["metrics"].get("generalization") for result in successful),
        "mean_reverse": mean_or_none(result["metrics"].get("reverse") for result in successful),
        "mean_neighbor": mean_or_none(result["metrics"].get("neighbor") for result in successful),
        "mean_fact_locality": mean_or_none(result["metrics"].get("fact_locality") for result in successful),
        "mean_global_locality": mean_or_none(result["metrics"].get("global_locality") for result in successful),
        "mean_domain_score": mean_or_none(result["metrics"].get("domain_score") for result in successful),
        "mean_edit_quality": mean_or_none(result["metrics"].get("edit_quality") for result in successful),
        "target_old_sources": target_old_sources,
        "failed_fact_ids": [result.get("fact_id") for result in failed],
    }


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "total_cases",
        "successful_cases",
        "failed_cases",
        "success_rate",
        "mean_time_sec",
        "mean_peak_gpu_gb",
        "mean_reliability",
        "mean_generalization",
        "mean_reverse",
        "mean_neighbor",
        "mean_fact_locality",
        "mean_global_locality",
        "mean_domain_score",
        "mean_edit_quality",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_method_worker(method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    torch, reset_gpu_peak_fn, gpu_stats_fn = gpu_helpers()
    BaseEditor, hparams_cls, hparams_path = method_components(method_name)
    data = load_oilgas_dataset(args.data_dir)
    fact_ids = fact_ids_for_run(data, args.max_facts)

    method_dir = os.path.join(args.output_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)
    stdout_path = os.path.join(method_dir, "stdout.log")
    stderr_path = os.path.join(method_dir, "stderr.log")

    with open(stdout_path, "w", encoding="utf-8") as method_log, open(stderr_path, "w", encoding="utf-8") as error_log:
        hparams = hparams_cls.from_hparams(hparams_path)
        hparams.model_name = args.model
        with suppress_stdio():
            editor = BaseEditor.from_hparams(hparams)

        results = []
        method_started = time.time()
        print("__PROGRESS__ " + json.dumps({"event": "method_start", "method": method_name, "total_cases": len(fact_ids)}), flush=True)

        for idx, fact_id in enumerate(fact_ids, start=1):
            case = build_single_edit_case(fact_id, data)
            try:
                with suppress_stdio():
                    result = run_case(
                        editor,
                        case,
                        method_name,
                        args.eval_scope,
                        args.general_limit,
                        args.domain_limit,
                        args.target_old_probes,
                        args.target_old_max_new_tokens,
                        args.disable_target_old_resolution,
                        gpu_stats_fn,
                        reset_gpu_peak_fn,
                    )
            except Exception as exc:
                result = {
                    "status": "error",
                    "fact_id": fact_id,
                    "method": method_name,
                    "time_sec": None,
                    "gpu": gpu_stats_fn(),
                    "input_edit_request": case.get("edit_request"),
                    "fact": case.get("fact"),
                    "grouped_eval_questions": {},
                    "metrics": {},
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                error_log.write(result["error"]["traceback"] + "\n")
                error_log.flush()

            results.append(result)
            write_json(os.path.join(method_dir, f"case_{fact_id}.json"), result)

            elapsed_total = time.time() - method_started
            avg = elapsed_total / idx
            eta = avg * (len(fact_ids) - idx)
            metrics = result.get("metrics", {})
            method_log.write(
                "[case] {idx}/{total} fact_id={fact_id} status={status} time={time} "
                "rel={rel} gen={gen} loc={loc} quality={quality} elapsed={elapsed} eta={eta}\n".format(
                    idx=idx,
                    total=len(fact_ids),
                    fact_id=fact_id,
                    status=result["status"],
                    time=format_eta(result.get("time_sec")),
                    rel=metrics.get("reliability"),
                    gen=metrics.get("generalization"),
                    loc=metrics.get("global_locality"),
                    quality=metrics.get("edit_quality"),
                    elapsed=format_eta(elapsed_total),
                    eta=format_eta(eta),
                )
            )
            method_log.flush()
            print(
                "__PROGRESS__ "
                + json.dumps(
                    {
                        "event": "case_done",
                        "method": method_name,
                        "fact_id": fact_id,
                        "status": result["status"],
                        "current": idx,
                        "total": len(fact_ids),
                    }
                ),
                flush=True,
            )

        summary = summarize_method(method_name, results)
        write_json(os.path.join(method_dir, "summary.json"), summary)
        method_log.write(f"[done] {json.dumps(summary, ensure_ascii=False)}\n")

        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("__PROGRESS__ " + json.dumps({"event": "method_done", "method": method_name}), flush=True)
        return summary


def run_method_subprocess(method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    method_dir = os.path.join(args.output_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)
    stderr_path = os.path.join(method_dir, "stderr.log")
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--data-dir",
        args.data_dir,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--eval-scope",
        args.eval_scope,
        "--general-limit",
        str(args.general_limit),
        "--domain-limit",
        str(args.domain_limit),
        "--target-old-probes",
        str(args.target_old_probes),
        "--target-old-max-new-tokens",
        str(args.target_old_max_new_tokens),
        "--worker-method",
        method_name,
    ]
    if args.disable_target_old_resolution:
        command.append("--disable-target-old-resolution")
    if args.max_facts is not None:
        command.extend(["--max-facts", str(args.max_facts)])

    progress_events: List[Dict[str, Any]] = []
    with open(stderr_path, "a", encoding="utf-8") as stderr_f:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("__PROGRESS__ "):
                try:
                    progress_events.append(json.loads(line[len("__PROGRESS__ ") :]))
                except json.JSONDecodeError:
                    stderr_f.write(line)
            else:
                stderr_f.write(line)
            stderr_f.flush()
        returncode = process.wait()
        if returncode != 0:
            stderr_f.write(f"Subprocess exited with return code {returncode}\n")

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
            "failed_fact_ids": [],
        }
    summary["returncode"] = returncode
    summary["stdout_log"] = os.path.join(method_dir, "stdout.log")
    summary["stderr_log"] = stderr_path
    summary["_progress_events"] = progress_events
    return summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.worker_method:
        run_method_worker(args.worker_method, args)
        return

    data = load_oilgas_dataset(args.data_dir)
    fact_ids = fact_ids_for_run(data, args.max_facts)
    total_units = len(fact_ids) * len(args.methods)
    progress = tqdm(total=total_units, desc="OilGas single-edit", unit="case")
    summaries = []

    for method_name in args.methods:
        summary = run_method_subprocess(method_name, args)
        for event in summary.pop("_progress_events", []):
            if event.get("event") == "case_done":
                progress.update(1)
                progress.set_postfix_str(f"{event.get('method')} {event.get('fact_id')} {event.get('status')}")
        summaries.append(summary)
    progress.close()

    overall = {
        "data_dir": os.path.abspath(args.data_dir),
        "model": os.path.abspath(args.model),
        "methods": args.methods,
        "facts_requested": args.max_facts,
        "facts_loaded": len(fact_ids),
        "eval_scope": args.eval_scope,
        "target_old_resolution": {
            "enabled": not args.disable_target_old_resolution,
            "methods_requiring_target_old": sorted(METHODS_REQUIRING_TARGET_OLD),
            "max_probes": args.target_old_probes,
            "max_new_tokens": args.target_old_max_new_tokens,
        },
        "process_model": "one-method-per-subprocess",
        "method_summaries": summaries,
        "limitations": [
            "MVP single-edit only.",
            "Post-edit scoring is based on EasyEdit internal metrics.",
            "SFT, LocFT-BF and sequential-edit are not included in this first run.",
        ],
    }
    write_json(os.path.join(args.output_dir, "summary.json"), overall)
    write_summary_csv(os.path.join(args.output_dir, "summary.csv"), summaries)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
