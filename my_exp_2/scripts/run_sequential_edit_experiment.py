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
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import build_easyedit_kwargs, build_single_edit_case, load_oilgas_dataset, write_json
from eval_utils import harmonic_or_none, list_accuracy, mean_or_none, normalize_easyedit_metric, score_answer
from run_baseline_eval import generate_answer
from run_single_edit_experiment import (
    METHODS_REQUIRING_TARGET_OLD,
    METHOD_HPARAM_PATHS,
    format_eta,
    gpu_helpers,
    json_safe,
    method_components,
    prepare_wise_runtime,
    suppress_stdio,
)
from target_old_resolver import resolve_target_old


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["ROME"], choices=sorted(METHOD_HPARAM_PATHS))
    parser.add_argument("--data-dir", default=os.path.join("my_exp_2", "data"))
    parser.add_argument("--model", default=os.path.join("my_exp", "models", "Qwen2.5-3B"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join("my_exp_2", "outputs", "sequential_edit", "qwen25_3b_seq"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-scope", choices=["fact-only", "fact-plus-general", "full"], default="fact-plus-general")
    parser.add_argument(
        "--eval-mode",
        choices=["sample", "full"],
        default="sample",
        help="sample uses configured limits; full evaluates all seen facts and all selected global/domain questions.",
    )
    parser.add_argument("--retention-fact-limit", type=int, default=5)
    parser.add_argument("--general-limit", type=int, default=5)
    parser.add_argument("--domain-limit", type=int, default=5)
    parser.add_argument("--target-old-probes", type=int, default=2)
    parser.add_argument("--target-old-max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--save-generation-details", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--disable-target-old-resolution",
        action="store_true",
        help="Use target_old from JSON/fallback only. Intended for debugging.",
    )
    parser.add_argument("--worker-method", choices=sorted(METHOD_HPARAM_PATHS), default=None)
    return parser.parse_args()


def sequential_fact_ids(data: Dict[str, Any], max_steps: Optional[int]) -> List[str]:
    fact_ids = [str(fact_id) for fact_id in data["sequential_order"]]
    return fact_ids[:max_steps] if max_steps is not None else fact_ids


def apply_eval_mode_defaults(args: argparse.Namespace) -> None:
    if args.eval_mode != "full":
        return
    args.retention_fact_limit = None
    args.general_limit = None
    args.domain_limit = None
    args.save_generation_details = True


def device_string(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def ensure_cuda_or_raise(allow_cpu: bool, context: str) -> None:
    if torch.cuda.is_available() or allow_cpu:
        return
    raise SystemExit(
        f"{context}: CUDA is not available. Sequential experiment is GPU-only by default. "
        "Use --allow-cpu only for explicit debug runs."
    )


def select_questions(questions: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    return questions[:limit] if limit is not None else questions


def evaluate_questions(
    model: Any,
    tokenizer: Any,
    questions: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
    save_details: bool,
) -> Dict[str, Any]:
    results = []
    with torch.inference_mode():
        for question in questions:
            answer = generate_answer(model, tokenizer, str(question["question"]), device, max_new_tokens)
            results.append(score_answer(question, answer))
    payload = {
        "total": len(results),
        "accuracy": list_accuracy(results),
    }
    if save_details:
        payload["results"] = results
    return payload


def evaluate_seen_facts(
    model: Any,
    tokenizer: Any,
    data: Dict[str, Any],
    seen_fact_ids: List[str],
    device: str,
    max_new_tokens: int,
    retention_fact_limit: Optional[int],
    save_details: bool,
) -> Dict[str, Any]:
    selected_fact_ids = seen_fact_ids[:retention_fact_limit] if retention_fact_limit is not None else seen_fact_ids
    per_fact = []
    direct_accs: List[Optional[float]] = []
    paraphrase_accs: List[Optional[float]] = []
    for fact_id in selected_fact_ids:
        case = build_single_edit_case(fact_id, data)
        direct_eval = evaluate_questions(
            model,
            tokenizer,
            case["direct_questions"],
            device,
            max_new_tokens,
            save_details=save_details,
        )
        paraphrase_eval = evaluate_questions(
            model,
            tokenizer,
            case["paraphrase_questions"],
            device,
            max_new_tokens,
            save_details=save_details,
        )
        direct_accs.append(direct_eval["accuracy"])
        paraphrase_accs.append(paraphrase_eval["accuracy"])
        fact_payload = {
            "fact_id": fact_id,
            "direct_accuracy": direct_eval["accuracy"],
            "paraphrase_accuracy": paraphrase_eval["accuracy"],
            "direct_total": direct_eval["total"],
            "paraphrase_total": paraphrase_eval["total"],
        }
        if save_details:
            fact_payload["direct_results"] = direct_eval.get("results", [])
            fact_payload["paraphrase_results"] = paraphrase_eval.get("results", [])
        per_fact.append(fact_payload)
    retention = mean_or_none(direct_accs)
    retained_generalization = mean_or_none(paraphrase_accs)
    return {
        "fact_count": len(selected_fact_ids),
        "retention": retention,
        "retained_generalization": retained_generalization,
        "per_fact": per_fact,
    }


def evaluate_global_sets(
    model: Any,
    tokenizer: Any,
    general_questions: List[Dict[str, Any]],
    domain_questions: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
    eval_scope: str,
    save_details: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if eval_scope in {"fact-plus-general", "full"}:
        general_eval = evaluate_questions(
            model,
            tokenizer,
            general_questions,
            device,
            max_new_tokens,
            save_details=save_details,
        )
        payload["general_questions"] = general_eval
    if eval_scope == "full":
        domain_eval = evaluate_questions(
            model,
            tokenizer,
            domain_questions,
            device,
            max_new_tokens,
            save_details=save_details,
        )
        payload["domain_questions"] = domain_eval
    return payload


def compact_error_result(
    method_name: str,
    fact_id: str,
    step_index: int,
    case: Dict[str, Any],
    gpu_stats_fn: Any,
    exc: Exception,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "fact_id": fact_id,
        "method": method_name,
        "step_index": step_index,
        "time_sec": None,
        "gpu": gpu_stats_fn(),
        "input_edit_request": case.get("edit_request"),
        "fact": case.get("fact"),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def compact_success_result(result: Dict[str, Any], save_generation_details: bool) -> Dict[str, Any]:
    payload = {
        "status": result.get("status"),
        "fact_id": result.get("fact_id"),
        "method": result.get("method"),
        "time_sec": result.get("time_sec"),
        "gpu": result.get("gpu"),
        "step_index": result.get("step_index"),
        "seen_fact_ids": result.get("seen_fact_ids"),
        "input_edit_request": result.get("input_edit_request"),
        "fact": result.get("fact"),
        "eval_scope": result.get("eval_scope"),
        "ground_truth_source": result.get("ground_truth_source"),
        "target_old_resolution": result.get("target_old_resolution"),
        "current_edit_metrics": result.get("current_edit_metrics"),
        "sequential_metrics": result.get("sequential_metrics"),
        "retention_eval": result.get("retention_eval"),
        "global_eval": result.get("global_eval"),
    }
    if save_generation_details:
        payload["raw_easyedit_metrics"] = result.get("raw_easyedit_metrics")
    return payload


def summary_row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status"),
        "fact_id": result.get("fact_id"),
        "method": result.get("method"),
        "step_index": result.get("step_index"),
        "time_sec": result.get("time_sec"),
        "gpu": result.get("gpu"),
        "target_old_resolution": result.get("target_old_resolution"),
        "sequential_metrics": result.get("sequential_metrics"),
        "error": result.get("error"),
    }


def run_step(
    editor: Any,
    case: Dict[str, Any],
    method_name: str,
    data: Dict[str, Any],
    seen_fact_ids: List[str],
    general_questions: List[Dict[str, Any]],
    domain_questions: List[Dict[str, Any]],
    args: argparse.Namespace,
    gpu_stats_fn: Any,
    reset_gpu_peak_fn: Any,
) -> Dict[str, Any]:
    reset_gpu_peak_fn()
    started = time.time()
    if args.disable_target_old_resolution:
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
            max_probes=args.target_old_probes,
            max_new_tokens=args.target_old_max_new_tokens,
        )
        if target_old_resolution["resolved_target_old"] is not None:
            case["ground_truth"] = target_old_resolution["resolved_target_old"]
            case["ground_truth_source"] = target_old_resolution["target_old_source"]

    edit_kwargs = build_easyedit_kwargs(
        case,
        eval_scope=args.eval_scope,
        general_limit=args.general_limit,
        domain_limit=args.domain_limit,
    )
    metrics, _, _ = editor.edit(**edit_kwargs, sequential_edit=True)
    raw_metric = metrics[0] if metrics else {}
    normalized = normalize_easyedit_metric(raw_metric)
    normalized["easyedit_edit_quality"] = normalized.get("edit_quality")

    model = editor.model if hasattr(editor.model, "generate") else editor.model.model
    device = device_string(editor.hparams.device)
    retention_eval = evaluate_seen_facts(
        model,
        editor.tok,
        data,
        seen_fact_ids,
        device,
        args.generation_max_new_tokens,
        args.retention_fact_limit,
        save_details=args.save_generation_details,
    )
    global_eval = evaluate_global_sets(
        model,
        editor.tok,
        general_questions,
        domain_questions,
        device,
        args.generation_max_new_tokens,
        args.eval_scope,
        save_details=args.save_generation_details,
    )

    seq_metrics = {
        "current_reliability": normalized.get("reliability"),
        "current_generalization": normalized.get("generalization"),
        "current_reverse": normalized.get("reverse"),
        "current_neighbor": normalized.get("neighbor"),
        "current_fact_locality": normalized.get("fact_locality"),
        "current_global_locality": normalized.get("global_locality"),
        "current_domain_score": normalized.get("domain_score"),
        "retention": retention_eval["retention"],
        "retained_generalization": retention_eval["retained_generalization"],
        "global_locality_generation": (global_eval.get("general_questions") or {}).get("accuracy"),
        "domain_score_generation": (global_eval.get("domain_questions") or {}).get("accuracy"),
    }
    locality_for_quality = mean_or_none(
        [
            normalized.get("fact_locality"),
            (global_eval.get("general_questions") or {}).get("accuracy"),
        ]
    )
    seq_metrics["sequential_quality"] = harmonic_or_none(
        [
            normalized.get("reliability"),
            normalized.get("generalization"),
            locality_for_quality,
            retention_eval["retention"],
        ]
    )

    elapsed = round(time.time() - started, 4)
    return {
        "status": "success",
        "fact_id": case["fact_id"],
        "method": editor.alg_name,
        "time_sec": elapsed,
        "gpu": gpu_stats_fn(),
        "step_index": len(seen_fact_ids),
        "seen_fact_ids": list(seen_fact_ids),
        "input_edit_request": case["edit_request"],
        "fact": case["fact"],
        "eval_scope": args.eval_scope,
        "ground_truth_source": case["ground_truth_source"],
        "target_old_resolution": target_old_resolution,
        "current_edit_metrics": normalized,
        "sequential_metrics": seq_metrics,
        "retention_eval": retention_eval,
        "global_eval": global_eval,
        "raw_easyedit_metrics": json_safe(metrics) if args.save_generation_details else None,
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
        "total_steps": len(results),
        "successful_steps": len(successful),
        "failed_steps": len(failed),
        "success_rate": round(len(successful) / len(results), 6) if results else None,
        "mean_time_sec": mean_or_none(result.get("time_sec") for result in successful),
        "mean_peak_gpu_gb": mean_or_none((result.get("gpu", {}) or {}).get("peak_allocated_gb") for result in successful),
        "mean_current_reliability": mean_or_none(result["sequential_metrics"].get("current_reliability") for result in successful),
        "mean_current_generalization": mean_or_none(result["sequential_metrics"].get("current_generalization") for result in successful),
        "mean_retention": mean_or_none(result["sequential_metrics"].get("retention") for result in successful),
        "mean_global_locality": mean_or_none(result["sequential_metrics"].get("global_locality_generation") for result in successful),
        "mean_domain_score": mean_or_none(result["sequential_metrics"].get("domain_score_generation") for result in successful),
        "mean_sequential_quality": mean_or_none(result["sequential_metrics"].get("sequential_quality") for result in successful),
        "target_old_sources": target_old_sources,
        "failed_fact_ids": [result.get("fact_id") for result in failed],
    }


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "total_steps",
        "successful_steps",
        "failed_steps",
        "success_rate",
        "mean_time_sec",
        "mean_peak_gpu_gb",
        "mean_current_reliability",
        "mean_current_generalization",
        "mean_retention",
        "mean_global_locality",
        "mean_domain_score",
        "mean_sequential_quality",
        "target_old_sources",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["target_old_sources"] = json.dumps(row.get("target_old_sources", {}), ensure_ascii=False)
            writer.writerow(rendered)


def run_method_worker(method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    torch, reset_gpu_peak_fn, gpu_stats_fn = gpu_helpers()
    BaseEditor, hparams_cls, hparams_path = method_components(method_name)
    data = load_oilgas_dataset(args.data_dir)
    fact_ids = sequential_fact_ids(data, args.max_steps)
    ensure_cuda_or_raise(args.allow_cpu, f"worker[{method_name}]")
    selected_general_questions = (
        select_questions(data["general_questions"], args.general_limit) if args.eval_scope in {"fact-plus-general", "full"} else []
    )
    selected_domain_questions = (
        select_questions(data["domain_questions"], args.domain_limit) if args.eval_scope == "full" else []
    )

    method_dir = os.path.join(args.output_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)
    stdout_path = os.path.join(method_dir, "stdout.log")
    stderr_path = os.path.join(method_dir, "stderr.log")

    with open(stdout_path, "w", encoding="utf-8") as method_log, open(stderr_path, "w", encoding="utf-8") as error_log:
        hparams = hparams_cls.from_hparams(hparams_path)
        hparams.model_name = args.model
        if method_name == "WISE":
            hparams.sequential_edit = True
            prepare_wise_runtime()
        if args.allow_cpu and not torch.cuda.is_available() and isinstance(getattr(hparams, "device", None), int):
            hparams.device = "cpu"
        with suppress_stdio():
            editor = BaseEditor.from_hparams(hparams)

        results = []
        method_started = time.time()
        print("__PROGRESS__ " + json.dumps({"event": "method_start", "method": method_name, "total_steps": len(fact_ids)}), flush=True)

        seen_fact_ids: List[str] = []
        for idx, fact_id in enumerate(fact_ids, start=1):
            case = build_single_edit_case(fact_id, data)
            seen_fact_ids.append(fact_id)
            try:
                with suppress_stdio():
                    result = run_step(
                        editor,
                        case,
                        method_name,
                        data,
                        seen_fact_ids,
                        selected_general_questions,
                        selected_domain_questions,
                        args,
                        gpu_stats_fn,
                        reset_gpu_peak_fn,
                    )
            except Exception as exc:
                result = compact_error_result(method_name, fact_id, idx, case, gpu_stats_fn, exc)
                error_log.write(result["error"]["traceback"] + "\n")
                error_log.flush()

            compact_result = compact_success_result(result, args.save_generation_details) if result["status"] == "success" else result
            results.append(summary_row_from_result(compact_result))
            write_json(os.path.join(method_dir, f"step_{idx:03d}_{fact_id}.json"), compact_result)

            elapsed_total = time.time() - method_started
            avg = elapsed_total / idx
            eta = avg * (len(fact_ids) - idx)
            metrics = compact_result.get("sequential_metrics", {})
            method_log.write(
                "[step] {idx}/{total} fact_id={fact_id} status={status} time={time} "
                "rel={rel} gen={gen} retention={ret} global={glob} domain={domain} quality={quality} elapsed={elapsed} eta={eta}\n".format(
                    idx=idx,
                    total=len(fact_ids),
                    fact_id=fact_id,
                    status=result["status"],
                    time=format_eta(result.get("time_sec")),
                    rel=metrics.get("current_reliability"),
                    gen=metrics.get("current_generalization"),
                    ret=metrics.get("retention"),
                    glob=metrics.get("global_locality_generation"),
                    domain=metrics.get("domain_score_generation"),
                    quality=metrics.get("sequential_quality"),
                    elapsed=format_eta(elapsed_total),
                    eta=format_eta(eta),
                )
            )
            method_log.flush()
            print(
                "__PROGRESS__ "
                + json.dumps(
                    {
                        "event": "step_done",
                        "method": method_name,
                        "fact_id": fact_id,
                        "status": compact_result["status"],
                        "current": idx,
                        "total": len(fact_ids),
                    }
                ),
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if compact_result["status"] != "success":
                break

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
        "--eval-mode",
        args.eval_mode,
        "--general-limit",
        str(args.general_limit if args.general_limit is not None else -1),
        "--domain-limit",
        str(args.domain_limit if args.domain_limit is not None else -1),
        "--target-old-probes",
        str(args.target_old_probes),
        "--target-old-max-new-tokens",
        str(args.target_old_max_new_tokens),
        "--generation-max-new-tokens",
        str(args.generation_max_new_tokens),
        "--worker-method",
        method_name,
    ]
    if args.retention_fact_limit is not None:
        command.extend(["--retention-fact-limit", str(args.retention_fact_limit)])
    if args.disable_target_old_resolution:
        command.append("--disable-target-old-resolution")
    if args.save_generation_details:
        command.append("--save-generation-details")
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])

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
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
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
    if args.general_limit == -1:
        args.general_limit = None
    if args.domain_limit == -1:
        args.domain_limit = None
    apply_eval_mode_defaults(args)
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_cuda_or_raise(args.allow_cpu, "parent")

    if args.worker_method:
        run_method_worker(args.worker_method, args)
        return

    data = load_oilgas_dataset(args.data_dir)
    fact_ids = sequential_fact_ids(data, args.max_steps)
    total_units = len(fact_ids) * len(args.methods)
    progress = tqdm(total=total_units, desc="OilGas sequential-edit", unit="step")
    summaries = []

    for method_name in args.methods:
        summary = run_method_subprocess(method_name, args)
        for event in summary.pop("_progress_events", []):
            if event.get("event") == "step_done":
                progress.update(1)
                progress.set_postfix_str(f"{event.get('method')} {event.get('fact_id')} {event.get('status')}")
        summaries.append(summary)
    progress.close()

    overall = {
        "data_dir": os.path.abspath(args.data_dir),
        "model": os.path.abspath(args.model),
        "methods": args.methods,
        "steps_requested": args.max_steps,
        "steps_loaded": len(fact_ids),
        "eval_scope": args.eval_scope,
        "eval_mode": args.eval_mode,
        "target_old_resolution": {
            "enabled": not args.disable_target_old_resolution,
            "methods_requiring_target_old": sorted(METHODS_REQUIRING_TARGET_OLD),
            "max_probes": args.target_old_probes,
            "max_new_tokens": args.target_old_max_new_tokens,
        },
        "generation_eval": {
            "max_new_tokens": args.generation_max_new_tokens,
            "retention_fact_limit": args.retention_fact_limit,
            "general_limit": args.general_limit,
            "domain_limit": args.domain_limit,
            "full_generation_details": args.save_generation_details,
        },
        "runtime_options": {
            "allow_cpu": args.allow_cpu,
            "save_generation_details": args.save_generation_details,
        },
        "process_model": "one-method-per-subprocess",
        "method_summaries": summaries,
        "limitations": [
            "Sequential evaluation uses generation-based retention/domain/general checks.",
            "Current edit quality still comes from EasyEdit internal metrics for the current fact.",
            "This MVP covers sequential LoRA/ROME/MEMIT/WISE only.",
        ],
    }
    write_json(os.path.join(args.output_dir, "summary.json"), overall)
    write_summary_csv(os.path.join(args.output_dir, "summary.csv"), summaries)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
