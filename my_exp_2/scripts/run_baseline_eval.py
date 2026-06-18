import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import load_oilgas_dataset, write_json
from eval_utils import build_short_answer_prompt, clean_generated_answer, list_accuracy, score_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("my_exp_2", "data"))
    parser.add_argument("--model", default=os.path.join("my_exp", "models", "Qwen2.5-3B"))
    parser.add_argument("--output-dir", default=os.path.join("my_exp_2", "outputs", "baseline"))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return build_short_answer_prompt(question)


def generate_answer(model: Any, tokenizer: Any, question: str, device: str, max_new_tokens: int) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_ids = generated[0][input_len:]
    decoded = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return clean_generated_answer(decoded, question=question)


def select_questions(questions: List[Dict[str, Any]], max_questions: Optional[int]) -> List[Dict[str, Any]]:
    return questions[:max_questions] if max_questions is not None else questions


def eval_questions(
    model: Any,
    tokenizer: Any,
    questions: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
    desc: str,
) -> Dict[str, Any]:
    results = []
    started = time.time()
    for question in tqdm(questions, desc=desc, unit="q"):
        answer = generate_answer(model, tokenizer, str(question["question"]), device, max_new_tokens)
        results.append(score_answer(question, answer))
    return {
        "total": len(results),
        "accuracy": list_accuracy(results),
        "time_sec": round(time.time() - started, 4),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_oilgas_dataset(args.data_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
    model.to(device)
    model.eval()

    datasets = {
        "fact_questions": select_questions(data["fact_questions"], args.max_questions),
        "domain_questions": select_questions(data["domain_questions"], args.max_questions),
        "general_questions": select_questions(data["general_questions"], args.max_questions),
    }

    summary: Dict[str, Any] = {
        "model": os.path.abspath(args.model),
        "data_dir": os.path.abspath(args.data_dir),
        "device": device,
        "max_questions": args.max_questions,
        "max_new_tokens": args.max_new_tokens,
        "datasets": {},
    }

    for name, questions in datasets.items():
        payload = eval_questions(model, tokenizer, questions, device, args.max_new_tokens, name)
        write_json(os.path.join(args.output_dir, f"{name}_results.json"), payload)
        summary["datasets"][name] = {
            "total": payload["total"],
            "accuracy": payload["accuracy"],
            "time_sec": payload["time_sec"],
        }

    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
