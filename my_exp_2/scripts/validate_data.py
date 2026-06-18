import argparse
import os
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_io import QUESTION_TYPES, load_oilgas_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("my_exp_2", "data"))
    parser.add_argument("--output", default=os.path.join("my_exp_2", "outputs", "metrics", "data_validation.json"))
    return parser.parse_args()


def require(condition: bool, errors: List[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_train_records(name: str, records: List[Dict[str, Any]], errors: List[str]) -> None:
    for idx, record in enumerate(records, start=1):
        require("fact_id" in record, errors, f"{name}[{idx}] has no fact_id")
        require("input" in record, errors, f"{name}[{idx}] has no input")
        require("output" in record, errors, f"{name}[{idx}] has no output")


def main() -> None:
    args = parse_args()
    data = load_oilgas_dataset(args.data_dir)
    errors: List[str] = []
    warnings: List[str] = []

    fact_ids = [str(record.get("fact_id")) for record in data["triplets"]]
    unique_fact_ids = set(fact_ids)
    require(len(fact_ids) == len(unique_fact_ids), errors, "triplets contain duplicate fact_id values")
    require(len(fact_ids) == 20, warnings, f"expected 20 triplets, got {len(fact_ids)}")

    edit_ids = set(data["edit_requests_by_fact_id"])
    for fact_id in unique_fact_ids:
        require(fact_id in edit_ids, errors, f"missing edit request for {fact_id}")
        edit_request = data["edit_requests_by_fact_id"].get(fact_id)
        if edit_request:
            subject = str(edit_request.get("subject", ""))
            prompt = str(edit_request.get("prompt", ""))
            require(
                subject in prompt or subject.lower() in prompt.lower(),
                errors,
                f"edit request {fact_id} subject must appear in prompt for EasyEdit compatibility",
            )

    for question in data["fact_questions"]:
        fact_id = str(question.get("fact_id", ""))
        require(fact_id in unique_fact_ids, errors, f"question {question.get('question_id')} references unknown fact_id={fact_id}")
        require(bool(question.get("expected_answer")), errors, f"question {question.get('question_id')} has no expected_answer")
        require(isinstance(question.get("aliases"), list), errors, f"question {question.get('question_id')} aliases must be a list")

    for fact_id in unique_fact_ids:
        grouped = data["fact_questions_by_fact_id"].get(fact_id, {})
        for question_type in QUESTION_TYPES:
            require(grouped.get(question_type), errors, f"{fact_id} has no {question_type} questions")

    sequential_order = data["sequential_order"]
    require(len(sequential_order) == len(set(sequential_order)), errors, "sequential_order contains duplicates")
    for fact_id in sequential_order:
        require(fact_id in unique_fact_ids, errors, f"sequential_order references unknown fact_id={fact_id}")

    validate_train_records("lora_train", data["lora_train"], errors)
    validate_train_records("lora_augmented_train", data["lora_augmented_train"], errors)

    summary = {
        "status": "ok" if not errors else "error",
        "data_dir": os.path.abspath(args.data_dir),
        "triplets": len(data["triplets"]),
        "edit_requests": len(data["edit_requests"]),
        "fact_questions": len(data["fact_questions"]),
        "domain_questions": len(data["domain_questions"]),
        "general_questions": len(data["general_questions"]),
        "sequential_order": len(sequential_order),
        "lora_train": len(data["lora_train"]),
        "lora_augmented_train": len(data["lora_augmented_train"]),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(args.output, summary)

    print(f"status={summary['status']}")
    print(f"triplets={summary['triplets']} edit_requests={summary['edit_requests']} fact_questions={summary['fact_questions']}")
    print(f"domain_questions={summary['domain_questions']} general_questions={summary['general_questions']}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
