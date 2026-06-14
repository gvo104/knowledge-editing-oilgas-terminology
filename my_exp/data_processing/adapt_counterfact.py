"""
Convert CounterFact-style JSON into the lightweight JSONL format used in my_exp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=Path, help="Path to counterfact-edit.json")
    parser.add_argument("--output-path", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--portability-path",
        type=Path,
        default=None,
        help="Optional path to counterfact portability JSON (for example counterfact_portability_gpt4.json)",
    )
    parser.add_argument(
        "--general-locality-path",
        type=Path,
        default=None,
        help="Optional path to unrelated/general locality JSON",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
    return None


def maybe_first_answer(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first and isinstance(first[0], str):
            return first[0]
    return None


def maybe_first_prompt_answer_pair(value: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(value, dict):
        prompt = first_nonempty(value.get("prompt"), value.get("question"), value.get("src"))
        answer = first_nonempty(
            value.get("ground_truth"),
            value.get("answer"),
            value.get("target"),
            value.get("expected"),
            maybe_first_answer(value.get("answers")),
        )
        return prompt, answer

    if isinstance(value, list):
        for item in value:
            prompt, answer = maybe_first_prompt_answer_pair(item)
            if prompt is not None and answer is not None:
                return prompt, answer

    return None, None


def lookup_case_mapping(side_data: Any, case_id: int) -> Optional[Dict[str, Any]]:
    if isinstance(side_data, list):
        for candidate in side_data:
            if isinstance(candidate, dict) and candidate.get("case_id") == case_id:
                return candidate
    elif isinstance(side_data, dict):
        direct = side_data.get(str(case_id))
        if isinstance(direct, dict):
            return direct
    return None


def extract_portability(record: Dict[str, Any], side_record: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    if side_record is not None and isinstance(side_record.get("portability"), dict):
        portability = side_record["portability"]
        prompt = first_nonempty(portability.get("New Question"))
        answer = first_nonempty(portability.get("New Answer"))
        if prompt is not None and answer is not None:
            return prompt, answer

    candidates: List[Any] = []
    if side_record is not None:
        candidates.extend(
            [
                side_record.get("portability"),
                side_record.get("one_hop"),
                side_record.get("One Hop"),
                side_record.get("Reasoning"),
                side_record,
            ]
        )
    candidates.extend(
        [
            record.get("portability"),
            record.get("one_hop"),
            record.get("One Hop"),
            record.get("Reasoning"),
        ]
    )

    for candidate in candidates:
        prompt, answer = maybe_first_prompt_answer_pair(candidate)
        if prompt is not None and answer is not None:
            return prompt, answer
    return None, None


def extract_general_locality(side_record: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    if side_record is None:
        return None, None

    if isinstance(side_record.get("unrelated_relation"), dict):
        unrelated = side_record["unrelated_relation"]
        prompt = first_nonempty(unrelated.get("question"))
        answer = first_nonempty(unrelated.get("object"))
        if prompt is not None and answer is not None:
            return prompt, answer

    candidates = [
        side_record.get("locality"),
        side_record.get("general_locality"),
        side_record.get("unrelated"),
        side_record,
    ]
    for candidate in candidates:
        prompt, answer = maybe_first_prompt_answer_pair(candidate)
        if prompt is not None and answer is not None:
            return prompt, answer
    return None, None


def build_record(
    record: Dict[str, Any],
    *,
    case_id: int,
    portability_side_data: Any = None,
    general_locality_side_data: Any = None,
) -> Dict[str, Any]:
    original_case_id = int(record.get("case_id", case_id))
    side_portability = (
        lookup_case_mapping(portability_side_data, original_case_id)
        if portability_side_data is not None
        else None
    )
    side_general_locality = (
        lookup_case_mapping(general_locality_side_data, original_case_id)
        if general_locality_side_data is not None
        else None
    )

    portability_prompt, portability_ground_truth = extract_portability(record, side_portability)
    general_locality_prompt, general_locality_ground_truth = extract_general_locality(side_general_locality)

    rephrase_prompt = first_nonempty(
        record.get("rephrase_prompt"),
        record.get("rephrase"),
        record.get("paraphrase_prompt"),
        record.get("paraphrase"),
    )

    target_new = first_nonempty(record.get("target_new"), record.get("alt"), record.get("target"))
    ground_truth = first_nonempty(
        record.get("ground_truth"),
        record.get("target_old"),
        maybe_first_answer(record.get("answers")),
    )
    prompt = first_nonempty(record.get("prompt"), record.get("src"))
    subject = first_nonempty(record.get("subject"))
    locality_prompt = first_nonempty(record.get("locality_prompt"), record.get("loc"))
    locality_ground_truth = first_nonempty(
        record.get("locality_ground_truth"),
        record.get("loc_ans"),
    )

    if prompt is None or target_new is None:
        raise ValueError(f"Record {case_id} is missing prompt or target_new")

    return {
        "case_id": case_id + 1,
        "source_case_id": original_case_id,
        "source_dataset": "counterfact",
        "prompt": prompt,
        "subject": subject,
        "ground_truth": ground_truth,
        "target_new": target_new,
        "rephrase_prompt": rephrase_prompt,
        "locality_prompt": locality_prompt,
        "locality_ground_truth": locality_ground_truth,
        "general_locality_prompt": general_locality_prompt,
        "general_locality_ground_truth": general_locality_ground_truth,
        "portability_prompt": portability_prompt,
        "portability_ground_truth": portability_ground_truth,
        "notes": "Converted from CounterFact-style JSON.",
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    raw = load_json(args.input_path)
    portability_side_data = load_json(args.portability_path) if args.portability_path else None
    general_locality_side_data = load_json(args.general_locality_path) if args.general_locality_path else None

    if not isinstance(raw, list):
        raise ValueError("CounterFact input must be a JSON list")

    records = []
    skipped_missing_portability = 0
    skipped_missing_general_locality = 0

    for i, record in enumerate(raw):
        built = build_record(
            record,
            case_id=len(records),
            portability_side_data=portability_side_data,
            general_locality_side_data=general_locality_side_data,
        )
        if portability_side_data is not None and built["portability_prompt"] is None:
            skipped_missing_portability += 1
            continue
        if general_locality_side_data is not None and built["general_locality_prompt"] is None:
            skipped_missing_general_locality += 1
            continue
        records.append(built)
        if args.max_cases is not None and len(records) >= args.max_cases:
            break

    write_jsonl(args.output_path, records)

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "portability_path": str(args.portability_path) if args.portability_path else None,
        "general_locality_path": str(args.general_locality_path) if args.general_locality_path else None,
        "records_total": len(records),
        "source_records_total": len(raw),
        "skipped_missing_portability": skipped_missing_portability,
        "skipped_missing_general_locality": skipped_missing_general_locality,
        "source_dataset": "counterfact",
    }
    args.output_path.with_suffix(args.output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
