import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


QUESTION_TYPES = ("direct", "paraphrase", "reverse", "neighbor", "locality")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_triplets(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_edit_requests(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_fact_questions(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_domain_questions(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_general_questions(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_sequential_order(path: str) -> List[str]:
    payload = load_json(path)
    if isinstance(payload, dict):
        return list(payload.get("order", []))
    return list(payload)


def by_key(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        result[str(record[key])] = record
    return result


def group_questions_by_fact_id(questions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for question in questions:
        fact_id = str(question.get("fact_id", ""))
        question_type = str(question.get("question_type", ""))
        grouped[fact_id][question_type].append(question)
    return {fact_id: dict(type_map) for fact_id, type_map in grouped.items()}


def load_oilgas_dataset(data_dir: str) -> Dict[str, Any]:
    triplets = load_triplets(os.path.join(data_dir, "triplets", "oilgas.json"))
    edit_requests = load_edit_requests(os.path.join(data_dir, "edit_requests", "oilgas_edit_requests.json"))
    fact_questions = load_fact_questions(os.path.join(data_dir, "evaluate_set", "fact_questions.json"))
    domain_questions = load_domain_questions(os.path.join(data_dir, "evaluate_set", "domain_questions.json"))
    general_questions = load_general_questions(os.path.join(data_dir, "evaluate_set", "general_questions.json"))
    sequential_order = load_sequential_order(os.path.join(data_dir, "sequential", "sequential_order.json"))

    lora_train_path = os.path.join(data_dir, "train_sets", "lora_train.json")
    lora_augmented_train_path = os.path.join(data_dir, "train_sets", "lora_augmented_train.json")

    return {
        "data_dir": data_dir,
        "triplets": triplets,
        "edit_requests": edit_requests,
        "fact_questions": fact_questions,
        "domain_questions": domain_questions,
        "general_questions": general_questions,
        "sequential_order": sequential_order,
        "lora_train": load_json(lora_train_path) if os.path.exists(lora_train_path) else [],
        "lora_augmented_train": load_json(lora_augmented_train_path) if os.path.exists(lora_augmented_train_path) else [],
        "facts_by_id": by_key(triplets, "fact_id"),
        "edit_requests_by_fact_id": by_key(edit_requests, "fact_id"),
        "fact_questions_by_fact_id": group_questions_by_fact_id(fact_questions),
    }


def first_question(questions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return questions[0] if questions else None


def question_texts(questions: List[Dict[str, Any]]) -> List[str]:
    return [str(question.get("question", "")) for question in questions if question.get("question")]


def subject_span_in_prompt(subject: str, prompt: str) -> str:
    if subject in prompt:
        return subject
    subject_lower = subject.lower()
    prompt_lower = prompt.lower()
    index = prompt_lower.find(subject_lower)
    if index >= 0:
        return prompt[index : index + len(subject)]
    return subject


def build_single_edit_case(fact_id: str, loaded_data: Dict[str, Any]) -> Dict[str, Any]:
    facts_by_id = loaded_data["facts_by_id"]
    requests_by_id = loaded_data["edit_requests_by_fact_id"]
    questions_by_fact_id = loaded_data["fact_questions_by_fact_id"]

    if fact_id not in facts_by_id:
        raise KeyError(f"Unknown fact_id: {fact_id}")
    if fact_id not in requests_by_id:
        raise KeyError(f"No edit request for fact_id: {fact_id}")

    edit_request = requests_by_id[fact_id]
    grouped_questions = questions_by_fact_id.get(fact_id, {})
    direct_questions = grouped_questions.get("direct", [])
    paraphrase_questions = grouped_questions.get("paraphrase", [])
    reverse_questions = grouped_questions.get("reverse", [])
    neighbor_questions = grouped_questions.get("neighbor", [])
    locality_questions = grouped_questions.get("locality", [])

    target_old = edit_request.get("target_old")
    ground_truth_fallback = "target_new"
    ground_truth = target_old
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        ground_truth = str(edit_request.get("target_new", ""))

    direct = first_question(direct_questions)
    if direct is not None and direct.get("expected_answer"):
        ground_truth = str(direct["expected_answer"]) if target_old else ground_truth

    prompt = str(edit_request["prompt"])
    original_subject = str(edit_request["subject"])
    easyedit_subject = subject_span_in_prompt(original_subject, prompt)

    return {
        "fact_id": fact_id,
        "fact": facts_by_id[fact_id],
        "edit_request": edit_request,
        "prompt": prompt,
        "subject": easyedit_subject,
        "original_subject": original_subject,
        "target_new": str(edit_request["target_new"]),
        "target_old": target_old,
        "ground_truth": ground_truth,
        "ground_truth_source": "target_old" if target_old else ground_truth_fallback,
        "direct_questions": direct_questions,
        "paraphrase_questions": paraphrase_questions,
        "reverse_questions": reverse_questions,
        "neighbor_questions": neighbor_questions,
        "locality_questions": locality_questions,
        "domain_questions": loaded_data["domain_questions"],
        "general_questions": loaded_data["general_questions"],
    }


def build_easyedit_kwargs(
    case: Dict[str, Any],
    eval_scope: str = "fact-plus-general",
    general_limit: Optional[int] = 5,
    domain_limit: Optional[int] = 5,
) -> Dict[str, Any]:
    # EasyEdit's single-edit API accepts one auxiliary prompt per request.
    # The full question groups stay in raw results for the richer evaluator.
    rephrase_prompts = question_texts(case["paraphrase_questions"])[:1] or None

    locality_inputs: Dict[str, Dict[str, List[str]]] = {}
    if case["neighbor_questions"]:
        questions = case["neighbor_questions"][:1]
        locality_inputs["neighbor"] = {
            "prompt": question_texts(questions),
            "ground_truth": [str(q["expected_answer"]) for q in questions],
        }
    if case["locality_questions"]:
        questions = case["locality_questions"][:1]
        locality_inputs["fact_locality"] = {
            "prompt": question_texts(questions),
            "ground_truth": [str(q["expected_answer"]) for q in questions],
        }
    if eval_scope in {"fact-plus-general", "full"}:
        general_questions = case["general_questions"]
        if general_limit is not None:
            general_questions = general_questions[:general_limit]
        general_questions = general_questions[:1]
        locality_inputs["global"] = {
            "prompt": question_texts(general_questions),
            "ground_truth": [str(q["expected_answer"]) for q in general_questions],
        }
    if eval_scope == "full":
        domain_questions = case["domain_questions"]
        if domain_limit is not None:
            domain_questions = domain_questions[:domain_limit]
        domain_questions = domain_questions[:1]
        locality_inputs["domain"] = {
            "prompt": question_texts(domain_questions),
            "ground_truth": [str(q["expected_answer"]) for q in domain_questions],
        }

    portability_inputs = None
    if case["reverse_questions"]:
        questions = case["reverse_questions"][:1]
        portability_inputs = {
            "reverse": {
                "prompt": question_texts(questions),
                "ground_truth": [str(q["expected_answer"]) for q in questions],
            }
        }

    kwargs: Dict[str, Any] = {
        "prompts": case["prompt"],
        "ground_truth": case["ground_truth"],
        "target_new": case["target_new"],
        "subject": case["subject"],
        # WISE expects loc_prompt in the prepared request and uses it as the in-scope phrase.
        "loc_prompts": case["subject"],
        "keep_original_weight": True,
        "verbose": False,
    }
    if rephrase_prompts:
        kwargs["rephrase_prompts"] = rephrase_prompts
    if locality_inputs:
        kwargs["locality_inputs"] = locality_inputs
    if portability_inputs:
        kwargs["portability_inputs"] = portability_inputs
    return kwargs
