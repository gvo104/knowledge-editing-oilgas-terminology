from typing import Any, Dict, List

import torch

from eval_utils import build_short_answer_prompt, clean_generated_answer, normalize_text


REFUSAL_MARKERS = (
    "не знаю",
    "не могу",
    "нельзя точно",
    "затрудняюсь",
    "нет информации",
    "неизвестно",
    "не уверен",
    "как ии",
    "как искусственный интеллект",
)

UNCERTAIN_MARKERS = (
    "возможно",
    "вероятно",
    "может быть",
    "зависит от контекста",
    "в общем случае",
)

GENERIC_MARKERS = (
    "связан с",
    "относится к области",
    "является важным",
    "это термин",
    "это понятие",
)

STOPWORDS = {
    "и",
    "или",
    "в",
    "во",
    "на",
    "к",
    "ко",
    "с",
    "со",
    "для",
    "из",
    "от",
    "по",
    "что",
    "это",
    "является",
    "являются",
    "есть",
}


def build_probe_prompt(question: str) -> str:
    return build_short_answer_prompt(question)


def ask_model(model: Any, tokenizer: Any, question: str, device: Any, max_new_tokens: int = 48) -> str:
    prompt = build_probe_prompt(question)
    if isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
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


def validate_target_old_answer(answer: str, max_words: int = 24, max_chars: int = 180) -> Dict[str, Any]:
    raw = (answer or "").strip()
    normalized = normalize_text(raw)
    reasons: List[str] = []

    if not raw:
        reasons.append("empty")
    if len(raw) > max_chars:
        reasons.append("too_long")
    if len(normalized.split()) > max_words:
        reasons.append("too_many_words")
    if not any(char.isalnum() for char in normalized):
        reasons.append("no_alnum")
    if "?" in raw:
        reasons.append("contains_question")
    if any(marker in normalized for marker in REFUSAL_MARKERS):
        reasons.append("refusal")
    if any(marker in normalized for marker in UNCERTAIN_MARKERS):
        reasons.append("uncertain")
    if any(marker in normalized for marker in GENERIC_MARKERS):
        reasons.append("generic")

    return {
        "is_valid": not reasons,
        "normalized_answer": normalized,
        "reasons": reasons,
    }


def answer_signature(normalized_answer: str) -> set[str]:
    return {token for token in normalized_answer.split() if token not in STOPWORDS and len(token) > 1}


def answers_are_stable(valid_results: List[Dict[str, Any]]) -> bool:
    if len(valid_results) <= 1:
        return True
    base = answer_signature(valid_results[0]["normalized_answer"])
    if not base:
        return False
    for result in valid_results[1:]:
        current = answer_signature(result["normalized_answer"])
        if not current:
            return False
        overlap = len(base & current) / max(1, min(len(base), len(current)))
        if overlap < 0.5:
            return False
    return True


def target_old_probe_questions(case: Dict[str, Any], max_probes: int) -> List[str]:
    questions = [case["prompt"]]
    for bucket in ("direct_questions", "paraphrase_questions"):
        for question in case.get(bucket, []):
            text = question.get("question")
            if text and text not in questions:
                questions.append(str(text))
            if len(questions) >= max_probes:
                return questions
    return questions[:max_probes]


def resolve_target_old(
    model: Any,
    tokenizer: Any,
    case: Dict[str, Any],
    method_requires_target_old: bool,
    device: Any,
    max_probes: int = 2,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    if not method_requires_target_old:
        return {
            "resolved_target_old": None,
            "target_old_source": "not_required",
            "target_old_is_valid": None,
            "target_old_is_stable": None,
            "raw_model_answer": None,
            "probe_results": [],
        }

    probe_results = []
    for question in target_old_probe_questions(case, max(1, max_probes)):
        raw_answer = ask_model(model, tokenizer, question, device=device, max_new_tokens=max_new_tokens)
        validation = validate_target_old_answer(raw_answer)
        probe_results.append(
            {
                "question": question,
                "raw_model_answer": raw_answer,
                "is_valid": validation["is_valid"],
                "normalized_answer": validation["normalized_answer"],
                "validation_reasons": validation["reasons"],
            }
        )

    valid_results = [result for result in probe_results if result["is_valid"]]
    all_probes_valid = len(valid_results) == len(probe_results)
    stable = answers_are_stable(valid_results)
    if valid_results and all_probes_valid and stable:
        first_valid = valid_results[0]
        return {
            "resolved_target_old": first_valid["raw_model_answer"],
            "target_old_source": "model_current_answer",
            "target_old_is_valid": True,
            "target_old_is_stable": True,
            "raw_model_answer": probe_results[0]["raw_model_answer"] if probe_results else None,
            "probe_results": probe_results,
        }

    return {
        "resolved_target_old": case["target_new"],
        "target_old_source": "fallback_to_target_new",
        "target_old_is_valid": False,
        "target_old_is_stable": stable,
        "raw_model_answer": probe_results[0]["raw_model_answer"] if probe_results else None,
        "probe_results": probe_results,
        "fallback_reasons": {
            "has_valid_answer": bool(valid_results),
            "all_probes_valid": all_probes_valid,
            "stable": stable,
        },
    }
