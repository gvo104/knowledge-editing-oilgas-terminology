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

DEFINITION_STYLE_MARKERS = (
    " это ",
    " является ",
    " называют ",
)

DEFINITION_OBJECT_MARKERS = (
    "процесс",
    "показатель",
    "способность",
    "смесь",
    "материал",
    "термин",
    "понятие",
)

PROMPT_LEAKAGE_PREFIXES = (
    "ответ",
    "ответ пол",
    "ответ полный",
    "ответ кратко",
)

DEFINITION_QUESTION_PREFIXES = (
    "что такое",
    "кто такой",
    "что представляет собой",
    "что называется",
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
    if looks_repetitive_answer(raw, normalized):
        reasons.append("repetitive_gibberish")
    if "?" in raw:
        reasons.append("contains_question")
    if any(normalized.startswith(prefix) for prefix in PROMPT_LEAKAGE_PREFIXES):
        reasons.append("prompt_leakage")
    if normalized in {"ответ", "ответ пол", "ответ полный", "ответ кратко"}:
        reasons.append("fragment")
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


def token_diversity_score(normalized_answer: str) -> float:
    tokens = normalized_answer.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repeated_pattern(token: str, max_pattern_len: int = 3, min_repeats: int = 3) -> bool:
    if len(token) < max_pattern_len * min_repeats:
        return False
    for pattern_len in range(1, min(max_pattern_len, len(token) // min_repeats) + 1):
        if len(token) % pattern_len != 0:
            continue
        pattern = token[:pattern_len]
        if pattern * (len(token) // pattern_len) == token:
            return True
    return False


def looks_repetitive_answer(raw: str, normalized_answer: str) -> bool:
    tokens = normalized_answer.split()
    if not tokens:
        return False

    if len(tokens) == 1:
        token = tokens[0]
        if len(token) >= 8 and repeated_pattern(token):
            return True
        if len(token) >= 8 and len(set(token)) <= 3:
            return True
        return False

    diversity = token_diversity_score(normalized_answer)
    if len(tokens) >= 4 and diversity < 0.34:
        return True

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    most_common = max(counts.values())
    if len(tokens) >= 4 and most_common / len(tokens) >= 0.8:
        return True

    return False


def is_definition_question(question: str) -> bool:
    normalized_question = normalize_text(question)
    return any(normalized_question.startswith(prefix) for prefix in DEFINITION_QUESTION_PREFIXES)


def answer_quality_score(case: Dict[str, Any], question: str, normalized_answer: str) -> int:
    score = 0
    answer_tokens = normalized_answer.split()
    subject = str(case.get("original_subject") or case.get("subject") or "")
    normalized_subject = normalize_text(subject)
    question_signature = answer_signature(normalize_text(question))
    answer_sig = answer_signature(normalized_answer)
    diversity = token_diversity_score(normalized_answer)

    if len(answer_tokens) <= 6:
        score += 3
    elif len(answer_tokens) <= 12:
        score += 2
    elif len(answer_tokens) <= 18:
        score += 1

    if normalized_subject and normalized_answer.startswith(normalized_subject):
        score -= 2
    if normalized_subject and normalized_subject in normalized_answer and not is_definition_question(question):
        score -= 1
    if any(normalized_answer.startswith(prefix) for prefix in PROMPT_LEAKAGE_PREFIXES):
        score -= 4
    if looks_repetitive_answer(normalized_answer, normalized_answer):
        score -= 6

    if not is_definition_question(question):
        if any(marker in f" {normalized_answer} " for marker in DEFINITION_STYLE_MARKERS):
            score -= 1
        if any(marker in normalized_answer for marker in DEFINITION_OBJECT_MARKERS):
            score -= 1

    if diversity < 0.34:
        score -= 3
    elif diversity < 0.5:
        score -= 1

    if question_signature and answer_sig:
        overlap = len(question_signature & answer_sig) / max(1, min(len(question_signature), len(answer_sig)))
        if overlap >= 0.75:
            score -= 2
        elif overlap >= 0.5:
            score -= 1

    return score


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


def invalid_results_are_generation_noise(probe_results: List[Dict[str, Any]]) -> bool:
    noise_reasons = {"empty", "contains_question", "no_alnum", "prompt_leakage", "fragment"}
    invalid_results = [result for result in probe_results if not result["is_valid"]]
    if not invalid_results:
        return True
    for result in invalid_results:
        reasons = set(result.get("validation_reasons", []))
        if not reasons or not reasons.issubset(noise_reasons):
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
        quality_score = answer_quality_score(case, question, validation["normalized_answer"])
        probe_results.append(
            {
                "question": question,
                "raw_model_answer": raw_answer,
                "is_valid": validation["is_valid"],
                "normalized_answer": validation["normalized_answer"],
                "validation_reasons": validation["reasons"],
                "quality_score": quality_score,
            }
        )

    valid_results = [result for result in probe_results if result["is_valid"]]
    valid_results = sorted(
        valid_results,
        key=lambda result: (
            result.get("quality_score", -999),
            -len(result.get("normalized_answer", "").split()),
        ),
        reverse=True,
    )
    all_probes_valid = len(valid_results) == len(probe_results)
    stable = answers_are_stable(valid_results)
    invalids_are_noise = invalid_results_are_generation_noise(probe_results)
    if valid_results and valid_results[0].get("quality_score", -999) >= 2 and stable and (all_probes_valid or invalids_are_noise):
        first_valid = valid_results[0]
        return {
            "resolved_target_old": first_valid["raw_model_answer"],
            "target_old_source": "model_current_answer",
            "target_old_is_valid": True,
            "target_old_is_stable": True,
            "raw_model_answer": probe_results[0]["raw_model_answer"] if probe_results else None,
            "probe_results": probe_results,
            "accepted_with_partial_probes": not all_probes_valid,
            "accepted_quality_score": first_valid.get("quality_score"),
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
            "invalids_are_generation_noise": invalids_are_noise,
            "stable": stable,
            "best_quality_score": valid_results[0].get("quality_score") if valid_results else None,
        },
    }
