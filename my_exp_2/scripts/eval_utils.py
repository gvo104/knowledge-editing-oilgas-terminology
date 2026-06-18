import re
from statistics import harmonic_mean, mean
from typing import Any, Dict, Iterable, List, Optional


ROLE_MARKERS = (
    "assistant",
    "user",
    "system",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end|>",
    "🐉",
    "🐙",
)


def build_short_answer_prompt(question: str) -> str:
    return f"Вопрос: {str(question).strip()}\nОтвет кратко:"


def clean_generated_answer(answer: Any, question: Optional[str] = None) -> str:
    value = str(answer or "")
    for marker in ROLE_MARKERS:
        value = value.replace(marker, "\n")
    value = value.replace("Ответ кратко:", "\n").replace("Ответ:", "\n").replace("Вопрос:", "\n")

    question_norm = normalize_text(question) if question else ""
    lines = []
    for raw_line in value.splitlines():
        line = raw_line.strip(" \t\r\n:;-")
        if not line:
            continue
        line_norm = normalize_text(line)
        if question_norm and (line_norm == question_norm or question_norm in line_norm):
            continue
        lines.append(line)

    cleaned = lines[0] if lines else value.strip()
    cleaned = re.split(r"\n|</s>|<pad>", cleaned, maxsplit=1)[0].strip()
    cleaned = re.split(r"(?i)\b(?:user|assistant|system)\b", cleaned, maxsplit=1)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n:;-")
    return cleaned


def normalize_text(text: Any) -> str:
    value = str(text or "").lower().replace("ё", "е")
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def answer_matches(answer: str, expected_answer: str, aliases: Optional[Iterable[str]] = None) -> bool:
    normalized_answer = normalize_text(answer)
    candidates = [expected_answer, *(aliases or [])]
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate and normalized_candidate in normalized_answer:
            return True
    return False


def score_answer(question: Dict[str, Any], answer: str) -> Dict[str, Any]:
    return {
        "question_id": question.get("question_id"),
        "question_type": question.get("question_type"),
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),
        "aliases": question.get("aliases", []),
        "answer": answer,
        "match": answer_matches(answer, question.get("expected_answer", ""), question.get("aliases", [])),
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(mean(clean), 6)


def harmonic_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean or any(value <= 0 for value in clean):
        return 0.0 if clean else None
    return round(harmonic_mean(clean), 6)


def list_accuracy(results: List[Dict[str, Any]]) -> Optional[float]:
    if not results:
        return None
    return round(sum(1 for item in results if item.get("match")) / len(results), 6)


def metric_mean(value: Any) -> Optional[float]:
    if isinstance(value, list):
        values = [metric_mean(item) for item in value]
        return mean_or_none(values)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nested_accuracy(metrics: Dict[str, Any], phase: str, bucket: str, wanted_prefix: Optional[str] = None) -> Optional[float]:
    bucket_value = metrics.get(phase, {}).get(bucket)
    if not isinstance(bucket_value, dict):
        return None
    values: List[float] = []
    for key, value in bucket_value.items():
        if not key.endswith("_acc"):
            continue
        if wanted_prefix is not None and not key.startswith(wanted_prefix):
            continue
        metric_value = metric_mean(value)
        if metric_value is not None:
            values.append(metric_value)
    return mean_or_none(values)


def normalize_easyedit_metric(raw_metric: Dict[str, Any]) -> Dict[str, Optional[float]]:
    reliability = metric_mean(raw_metric.get("post", {}).get("rewrite_acc"))
    generalization = metric_mean(raw_metric.get("post", {}).get("rephrase_acc"))
    reverse = nested_accuracy(raw_metric, "post", "portability", "reverse")
    neighbor = nested_accuracy(raw_metric, "post", "locality", "neighbor")
    fact_locality = nested_accuracy(raw_metric, "post", "locality", "fact_locality")
    global_locality = nested_accuracy(raw_metric, "post", "locality", "global")
    domain_score = nested_accuracy(raw_metric, "post", "locality", "domain")

    locality_for_quality = mean_or_none([fact_locality, global_locality])
    return {
        "pre_rewrite_acc": metric_mean(raw_metric.get("pre", {}).get("rewrite_acc")),
        "reliability": reliability,
        "generalization": generalization,
        "reverse": reverse,
        "neighbor": neighbor,
        "fact_locality": fact_locality,
        "global_locality": global_locality,
        "domain_score": domain_score,
        "edit_quality": harmonic_or_none([reliability, generalization, locality_for_quality]),
    }
