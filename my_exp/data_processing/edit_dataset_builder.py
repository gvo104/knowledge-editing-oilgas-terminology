"""
Сборка универсального edit-dataset из триплетов и сгенерированных запросов.

Цель:
- получить единый JSONL-формат для FT / LoRA / ROME / MEMIT
- сохранить прямой вопрос, парафразы, reverse-вопрос и locality
- отдельно пометить неоднозначные случаи, которые плохо подходят для knowledge editing
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Triplet = Tuple[str, str, str]


def build_subject_object_stats(
    triplets: Iterable[Triplet],
    *,
    predicate: str = "mentioned in",
) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Возвращает:
    - subject -> set(objects)
    - object -> set(subjects)
    """
    objects_by_subject: Dict[str, set] = defaultdict(set)
    subjects_by_object: Dict[str, set] = defaultdict(set)

    for subj, pred, obj in triplets:
        if pred != predicate:
            continue
        objects_by_subject[subj].add(obj)
        subjects_by_object[obj].add(subj)

    return dict(objects_by_subject), dict(subjects_by_object)


def build_query_indexes(generated_queries: Dict[str, List[Dict]]) -> Dict[str, Dict[Triplet, List[Dict]]]:
    """Индексирует direct / inverse / paraphrase по triplet."""
    indexed = {}
    for split_name, rows in generated_queries.items():
        bucket: Dict[Triplet, List[Dict]] = defaultdict(list)
        for row in rows:
            triplet = tuple(row["triplet"])
            bucket[triplet].append(row)
        indexed[split_name] = dict(bucket)
    return indexed


def pick_counterfactual_target(
    truth: str,
    pool: Sequence[str],
    *,
    rng: random.Random,
) -> Optional[str]:
    candidates = [item for item in pool if item != truth]
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_neighbor_locality(
    current_triplet: Triplet,
    direct_rows: Dict[Triplet, List[Dict]],
    *,
    predicate: str,
    rng: random.Random,
) -> Optional[Dict[str, str]]:
    """Берет соседний факт той же формы, чтобы проверять, что правка не ломает близкие кейсы."""
    candidates: List[Tuple[Triplet, Dict]] = []
    current_subject, _, current_object = current_triplet

    for triplet, rows in direct_rows.items():
        subj, pred, obj = triplet
        if pred != predicate:
            continue
        if subj == current_subject or obj == current_object:
            continue
        if not rows:
            continue
        candidates.append((triplet, rows[0]))

    if not candidates:
        return None

    _, row = rng.choice(candidates)
    return {"question": row["question"], "expected": row["expected"]}


def build_edit_records(
    generated_queries: Dict[str, List[Dict]],
    locality_queries: List[Dict[str, str]],
    *,
    entity_type: str,
    max_cases: Optional[int] = None,
    paraphrase_limit: int = 3,
    predicate: str = "mentioned in",
    max_objects_per_subject: Optional[int] = 1,
    max_subjects_per_object: Optional[int] = 1,
    seed: int = 42,
) -> List[Dict]:
    """
    Строит edit-records под универсальный JSONL-формат.

    Важная логика:
    - берем только прямые fact-style triplets по predicate
    - фильтруем неоднозначные кейсы, если задан лимит по числу истинных объектов/субъектов
    - target_new выбираем как контрфакт из пула других объектов
    """
    rng = random.Random(seed)
    idx = build_query_indexes(generated_queries)
    direct = idx.get("direct", {})
    inverse = idx.get("inverse", {})
    paraphrase = idx.get("paraphrase", {})

    all_triplets = [tuple(row["triplet"]) for rows in generated_queries.values() for row in rows]
    objects_by_subject, subjects_by_object = build_subject_object_stats(all_triplets, predicate=predicate)
    object_pool = sorted({obj for _, pred, obj in all_triplets if pred == predicate})

    records = []
    locality_cursor = 0

    direct_triplets = sorted(
        triplet for triplet in direct.keys()
        if triplet[1] == predicate
    )

    for triplet in direct_triplets:
        subj, pred, obj = triplet
        subj_truths = objects_by_subject.get(subj, set())
        obj_subjects = subjects_by_object.get(obj, set())

        if max_objects_per_subject is not None and len(subj_truths) > max_objects_per_subject:
            continue
        if max_subjects_per_object is not None and len(obj_subjects) > max_subjects_per_object:
            continue

        target_new = pick_counterfactual_target(obj, object_pool, rng=rng)
        if target_new is None:
            continue

        direct_row = direct[triplet][0]
        paraphrase_rows = paraphrase.get(triplet, [])[:paraphrase_limit]
        inverse_rows = inverse.get(triplet, [])
        inverse_row = inverse_rows[0] if inverse_rows else None

        general_locality = locality_queries[locality_cursor % len(locality_queries)] if locality_queries else None
        locality_cursor += 1
        neighbor_locality = _pick_neighbor_locality(
            triplet,
            direct,
            predicate=predicate,
            rng=rng,
        )

        rephrase_prompts = [row["question"] for row in paraphrase_rows]
        locality_bundle = [item for item in (neighbor_locality, general_locality) if item]

        record = {
            "case_id": len(records) + 1,
            "entity_type": entity_type,
            "relation": pred,
            "source_triplet": list(triplet),
            "prompt": direct_row["question"],
            "subject": subj,
            "ground_truth": obj,
            "target_new": target_new,
            "rephrase_prompt": paraphrase_rows[0]["question"] if paraphrase_rows else None,
            "rephrase_prompts": rephrase_prompts,
            "locality_prompt": neighbor_locality["question"] if neighbor_locality else (
                general_locality["question"] if general_locality else None
            ),
            "locality_ground_truth": neighbor_locality["expected"] if neighbor_locality else (
                general_locality["expected"] if general_locality else None
            ),
            "locality_prompts": [item["question"] for item in locality_bundle],
            "locality_ground_truths": [item["expected"] for item in locality_bundle],
            "neighborhood_prompt": neighbor_locality["question"] if neighbor_locality else None,
            "neighborhood_ground_truth": neighbor_locality["expected"] if neighbor_locality else None,
            "general_locality_prompt": general_locality["question"] if general_locality else None,
            "general_locality_ground_truth": general_locality["expected"] if general_locality else None,
            "portability_prompt": paraphrase_rows[1]["question"] if len(paraphrase_rows) > 1 else None,
            "portability_ground_truth": target_new if len(paraphrase_rows) > 1 else None,
            "reverse_prompt": inverse_row["question"] if inverse_row else None,
            "reverse_ground_truth": inverse_row["expected"] if inverse_row else None,
            "evaluation_prompts": {
                "rewrite": [direct_row["question"]],
                "rephrase": rephrase_prompts,
                "locality": [item["question"] for item in locality_bundle],
                "reverse": [inverse_row["question"]] if inverse_row else [],
            },
            "notes": (
                "Auto-generated from PubTator triplets. "
                "target_new is a counterfactual sampled from another object of the same relation."
            ),
        }
        records.append(record)

        if max_cases is not None and len(records) >= max_cases:
            break

    return records


def save_jsonl(records: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
