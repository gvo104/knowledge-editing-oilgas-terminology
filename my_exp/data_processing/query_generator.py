"""
Генерация тестовых запросов (direct, inverse, paraphrase) и locality-вопросов.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

Triplet = Tuple[str, str, str]

# Полные шаблоны (7 парафраз для каждого типа)
TEMPLATES = {
    "gene": {
        "direct": "In which PubMed article is {entity} mentioned?",
        "inverse": "Which gene is mentioned in {pmid}?",
        "paraphrase": [
            "What publication cites {entity}?",
            "Which paper mentions the gene {entity}?",
            "Give the PubMed ID of a study discussing {entity}.",
            "In which article does {entity} appear?",
            "What is the PubMed ID for the paper about {entity}?",
            "Which scientific article reports on {entity}?",
            "Provide the PMID of a study that examines {entity}.",
        ]
    },
    "disease": {
        "direct": "In which PubMed article is the disease {entity} mentioned?",
        "inverse": "Which disease is mentioned in {pmid}?",
        "paraphrase": [
            "What publication discusses the disease {entity}?",
            "Which paper mentions the condition {entity}?",
            "Give the PubMed ID of a study about {entity}.",
            "In which article does the disease {entity} appear?",
            "What is the PMID for the paper focusing on {entity}?",
            "Which scientific article investigates {entity}?",
            "Provide the PubMed ID of research concerning {entity}.",
        ]
    },
    "mutation": {
        "direct": "In which PubMed article is the mutation {entity} mentioned?",
        "inverse": "Which mutation is mentioned in {pmid}?",
        "paraphrase": [
            "What publication describes the mutation {entity}?",
            "Which paper mentions the variant {entity}?",
            "Give the PubMed ID of a study discussing the mutation {entity}.",
            "In which article does the mutation {entity} appear?",
            "What is the PMID for the paper about the mutation {entity}?",
            "Which scientific article reports on {entity}?",
            "Provide the PubMed ID of a study that examines the mutation {entity}.",
        ]
    }
}

VAL_TEMPLATES = {
    "gene": {
        "direct": "Return only the PMID for the PubMed article that mentions gene {entity}.",
        "inverse": "Return only the gene mentioned in PubMed article {pmid}.",
        "paraphrase": [
            "Return exactly one PMID associated with gene {entity}.",
            "Return only the PubMed identifier linked to {entity}.",
            "Return exactly one PubMed ID for an article mentioning {entity}.",
            "Return only the PMID for a publication where {entity} appears.",
        ],
    },
    "disease": {
        "direct": "Return only the PMID for the PubMed article that mentions disease {entity}.",
        "inverse": "Return only the disease mentioned in PubMed article {pmid}.",
        "paraphrase": [
            "Return exactly one PMID associated with disease {entity}.",
            "Return only the PubMed identifier linked to disease {entity}.",
            "Return exactly one PubMed ID for an article discussing {entity}.",
            "Return only the PMID for a publication where disease {entity} appears.",
        ],
    },
    "mutation": {
        "direct": "Return only the PMID for the PubMed article that mentions mutation {entity}.",
        "inverse": "Return only the mutation mentioned in PubMed article {pmid}.",
        "paraphrase": [
            "Return exactly one PMID associated with mutation {entity}.",
            "Return only the PubMed identifier linked to mutation {entity}.",
            "Return exactly one PubMed ID for an article describing {entity}.",
            "Return only the PMID for a publication where mutation {entity} appears.",
        ],
    },
}

TRAIN_TEMPLATES = {
    "gene": {
        "direct": [
            "Return exactly one token in the form PMID:<digits> for the article mentioning gene {entity}.",
        ],
        "inverse": [
            "Return exactly the gene name mentioned in PubMed article {pmid}. No extra text.",
        ],
        "paraphrase": [
            "Return only the PMID associated with gene {entity}.",
            "Return exactly the article identifier linked to gene {entity}.",
            "Return one PMID where {entity} is mentioned. No explanation.",
            "Return only the PubMed record that contains gene {entity}.",
            "Return exactly one PubMed article ID connected to {entity}.",
            "Return only the PMID for a publication that includes {entity}.",
        ],
        "cooccur": [
            "Return only the co-occurring gene for {subj}.",
            "Return exactly the gene paired with {subj} in the same PubMed context.",
            "Return only one gene found alongside {subj}.",
            "Return exactly one gene linked with {subj} by co-occurrence.",
            "Return only one gene co-mentioned with {subj}.",
        ],
    },
    "disease": {
        "direct": [
            "Return exactly one token in the form PMID:<digits> for the article mentioning disease {entity}.",
        ],
        "inverse": [
            "Return exactly the disease name mentioned in PubMed article {pmid}. No extra text.",
        ],
        "paraphrase": [
            "Return only the PMID associated with disease {entity}.",
            "Return exactly the article identifier linked to disease {entity}.",
            "Return one PMID where {entity} is discussed. No explanation.",
            "Return only the PubMed record that contains disease {entity}.",
            "Return exactly one PubMed article ID connected to {entity}.",
            "Return only the PMID for a publication that includes disease {entity}.",
        ],
        "cooccur": [
            "Return only the co-occurring disease for {subj}.",
            "Return exactly the disease paired with {subj} in the same PubMed context.",
            "Return only one disease found alongside {subj}.",
            "Return exactly one disease linked with {subj} by co-occurrence.",
            "Return only one disease co-mentioned with {subj}.",
        ],
    },
    "mutation": {
        "direct": [
            "Return exactly one token in the form PMID:<digits> for the article mentioning mutation {entity}.",
        ],
        "inverse": [
            "Return exactly the mutation name mentioned in PubMed article {pmid}. No extra text.",
        ],
        "paraphrase": [
            "Return only the PMID associated with mutation {entity}.",
            "Return exactly the article identifier linked to mutation {entity}.",
            "Return one PMID where {entity} is described. No explanation.",
            "Return only the PubMed record that contains mutation {entity}.",
            "Return exactly one PubMed article ID connected to {entity}.",
            "Return only the PMID for a publication that includes mutation {entity}.",
        ],
        "cooccur": [
            "Return only the co-occurring mutation for {subj}.",
            "Return exactly the mutation paired with {subj} in the same PubMed context.",
            "Return only one mutation found alongside {subj}.",
            "Return exactly one mutation linked with {subj} by co-occurrence.",
            "Return only one mutation co-mentioned with {subj}.",
        ],
    },
}


def _ordered_triplets(triplets: Set[Triplet], limit: int) -> List[Triplet]:
    return sorted(triplets)[:limit]


def _normalize_inverse_max(max_entities_per_pmid: Optional[int]) -> Optional[int]:
    if max_entities_per_pmid is None:
        return None
    value = int(max_entities_per_pmid)
    if value <= 0:
        return None
    return value


def _mentioned_entities_by_pmid(triplets: Set[Triplet]) -> Dict[str, Set[str]]:
    by_pmid: Dict[str, Set[str]] = defaultdict(set)
    for subj, pred, obj in triplets:
        if pred == "mentioned in":
            by_pmid[obj].add(subj)
    return dict(by_pmid)


def _ordered_inverse_triplets(
    triplets: Set[Triplet],
    limit: int,
    max_entities_per_pmid: Optional[int],
    fallback_triplets: List[Triplet],
) -> List[Triplet]:
    normalized_max = _normalize_inverse_max(max_entities_per_pmid)
    if normalized_max is None:
        return [t for t in fallback_triplets if t[1] == "mentioned in"]

    by_pmid = _mentioned_entities_by_pmid(triplets)
    allowed_pmids = {
        pmid
        for pmid, entities in by_pmid.items()
        if len(entities) <= normalized_max
    }
    return [
        triplet
        for triplet in sorted(triplets)
        if triplet[1] == "mentioned in" and triplet[2] in allowed_pmids
    ][:limit]


def inverse_sampling_info(
    triplets: Set[Triplet],
    limit: int,
    max_entities_per_pmid: Optional[int],
) -> Dict[str, object]:
    """Возвращает диагностику inverse-сэмплинга для сохранения в triplets.json."""
    normalized_max = _normalize_inverse_max(max_entities_per_pmid)
    selected = _ordered_triplets(triplets, limit)
    by_pmid = _mentioned_entities_by_pmid(triplets)
    candidate_triplets = [t for t in sorted(triplets) if t[1] == "mentioned in"]
    selected_mentions = [t for t in selected if t[1] == "mentioned in"]
    sampled = _ordered_inverse_triplets(triplets, limit, normalized_max, selected)
    allowed_pmids = {
        pmid
        for pmid, entities in by_pmid.items()
        if normalized_max is None or len(entities) <= normalized_max
    }
    allowed_candidate_triplets = [
        t for t in candidate_triplets if t[2] in allowed_pmids
    ]
    entity_count_histogram = Counter(len(entities) for entities in by_pmid.values())
    return {
        "enabled": normalized_max is not None,
        "max_entities_per_pmid": normalized_max,
        "limit": limit,
        "selected_mentioned_triplets": len(selected_mentions),
        "candidate_mentioned_triplets": len(candidate_triplets),
        "allowed_candidate_triplets": len(allowed_candidate_triplets),
        "sampled_inverse_triplets": len(sampled),
        "filtered_ambiguous_triplets": len(candidate_triplets) - len(allowed_candidate_triplets),
        "unsampled_allowed_triplets_due_to_limit": max(
            0,
            len(allowed_candidate_triplets) - len(sampled),
        ),
        "pmids_with_mentions": len(by_pmid),
        "allowed_pmids": len(allowed_pmids),
        "filtered_pmids": len(by_pmid) - len(allowed_pmids),
        "pmid_entity_count_histogram": {
            str(k): v for k, v in sorted(entity_count_histogram.items())
        },
    }


def generate_queries(triplets: Set[Triplet], entity_type: str = "gene",
                     limit: int = 100, augment: bool = False,
                     inverse_max_entities_per_pmid: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Генерирует direct, inverse и paraphrase запросы.
    Если augment=True, используется расширенный набор парафраз (7 вместо 4).
    """
    tmpl = TEMPLATES.get(entity_type, TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}

    # Определяем количество парафраз в зависимости от флага
    paraphrase_templates = tmpl["paraphrase"] if augment else tmpl["paraphrase"][:4]
    selected_triplets = _ordered_triplets(triplets, limit)
    inverse_triplets = _ordered_inverse_triplets(
        triplets,
        limit,
        inverse_max_entities_per_pmid,
        fallback_triplets=selected_triplets,
    )

    for subj, pred, obj in selected_triplets:
        if pred == "mentioned in":
            # прямой вопрос
            queries['direct'].append({
                "question": tmpl["direct"].format(entity=subj),
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            for pt in paraphrase_templates:
                queries['paraphrase'].append({
                    "question": pt.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj)
                })
        elif pred == "co-occurs with":
            queries['direct'].append({
                "question": f"With which {entity_type} does {subj} co-occur?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            # Ко-оккуренционные парафразы (тоже зависят от augment)
            cooccur_templates = [
                f"What {entity_type} is co-mentioned with {subj} in the same article?",
                f"Which {entity_type} appears together with {subj} in a PubMed article?",
                f"Name a {entity_type} that co-occurs with {subj} in a publication.",
                f"With which {entity_type} does {subj} share a PubMed article?",
            ]
            if augment:
                cooccur_templates += [
                    f"Identify a {entity_type} that is mentioned alongside {subj} in a paper.",
                    f"List a {entity_type} that co-appears with {subj} in PubMed.",
                ]
            for pt in cooccur_templates:
                queries['paraphrase'].append({
                    "question": pt,
                    "expected": obj,
                    "triplet": (subj, pred, obj)
                })

    for subj, pred, obj in inverse_triplets:
        queries['inverse'].append({
            "question": tmpl["inverse"].format(pmid=obj),
            "expected": subj,
            "triplet": (subj, pred, obj)
        })

    return queries


def generate_validation_queries(
    triplets: Set[Triplet],
    entity_type: str = "gene",
    limit: int = 25,
    inverse_max_entities_per_pmid: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Генерирует небольшой validation split на held-out prompt-шаблонах.
    Он нужен для подбора гиперпараметров без просмотра test-метрик.
    """
    tmpl = VAL_TEMPLATES.get(entity_type, VAL_TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}
    selected_triplets = _ordered_triplets(triplets, limit)
    inverse_triplets = _ordered_inverse_triplets(
        triplets,
        limit,
        inverse_max_entities_per_pmid,
        fallback_triplets=selected_triplets,
    )

    for subj, pred, obj in selected_triplets:
        if pred == "mentioned in":
            queries["direct"].append({
                "question": tmpl["direct"].format(entity=subj),
                "expected": obj,
                "triplet": (subj, pred, obj),
                "split": "validation",
            })
            for question_template in tmpl["paraphrase"]:
                queries["paraphrase"].append({
                    "question": question_template.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj),
                    "split": "validation",
                })
        elif pred == "co-occurs with":
            queries["direct"].append({
                "question": f"Return only the {entity_type} that co-occurs with {subj}.",
                "expected": obj,
                "triplet": (subj, pred, obj),
                "split": "validation",
            })
            queries["paraphrase"].append({
                "question": f"Which {entity_type} is mentioned alongside {subj}?",
                "expected": obj,
                "triplet": (subj, pred, obj),
                "split": "validation",
            })

    for subj, pred, obj in inverse_triplets:
        queries["inverse"].append({
            "question": tmpl["inverse"].format(pmid=obj),
            "expected": subj,
            "triplet": (subj, pred, obj),
            "split": "validation",
        })

    return queries


def generate_training_queries(
    triplets: Set[Triplet],
    entity_type: str = "gene",
    limit: int = 100,
    augment: bool = False,
    inverse_max_entities_per_pmid: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Генерирует обучающие формулировки для SFT.
    Они намеренно отличаются от evaluation prompts в generate_queries(),
    чтобы не обучаться на тех же строках, которыми затем меряется качество.
    """
    tmpl = TRAIN_TEMPLATES.get(entity_type, TRAIN_TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}
    num_paraphrases = len(tmpl["paraphrase"]) if augment else 2
    num_cooccur = len(tmpl["cooccur"]) if augment else 2
    selected_triplets = _ordered_triplets(triplets, limit)
    inverse_triplets = _ordered_inverse_triplets(
        triplets,
        limit,
        inverse_max_entities_per_pmid,
        fallback_triplets=selected_triplets,
    )

    for subj, pred, obj in selected_triplets:
        if pred == "mentioned in":
            for question_template in tmpl["direct"]:
                queries["direct"].append({
                    "question": question_template.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj),
                    "split": "train",
                })
            for question_template in tmpl["paraphrase"][:num_paraphrases]:
                queries["paraphrase"].append({
                    "question": question_template.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj),
                    "split": "train_augmented" if augment else "train",
                })
        elif pred == "co-occurs with":
            for question_template in tmpl["cooccur"][:num_cooccur]:
                queries["paraphrase"].append({
                    "question": question_template.format(subj=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj),
                    "split": "train_augmented" if augment else "train",
                })

    for subj, pred, obj in inverse_triplets:
        for question_template in tmpl["inverse"]:
            queries["inverse"].append({
                "question": question_template.format(pmid=obj),
                "expected": subj,
                "triplet": (subj, pred, obj),
                "split": "train",
            })

    return queries


def generate_locality_queries() -> List[Dict[str, str]]:
    common_knowledge = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Who wrote the play 'Romeo and Juliet'?", "expected": "William Shakespeare"},
        {"question": "What is the chemical symbol for water?", "expected": "H2O"},
        {"question": "In which year did the Titanic sink?", "expected": "1912"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter"},
        {"question": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
        {"question": "What is the capital of Japan?", "expected": "Tokyo"},
        {"question": "How many continents are there on Earth?", "expected": "7"},
        {"question": "What is the speed of light in vacuum (km/s)?", "expected": "300000"},
        {"question": "What is the main language spoken in Brazil?", "expected": "Portuguese"},
    ]
    return common_knowledge
