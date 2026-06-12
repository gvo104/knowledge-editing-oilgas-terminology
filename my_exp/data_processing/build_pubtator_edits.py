"""
CLI для сборки универсального edit-dataset из локального PubTator-файла.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_reader import iter_records
from edit_dataset_builder import build_edit_records, save_jsonl
from query_generator import generate_locality_queries, generate_queries, inverse_sampling_info
from triplet_extractor import build_triplets, extract_sample, group_entities_by_pmid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--entity-type", choices=["gene", "disease", "mutation"], default="gene")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--query-limit", type=int, default=500)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--paraphrase-limit", type=int, default=5)
    parser.add_argument("--max-cooccur-entities", type=int, default=5)
    parser.add_argument("--inverse-max-entities-per-pmid", type=int, default=1)
    parser.add_argument("--max-objects-per-subject", type=int, default=1)
    parser.add_argument("--max-subjects-per-object", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records_iter = iter_records(args.input_path)
    sample_records = extract_sample(records_iter, sample_size=args.sample_size)
    entities_by_pmid = group_entities_by_pmid(sample_records)
    triplets = build_triplets(
        entities_by_pmid,
        max_cooccur_entities=args.max_cooccur_entities,
    )

    generated_queries = generate_queries(
        triplets,
        entity_type=args.entity_type,
        limit=args.query_limit,
        augment=args.augment,
        inverse_max_entities_per_pmid=args.inverse_max_entities_per_pmid,
    )
    locality_queries = generate_locality_queries()

    records = build_edit_records(
        generated_queries,
        locality_queries,
        entity_type=args.entity_type,
        max_cases=args.max_cases,
        paraphrase_limit=args.paraphrase_limit,
        max_objects_per_subject=args.max_objects_per_subject,
        max_subjects_per_object=args.max_subjects_per_object,
        seed=args.seed,
    )
    save_jsonl(records, args.output_path)

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "entity_type": args.entity_type,
        "sample_size": args.sample_size,
        "query_limit": args.query_limit,
        "max_cases": args.max_cases,
        "triplets_total": len(triplets),
        "records_total": len(records),
        "inverse_sampling": inverse_sampling_info(
            triplets,
            limit=args.query_limit,
            max_entities_per_pmid=args.inverse_max_entities_per_pmid,
        ),
    }
    metadata_path = args.output_path.with_suffix(args.output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(records)} edit records to {args.output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
