# my_exp_2: Research Layer Overview

`my_exp_2/` is the main experiment module of this repository. It contains the oil and gas dataset, experiment runners, evaluation scripts, and reporting utilities built on top of `EasyEdit`.

## Purpose

The goal is to evaluate domain-oriented knowledge editing on `Qwen2.5-3B` in two regimes:

- `single-edit`: one fact is edited, evaluated, and discarded before the next fact;
- `sequential-edit`: facts are edited one after another on the same evolving model.

This lets us compare:

- immediate edit success;
- paraphrase generalization;
- locality preservation;
- retention under sequential updates;
- effect on domain and general knowledge.

## Data Layout

```text
my_exp_2/data/
├── triplets/oilgas.json
├── edit_requests/oilgas_edit_requests.json
├── train_sets/lora_train.json
├── train_sets/lora_augmented_train.json
├── evaluate_set/fact_questions.json
├── evaluate_set/domain_questions.json
├── evaluate_set/general_questions.json
└── sequential/sequential_order.json
```

### Main files

- `triplets/oilgas.json`: the 20 core oil and gas facts.
- `edit_requests/oilgas_edit_requests.json`: edit prompts and targets for the editors.
- `fact_questions.json`: direct, paraphrase, reverse, neighbor, and locality questions for each fact.
- `domain_questions.json`: broader oil and gas evaluation questions.
- `general_questions.json`: non-domain locality checks.
- `sequential_order.json`: fixed order for sequential editing.

## Scripts

```text
my_exp_2/scripts/
├── validate_data.py
├── run_baseline_eval.py
├── run_single_edit_experiment.py
├── run_sequential_edit_experiment.py
├── compute_metrics.py
├── compute_sequential_metrics.py
├── generate_report.py
└── generate_sequential_report.py
```

### Execution flow

1. Validate JSON inputs.
2. Run baseline evaluation on the untouched model.
3. Run single-edit or sequential-edit experiments.
4. Aggregate metrics into JSON and CSV.
5. Generate markdown reports.

## Metrics

### Single-edit metrics

- `reliability`: direct fact accuracy after editing.
- `generalization`: paraphrase accuracy after editing.
- `reverse`: reverse-question accuracy.
- `neighbor`: nearby-domain behavior.
- `fact_locality`: locality around the edited fact.
- `global_locality`: general knowledge preservation.
- `domain_score`: broader domain knowledge preservation.
- `edit_quality`: combined EasyEdit-based quality indicator.

### Sequential-edit metrics

- `current_reliability`: current-step direct success.
- `current_generalization`: current-step paraphrase success.
- `retention`: preservation of previously edited facts.
- `global_locality`: preservation on general questions.
- `domain_score`: preservation on domain questions.
- `sequential_quality`: compact sequential quality indicator.

## Outputs

Default output root:

```text
my_exp_2/outputs/
├── baseline/
├── single_edit/
├── sequential_edit/
└── metrics/
```

Typical artifacts:

- raw case JSON files for single-edit runs;
- raw step JSON files for sequential runs;
- `summary.json` and `summary.csv`;
- generated `report.md`.

Outputs are ignored by git by default.

## Notes on EasyEdit Integration

`my_exp_2` does not replace EasyEdit internals. Instead, it adapts data loading, orchestration, target-old handling, and reporting around the framework.

That means:

- the oil and gas JSON format stays local to `my_exp_2`;
- the editing backend still uses EasyEdit editors and hparams;
- the repository remains reproducible because both layers are versioned together.
