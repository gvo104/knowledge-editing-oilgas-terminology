# my_exp_2: Technical Documentation

`my_exp_2/` is the main research layer of this repository. It contains the domain dataset, experiment orchestration, evaluation scripts, and reporting utilities built on top of `EasyEdit`.

The goal of this module is not to replace EasyEdit, but to adapt it to a reproducible domain-specific study of knowledge editing in the oil and gas area.

## Research Goal

The main question is whether a relatively small language model, `Qwen2.5-3B`, can absorb new domain facts without full retraining while preserving acceptable behavior outside the edited fact.

The current study focuses on five properties:

- `edit success`: did the model learn the target fact after editing;
- `generalization`: does the edit transfer to paraphrased formulations;
- `locality`: did nearby knowledge remain stable;
- `retention`: in sequential mode, are earlier inserted facts preserved;
- `global robustness`: did the model keep general and domain knowledge outside the edited fact.

## Implemented Scope

At the current stage `my_exp_2` already supports:

- validation of the oil and gas JSON dataset;
- baseline evaluation of the unedited model;
- `single-edit` experiments;
- `sequential-edit` experiments;
- metric aggregation to JSON and CSV;
- markdown report generation.

The working editing methods in this layer are:

- `LoRA`
- `ROME`
- `MEMIT`
- `WISE`

The current public MVP does not yet cover:

- `SFT` as a full experimental branch;
- `LocFT-BF`;
- richer judge-based evaluation;
- large-scale plotting and final benchmark packaging.

## Experimental Design

### 1. Baseline evaluation

The untouched base model is evaluated before any editing.

Purpose:

- estimate which oil and gas facts the model already knows;
- detect unstable or noisy answers before editing;
- record baseline domain accuracy;
- record baseline general-knowledge accuracy.

### 2. Single-edit experiment

Each fact is edited independently.

Pipeline:

1. Load the base model.
2. Edit one fact.
3. Evaluate the edited model.
4. Save results.
5. Discard the edited state.
6. Reload the base model for the next fact.

This setup measures how well each method performs on isolated fact insertion or replacement.

### 3. Sequential-edit experiment

Facts are edited one after another on the same evolving model.

Pipeline:

1. Load the base model once.
2. Apply the first edit.
3. Evaluate the current edited fact and previously seen facts.
4. Apply the next edit on top of the already modified model.
5. Repeat for the configured sequence length.

This setup measures whether the model can accumulate multiple edits without quickly forgetting earlier ones or degrading on unrelated questions.

## Method Role in This Project

This module uses EasyEdit as the backend, but the experiment logic is project-specific.

### `LoRA`

In this project, LoRA acts as a parameter-efficient editing baseline. It updates a low-rank adapter instead of directly rewriting the full model weights.

### `ROME`

ROME is used as a direct localized editing method for a single factual association. In the sequential regime it lets us test whether repeated local rewrites accumulate cleanly or destabilize previous edits.

### `MEMIT`

MEMIT is used as another direct weight-editing method, but with a broader memory-update style than ROME. In practice it is useful here as a comparison point for edit strength versus locality and retention.

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

### Main datasets

- `triplets/oilgas.json`: 20 core oil and gas facts.
- `edit_requests/oilgas_edit_requests.json`: edit prompts, subjects, and edit targets.
- `fact_questions.json`: direct, paraphrase, reverse, neighbor, and locality questions for each fact.
- `domain_questions.json`: broader oil and gas evaluation set.
- `general_questions.json`: general non-domain locality set.
- `sequential_order.json`: fixed edit order for sequential runs.

## Scripts

```text
my_exp_2/scripts/
├── validate_data.py
├── data_io.py
├── eval_utils.py
├── target_old_resolver.py
├── run_baseline_eval.py
├── run_single_edit_experiment.py
├── run_sequential_edit_experiment.py
├── run_full_experiment_pipeline.py
├── compute_metrics.py
├── compute_sequential_metrics.py
├── generate_report.py
└── generate_sequential_report.py
```

### Script roles

- `validate_data.py`: checks dataset integrity and EasyEdit compatibility constraints.
- `data_io.py`: loads JSON files and builds runtime cases.
- `eval_utils.py`: normalization, scoring helpers, and compact metric utilities.
- `target_old_resolver.py`: resolves `target_old` from the current model state when needed.
- `run_baseline_eval.py`: evaluates the untouched model.
- `run_single_edit_experiment.py`: isolated single-edit benchmark.
- `run_sequential_edit_experiment.py`: sequential benchmark with retention/general/domain checks.
- `run_full_experiment_pipeline.py`: unified launcher for preflight, baseline, single-edit, sequential-edit, metrics, and reports.
- `compute_metrics.py`: aggregates single-edit outputs.
- `compute_sequential_metrics.py`: aggregates sequential outputs.
- `generate_report.py`: builds single-edit markdown reports.
- `generate_sequential_report.py`: builds sequential markdown reports.

## Metrics

### Single-edit metrics

- `reliability`: direct fact accuracy after editing.
- `generalization`: paraphrase accuracy after editing.
- `reverse`: reverse-question accuracy.
- `neighbor`: behavior on nearby domain questions.
- `fact_locality`: preservation around the edited fact.
- `global_locality`: preservation on general non-domain questions.
- `domain_score`: preservation on the broader oil and gas set.
- `edit_quality`: compact combined indicator derived from the normalized post-edit metrics.

Interpretation:

- high `reliability` with low `fact_locality` means the edit worked but damaged nearby knowledge;
- high `generalization` means the model did not only memorize one exact prompt;
- high `global_locality` means the edit did not spill too aggressively into general knowledge.

### Sequential-edit metrics

- `current_reliability`: direct success on the current step.
- `current_generalization`: paraphrase success on the current step.
- `retention`: preservation of earlier edited facts.
- `global_locality`: preservation on general questions after accumulated edits.
- `domain_score`: preservation on domain questions after accumulated edits.
- `sequential_quality`: compact aggregate of current-step success and preservation behavior.

Interpretation:

- high `current_reliability` with low `retention` means the method can write new facts but forgets older ones;
- high `retention` with low `global_locality` means previous edits remain, but unrelated knowledge degrades;
- `sequential_quality` is useful as a compact ranking signal, but the underlying metrics should still be inspected separately.

## Target Old Resolution

Some editing methods require `target_old`, the value that the model currently associates with the edited prompt.

In this project, `target_old` is not treated as permanently fixed:

- in `single-edit`, it is resolved against the base model before the edit;
- in `sequential-edit`, it is resolved again before every step on the current edited model state.

If the current model answer is empty, unstable, or clearly noisy, the pipeline falls back to `target_new`. This lets the experiment distinguish between:

- `knowledge replacement`: overwrite an existing model belief;
- `knowledge insertion`: add a fact the model does not reliably know yet.

## Eval Scopes

The runners support three evaluation scopes:

- `fact-only`: evaluate only fact-level edit behavior;
- `fact-plus-general`: evaluate fact-level behavior and general locality;
- `full`: evaluate fact-level behavior, general locality, and domain-level preservation.

This is useful because `full` runs are noticeably heavier than fact-only smoke checks.

For practical execution there are also two run modes:

- `sample`: bounded retention/general/domain subsets for cheaper experiments;
- `full`: full configured coverage with more detailed saved artifacts.

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

- raw single-edit case JSON files;
- raw sequential step JSON files;
- `summary.json` and `summary.csv`;
- generated `report.md`;
- grouped CSV/JSON analytics by method, relation, level, and edit mode;
- unified pipeline outputs under `my_exp_2/outputs/full_pipeline/<run_name>/`.

## Unified Pipeline

For server-side or long GPU runs the main entry point is:

```bash
python my_exp_2/scripts/run_full_experiment_pipeline.py \
  --run-name oilgas_qwen25_3b_full_20facts \
  --methods LoRA ROME MEMIT WISE \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --max-facts 20 \
  --eval-scope full \
  --single-eval-mode full \
  --sequential-eval-mode full
```

This launcher runs:

1. `preflight`
2. `baseline`
3. `single`
4. `sequential`
5. `reports`

Supported operational flags:

- `--resume`
- `--overwrite`
- `--stop-after`
- `--skip-preflight`
- `--skip-baseline`

If you prefer wrappers instead of raw Python commands, use the top-level `Makefile`:

```bash
make big-preflight
make big-smoke-pipeline
make big-run
make big-run-resume
make big-reports
```

For DGX/container execution see [docs/DGX_RUNBOOK_RU.md](../docs/DGX_RUNBOOK_RU.md).

Outputs are ignored by git by default.

## Relation to EasyEdit

`my_exp_2` does not reimplement the editing algorithms themselves. Instead, it adds a project-specific layer around EasyEdit:

- custom domain dataset format;
- experiment orchestration;
- target-old resolution logic;
- baseline evaluation;
- aggregation and markdown reporting.

So the repository remains reproducible while keeping a clean distinction:

- `EasyEdit` provides the editing backend;
- `my_exp_2` provides the research workflow.
