# Data Documentation

This directory contains the domain data used by `my_exp_2` for knowledge editing experiments in the oil and gas domain.

The dataset is intentionally small and controlled. It is designed not as a large benchmark, but as a reproducible experimental set for comparing editing methods under the same conditions.

## Design Goal

The data is organized so that each edited fact can be evaluated from several angles:

- direct success on the target fact;
- generalization to paraphrases;
- consistency on reverse formulations;
- behavior on nearby domain knowledge;
- locality with respect to related facts;
- preservation of broader domain and general knowledge.

## Directory Layout

```text
data/
  triplets/
    oilgas.json

  edit_requests/
    oilgas_edit_requests.json

  train_sets/
    lora_train.json
    lora_augmented_train.json

  evaluate_set/
    fact_questions.json
    domain_questions.json
    general_questions.json

  sequential/
    sequential_order.json
```

## Core Fact Layer

### `triplets/oilgas.json`

This file stores the 20 base oil and gas facts in the form:

```text
subject -> relation -> object
```

Each record provides the conceptual fact to be inserted or checked.

Main fields:

- `fact_id`
- `subject`
- `relation`
- `object`
- `level`

Example idea:

```text
Кероген -> является источником -> углеводородов
```

This file is the source of truth for the fact inventory.

## Edit Layer

### `edit_requests/oilgas_edit_requests.json`

This file contains the runtime edit requests used by the editing methods.

Main fields:

- `fact_id`: link to the fact in `triplets/oilgas.json`
- `prompt`: the edit prompt used for the method
- `subject`: the entity that should be edited
- `target_new`: the new target answer
- `target_old`: old answer if known in advance

Important note:

In the current pipeline, `target_old` is not always trusted as static. For methods that need it, the runtime may resolve `target_old` from the current model state before applying the edit.

## LoRA Training Layer

### `train_sets/lora_train.json`

This is the minimal LoRA training set.

Each item is a simple QA pair:

```text
input -> output
```

Its role is to provide a compact parameter-efficient training baseline that is simpler than targeted knowledge editing.

### `train_sets/lora_augmented_train.json`

This is the expanded LoRA training set with additional paraphrased inputs.

Its role is to test whether training on several formulations improves generalization compared to the minimal LoRA set.

At the current project stage, these train files are included for the broader research roadmap, while the main implemented comparison focuses on `LoRA`, `ROME`, and `MEMIT` through the editing pipeline.

## Evaluation Layer

### `evaluate_set/fact_questions.json`

This is the main evaluation set around the edited facts.

For each `fact_id`, the file groups several question types:

- `direct`
- `paraphrase`
- `reverse`
- `neighbor`
- `locality`

These question types are used differently:

- `direct` checks immediate fact insertion success;
- `paraphrase` checks whether the edit transfers beyond one exact wording;
- `reverse` checks consistency from the opposite angle;
- `neighbor` checks nearby domain knowledge;
- `locality` checks whether closely related knowledge was damaged.

Typical fields per question include:

- `question_id`
- `fact_id`
- `question_type`
- `question`
- `expected_answer`
- `aliases`

### `evaluate_set/domain_questions.json`

This file contains broader oil and gas questions not limited to one edited fact.

Its role is to measure whether domain competence is preserved after editing.

This is especially useful in the sequential setting, where multiple edits may gradually distort the model's broader behavior in-domain.

### `evaluate_set/general_questions.json`

This file contains non-domain general knowledge questions.

Its role is to evaluate global locality: after editing oil and gas facts, does the model still answer ordinary questions correctly?

The set includes general topics such as:

- geography
- history
- literature
- biology
- mathematics
- programming
- basic science

## Sequential Layer

### `sequential/sequential_order.json`

This file defines the fixed order of facts for sequential editing.

Its role is:

- make the sequential experiment reproducible;
- ensure that the order does not depend on JSON file ordering;
- let all methods be compared on the same edit sequence.

## How the Data Is Used in the Pipeline

### Baseline evaluation

- `fact_questions.json`
- `domain_questions.json`
- `general_questions.json`

These are used before any edit to estimate the initial model behavior.

### Single-edit experiment

- `triplets/oilgas.json`
- `edit_requests/oilgas_edit_requests.json`
- `fact_questions.json`
- optionally `domain_questions.json`
- optionally `general_questions.json`

Each fact is edited independently and evaluated in isolation.

### Sequential-edit experiment

- `triplets/oilgas.json`
- `edit_requests/oilgas_edit_requests.json`
- `fact_questions.json`
- `domain_questions.json`
- `general_questions.json`
- `sequential_order.json`

Facts are edited one after another on the same model state.

## Why This Format Works Well for the Study

This data layout is useful because it separates concerns:

- fact definition;
- edit request representation;
- method-specific training support for LoRA-style baselines;
- local fact evaluation;
- broader domain evaluation;
- global non-domain evaluation;
- fixed sequential scheduling.

That makes it possible to compare editing methods under a common and reproducible protocol instead of evaluating them only on one prompt per fact.
