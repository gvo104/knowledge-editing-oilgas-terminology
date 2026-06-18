# Knowledge Editing for Domain Terminology on Top of EasyEdit

This repository contains a research-oriented implementation of knowledge editing experiments for domain terminology, with a current focus on the oil and gas domain.

The project studies whether a small language model can absorb new domain facts without full retraining, while keeping:

- the edited fact itself;
- generalization to paraphrases;
- retention of previously edited facts;
- locality with respect to nearby domain facts;
- locality with respect to general non-domain knowledge.

The experimental layer lives in `my_exp_2/`. The underlying editing framework is based on `EasyEdit`, which is included in this repository because the experiments reuse its internals and contain local adaptations required for reproducible runs.

## What Is Implemented

The current repository already contains a working experimental pipeline for `Qwen2.5-3B`:

- dataset validation for the oil and gas JSON files;
- baseline evaluation of the unedited model;
- single-edit experiments for `LoRA`, `ROME`, and `MEMIT`;
- sequential-edit experiments for `LoRA`, `ROME`, and `MEMIT`;
- metric aggregation to JSON and CSV;
- markdown report generation for both single-edit and sequential runs.

The research entry point is `my_exp_2/`, not the generic EasyEdit examples.

Additional project docs:

- Technical experiment overview: [my_exp_2/README.md](my_exp_2/README.md)
- Reproducibility notes: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Upstream framework reference: [docs/EASYEDIT_UPSTREAM_REFERENCE.md](docs/EASYEDIT_UPSTREAM_REFERENCE.md)

## Repository Layout

```text
.
├── my_exp_2/                  # Main research module
│   ├── data/                  # Oil and gas facts, edit requests, eval sets
│   ├── scripts/               # Validation, baseline, single/sequential runs, reports
│   └── outputs/               # Ignored experiment outputs
├── easyeditor/                # Embedded EasyEdit framework code
├── hparams/                   # Framework method configurations
├── my_exp/                    # Local model/hparams area reused by experiments
└── docs/                      # Upstream and reproducibility reference docs
```

## Based on EasyEdit

This repository is built on top of the `EasyEdit` framework:

- Upstream project: <https://github.com/zjunlp/EasyEdit>
- Local preserved upstream README snapshot: [docs/README_EASYEDIT_UPSTREAM.md](docs/README_EASYEDIT_UPSTREAM.md)
- Additional upstream steering README: [README_2.md](README_2.md)

In this repository, `EasyEdit` is used as the framework base, while `my_exp_2/` contains the research-specific data layout, experiment orchestration, and reporting logic.

## Requirements

- Linux or WSL
- NVIDIA GPU
- CUDA-enabled PyTorch environment
- Conda
- Local model weights for `Qwen2.5-3B`

Heavy runs are GPU-intensive. Sequential experiments should be treated as long-running jobs.

## Environment Setup

Create the environment:

```bash
conda create -n EasyEdit python=3.10 -y
conda activate EasyEdit
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count())"
```

If PyTorch was installed without CUDA support, reinstall the CUDA-enabled build appropriate for your system before running the experiments.

## Model Placement

The repository does not include model weights.

Place the local model under:

```text
my_exp/models/Qwen2.5-3B
```

Quick load check:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='my_exp/models/Qwen2.5-3B'; AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype='auto'); print('ok')"
```

## Quick Start

Validate the JSON data:

```bash
python my_exp_2/scripts/validate_data.py \
  --data-dir my_exp_2/data \
  --output my_exp_2/outputs/metrics/data_validation.json
```

Run a small baseline smoke:

```bash
python my_exp_2/scripts/run_baseline_eval.py \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/baseline_smoke \
  --max-questions 5
```

Run a single-edit smoke on 3 facts:

```bash
python my_exp_2/scripts/run_single_edit_experiment.py \
  --methods LoRA ROME MEMIT \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/single_edit/smoke_all_3facts \
  --max-facts 3 \
  --eval-scope fact-only
```

Build the single-edit report:

```bash
python my_exp_2/scripts/compute_metrics.py \
  --single-edit-dir my_exp_2/outputs/single_edit/smoke_all_3facts \
  --baseline-dir my_exp_2/outputs/baseline_smoke \
  --output-dir my_exp_2/outputs/metrics/smoke_all_3facts

python my_exp_2/scripts/generate_report.py \
  --single-edit-dir my_exp_2/outputs/single_edit/smoke_all_3facts \
  --baseline-dir my_exp_2/outputs/baseline_smoke \
  --output my_exp_2/outputs/metrics/smoke_all_3facts/report.md
```

Run a sequential smoke on 2 steps:

```bash
python my_exp_2/scripts/run_sequential_edit_experiment.py \
  --methods LoRA ROME MEMIT \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/sequential_edit/smoke_all_3methods_2steps_full \
  --max-steps 2 \
  --eval-scope full \
  --retention-fact-limit 2 \
  --general-limit 2 \
  --domain-limit 2 \
  --generation-max-new-tokens 32 \
  --target-old-max-new-tokens 32
```

Build the sequential report:

```bash
python my_exp_2/scripts/compute_sequential_metrics.py \
  --sequential-edit-dir my_exp_2/outputs/sequential_edit/smoke_all_3methods_2steps_full \
  --baseline-dir my_exp_2/outputs/baseline_smoke \
  --output-dir my_exp_2/outputs/metrics/sequential/smoke_all_3methods_2steps_full

python my_exp_2/scripts/generate_sequential_report.py \
  --sequential-edit-dir my_exp_2/outputs/sequential_edit/smoke_all_3methods_2steps_full \
  --baseline-dir my_exp_2/outputs/baseline_smoke \
  --output my_exp_2/outputs/metrics/sequential/smoke_all_3methods_2steps_full/report.md
```

## Full Experiment Commands

Baseline on the full evaluation set:

```bash
python my_exp_2/scripts/run_baseline_eval.py \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/baseline_full_20facts
```

Single-edit full run on 20 facts:

```bash
python my_exp_2/scripts/run_single_edit_experiment.py \
  --methods LoRA ROME MEMIT \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/single_edit/full_20facts \
  --max-facts 20 \
  --eval-scope full
```

Sequential full run on 20 steps:

```bash
python my_exp_2/scripts/run_sequential_edit_experiment.py \
  --methods LoRA ROME MEMIT \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/sequential_edit/full_20steps_all_3methods \
  --max-steps 20 \
  --eval-scope full \
  --retention-fact-limit 5 \
  --general-limit 5 \
  --domain-limit 5 \
  --generation-max-new-tokens 32 \
  --target-old-max-new-tokens 32
```

Aggregate and report the full runs:

```bash
python my_exp_2/scripts/compute_metrics.py \
  --single-edit-dir my_exp_2/outputs/single_edit/full_20facts \
  --baseline-dir my_exp_2/outputs/baseline_full_20facts \
  --output-dir my_exp_2/outputs/metrics/full_20facts

python my_exp_2/scripts/generate_report.py \
  --single-edit-dir my_exp_2/outputs/single_edit/full_20facts \
  --baseline-dir my_exp_2/outputs/baseline_full_20facts \
  --output my_exp_2/outputs/metrics/full_20facts/report.md

python my_exp_2/scripts/compute_sequential_metrics.py \
  --sequential-edit-dir my_exp_2/outputs/sequential_edit/full_20steps_all_3methods \
  --baseline-dir my_exp_2/outputs/baseline_full_20facts \
  --output-dir my_exp_2/outputs/metrics/sequential/full_20steps_all_3methods

python my_exp_2/scripts/generate_sequential_report.py \
  --sequential-edit-dir my_exp_2/outputs/sequential_edit/full_20steps_all_3methods \
  --baseline-dir my_exp_2/outputs/baseline_full_20facts \
  --output my_exp_2/outputs/metrics/sequential/full_20steps_all_3methods/report.md
```

## Where to Look at Results

- Single-edit raw cases: `my_exp_2/outputs/single_edit/<run_name>/<method>/case_*.json`
- Sequential raw steps: `my_exp_2/outputs/sequential_edit/<run_name>/<method>/step_*.json`
- Single-edit summaries: `my_exp_2/outputs/metrics/<run_name>/`
- Sequential summaries: `my_exp_2/outputs/metrics/sequential/<run_name>/`

## Current Limitations

- The repository does not ship model weights.
- Experiment outputs are ignored by default and are not intended to be versioned.
- Full runs require substantial GPU time and memory.
- The current public research layer is centered on `LoRA`, `ROME`, and `MEMIT`.
- The oil and gas data format is custom to `my_exp_2`, while the execution backend is EasyEdit-based.

## Acknowledgements

This work reuses and adapts the `EasyEdit` framework for a domain-specific research setting.

- Framework base: `EasyEdit`
- Local experiment layer: `my_exp_2`
- Some internal framework behavior and orchestration were adapted for this project, especially around evaluation, target-old resolution, and sequential experiment reporting.

If you use this repository, please also acknowledge the original `EasyEdit` project where appropriate.
