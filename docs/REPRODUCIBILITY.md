# Reproducibility Guide

## Minimal Reproduction Order

1. Create and activate the `EasyEdit` conda environment.
2. Install Python dependencies from `requirements.txt`.
3. Place `Qwen2.5-3B` at `my_exp/models/Qwen2.5-3B`.
4. Verify CUDA with a short `torch.cuda.is_available()` check.
5. Run data validation.
6. Run a baseline smoke.
7. Run a single-edit smoke.
8. Run a sequential smoke.
9. Generate markdown reports.
10. For heavy server runs, execute unified pipeline preflight and then the full pipeline.

## Recommended First Commands

```bash
python my_exp_2/scripts/validate_data.py

python my_exp_2/scripts/run_baseline_eval.py \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/baseline_smoke \
  --max-questions 5

python my_exp_2/scripts/run_single_edit_experiment.py \
  --methods LoRA ROME MEMIT WISE \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/single_edit/smoke_all_3facts \
  --max-facts 3 \
  --eval-scope fact-only

python my_exp_2/scripts/run_sequential_edit_experiment.py \
  --methods ROME WISE \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --output-dir my_exp_2/outputs/sequential_edit/smoke_rome_wise_2steps \
  --max-steps 2 \
  --eval-scope fact-only \
  --retention-fact-limit 2
```

## Unified Heavy Run

Preflight:

```bash
python my_exp_2/scripts/run_full_experiment_pipeline.py \
  --run-name oilgas_qwen25_3b_full_20facts \
  --methods LoRA ROME MEMIT WISE \
  --data-dir my_exp_2/data \
  --model my_exp/models/Qwen2.5-3B \
  --max-facts 20 \
  --eval-scope full \
  --single-eval-mode full \
  --sequential-eval-mode full \
  --stop-after preflight
```

Full pipeline:

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

Equivalent `Makefile` targets:

```bash
make big-preflight
make big-smoke-pipeline
make big-run
make big-run-resume
make big-reports
```

## Expected Output Areas

- Baseline: `my_exp_2/outputs/baseline_*`
- Single-edit: `my_exp_2/outputs/single_edit/*`
- Sequential-edit: `my_exp_2/outputs/sequential_edit/*`
- Aggregated metrics: `my_exp_2/outputs/metrics/*`
- Unified pipeline: `my_exp_2/outputs/full_pipeline/*`

## Important Constraints

- Full runs are GPU-heavy.
- Sequential full runs are long-running jobs.
- Model weights are local-only and are not included in the repository.
- Lightweight sequential runs can intentionally use sampled retention/general/domain subsets.
- For DGX/container execution, use [DGX_RUNBOOK_RU.md](DGX_RUNBOOK_RU.md).
