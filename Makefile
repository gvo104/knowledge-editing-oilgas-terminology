PYTHON := conda run -n EasyEdit python

DATA_DIR := my_exp_2/data
MODEL := my_exp/models/Qwen2.5-3B

BASELINE_SMOKE_DIR := my_exp_2/outputs/baseline_smoke
BASELINE_FULL_DIR := my_exp_2/outputs/baseline_full_20facts

SINGLE_SMOKE_WISE_DIR := my_exp_2/outputs/single_edit/smoke_wise
SINGLE_FULL_DIR := my_exp_2/outputs/single_edit/full_20facts_all_4methods
SINGLE_METRICS_FULL_DIR := my_exp_2/outputs/metrics/full_20facts_all_4methods

SEQUENTIAL_SMOKE_WISE_DIR := my_exp_2/outputs/sequential_edit/smoke_wise_2steps
SEQUENTIAL_FULL_DIR := my_exp_2/outputs/sequential_edit/full_20steps_all_4methods
SEQUENTIAL_METRICS_FULL_DIR := my_exp_2/outputs/metrics/sequential/full_20steps_all_4methods

.PHONY: help \
	baseline-smoke baseline-full \
	single-smoke-wise single-full single-metrics single-report single-full-pipeline \
	sequential-smoke-wise sequential-full sequential-metrics sequential-report sequential-full-pipeline \
	full-pipeline

help:
	@printf "%s\n" \
	"Available targets:" \
	"  baseline-smoke              Small baseline run" \
	"  baseline-full               Full baseline run on all current questions" \
	"  single-smoke-wise           Smoke single-edit run for WISE (1 fact, fact-only)" \
	"  single-full                 Full single-edit run for LoRA ROME MEMIT WISE (20 facts, full)" \
	"  single-metrics              Aggregate full single-edit results" \
	"  single-report               Generate markdown report for full single-edit results" \
	"  single-full-pipeline        baseline-full + single-full + single-metrics + single-report" \
	"  sequential-smoke-wise       Smoke sequential run for WISE (2 steps, fact-only)" \
	"  sequential-full             Full sequential run for LoRA ROME MEMIT WISE (20 steps, full)" \
	"  sequential-metrics          Aggregate full sequential results" \
	"  sequential-report           Generate markdown report and graphs for full sequential results" \
	"  sequential-full-pipeline    baseline-full + sequential-full + sequential-metrics + sequential-report" \
	"  full-pipeline               baseline-full + single-full-pipeline + sequential-full-pipeline"

baseline-smoke:
	$(PYTHON) my_exp_2/scripts/run_baseline_eval.py \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(BASELINE_SMOKE_DIR) \
		--max-questions 5 \
		--max-new-tokens 32

baseline-full:
	$(PYTHON) my_exp_2/scripts/run_baseline_eval.py \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(BASELINE_FULL_DIR) \
		--max-new-tokens 32

single-smoke-wise:
	$(PYTHON) my_exp_2/scripts/run_single_edit_experiment.py \
		--methods WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(SINGLE_SMOKE_WISE_DIR) \
		--max-facts 1 \
		--eval-scope fact-only

single-full:
	$(PYTHON) my_exp_2/scripts/run_single_edit_experiment.py \
		--methods LoRA ROME MEMIT WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(SINGLE_FULL_DIR) \
		--max-facts 20 \
		--eval-scope full

single-metrics:
	$(PYTHON) my_exp_2/scripts/compute_metrics.py \
		--single-edit-dir $(SINGLE_FULL_DIR) \
		--baseline-dir $(BASELINE_FULL_DIR) \
		--output-dir $(SINGLE_METRICS_FULL_DIR)

single-report:
	$(PYTHON) my_exp_2/scripts/generate_report.py \
		--single-edit-dir $(SINGLE_FULL_DIR) \
		--baseline-dir $(BASELINE_FULL_DIR) \
		--output $(SINGLE_METRICS_FULL_DIR)/report.md \
		--num-cases 5

single-full-pipeline: baseline-full single-full single-metrics single-report

sequential-smoke-wise:
	$(PYTHON) my_exp_2/scripts/run_sequential_edit_experiment.py \
		--methods WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(SEQUENTIAL_SMOKE_WISE_DIR) \
		--max-steps 2 \
		--eval-scope fact-only \
		--retention-fact-limit 2

sequential-full:
	$(PYTHON) my_exp_2/scripts/run_sequential_edit_experiment.py \
		--methods LoRA ROME MEMIT WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-dir $(SEQUENTIAL_FULL_DIR) \
		--max-steps 20 \
		--eval-scope full \
		--retention-fact-limit 5 \
		--general-limit 5 \
		--domain-limit 5 \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32

sequential-metrics:
	$(PYTHON) my_exp_2/scripts/compute_sequential_metrics.py \
		--sequential-edit-dir $(SEQUENTIAL_FULL_DIR) \
		--baseline-dir $(BASELINE_FULL_DIR) \
		--output-dir $(SEQUENTIAL_METRICS_FULL_DIR)

sequential-report:
	$(PYTHON) my_exp_2/scripts/generate_sequential_report.py \
		--sequential-edit-dir $(SEQUENTIAL_FULL_DIR) \
		--baseline-dir $(BASELINE_FULL_DIR) \
		--output $(SEQUENTIAL_METRICS_FULL_DIR)/report.md \
		--num-steps 5

sequential-full-pipeline: baseline-full sequential-full sequential-metrics sequential-report

full-pipeline: baseline-full single-full single-metrics single-report sequential-full sequential-metrics sequential-report
