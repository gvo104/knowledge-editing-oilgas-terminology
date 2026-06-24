PYTHON := conda run -n EasyEdit python

DATA_DIR := my_exp_2/data
MODEL := my_exp/models/Qwen2.5-3B
METHODS := LoRA ROME MEMIT WISE
RUN_NAME := oilgas_qwen25_3b_full_20facts
MAX_FACTS := 20
FULL_PIPELINE_ROOT := my_exp_2/outputs/full_pipeline

DOCKER_IMAGE := galkin-easyedit:dgx
DOCKER_CONTAINER := galkin-easyedit
# GPU 7 is reserved for another group on DGX. Do not use DGX_GPU=7.
DGX_GPU := 6
DOCKER_SHM_SIZE := 32g
DGX_METHODS := LoRA ROME MEMIT WISE

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
	full-pipeline \
	big-preflight big-smoke-single big-smoke-sequential big-smoke-pipeline \
	big-run big-run-resume big-reports \
	docker-build-dgx docker-run-dgx docker-shell-dgx docker-stop-dgx \
	download-model dgx-preflight dgx-smoke dgx-run

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
	"  full-pipeline               baseline-full + single-full-pipeline + sequential-full-pipeline" \
	"" \
	"Big unified pipeline targets:" \
	"  big-preflight               Run only preflight for the unified full pipeline" \
	"  big-smoke-single            Run unified pipeline through single stage on 1 WISE fact" \
	"  big-smoke-sequential        Run unified pipeline through sequential stage on 2 ROME/WISE facts" \
	"  big-smoke-pipeline          Run small end-to-end unified pipeline with selected METHODS/MAX_FACTS" \
	"  big-run                     Run full unified single + sequential experiment" \
	"  big-run-resume              Resume full unified experiment" \
	"  big-reports                 Rebuild reports for an existing full unified run" \
	"" \
	"DGX Docker targets:" \
	"  docker-build-dgx            Build Docker image from Dockerfile.dgx" \
	"  docker-run-dgx              Start detached DGX container on selected GPU" \
	"                              DGX_GPU=7 is forbidden: reserved for another group" \
	"  docker-shell-dgx            Open bash inside the DGX container" \
	"  docker-stop-dgx             Stop the DGX container" \
	"  download-model              Download Qwen/Qwen2.5-3B into my_exp/models/Qwen2.5-3B" \
	"  dgx-preflight               Run big-preflight with all DGX methods" \
	"  dgx-smoke                   Run big-smoke-pipeline with all DGX methods on 2 facts" \
	"  dgx-run                     Run full DGX pipeline with all DGX methods" \
	"" \
	"Override examples:" \
	"  make big-run RUN_NAME=my_run MODEL=/path/to/model" \
	"  make big-run-resume RUN_NAME=my_run" \
	"  make big-smoke-pipeline METHODS='ROME WISE' MAX_FACTS=2" \
	"  make docker-run-dgx DGX_GPU=6 DOCKER_CONTAINER=galkin-easyedit"

docker-build-dgx:
	docker build -f Dockerfile.dgx -t $(DOCKER_IMAGE) .

docker-run-dgx:
	docker run -dit \
		--name $(DOCKER_CONTAINER) \
		--gpus '"device=$(DGX_GPU)"' \
		--ipc=host \
		--shm-size=$(DOCKER_SHM_SIZE) \
		-v "$(CURDIR)":/workspace/EasyEdit \
		-w /workspace/EasyEdit \
		$(DOCKER_IMAGE)

docker-shell-dgx:
	docker exec -it $(DOCKER_CONTAINER) bash

docker-stop-dgx:
	docker stop $(DOCKER_CONTAINER)

download-model:
	mkdir -p my_exp/models
	conda run -n EasyEdit hf download Qwen/Qwen2.5-3B --local-dir $(MODEL)

dgx-preflight:
	$(MAKE) big-preflight METHODS="$(DGX_METHODS)" MODEL=$(MODEL) MAX_FACTS=$(MAX_FACTS) RUN_NAME=$(RUN_NAME)

dgx-smoke:
	$(MAKE) big-smoke-pipeline METHODS="$(DGX_METHODS)" MODEL=$(MODEL) MAX_FACTS=2

dgx-run:
	$(MAKE) big-run METHODS="$(DGX_METHODS)" MODEL=$(MODEL) MAX_FACTS=$(MAX_FACTS) RUN_NAME=$(RUN_NAME)

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

# Цели для тяжелого единого pipeline.
# Рекомендуемый порядок на новой машине:
#   1. make big-preflight
#   2. make big-smoke-pipeline
#   3. make big-run
# Если полный запуск прервался, использовать:
#   make big-run-resume
# Если raw-результаты уже есть и нужно только пересобрать метрики/отчеты:
#   make big-reports
#
# Часто используемые переопределения параметров:
#   make big-run RUN_NAME=my_run MODEL=/path/to/model
#   make big-run METHODS='ROME WISE' MAX_FACTS=2

# Проверяет CUDA, загрузку модели, данные и smoke-прогоны перед тяжелым запуском.
big-preflight:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name $(RUN_NAME) \
		--methods $(METHODS) \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts $(MAX_FACTS) \
		--eval-scope full \
		--single-eval-mode full \
		--sequential-eval-mode full \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--stop-after preflight

# Маленькая проверка WISE в single-edit: hparams и интеграция с editor.
big-smoke-single:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name smoke_single_wise \
		--methods WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts 1 \
		--eval-scope fact-only \
		--single-eval-mode sample \
		--sequential-eval-mode sample \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--stop-after single \
		--overwrite

# Маленькая sequential-проверка ROME + WISE на двух фактах без полной стоимости eval.
big-smoke-sequential:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name smoke_seq_rome_wise \
		--methods ROME WISE \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts 2 \
		--eval-scope fact-only \
		--single-eval-mode sample \
		--sequential-eval-mode sample \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--stop-after sequential \
		--overwrite

# Маленькая end-to-end проверка: preflight, baseline, single, sequential и отчеты.
big-smoke-pipeline:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name smoke_full_pipeline_2facts \
		--methods $(METHODS) \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts $(MAX_FACTS) \
		--eval-scope full \
		--single-eval-mode full \
		--sequential-eval-mode full \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--overwrite

# Основной тяжелый серверный запуск: single-edit + sequential-edit + аналитика + отчеты.
big-run:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name $(RUN_NAME) \
		--methods $(METHODS) \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts $(MAX_FACTS) \
		--eval-scope full \
		--single-eval-mode full \
		--sequential-eval-mode full \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32

# Продолжает тяжелый запуск после обрыва, используя маркеры завершенных стадий.
big-run-resume:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name $(RUN_NAME) \
		--methods $(METHODS) \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts $(MAX_FACTS) \
		--eval-scope full \
		--single-eval-mode full \
		--sequential-eval-mode full \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--resume

# Пересобирает метрики и markdown-отчеты из уже существующих raw single/sequential результатов.
big-reports:
	$(PYTHON) my_exp_2/scripts/run_full_experiment_pipeline.py \
		--run-name $(RUN_NAME) \
		--methods $(METHODS) \
		--data-dir $(DATA_DIR) \
		--model $(MODEL) \
		--output-root $(FULL_PIPELINE_ROOT) \
		--max-facts $(MAX_FACTS) \
		--eval-scope full \
		--single-eval-mode full \
		--sequential-eval-mode full \
		--generation-max-new-tokens 32 \
		--target-old-max-new-tokens 32 \
		--skip-preflight \
		--skip-baseline \
		--resume \
		--stop-after reports
