# my_exp_2: исследовательский контур Knowledge Editing

`my_exp_2/` — основной исследовательский слой проекта поверх `EasyEdit`. Он нужен для воспроизводимого сравнения методов редактирования знаний в задаче доменной адаптации терминологии без полного переобучения модели.

Текущий домен: нефтегазовая терминология.  
Базовая модель: `Qwen2.5-3B`.  
Методы: `LoRA`, `ROME`, `MEMIT`, `WISE`.  
Полный актуальный прогон: `my_exp_2/outputs/oilgas_qwen25_3b_full_20facts`.

## Постановка проблемы

Большие языковые модели могут уверенно отвечать на общие вопросы, но плохо работать с узкой терминологией: не знать факты, отвечать нестабильно или заменять точный ответ общими рассуждениями. Полное дообучение модели решает часть проблемы, но требует ресурсов, данных и времени.

`Knowledge editing` рассматривается как более локальная альтернатива: изменить поведение модели относительно ограниченного набора фактов и проверить, насколько это изменение переносится на другие формулировки, не разрушает соседние знания и сохраняется при серии правок.

В этом проекте knowledge editing проверяется как инструмент controlled domain adaptation для терминологии нефтегазовой отрасли.

## Исследовательские цели

Основные вопросы:

- можно ли добавить или скорректировать доменный факт без полного дообучения модели;
- какой метод лучше работает для одиночного редактирования;
- какой метод лучше выдерживает последовательное накопление правок;
- как отличаются режимы `knowledge insertion` и `knowledge replacement`;
- какие типы фактов, `subject`, `relation` и `level` хуже редактируются;
- насколько редактирование портит соседние, доменные и общие знания.

## Данные

Основной набор содержит 20 нефтегазовых фактов в формате:

```text
subject -> relation -> object
```

Примеры:

```text
Кероген -> является источником -> углеводородов
Гидроразрыв пласта -> создаёт -> трещины в продуктивном пласте
Проппант -> удерживает -> трещины гидроразрыва открытыми
```

Структура:

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

Размеры текущих eval-наборов:

- `triplets/oilgas.json`: 20 фактов;
- `fact_questions.json`: 260 вопросов вокруг редактируемых фактов;
- `domain_questions.json`: 40 общих нефтегазовых вопросов;
- `general_questions.json`: 50 недоменных вопросов общего знания;
- `sequential_order.json`: фиксированный порядок 20 sequential-шагов.

Типы fact-вопросов:

- `direct`: прямое усвоение факта;
- `paraphrase`: перенос на переформулировки;
- `reverse`: обратная формулировка;
- `neighbor`: соседние доменные знания;
- `locality`: близкие знания, которые не должны ломаться.

Поля `subject`, `relation` и `level` используются не только для редактирования, но и для последующей аналитики: можно смотреть, какие типы терминов и отношений хуже усваиваются или быстрее забываются.

## Методы

В текущем полном сравнении участвуют четыре метода:

- `LoRA` — parameter-efficient baseline, который часто хорошо записывает целевой факт, но может быть агрессивным к locality.
- `ROME` — локальный targeted editing method, хорошо подходящий для одиночных factual правок.
- `MEMIT` — targeted weight-editing method, близкий по роли к ROME, но с другим профилем силы правки и сохранности.
- `WISE` — метод редактирования, добавленный как четвертый полноценный участник сравнения; в текущем полном запуске он особенно силен в sequential-режиме.

Методы не переписываются в `my_exp_2`: их реализация берется из `EasyEdit`. Этот слой отвечает за данные, orchestration, оценку и отчеты.

## Экспериментальные режимы

### Baseline

Исходная модель оценивается до редактирования на:

- `fact_questions`;
- `domain_questions`;
- `general_questions`.

Цель baseline — понять, что модель уже знает, где отвечает шумно и насколько слабым является исходное доменное знание.

### Single-edit

Каждый факт редактируется независимо:

```text
base model -> edit one fact -> evaluate -> discard edited state
```

Этот режим показывает, насколько метод способен внести один факт без накопления предыдущих изменений.

### Sequential-edit

Факты вносятся последовательно в одну изменяющуюся модель:

```text
model_step_0 -> edit fact_1 -> eval
model_step_1 -> edit fact_2 -> eval
...
model_step_19 -> edit fact_20 -> eval
```

Этот режим проверяет, может ли метод накапливать серию доменных правок и удерживать ранее внесенные знания.

В полном запуске `sequential-edit` выполнен с `eval_mode=full`:

- retention считается по всем уже внесенным фактам;
- general/domain оцениваются по полным наборам;
- generation details сохраняются в raw step-файлах.

## Target Old, Replacement и Insertion

Некоторым методам нужен `target_old`: текущее значение, которое модель связывает с редактируемым prompt. В проекте оно не считается статичным.

- В `single-edit` `target_old` уточняется на исходной модели.
- В `sequential-edit` `target_old` уточняется перед каждым шагом на текущем измененном состоянии модели.

Если ответ модели пустой, шумный или нестабильный, pipeline использует fallback к `target_new`. Это разделяет два режима:

- `knowledge replacement`: модель уже что-то устойчиво отвечала, и это заменяется;
- `knowledge insertion`: модель фактически не знала факт, и новое знание добавляется.

В текущем домене большинство кейсов оказываются ближе к `knowledge insertion`, потому что исходная модель слабо знает нефтегазовые факты.

## Актуальные результаты полного запуска

Источник:

```text
my_exp_2/outputs/oilgas_qwen25_3b_full_20facts
```

### Baseline

Исходная `Qwen2.5-3B`:

| Набор | Accuracy |
|---|---:|
| `fact_questions` | `0.053846` |
| `domain_questions` | `0.025` |
| `general_questions` | `0.68` |

Интерпретация: модель почти не знает подготовленный нефтегазовый benchmark, но относительно уверенно отвечает на общие недоменные вопросы.

### Single-edit, 20 фактов

| Метод | Reliability | Generalization | Global locality | Domain score | Edit quality |
|---|---:|---:|---:|---:|---:|
| `LoRA` | `0.765` | `0.744` | `0.050` | `0.094` | `0.075` |
| `ROME` | `0.826` | `0.646` | `0.933` | `0.933` | `0.753` |
| `MEMIT` | `0.774` | `0.606` | `0.933` | `0.744` | `0.725` |
| `WISE` | `1.000` | `0.546` | `1.000` | `0.983` | `0.738` |

Лучший метод по `edit_quality`: `ROME = 0.753`.  
`WISE` близок по итоговой single-edit оценке и лучше всех по reliability/locality, но уступает ROME по generalization.

### Sequential-edit, 20 шагов

| Метод | Current reliability | Current generalization | Retention | Global locality | Domain score | Sequential quality |
|---|---:|---:|---:|---:|---:|---:|
| `LoRA` | `0.599` | `0.513` | `0.031` | `0.009` | `0.001` | `0.017` |
| `ROME` | `0.827` | `0.634` | `0.271` | `0.698` | `0.058` | `0.416` |
| `MEMIT` | `0.411` | `0.285` | `0.063` | `0.212` | `0.005` | `0.073` |
| `WISE` | `0.742` | `0.553` | `0.440` | `0.701` | `0.135` | `0.503` |

Лучший метод по `sequential_quality`: `WISE = 0.503`.  
`ROME` остается сильным, но уступает WISE по retention и итоговой sequential-оценке.

### Основные выводы

- `ROME` — лучший метод для одиночного редактирования по балансу успешности и локальности.
- `WISE` — лучший метод для последовательного редактирования в текущем полном прогоне.
- `LoRA` хорошо записывает целевой факт, но сильно портит locality и почти не удерживает sequential-цепочку.
- `MEMIT` конкурентен в single-edit, но существенно деградирует в sequential.
- Single-edit и sequential-edit дают разные ранжирования методов, поэтому их нельзя заменять друг другом.

## Единый pipeline

Основной запуск:

```bash
make big-run
```

Рекомендуемый порядок на новой машине:

```bash
make big-preflight
make big-smoke-pipeline
make big-run
```

Команды:

- `make big-preflight`: проверяет CUDA, модель, данные, hparams и smoke-прогоны.
- `make big-smoke-pipeline`: маленький end-to-end запуск на 2 фактах.
- `make big-run`: полный запуск single + sequential + metrics + reports.
- `make big-run-resume`: продолжение после обрыва.
- `make big-reports`: пересборка метрик и отчетов из raw-результатов.

Python entrypoint:

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

## Outputs

Текущая структура полного запуска:

```text
my_exp_2/outputs/oilgas_qwen25_3b_full_20facts/
  baseline/
  single_edit/
  sequential_edit/
  metrics/
  reports/
  preflight/
  logs/
  run_manifest.json
```

Ключевые файлы:

- `reports/overall_summary.md`: общий вывод single vs sequential;
- `reports/single_edit_report.md`: подробный отчет по одиночным правкам;
- `reports/sequential_edit_report.md`: подробный отчет по последовательным правкам;
- `metrics/overall/single_vs_sequential.csv`: сравнение режимов;
- `metrics/single/case_metrics.csv`: single-edit метрики по кейсам;
- `metrics/sequential/step_metrics.csv`: sequential-метрики по шагам;
- `metrics/sequential/retention_matrix.csv`: удержание фактов по шагам;
- `preflight/preflight_report.json`: проверка железа, модели и данных;
- `run_manifest.json`: конфигурация и статус стадий.

Preflight полного запуска:

- GPU: `NVIDIA H100 PCIe`;
- VRAM: около `79 GB`;
- CUDA: `12.8`;
- модель загружалась на `cuda:0`;
- данные: 20 фактов, 260 fact questions, 40 domain questions, 50 general questions.

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

Основные роли:

- `validate_data.py`: проверка данных;
- `run_baseline_eval.py`: baseline исходной модели;
- `run_single_edit_experiment.py`: isolated single-edit benchmark;
- `run_sequential_edit_experiment.py`: sequential benchmark;
- `run_full_experiment_pipeline.py`: единый запуск preflight -> baseline -> single -> sequential -> reports;
- `compute_metrics.py`: агрегация single-edit;
- `compute_sequential_metrics.py`: агрегация sequential-edit;
- `generate_report.py`: single-edit markdown report;
- `generate_sequential_report.py`: sequential markdown report и графики.

## Ограничения и следующие вопросы

Текущие ограничения:

- датасет небольшой: 20 фактов;
- домен ограничен нефтегазовой терминологией;
- single-edit post-eval в основном опирается на EasyEdit internal metrics;
- метрики основаны на exact/alias matching, без LLM-judge;
- большинство кейсов ближе к `knowledge insertion`, чем к `replacement`.

Дальнейшие вопросы:

- какие `relation` и `level` систематически хуже редактируются;
- влияет ли длина или составность `subject` на качество;
- почему `WISE` лучше удерживает sequential-цепочку;
- можно ли улучшить `LoRA` через отдельный SFT/augmented режим;
- насколько результаты сохранятся на большем домене и другой модели.

## Relation to EasyEdit

`EasyEdit` предоставляет backend методов редактирования.  
`my_exp_2` предоставляет исследовательский протокол:

- доменные данные;
- сбор runtime cases;
- target-old resolution;
- запуск baseline/single/sequential;
- агрегация;
- отчеты;
- аналитика по фактам и режимам.
