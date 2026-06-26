# Документация по данным my_exp_2

Эта директория содержит нефтегазовый benchmark для экспериментов по `knowledge editing`.

Данные специально сделаны небольшими и контролируемыми. Это не большой отраслевой датасет, а воспроизводимый набор для сравнения методов редактирования знаний в одинаковых условиях.

## Цель данных

Данные организованы так, чтобы каждый редактируемый факт можно было проверить с нескольких сторон:

- усвоение прямого факта;
- перенос на парафразы;
- согласованность на обратных вопросах;
- влияние на соседние доменные знания;
- локальность относительно близких фактов;
- сохранение более широких доменных и общих знаний.

## Размеры текущего набора

- `triplets/oilgas.json`: 20 базовых фактов;
- `edit_requests/oilgas_edit_requests.json`: 20 edit requests;
- `fact_questions.json`: 260 вопросов вокруг фактов;
- `domain_questions.json`: 40 общих нефтегазовых вопросов;
- `general_questions.json`: 50 недоменных вопросов общего знания;
- `sequential_order.json`: порядок из 20 sequential-шагов.

## Структура

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

## Fact layer

### `triplets/oilgas.json`

Содержит 20 базовых фактов в формате:

```text
subject -> relation -> object
```

Пример:

```text
Кероген -> является источником -> углеводородов
```

Основные поля:

- `fact_id`: стабильный идентификатор факта;
- `subject`: редактируемая сущность или термин;
- `relation`: тип отношения;
- `object`: целевое значение факта;
- `level`: смысловая группа факта.

`level` используется для анализа того, какие категории нефтегазовых знаний хуже редактируются или быстрее забываются.

Текущие группы:

- `general`;
- `basic_geology`;
- `petroleum_geology`;
- `reservoir_properties`;
- `well_construction`;
- `production_stimulation`.

## Edit layer

### `edit_requests/oilgas_edit_requests.json`

Содержит runtime-запросы для методов редактирования.

Основные поля:

- `fact_id`: связь с `triplets/oilgas.json`;
- `prompt`: prompt, на котором выполняется правка;
- `subject`: сущность, которая должна присутствовать в prompt для совместимости с EasyEdit;
- `target_new`: новое целевое знание;
- `target_old`: старое значение, если оно известно заранее.

В текущем pipeline `target_old` не считается жестко заданным. Для методов, которым он нужен, значение уточняется по текущему состоянию модели:

- в single-edit — на исходной модели;
- в sequential-edit — перед каждым шагом на текущей измененной модели.

Если модель отвечает неустойчиво или шумно, используется fallback к `target_new`. Это позволяет различать:

- `knowledge replacement`;
- `knowledge insertion`.

## Evaluation layer

### `evaluate_set/fact_questions.json`

Главный набор вопросов вокруг редактируемых фактов.

Типы вопросов:

- `direct`: прямой вопрос на целевой факт;
- `paraphrase`: переформулировки прямого вопроса;
- `reverse`: вопрос в обратном направлении;
- `neighbor`: соседние доменные факты;
- `locality`: близкие знания, которые не должны ломаться.

Типичные поля:

- `question_id`;
- `fact_id`;
- `question_type`;
- `question`;
- `expected_answer`;
- `aliases`.

Эти вопросы используются в single-edit и sequential-edit для оценки reliability, generalization, reverse, neighbor и fact locality.

### `evaluate_set/domain_questions.json`

Содержит 40 более общих нефтегазовых вопросов, не привязанных к одному редактируемому факту.

Назначение:

- проверить сохранение более широкой доменной компетентности;
- увидеть, не начинает ли модель после правок отвечать хуже на общие нефтегазовые вопросы;
- оценивать `domain_score` в full-режиме.

### `evaluate_set/general_questions.json`

Содержит 50 недоменных вопросов общего знания.

Назначение:

- проверить global locality;
- убедиться, что нефтегазовые правки не ломают общие знания модели.

Вопросы покрывают разные темы: географию, историю, литературу, биологию, математику, программирование и базовую науку.

## Sequential layer

### `sequential/sequential_order.json`

Задает фиксированный порядок внесения 20 фактов в sequential-edit.

Назначение:

- обеспечить воспроизводимость;
- сравнивать все методы на одинаковом порядке;
- анализировать forgetting/retention по шагам.

## Training layer

### `train_sets/lora_train.json`

Минимальный набор QA-пар для будущих LoRA/SFT baseline.

### `train_sets/lora_augmented_train.json`

Расширенный набор с дополнительными формулировками.

В текущем полном сравнении основные методы запускаются через EasyEdit editing pipeline: `LoRA`, `ROME`, `MEMIT`, `WISE`. Train sets остаются полезными для дальнейшего сравнения с SFT/augmented LoRA.

## Как данные используются

### Baseline

До редактирования модель оценивается на:

- `fact_questions.json`;
- `domain_questions.json`;
- `general_questions.json`.

### Single-edit

Каждый факт редактируется отдельно. Используются:

- `triplets/oilgas.json`;
- `edit_requests/oilgas_edit_requests.json`;
- `fact_questions.json`;
- `domain_questions.json`;
- `general_questions.json`.

### Sequential-edit

Факты редактируются по порядку из `sequential_order.json`. После каждого шага оцениваются:

- текущий факт;
- все ранее внесенные факты;
- полный domain set;
- полный general set.

## Использование в аналитике

После полного запуска данные агрегируются в CSV/JSON:

- `metrics/single/case_metrics.csv`;
- `metrics/single/subject_relation_analysis.csv`;
- `metrics/single/replacement_vs_insertion.csv`;
- `metrics/sequential/step_metrics.csv`;
- `metrics/sequential/retention_matrix.csv`;
- `metrics/sequential/subject_relation_analysis.csv`;
- `metrics/sequential/replacement_vs_insertion.csv`.

Поля `subject`, `relation`, `object`, `level`, `question_type`, `expected_answer` и `aliases` нужны для поиска закономерностей:

- какие relation редактируются хуже;
- какие subject сложнее;
- какие level быстрее забываются;
- чем отличается insertion от replacement;
- какие вопросы чаще всего ломаются после последовательных правок.
