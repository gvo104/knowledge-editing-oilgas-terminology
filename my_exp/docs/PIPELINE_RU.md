# Как Доработать `data_processing` Под Knowledge Editing

## Что Уже Есть

Сейчас в `my_exp/data_processing` уже реализованы полезные части:

- чтение PubTator
- группировка сущностей по статье
- построение триплетов
- генерация direct / inverse / paraphrase-запросов
- генерация locality-вопросов

Этого почти достаточно, чтобы построить edit-dataset.

## Главная Проблема Текущих Данных

Текущая связь вида:

- `entity mentioned in PMID`

плохо подходит для knowledge editing напрямую.

Почему:

- у одной сущности часто много PMID
- один PMID часто содержит много сущностей
- значит `ground_truth` часто не уникален

Для методов редактирования это плохо, потому что они лучше работают в режиме:

- один `subject`
- одна целевая связь
- один канонический `ground_truth`

## Что Нужно Для Универсального Формата

Для всех методов (`FT`, `LoRA`, `ROME`, `MEMIT`) нужен единый edit-record:

- `prompt`
- `subject`
- `ground_truth`
- `target_new`

Для хорошей оценки желательно добавить:

- `rephrase_prompt` / `rephrase_prompts`
- `locality_prompt`
- `locality_ground_truth`
- `portability_prompt`
- `portability_ground_truth`
- `reverse_prompt`
- `reverse_ground_truth`

## Что Нужно Изменить В Pipeline

### 1. Фильтрация Неоднозначных Фактов

Нужно оставлять только те записи, где факт достаточно однозначен.

Практическое правило:

- `subject -> object` должен быть почти one-to-one
- `object -> subject` тоже желательно не слишком многозначен

Иначе вы будете редактировать не факт, а шумную many-to-many связь.

### 2. Генерация Counterfactual `target_new`

В PubTator обычно нет готового поля `target_new`.

Поэтому `target_new` надо генерировать отдельно:

- брать другой объект того же типа
- который не равен исходному `ground_truth`

Важно понимать:

- это уже синтетический контрфакт
- он подходит для сравнения методов
- но не всегда соответствует реальному биомедицинскому знанию

### 3. Парафразы

Ваш `query_generator.py` уже генерирует хорошие парафразы.

Их надо использовать так:

- первый парафраз -> `rephrase_prompt`
- еще 2-4 парафраза -> `rephrase_prompts`
- один из них можно использовать как `portability_prompt`

### 4. Reverse-вопросы

Reverse-вопросы не надо смешивать с основным `rewrite`.

Они полезны как отдельная диагностика:

- переносится ли редактирование на обратную формулировку связи

Поэтому их лучше хранить отдельными полями:

- `reverse_prompt`
- `reverse_ground_truth`

### 5. Locality

`generate_locality_queries()` уже дает набор несвязанных вопросов.

Этого достаточно для первого прогона.

Но для хорошей оценки лучше иметь два типа locality:

- общие несвязанные факты
- близкие факты того же домена

Например:

- общий: "What is the capital of France?"
- доменный: другой gene/disease/mutation вопрос, не связанный с редактируемой сущностью

## Что Уже Добавлено

Добавлен модуль:

- [edit_dataset_builder.py](/home/penguin/project/NIR_fast_tray/EasyEdit/my_exp/data_processing/edit_dataset_builder.py)

Он умеет:

- брать `generated_queries`
- брать `locality_queries`
- фильтровать неоднозначные triplets
- выбирать `target_new` как контрфакт
- собирать универсальные edit-records
- сохранять их в `jsonl`

## Как Я Бы Строил Исследование

### Для Сравнения Методов

Нужны:

- clean test set из однозначных фактов
- одинаковый формат данных для всех методов
- одинаковые метрики

Минимальный набор метрик:

- `rewrite`
- `rephrase`
- `locality`
- `portability`

### Для LoRA / SFT Идей

Из ваших требований видно, что вам нужен не только one-shot edit, но и обучение на множестве примеров.

Для этого надо хранить два набора:

1. `edit dataset`
   для точечных методов (`ROME`, `MEMIT`, `FT`)

2. `sft dataset`
   для `LoRA`/`LocFT-BF`

Их не стоит смешивать в один файл.

### Предлагаемая Структура

```text
my_exp/
  data/
    edits/
      template.jsonl
      biomedical_edit_cases.jsonl
    sft/
      biomedical_sft_train.jsonl
      biomedical_sft_val.jsonl
    corpus/
      wiki_20220301_simple/
```

## Практический Вывод

Текущие PubTator-данные можно использовать, но после доработки:

1. отфильтровать неоднозначные связи
2. построить синтетический `target_new`
3. сохранить direct / paraphrase / reverse / locality в одном record
4. отдельно держать SFT-датасет и edit-датасет

Без этих шагов сравнение методов будет шумным и трудно интерпретируемым.

## Следующий Правильный Шаг

Нужно сделать runner, который:

- читает output из `edit_dataset_builder.py`
- прогоняет `FT / LoRA / ROME / MEMIT`
- сохраняет результаты по каждому `case_id`

И отдельно полезно сделать split:

- `train` для LoRA/SFT
- `validation`
- `test` для окончательного сравнения методов
