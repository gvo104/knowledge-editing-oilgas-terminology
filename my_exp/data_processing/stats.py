"""
Потоковый подсчёт статистики по всему файлу.
Использует множества для уникальных значений, не хранит сами строки.
"""
from typing import Iterator, Dict, Set


def compute_statistics(records: Iterator[Dict[str, str]],
                       verbose: bool = True) -> dict:
    """
    Принимает итератор записей, возвращает словарь со статистикой:
        total_lines: общее число записей
        unique_pmids: количество уникальных PMID
        unique_entities: количество уникальных сущностей
    """
    unique_pmids: Set[str] = set()
    unique_entities: Set[str] = set()
    total_lines = 0

    for rec in records:
        total_lines += 1
        unique_pmids.add(rec['pmid'])
        unique_entities.add(rec['entity_name'])

        if verbose and total_lines % 100_000 == 0:
            print(f"  Обработано {total_lines} строк...")

    return {
        'total_lines': total_lines,
        'unique_pmids': len(unique_pmids),
        'unique_entities': len(unique_entities)
    }