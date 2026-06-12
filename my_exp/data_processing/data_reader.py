"""
Потоковое чтение данных из gene2pubtatorcentral (поддерживает .gz).
Возвращает итератор словарей с ключами: pmid, gene_id, gene_name.
"""
import gzip
from pathlib import Path
from typing import Iterator, Dict, Optional


def detect_file(path: Path) -> Path:
    """Если файла нет, проверяем .gz и возвращаем актуальный путь."""
    if path.exists():
        return path
    gz_path = path.with_suffix(path.suffix + '.gz')
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Не найден ни {path}, ни {gz_path}")


def parse_line(line: str) -> Optional[Dict[str, str]]:
    """
    Разбирает одну строку табулированного файла.
    Ожидаемый формат: pmid\t...\tgene_id\tgene_name|суффикс\t...
    Возвращает словарь с pmid, gene_id, gene_name (без |суффикса)
    или None, если строка не содержит нужных полей.
    """
    parts = line.strip().split('\t')
    if len(parts) < 4:
        return None
    pmid = parts[0]
    entity_id = parts[2]
    entity_name = parts[3]
    if '|' in entity_name:
        entity_name = entity_name.split('|', 1)[0]
    return {'pmid': pmid, 'entity_id': entity_id, 'entity_name': entity_name}

def iter_records(file_path: Path) -> Iterator[Dict[str, str]]:
    """
    Генератор, построчно отдающий распарсенные записи.
    Автоматически открывает обычный или gzip-файл.
    """
    actual_path = detect_file(file_path)
    open_func = gzip.open if actual_path.suffix == '.gz' else open
    mode = 'rt' if actual_path.suffix == '.gz' else 'r'

    with open_func(actual_path, mode) as f:
        for line in f:
            rec = parse_line(line)
            if rec:
                yield rec