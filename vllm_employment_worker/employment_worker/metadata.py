from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PATH_KEYS = (
    "relative_path",
    "source_relative_path",
    "file_path",
    "filename",
    "文件路径",
    "材料路径",
    "文件名",
)


def normalize_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


class MetadataIndex:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._by_path: dict[str, dict[str, Any]] = {}
        basenames: list[str] = []
        prepared: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            path_key = next((key for key in PATH_KEYS if row.get(key)), None)
            if not path_key:
                continue
            relative_path = normalize_path(str(row[path_key]))
            payload = {
                str(key): value
                for key, value in row.items()
                if key != path_key and value not in (None, "")
            }
            self._by_path[relative_path] = payload
            basename = Path(relative_path).name
            basenames.append(basename)
            prepared.append((basename, payload))
        counts = Counter(basenames)
        self._by_unique_name = {
            basename: payload
            for basename, payload in prepared
            if counts[basename] == 1
        }

    def get(self, relative_path: str) -> dict[str, Any]:
        normalized = normalize_path(relative_path)
        return dict(
            self._by_path.get(normalized)
            or self._by_unique_name.get(Path(normalized).name)
            or {}
        )


def load_metadata(path: Path | None) -> MetadataIndex:
    if path is None:
        return MetadataIndex([])
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"元数据第 {line_number} 行不是 JSON 对象")
                rows.append(value)
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            rows = [dict(item, relative_path=key) if isinstance(item, dict) else {} for key, item in value.items()]
        elif isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
        else:
            raise ValueError("元数据 JSON 必须是对象或对象数组")
    else:
        raise ValueError("登记元数据仅支持 CSV、JSONL 或 JSON")
    return MetadataIndex(rows)

