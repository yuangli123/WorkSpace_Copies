from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_schema(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("输出 Schema 必须是 JSON 对象")
    return value


def extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型回答中未找到完整 JSON 对象")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型回答的顶层 JSON 不是对象")
    return value


def validate_analysis(value: dict[str, Any], schema: dict[str, Any]) -> None:
    required = {str(item) for item in schema.get("required", [])}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"模型 JSON 缺少字段：{', '.join(missing)}")
    try:
        from jsonschema import validate
    except ImportError:
        return
    validate(instance=value, schema=schema)


def response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "employment_evidence_review",
            "strict": True,
            "schema": schema,
        },
    }
