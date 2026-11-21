from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    prompt_ids: list[int]
    response_ids: list[int]


def parse_calls(text):
    """Reject partially valid completions instead of silently rewarding a valid fragment."""
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    if not matches or text.count("<tool_call>") != len(matches) or text.count("</tool_call>") != len(matches):
        return []
    calls = []
    try:
        for match in matches:
            call = json.loads(match.group(1))
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                return []
            if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                return []
            calls.append(call)
    except json.JSONDecodeError:
        return []
    return calls


def validate_call(call, tool):
    if call["name"] != tool.name:
        return False
    args = call["arguments"]
    if not set(tool.required) <= args.keys() or not set(args) <= tool.parameters.keys():
        return False
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for name, value in args.items():
        if name == "top_k" and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 50):
            return False
        expected = tool.parameters[name].get("type")
        if expected in types and (not isinstance(value, types[expected]) or
                                  expected in {"integer", "number"} and isinstance(value, bool)):
            return False
    return True


def flatten(values):
    result = []
    for item in values or []:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
