"""就业材料识别任务的输出结构、提示词与结果清洗。"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable


EMPLOYMENT_TYPES = (
    "升学",
    "创业",
    "就业-就业协议",
    "就业-劳动合同",
    "就业-其他录用形式",
    "就业-基层项目",
    "就业-科研助理或博士后入站",
    "自由职业",
    "无法判断",
)

TYPE_ALIASES = {
    "就业协议": "就业-就业协议",
    "签订就业协议": "就业-就业协议",
    "劳动合同": "就业-劳动合同",
    "签订劳动合同": "就业-劳动合同",
    "其他录用": "就业-其他录用形式",
    "其他录用形式": "就业-其他录用形式",
    "基层项目": "就业-基层项目",
    "科研助理": "就业-科研助理或博士后入站",
    "博士后入站": "就业-科研助理或博士后入站",
    "科研助理/博士后入站": "就业-科研助理或博士后入站",
    "无法识别": "无法判断",
    "不确定": "无法判断",
}

PROMPT = """你是高校毕业生就业佐证材料的信息抽取助手。

请只完成三项任务：
1. 判断本页展示的就业协议或佐证材料种类 agreement_type。优先使用材料正式标题，例如“普通高等学校毕业生就业协议书（三方协议）”“劳动合同”“录用通知书”“研究生录取通知书”“营业执照”“基层项目录用材料”。看不清时写“无法判断”。
2. 认定就业类型 employment_type，只能从以下类别中选择一个：
   - 升学
   - 创业
   - 就业-就业协议
   - 就业-劳动合同
   - 就业-其他录用形式
   - 就业-基层项目
   - 就业-科研助理或博士后入站
   - 自由职业
   - 无法判断
3. 按自然阅读顺序提取本页所有有实际内容的业务字段 fields，包括印刷填写内容、手写内容和印章文字。

字段提取规则：
- 每个字段使用紧凑二元数组 ["字段名","字段值"]，例如 ["毕业生姓名","张三"]。
- 标题写作 ["文档标题","……"]，签字和印章分别写作 ["签字1","……"]、["印章1","……"]。
- 提取姓名、学号、证件号、学校、专业、学历、单位、岗位、地址、电话、日期、合同期限、薪资、签字、印章等具有业务意义的信息。
- 没有填写内容的空白字段不要输出；固定模板说明、法律条款和通用正文不要逐字抄写，除非其中包含影响就业认定的具体信息。
- 保留原文，不要根据常识补写；无法辨认的字符写“[无法辨认]”。
- 不要输出置信度、分析过程、分类理由、Markdown 或 JSON 之外的内容。

必须只输出以下结构的合法 JSON：
{
  "agreement_type": "材料种类",
  "employment_type": "一个允许的就业类型",
  "fields": [
    ["字段名", "字段值"]
  ]
}
"""


def has_complete_json_object(raw_text: str) -> bool:
    text = raw_text.strip()
    start = text.find("{")
    return start >= 0 and text.rfind("}") > start


def close_truncated_json(candidate: str) -> str:
    """保守补齐被 max_new_tokens 截断的引号、数组和对象结尾。"""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack and char == stack[-1]:
            stack.pop()

    repaired = candidate.rstrip()
    if in_string:
        if repaired.endswith("\\"):
            repaired = repaired[:-1]
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(":"):
        repaired += " null"
    if repaired.endswith(","):
        repaired = repaired[:-1]
    repaired += "".join(reversed(stack))
    return repaired


def extract_json(raw_text: str) -> dict[str, Any]:
    """提取JSON，并修复常见缺逗号、缺引号和输出截断。"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象起始符")
    candidate = text[start : end + 1] if end > start else text[start:]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as original_error:
        try:
            from json_repair import repair_json  # type: ignore
        except ImportError:
            try:
                value = json.loads(close_truncated_json(candidate))
            except json.JSONDecodeError as repaired_error:
                raise ValueError(
                    f"模型输出不是合法 JSON：{original_error}; "
                    f"内置截断修复也失败：{repaired_error}"
                ) from original_error
        else:
            value = repair_json(candidate, return_objects=True)
    if not isinstance(value, dict):
        raise ValueError("模型输出 JSON 不是对象")
    return value


def normalize_employment_type(value: Any) -> str:
    text = str(value or "").strip()
    if text in EMPLOYMENT_TYPES:
        return text
    if text in TYPE_ALIASES:
        return TYPE_ALIASES[text]
    for category in EMPLOYMENT_TYPES:
        if category != "无法判断" and category in text:
            return category
    return "无法判断"


def normalize_fields(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        value = [{"name": key, "value": item} for key, item in value.items()]
    if not isinstance(value, list):
        return []
    fields: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            name = str(
                item.get("name")
                or item.get("field_name")
                or item.get("key")
                or f"未命名字段{index}"
            ).strip()
            raw_value = item.get("value", item.get("field_value", item.get("text", "")))
            if isinstance(raw_value, (dict, list)):
                field_value = json.dumps(raw_value, ensure_ascii=False)
            else:
                field_value = str(raw_value if raw_value is not None else "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = str(item[0] if item[0] is not None else "").strip()
            raw_value = item[1]
            if isinstance(raw_value, (dict, list)):
                field_value = json.dumps(raw_value, ensure_ascii=False)
            else:
                field_value = str(raw_value if raw_value is not None else "").strip()
        elif isinstance(item, str):
            name, field_value = f"未命名字段{index}", item.strip()
        else:
            continue
        fields.append({"name": name or f"未命名字段{index}", "value": field_value})
    return fields


def normalize_response(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "agreement_type": str(
            parsed.get("agreement_type")
            or parsed.get("document_kind")
            or "无法判断"
        ).strip() or "无法判断",
        "employment_type": normalize_employment_type(parsed.get("employment_type")),
        "fields": normalize_fields(parsed.get("fields")),
    }


def choose_majority(values: Iterable[str], unknown: str = "无法判断") -> str:
    """选取多数结果；票数相同时按首次出现顺序。"""
    ordered = [str(value).strip() for value in values if str(value).strip()]
    usable = [value for value in ordered if value != unknown]
    if not usable:
        return unknown
    counts = Counter(usable)
    highest = max(counts.values())
    return next(value for value in usable if counts[value] == highest)
