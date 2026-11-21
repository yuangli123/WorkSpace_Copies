from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class PromptBuilder:
    def __init__(self, prompt_dir: str | Path, business_rules: dict[str, Any]) -> None:
        root = Path(prompt_dir)
        self.system_prompt = (root / "system_prompt.txt").read_text(encoding="utf-8")
        self.review_template = (root / "document_review_prompt.txt").read_text(encoding="utf-8")
        self.merge_template = (root / "document_merge_prompt.txt").read_text(encoding="utf-8")
        self.business_rules = business_rules

    def build_review(
        self,
        relative_path: str,
        page_numbers: Iterable[int],
        registered_fields: dict[str, Any],
        compact: bool = False,
    ) -> str:
        page_list = list(page_numbers)
        replacements = {
            "{{RELATIVE_PATH}}": relative_path,
            "{{PAGE_NUMBERS}}": json.dumps(page_list, ensure_ascii=False),
            "{{REGISTERED_FIELDS}}": json.dumps(registered_fields, ensure_ascii=False, indent=2),
            "{{BUSINESS_RULES}}": json.dumps(self.business_rules, ensure_ascii=False, indent=2),
        }
        text = self.review_template
        for marker, value in replacements.items():
            text = text.replace(marker, value)
        if compact:
            text += """

【长度受限重试模式】
上一轮回答因长度或上下文限制失败。本轮必须压缩非 OCR 内容：
1. printed_text 仍完整保留页面可辨认原文，但不要重复同一句或逐字解释。
2. handwritten_text、seal_text、signature_text 只转写实际可见内容。
3. key_fields 合并同名同值字段，不输出重复字段。
4. rule_checks 只输出与当前材料直接相关且最重要的规则，最多 8 项；每项 evidence 最多 1 条。
5. document_classification.evidence 最多 2 条；registration_consistency 没有登记字段时必须为空数组。
6. reason、summary、limitations 每项使用短句，summary 不超过 120 个汉字。
7. 不得为了填满输出而复述业务规则，也不得省略 JSON 必需字段。
"""
        return text

    def build_merge(
        self,
        relative_path: str,
        chunk_results: list[dict[str, Any]],
        registered_fields: dict[str, Any],
    ) -> str:
        compact = [self._compact_for_merge(item) for item in chunk_results]
        replacements = {
            "{{RELATIVE_PATH}}": relative_path,
            "{{REGISTERED_FIELDS}}": json.dumps(registered_fields, ensure_ascii=False, indent=2),
            "{{BUSINESS_RULES}}": json.dumps(self.business_rules, ensure_ascii=False, indent=2),
            "{{CHUNK_RESULTS}}": json.dumps(compact, ensure_ascii=False, indent=2),
        }
        text = self.merge_template
        for marker, value in replacements.items():
            text = text.replace(marker, value)
        return text

    @staticmethod
    def _compact_for_merge(result: dict[str, Any]) -> dict[str, Any]:
        # 完整逐页 OCR 会由 pipeline 原样拼回最终结果；合并模型只需要分类、字段、
        # 视觉证据和规则结论。避免在长文档合并时重复携带数万字符 OCR。
        return {
            "document_classification": result.get("document_classification", {}),
            "page_numbers": [
                page.get("page")
                for page in result.get("ocr_pages", [])
                if isinstance(page, dict)
            ],
            "key_fields": result.get("key_fields", []),
            "visual_evidence": result.get("visual_evidence", {}),
            "rule_checks": result.get("rule_checks", []),
            "registration_consistency": result.get("registration_consistency", []),
            "decision": result.get("decision", {}),
        }
