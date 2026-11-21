from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from .prompt_router import PromptRoute


DECISION_POLICY_VERSION = "evidence-v11-seal-consistent-specific"

STUDENT_ID_POSITIVE_LABELS = (
    "学号",
    "学生编号",
    "student id",
    "student no",
    "student number",
)
STUDENT_ID_WRONG_LABELS = (
    "准考证号",
    "准考证",
    "考生号",
    "报名号",
    "身份证号",
    "身份证",
    "公民身份号码",
    "身份号码",
    "证件号",
    "手机号",
    "联系电话",
    "电话号码",
    "电话",
    "职工号",
    "工号",
    "档案号",
)


@dataclass(frozen=True)
class FieldSpec:
    label: str
    section: str
    database_key: str
    aliases: tuple[str, ...] = ()
    strict: bool = False
    importance: str = "advisory"
    comparison: str = "text"


FIELD_SPECS: dict[str, FieldSpec] = {
    "student_name": FieldSpec(
        "学生姓名", "student", "xm", strict=True, importance="core"
    ),
    "student_id": FieldSpec(
        "学号", "student", "xh", strict=True, importance="core"
    ),
    "identity_last4": FieldSpec(
        "证件号后四位", "student", "sfzh_masked", strict=True,
        importance="core",
    ),
    "school_name": FieldSpec("学校", "student", "xxmc", importance="supporting"),
    "college_name": FieldSpec("学院", "student", "xymc"),
    "major_name": FieldSpec("专业", "student", "zymc"),
    "graduation_year": FieldSpec("毕业年份", "student", "bynf"),
    "graduation_date": FieldSpec(
        "毕业日期", "student", "byrq", comparison="date"
    ),
    "organization_name": FieldSpec(
        "单位或院校名称", "employment", "dwmc",
        aliases=("employer_name",), importance="core", comparison="organization",
    ),
    "credit_code": FieldSpec(
        "统一社会信用代码", "employment", "zzjgdm", strict=True,
        importance="core",
    ),
    "organization_nature": FieldSpec(
        "单位性质", "employment", "dwxz", aliases=("employer_nature",)
    ),
    "industry": FieldSpec("行业", "employment", "hylb"),
    "work_or_study_location": FieldSpec(
        "工作或学习地点", "employment", "dwszd",
        aliases=("work_location",), comparison="location",
    ),
    "address": FieldSpec("单位地址", "employment", "dwdz", comparison="location"),
    "contact_name": FieldSpec("单位联系人", "employment", "dwlxr"),
    "contact_phone": FieldSpec("单位联系电话", "employment", "dwdh"),
    "signing_or_admission_date": FieldSpec(
        "签约或录取日期", "employment", "qysj",
        aliases=("date",), comparison="date",
    ),
    "report_date": FieldSpec(
        "报到证签发日期", "employment", "bdzqfrq", comparison="date"
    ),
}

ADVISORY_TEXT_KEYWORDS = (
    "工作地点", "学习地点", "所在地", "地址", "专业", "学院", "毕业年份",
    "毕业日期", "签约日期", "录取日期", "报到日期", "日期", "单位性质",
    "行业", "岗位", "职位", "薪酬", "工资", "联系人", "联系电话", "电话",
    "印章", "盖章", "签名", "签字",
)

NEGATIVE_EVIDENCE_PATTERNS = (
    r"未(?:体现|填写|提供|显示|载明|注明|识别|找到|发现|确认|签署|盖章)",
    r"(?:没有|无任何|缺少|缺失|为空|空白|不足以|无法)(?:证明|确认|判断|建立|核对|支持|体现|识别)?",
    r"不构成(?:录用|聘用|就业|接收|入学|入站|经营)",
    r"不能证明|无法证明|尚未(?:录用|聘用|就业|接收|入学|入站|签约)",
)

BENIGN_NEGATIVE_PHRASES = (
    "未发现明显异常", "未见明显异常", "未发现异常", "未见异常",
    "不存在明确冲突", "无明确冲突", "不存在明显冲突", "无明显冲突",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.lower())


def _last_four(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return compact[-4:] if len(compact) >= 4 else compact


def _compare(database_value: str, model_value: str, strict: bool) -> tuple[str, str]:
    left = _normalized(database_value)
    right = _normalized(model_value)
    if not database_value:
        return "extracted_only", "数据库未登记该字段，仅展示材料提取值"
    if not model_value:
        return "not_observed", "材料中未识别到可核对值，不作为不一致"
    if not left or not right:
        return "not_comparable", "字段内容不足，无法可靠比较"
    if left == right:
        return "match", "数据库值与材料识别值一致"
    if strict:
        return "mismatch", "数据库非空值与材料清晰识别值不一致"
    left_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", database_value))
    right_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", model_value))
    if left_has_chinese != right_has_chinese:
        return "not_comparable", "中英文名称无法仅凭字符串可靠比较，保留人工确认"
    if min(len(left), len(right)) >= 4 and (left in right or right in left):
        return "compatible", "名称存在简称、全称或包含关系，可合理对应"
    return "mismatch", "数据库非空值与材料清晰识别值无合理对应"


def _organization_core(value: str) -> str:
    without_region = re.sub(r"[（(][^）)]{1,12}[）)]", "", value)
    normalized = _normalized(without_region)
    for suffix in (
        "股份有限责任公司", "股份有限公司", "集团有限公司", "有限责任公司",
        "集团公司", "有限公司", "分公司", "公司",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _compare_organization(database_value: str, model_value: str) -> tuple[str, str]:
    basic_status, basic_basis = _compare(database_value, model_value, strict=False)
    if basic_status != "mismatch":
        return basic_status, basic_basis
    left = _organization_core(database_value)
    right = _organization_core(model_value)
    if not left or not right:
        return "not_comparable", "单位名称信息不足，保留人工确认"
    if min(len(left), len(right)) >= 4 and (left in right or right in left):
        return "compatible", "忽略行政区括号和企业法律形式后，单位核心名称可对应"
    ratio = SequenceMatcher(None, left, right).ratio()
    if min(len(left), len(right)) >= 6 and ratio >= 0.88:
        return "compatible", "单位核心名称高度相似，差异可解释为简称或轻微 OCR 误差"
    return "mismatch", "单位核心名称无法合理对应"


def _city_token(value: str) -> str:
    normalized = _normalized(value)
    for municipality in ("北京", "上海", "天津", "重庆", "香港", "澳门"):
        if municipality in normalized:
            return municipality
    tail = re.split(r"省|自治区|特别行政区", normalized)[-1]
    match = re.search(r"([\u4e00-\u9fff]{2,8}?)(?:市|自治州|地区|盟)", tail)
    return match.group(1) if match else ""


def _compare_location(database_value: str, model_value: str) -> tuple[str, str]:
    basic_status, basic_basis = _compare(database_value, model_value, strict=False)
    if basic_status != "mismatch":
        return basic_status, basic_basis
    left_city = _city_token(database_value)
    right_city = _city_token(model_value)
    left_text = _normalized(database_value)
    right_text = _normalized(model_value)
    if left_city and (
        left_city == right_city or (not right_city and left_city in right_text)
    ):
        return "compatible", "工作或学习地点位于同一地级市，区县和详细地址差异不影响一致性"
    if right_city and not left_city and right_city in left_text:
        return "compatible", "工作或学习地点位于同一地级市，区县和详细地址差异不影响一致性"
    return "mismatch", "无法确认两个地点位于同一地级市"


def _parse_date(value: str) -> tuple[date, str] | None:
    separated = re.search(
        r"((?:19|20)\d{2})\s*[年./-]\s*(\d{1,2})"
        r"(?:\s*[月./-]\s*(\d{1,2}))?",
        value,
    )
    if separated:
        year = int(separated.group(1))
        month = int(separated.group(2))
        day_text = separated.group(3)
        parts = (year, month, int(day_text) if day_text else 1)
        precision = "day" if day_text else "month"
        try:
            return date(*parts), precision
        except ValueError:
            return None
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 8:
        parts = (int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        precision = "day"
    elif len(digits) >= 6:
        parts = (int(digits[:4]), int(digits[4:6]), 1)
        precision = "month"
    elif len(digits) == 4:
        parts = (int(digits), 1, 1)
        precision = "year"
    else:
        return None
    try:
        return date(*parts), precision
    except ValueError:
        return None


def _compare_date(database_value: str, model_value: str) -> tuple[str, str]:
    basic_status, basic_basis = _compare(database_value, model_value, strict=True)
    if basic_status != "mismatch":
        return basic_status, basic_basis
    left = _parse_date(database_value)
    right = _parse_date(model_value)
    if not left or not right:
        return "not_comparable", "日期格式不足，无法进行可靠时间比较"
    left_date, left_precision = left
    right_date, right_precision = right
    if "year" in {left_precision, right_precision}:
        if left_date.year == right_date.year:
            return "compatible", "日期年份一致，材料未提供足够的月日精度"
        return "mismatch", "日期年份不一致"
    if "month" in {left_precision, right_precision}:
        month_gap = abs(
            (left_date.year * 12 + left_date.month)
            - (right_date.year * 12 + right_date.month)
        )
        if month_gap <= 1:
            return "compatible", "日期相差不超过一个月，处于业务容错范围"
    elif abs((left_date - right_date).days) <= 31:
        return "compatible", "日期相差不超过 31 天，处于业务容错范围"
    return "mismatch", "日期差异超过一个月容错范围"


def _compare_for_spec(
    spec: FieldSpec,
    database_value: str,
    model_value: str,
) -> tuple[str, str]:
    if spec.comparison == "organization":
        return _compare_organization(database_value, model_value)
    if spec.comparison == "location":
        return _compare_location(database_value, model_value)
    if spec.comparison == "date":
        return _compare_date(database_value, model_value)
    return _compare(database_value, model_value, spec.strict)


def _model_field(fields: dict[str, Any], name: str, aliases: tuple[str, ...]) -> str:
    for key in (name, *aliases):
        value = _text(fields.get(key))
        if value:
            return value
    return ""


def _field_evidence(result: dict[str, Any], field: str, aliases: tuple[str, ...]) -> dict[str, Any]:
    accepted = {field, *aliases}
    for item in result.get("field_evidence", []):
        if isinstance(item, dict) and _text(item.get("field")) in accepted:
            return item
    return {}


def _student_id_semantic_guard(
    model_value: str,
    evidence: dict[str, Any],
) -> tuple[str, str] | None:
    """学号必须有明确字段标签；其他编号即使 OCR 正确也不参与学号比对。"""
    if not model_value:
        return None
    observed = _text(evidence.get("observed_text")).lower()
    explanation = _text(evidence.get("explanation")).lower()
    value_text = model_value.lower()

    observed_wrong = next(
        (label for label in STUDENT_ID_WRONG_LABELS if label in observed),
        "",
    )
    if observed_wrong:
        return (
            "not_comparable",
            f"原文标注为“{observed_wrong}”，该号码不作为学号参与一致性判定",
        )
    if any(label in observed for label in STUDENT_ID_POSITIVE_LABELS):
        return None

    contextual_wrong = next(
        (
            label
            for label in STUDENT_ID_WRONG_LABELS
            if label in explanation or label in value_text
        ),
        "",
    )
    if contextual_wrong:
        return (
            "not_comparable",
            f"字段依据指向“{contextual_wrong}”，该号码不作为学号参与一致性判定",
        )
    return (
        "not_comparable",
        "引用的原文没有明确“学号”标签，该号码仅保留展示，不参与学号一致性判定",
    )


def build_field_comparisons(
    result: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = result.get("extracted_fields", {})
    if not isinstance(fields, dict):
        fields = {}
    comparisons: list[dict[str, Any]] = []
    for field, spec in FIELD_SPECS.items():
        section = context.get(spec.section, {})
        database_value = _text(section.get(spec.database_key)) if isinstance(section, dict) else ""
        if field == "identity_last4":
            database_value = _last_four(database_value)
        model_value = _model_field(fields, field, spec.aliases)
        if field == "identity_last4":
            model_value = _last_four(model_value)
        if not database_value and not model_value:
            continue
        evidence = _field_evidence(result, field, spec.aliases)
        semantic_guard = (
            _student_id_semantic_guard(model_value, evidence)
            if field == "student_id"
            else None
        )
        status, basis = semantic_guard or _compare_for_spec(
            spec,
            database_value,
            model_value,
        )
        if status == "mismatch" and spec.importance != "core":
            status = "advisory_difference"
            basis += "；该字段属于辅助信息，差异仅作提示，不单独阻断通过"
        comparisons.append(
            {
                "field": field,
                "label": spec.label,
                "database_field": f"{spec.section}.{spec.database_key}",
                "database_value": database_value,
                "model_value": model_value,
                "status": status,
                "importance": spec.importance,
                "blocking": status == "mismatch" and spec.importance == "core",
                "page_number": int(evidence.get("page_number", 0) or 0),
                "observed_text": _text(evidence.get("observed_text")),
                "basis": basis,
            }
        )
    _apply_corroborating_matches(comparisons)
    return comparisons


def _apply_corroborating_matches(comparisons: list[dict[str, Any]]) -> None:
    """唯一标识已匹配时，降低同组单个 OCR 差异的阻断权重。"""
    by_field = {str(item["field"]): item for item in comparisons}

    def matched(field: str) -> bool:
        return by_field.get(field, {}).get("status") in {"match", "compatible"}

    def downgrade(field: str, basis: str) -> None:
        item = by_field.get(field)
        if not item or item.get("status") != "mismatch":
            return
        item["status"] = "advisory_difference"
        item["blocking"] = False
        item["basis"] = basis

    if matched("student_id") or matched("identity_last4"):
        downgrade(
            "student_name",
            "已有学号或证件号后四位准确对应，姓名差异可能来自 OCR，仅提示人工查看",
        )
    if matched("student_name") and matched("identity_last4"):
        downgrade(
            "student_id",
            "姓名和证件号后四位均一致，单个学号差异按 OCR 提示处理",
        )
    if matched("student_name") and matched("student_id"):
        downgrade(
            "identity_last4",
            "姓名和学号均一致，单个证件号后四位差异按 OCR 提示处理",
        )
    if matched("credit_code"):
        downgrade(
            "organization_name",
            "统一社会信用代码一致，单位名称差异按简称、法律形式或 OCR 提示处理",
        )


def _is_advisory_conflict_text(value: str) -> bool:
    text = _text(value)
    return any(keyword in text for keyword in ADVISORY_TEXT_KEYWORDS)


def _is_negative_business_evidence(value: str) -> bool:
    text = _text(value)
    for phrase in BENIGN_NEGATIVE_PHRASES:
        text = text.replace(phrase, "")
    return any(re.search(pattern, text) for pattern in NEGATIVE_EVIDENCE_PATTERNS)


def _normalize_content_evidence(result: dict[str, Any]) -> None:
    """把“支持一条否定描述”转换为“对核验要求不支持”，保留模型原标签。"""
    content = result.get("content_assessment")
    if isinstance(content, dict):
        for point in content.get("evidence_points", []):
            if not isinstance(point, dict):
                continue
            model_assessment = _text(point.get("model_assessment")) or _text(
                point.get("assessment")
            )
            point["model_assessment"] = model_assessment
            detail = " ".join(
                [_text(point.get("claim")), _text(point.get("observed_text"))]
            )
            if any(marker in detail for marker in ("印章", "公章", "盖章", "章位")):
                decision_assessment = "advisory"
            elif _is_advisory_conflict_text(detail) and (
                model_assessment == "contradicts"
                or _is_negative_business_evidence(detail)
            ):
                decision_assessment = "advisory"
            elif model_assessment == "supports" and _is_negative_business_evidence(detail):
                decision_assessment = "contradicts"
            else:
                decision_assessment = model_assessment or "unclear"
            point["decision_assessment"] = decision_assessment

    for rule in result.get("rule_results", []):
        if not isinstance(rule, dict):
            continue
        model_result = _text(rule.get("model_result")) or _text(rule.get("result"))
        rule["model_result"] = model_result
        detail = " ".join([_text(rule.get("rule")), _text(rule.get("evidence"))])
        if any(marker in detail for marker in ("印章", "公章", "盖章", "章位")) and (
            model_result in {"contradicted", "not_observed"}
            or _is_negative_business_evidence(detail)
        ):
            # Seal absence/completeness is decided only by seal_assessment.
            decision_result = "advisory"
        elif rule.get("importance") != "essential" and (
            model_result == "contradicted" or _is_negative_business_evidence(detail)
        ):
            decision_result = "advisory"
        elif model_result == "supported" and _is_negative_business_evidence(detail):
            decision_result = "contradicted"
        else:
            decision_result = model_result or "not_observed"
        rule["decision_result"] = decision_result


def _analyze_seals(
    result: dict[str, Any],
    route: PromptRoute,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Determine seal presence/completeness while retaining evidence-specific wording."""
    risks = [
        copy.deepcopy(item)
        for item in result.get("risk_flags", [])
        if isinstance(item, dict)
        and _text(item.get("type")) != "seal"
        and not (
            _text(item.get("type")) in {"image_quality", "other"}
            and any(
                marker in " ".join(
                    [_text(item.get("description")), _text(item.get("evidence"))]
                )
                for marker in ("印章", "公章", "盖章", "章面")
            )
        )
    ]
    checks: list[dict[str, Any]] = []
    seal_regions = [
        region
        for region in result.get("visual_regions", [])
        if isinstance(region, dict) and region.get("region_type") == "seal"
    ]
    seal_sources: list[tuple[dict[str, Any] | None, str]] = [
        (region, _text(region.get("visible_text"))) for region in seal_regions
    ]
    aggregate_seal_text = _text(
        (result.get("extracted_fields") or {}).get("seal_text")
        if isinstance(result.get("extracted_fields"), dict)
        else ""
    )
    if not seal_sources and aggregate_seal_text:
        seal_sources.append((None, aggregate_seal_text))

    generic_observations = {
        "检测到印章", "已检测到印章", "不检查印章内容和外观",
    }
    generic_bases = {
        "仅记录印章存在", "仅记录印章是否存在",
        "仅记录印章存在，不检查章面内容或外观",
    }

    def region_location(region: dict[str, Any], page_number: int) -> str:
        box = region.get("bbox") or {}
        center_x = (int(box.get("x1", 0)) + int(box.get("x2", 0))) / 2
        center_y = (int(box.get("y1", 0)) + int(box.get("y2", 0))) / 2
        horizontal = "左侧" if center_x < 340 else "右侧" if center_x > 660 else "中部"
        vertical = "上方" if center_y < 340 else "下方" if center_y > 660 else "中部"
        return f"第 {page_number or 1} 页{vertical}{horizontal}"

    def is_specific_statement(value: str) -> bool:
        return any(
            marker in value
            for marker in (
                "第", "页", "栏", "落款", "签署", "甲方", "乙方",
                "单位", "学校", "学院", "院校", "审核", "盖章处",
            )
        )

    for index, (region, visible_text) in enumerate(seal_sources, start=1):
        region_id = _text(region.get("region_id")) if region else f"seal-text-{index}"
        page_number = int(region.get("page_number", 0) or 0) if region else 0
        if region:
            # A region labelled as a seal already represents a positive presence
            # observation.  Appearance/content judgements are deliberately ignored,
            # but the model's page-specific observation is retained for presentation.
            region["model_assessment"] = _text(region.get("assessment"))
            region["assessment"] = "no_obvious_anomaly"
            observation = _text(region.get("observation"))
            basis = _text(region.get("basis"))
            location = region_location(region, page_number)
            if (
                not observation
                or observation in generic_observations
                or not is_specific_statement(observation)
            ):
                observation = f"{location}定位到一枚实际印章"
            if not basis or basis in generic_bases or not is_specific_statement(basis):
                basis = f"{location}可确认存在盖章痕迹"
            region["observation"] = observation
            region["basis"] = basis
            visible_text = _text(region.get("visible_text"))
        findings = list(dict.fromkeys(
            value for value in (
                _text(region.get("observation")) if region else "",
                _text(region.get("basis")) if region else "",
            ) if value
        )) or [f"第 {page_number or 1} 页检测到印章"]
        checks.append(
            {
                "region_id": region_id,
                "page_number": page_number,
                "status": "present",
                "visible_text": visible_text,
                "seal_codes": [],
                "entity_status": "not_checked",
                "expected_entity": "",
                "findings": findings,
            }
        )

    extracted_fields = result.get("extracted_fields")
    if isinstance(extracted_fields, dict):
        extracted_fields["seal_text"] = ""

    result["risk_flags"] = risks
    model_assessment = result.get("seal_assessment")
    if not isinstance(model_assessment, dict):
        model_assessment = {}
    requirement = route.seal_requirement or {}
    mode = _text(requirement.get("mode")) or "not_required"
    configured_required = [
        _text(value) for value in requirement.get("required_seals", [])
        if _text(value)
    ]
    model_required = [
        _text(value) for value in model_assessment.get("required_seals", [])
        if _text(value)
    ]
    required_seals = model_required or configured_required
    detected_count = len(seal_sources)
    signature_present = any(
        isinstance(region, dict) and region.get("region_type") == "signature"
        for region in result.get("visual_regions", [])
    ) or bool(_text((result.get("extracted_fields") or {}).get("signature")))

    if mode == "required":
        required_count = max(1, len(required_seals))
        completeness = "complete" if detected_count >= required_count else "incomplete"
    elif mode == "seal_or_signature":
        completeness = "complete" if detected_count else (
            "not_required" if signature_present else "incomplete"
        )
    elif mode == "conditional":
        model_completeness = _text(model_assessment.get("completeness"))
        model_missing = [
            _text(value)
            for value in model_assessment.get("missing_required_seals", [])
            if _text(value)
        ]
        # Conditional materials only fail this dimension when the page visibly
        # contains an unfilled required seal slot.  A generic route hint is not
        # itself proof that this concrete material must be stamped.
        if model_required and detected_count < len(model_required):
            completeness = "incomplete"
            required_seals = model_required
        elif model_completeness == "incomplete" or model_missing:
            completeness = "incomplete"
            required_seals = model_required or model_missing or configured_required
        elif model_completeness == "unclear":
            completeness = "unclear"
            required_seals = model_required
        elif detected_count:
            completeness = "complete"
            required_seals = model_required or configured_required
        else:
            completeness = "not_required"
            required_seals = []
    else:
        completeness = "not_required"

    missing_required = []
    if completeness == "incomplete":
        missing_required = [
            _text(value)
            for value in model_assessment.get("missing_required_seals", [])
            if _text(value)
        ] or required_seals[detected_count:] or ["材料明确要求的印章"]
    presence = (
        "present" if detected_count
        else "unclear" if completeness == "unclear"
        else "absent"
    )
    if completeness == "complete":
        basis = f"检测到 {detected_count} 枚印章，业务所需章位齐全"
    elif completeness == "incomplete":
        basis = "未检测到全部必需印章：" + "、".join(missing_required)
    elif completeness == "unclear":
        basis = "当前页面不足以判断明确要求的章位是否已盖齐"
    elif detected_count:
        basis = f"检测到 {detected_count} 枚印章；当前材料不设置强制章位"
    else:
        basis = "当前材料不要求以印章作为必要条件"
    assessment = {
        "presence": presence,
        "detected_count": detected_count,
        "completeness": completeness,
        "required_seals": required_seals,
        "missing_required_seals": missing_required,
        "basis": basis,
    }
    return checks, assessment


def _sanitize_regions(result: dict[str, Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, raw in enumerate(result.get("visual_regions", []), start=1):
        if not isinstance(raw, dict):
            continue
        box = raw.get("bbox")
        if not isinstance(box, dict):
            continue
        try:
            x1 = max(0, min(1000, int(box.get("x1", 0))))
            y1 = max(0, min(1000, int(box.get("y1", 0))))
            x2 = max(0, min(1000, int(box.get("x2", 0))))
            y2 = max(0, min(1000, int(box.get("y2", 0))))
            page_number = max(1, int(raw.get("page_number", 1)))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        region = copy.deepcopy(raw)
        region["region_id"] = _text(region.get("region_id")) or f"region-{index}"
        region["page_number"] = page_number
        region["bbox"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        regions.append(region)
    return regions


def finalize_material_result(
    raw_result: dict[str, Any],
    context: dict[str, Any],
    route: PromptRoute,
) -> dict[str, Any]:
    """把模型负责的证据提取与服务器负责的一致性决策分离。"""
    result = copy.deepcopy(raw_result)
    model_status = _text(result.get("status")) or "review"
    result["visual_regions"] = _sanitize_regions(result)
    result["seal_checks"], result["seal_assessment"] = _analyze_seals(
        result, route
    )
    _normalize_content_evidence(result)
    comparisons = build_field_comparisons(result, context)
    result["field_comparisons"] = comparisons

    destination = result.get("destination_assessment", {})
    content = result.get("content_assessment", {})
    destination_verdict = _text(destination.get("verdict")) if isinstance(destination, dict) else ""
    content_verdict = (
        _text(content.get("model_verdict")) or _text(content.get("verdict"))
        if isinstance(content, dict) else ""
    )
    if isinstance(destination, dict):
        registered_destination = _text(context.get("student", {}).get("byqxlb"))
        destination["registered_destination"] = registered_destination
        if not _text(destination.get("recognized_destination")):
            destination["recognized_destination"] = _text(result.get("material_type"))
        recognized = _normalized(_text(destination.get("recognized_destination")))
        route_title = _normalized(route.title)
        route_compatible = bool(
            recognized
            and route_title
            and (recognized in route_title or route_title in recognized)
        )
        if (
            destination_verdict == "uncertain"
            and content_verdict == "supported"
            and route_compatible
        ):
            destination["verdict"] = "consistent"
            destination["summary"] = (
                _text(destination.get("summary"))
                or "材料识别类型与当前毕业去向规则相容"
            )
            destination_verdict = "consistent"
        destination_detail = " ".join(
            [
                _text(destination.get("summary")),
                *[
                    " ".join(
                        [
                            _text(item.get("observed_text")),
                            _text(item.get("explanation")),
                        ]
                    )
                    for item in destination.get("evidence_points", [])
                    if isinstance(item, dict)
                ],
            ]
        )
        if (
            destination_verdict == "inconsistent"
            and route_compatible
            and _is_advisory_conflict_text(destination_detail)
        ):
            destination["model_verdict"] = "inconsistent"
            destination["verdict"] = "consistent"
            destination["summary"] = (
                "材料类型与登记毕业去向相容；原判定差异仅涉及辅助字段"
            )
            destination_verdict = "consistent"

    explicit_mismatches = [
        item
        for item in comparisons
        if item["status"] == "mismatch" and item.get("blocking", True)
    ]
    advisory_differences = [
        item for item in comparisons if item["status"] == "advisory_difference"
    ]
    all_essential_conflicts = [
        item
        for item in result.get("rule_results", [])
        if isinstance(item, dict)
        and item.get("importance") == "essential"
        and (_text(item.get("decision_result")) or _text(item.get("result"))) == "contradicted"
    ]
    advisory_rule_conflicts = [
        item
        for item in all_essential_conflicts
        if _is_advisory_conflict_text(
            _text(item.get("evidence")) or _text(item.get("rule"))
        )
    ]
    essential_conflicts = [
        item for item in all_essential_conflicts if item not in advisory_rule_conflicts
    ]
    visual_suspicion = [
        item
        for item in result["visual_regions"]
        if item.get("region_type") != "seal"
        and item.get("assessment") == "suspicious"
    ]
    high_risks = [
        item
        for item in result.get("risk_flags", [])
        if isinstance(item, dict) and item.get("severity") == "high"
    ]
    supported_rules = [
        item
        for item in result.get("rule_results", [])
        if isinstance(item, dict)
        and (_text(item.get("decision_result")) or _text(item.get("result"))) == "supported"
        and not any(
            marker in " ".join([_text(item.get("rule")), _text(item.get("evidence"))])
            for marker in ("印章", "公章", "盖章", "章位")
        )
    ]
    content_points = content.get("evidence_points", []) if isinstance(content, dict) else []
    supporting_points = [
        item
        for item in content_points
        if isinstance(item, dict)
        and (_text(item.get("decision_assessment")) or _text(item.get("assessment"))) == "supports"
    ]
    all_contradicting_points = [
        item
        for item in content_points
        if isinstance(item, dict)
        and (_text(item.get("decision_assessment")) or _text(item.get("assessment"))) == "contradicts"
    ]
    advisory_contradicting_points = [
        item
        for item in all_contradicting_points
        if _is_advisory_conflict_text(
            " ".join(
                [
                    _text(item.get("claim")),
                    _text(item.get("observed_text")),
                ]
            )
        )
    ]
    contradicting_points = [
        item
        for item in all_contradicting_points
        if item not in advisory_contradicting_points
    ]
    positive = bool(supporting_points or supported_rules)
    content_supported_by_evidence = bool(
        positive and not contradicting_points and not essential_conflicts
    )
    if isinstance(content, dict):
        content["model_verdict"] = content_verdict
        if content_supported_by_evidence:
            content["verdict"] = "supported"
        elif contradicting_points or essential_conflicts:
            content["verdict"] = "questionable"
        else:
            content["verdict"] = "insufficient"

    blockers: list[str] = []
    if destination_verdict == "inconsistent":
        blockers.append("材料类型与登记毕业去向存在明确不一致")
    if not content_supported_by_evidence:
        blockers.append("材料中的关键结果性事实不足或存在疑问")
    if explicit_mismatches:
        labels = "、".join(item["label"] for item in explicit_mismatches[:4])
        blockers.append(f"数据库非空字段存在明确差异：{labels}")
    if essential_conflicts:
        blockers.append("材料证据与核心类别规则存在明确冲突")
    seal_assessment = result.get("seal_assessment") or {}
    if seal_assessment.get("completeness") == "incomplete":
        missing_seals = "、".join(
            _text(value)
            for value in seal_assessment.get("missing_required_seals", [])
            if _text(value)
        )
        blockers.append(
            "业务必需印章不齐全"
            + (f"：{missing_seals}" if missing_seals else "")
        )
    elif seal_assessment.get("completeness") == "unclear":
        blockers.append("当前材料无法确认业务必需印章是否齐全")
    if visual_suspicion or high_risks:
        blockers.append("材料存在有具体依据的视觉或业务高风险项")

    can_pass = (
        content_supported_by_evidence
        and destination_verdict == "consistent"
        and positive
        and not blockers
    )
    if can_pass:
        status = "pass"
    elif model_status == "unsupported":
        status = "unsupported"
    else:
        status = "review"

    generated_mismatches = [
        {
            "field": item["label"],
            "registered_value": item["database_value"],
            "evidence_value": item["model_value"],
            "description": item["basis"],
        }
        for item in explicit_mismatches
    ]
    result["mismatches"] = generated_mismatches

    decision_basis = [
        _text(item)
        for item in result.get("decision_basis", [])
        if _text(item)
        and not _text(item).startswith(("字段核对：", "存疑依据：", "提示差异："))
    ]
    if isinstance(content, dict) and _text(content.get("summary")):
        decision_basis.append("内容依据：" + _text(content.get("summary")))
    if isinstance(destination, dict) and _text(destination.get("summary")):
        decision_basis.append("去向依据：" + _text(destination.get("summary")))
    matches = [item for item in comparisons if item["status"] in {"match", "compatible"}]
    if matches:
        decision_basis.append(
            "字段核对：" + "、".join(item["label"] for item in matches[:6]) + "一致或可合理对应"
        )
    for blocker in blockers:
        decision_basis.append("存疑依据：" + blocker)
    advisory_messages = [
        f"{item['label']}存在差异，但属于辅助字段，不单独影响结论"
        for item in advisory_differences[:6]
    ]
    advisory_messages.extend(
        _text(item.get("evidence")) or _text(item.get("rule"))
        for item in advisory_rule_conflicts[:3]
    )
    advisory_messages.extend(
        _text(item.get("claim"))
        for item in advisory_contradicting_points[:3]
        if _text(item.get("claim"))
    )
    for message in advisory_messages:
        decision_basis.append("提示差异：" + message)

    result["model_suggested_status"] = model_status
    result["status"] = status
    result["decision_policy_version"] = DECISION_POLICY_VERSION
    result["decision_basis"] = list(dict.fromkeys(decision_basis))
    result["review_reasons"] = blockers
    if status == "pass":
        result["summary"] = (
            f"材料关键事实有可见证据支持，与登记毕业去向“"
            f"{_text(context.get('student', {}).get('byqxlb'))}”相容，"
            "未发现核心身份或核心单位字段冲突。"
        )
        result["manual_review_questions"] = []
    else:
        result["summary"] = "；".join(blockers) or _text(result.get("summary")) or "关键证据不足，建议人工复核"
        if blockers:
            result["manual_review_questions"] = blockers
    result["route_title"] = route.title
    return result


def empty_review_result(summary: str, unsupported: bool = False) -> dict[str, Any]:
    status = "unsupported" if unsupported else "review"
    return {
        "status": status,
        "confidence": 1.0,
        "material_type": "",
        "summary": summary,
        "visual_regions": [],
        "seal_assessment": {
            "presence": "unclear", "detected_count": 0,
            "completeness": "unclear", "required_seals": [],
            "missing_required_seals": [], "basis": summary,
        },
        "extracted_fields": {
            "student_name": "", "student_id": "", "identity_last4": "",
            "organization_name": "", "credit_code": "", "organization_nature": "",
            "industry": "", "school_name": "", "college_name": "", "major_name": "",
            "graduation_year": "", "graduation_date": "", "position_or_program": "",
            "work_or_study_location": "", "address": "", "salary": "",
            "employment_period": "", "signing_or_admission_date": "", "start_date": "",
            "contact_name": "", "contact_phone": "", "report_date": "",
            "project_or_business_name": "", "business_owner": "", "business_status": "",
            "seal_text": "", "signature": "", "keywords": [], "additional_fields": [],
        },
        "field_evidence": [],
        "field_comparisons": [],
        "content_assessment": {
            "verdict": "insufficient", "summary": summary, "evidence_points": []
        },
        "destination_assessment": {
            "registered_destination": "", "recognized_destination": "",
            "verdict": "uncertain", "summary": summary, "evidence_points": [],
        },
        "rule_results": [],
        "mismatches": [],
        "risk_flags": [],
        "decision_basis": [summary],
        "review_reasons": [summary] if not unsupported else [],
        "manual_review_questions": [summary] if not unsupported else [],
        "decision_policy_version": DECISION_POLICY_VERSION,
        "model_suggested_status": status,
        "route_title": "",
    }
