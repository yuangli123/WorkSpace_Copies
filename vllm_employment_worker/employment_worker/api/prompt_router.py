from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRoute:
    route_id: str
    title: str
    destinations: tuple[str, ...]
    rules: tuple[str, ...]
    seal_requirement: dict[str, Any]


class PromptRouter:
    def __init__(self, routes_file: Path, system_prompt_file: Path) -> None:
        value = json.loads(routes_file.read_text(encoding="utf-8"))
        self.version = str(value.get("version", "unknown"))
        self.decision_policy = {
            str(key): str(item)
            for key, item in value.get("decision_policy", {}).items()
        }
        self.system_prompt = system_prompt_file.read_text(encoding="utf-8")
        self._routes: dict[str, PromptRoute] = {}
        for raw in value.get("routes", []):
            route = PromptRoute(
                route_id=str(raw["route_id"]),
                title=str(raw["title"]),
                destinations=tuple(str(item) for item in raw["destinations"]),
                rules=tuple(str(item) for item in raw["rules"]),
                seal_requirement={
                    str(key): item
                    for key, item in raw.get(
                        "seal_requirement",
                        {"mode": "not_required", "required_seals": []},
                    ).items()
                },
            )
            for destination in route.destinations:
                if destination in self._routes:
                    raise ValueError(f"毕业去向重复配置：{destination}")
                self._routes[destination] = route
        self.unsupported = {
            str(item) for item in value.get("unsupported_destinations", [])
        }
        self._fallback_route = PromptRoute(
            route_id="generic_evidence_review",
            title="通用就业去向佐证材料",
            destinations=(),
            rules=(
                "提取材料中能够确认的学生身份、单位或院校以及结果性事实",
                "根据材料实际内容判断其能够证明的就业、升学、创业或其他去向",
                "未登记或未明确毕业去向时仍应完成核验，不得直接标记为不支持",
                "印章仅检查是否存在以及材料明确要求的章位是否齐全",
            ),
            seal_requirement={
                "mode": "conditional",
                "required_seals": ["材料画面中明确标注的必需盖章栏位"],
            },
        )

    def resolve(self, destination: str) -> PromptRoute:
        return self._routes.get(destination.strip(), self._fallback_route)

    def is_explicitly_unsupported(self, destination: str) -> bool:
        # 所有毕业生均进入核验；未配置去向使用通用材料路由。
        return False

    def build_material_prompt(
        self,
        route: PromptRoute,
        student_context: dict[str, Any],
        page_numbers: list[int],
    ) -> str:
        rules = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(route.rules, start=1)
        )
        context = json.dumps(student_context, ensure_ascii=False, indent=2)
        policy = json.dumps(self.decision_policy, ensure_ascii=False, indent=2)
        seal_requirement = json.dumps(
            route.seal_requirement, ensure_ascii=False, indent=2
        )
        return f"""当前核查模板：{route.title}
模板版本：{self.version}
本次图像对应材料实际页码：{page_numbers}

学生、就业登记和材料上下文：
{context}

核查规则：
{rules}

综合判定政策：
{policy}

印章齐全性要求：
{seal_requirement}

请完成：
1. 首先定位印章、签名、手写和有判定价值的图像区域，输出 visual_regions。bbox 使用当前原始页的 0—1000 相对坐标，x2>x1、y2>y1；不要把整页正文框成图像区域。每个实际可见的印章单独输出一个 seal 区域；印刷的“盖章处/公章”文字、空白圆框或五角星图标不算实际印章。对 seal 区域只做简短、具体的业务观察：observation 必须写清当前页的实际位置或对应栏位，例如“第1页甲方盖章栏可见一枚印章”；basis 必须说明该区域如何支持“已盖章”或为何仍无法确认。不得输出“仅记录印章存在”“不检查印章内容和外观”等提示词说明。visible_text 只填写确实可辨的少量文字，看不清就留空；文字内容不作为通过条件。实际印章区域使用 no_obvious_anomaly，无法区分实际印迹与印刷占位符时使用 unclear。不得检查或评价编号、主体对应关系、颜色、清晰度、形态、透视、分辨率、拼接、真伪或样章属性，也不得因此生成 seal 风险。根据“印章齐全性要求”和画面中明确可见的必需盖章栏位输出 seal_assessment：必需章位均有实际印章为 complete，至少一个明确必需章位空缺为 incomplete，不要求盖章或允许官方来源/签字替代时为 not_required，页面不足以判断时为 unclear。不得仅根据模板名称或通用规则臆测缺章。
2. 提取当前类别能够从材料中判断出的字段。表格或名单含多名人员时，先用数据库姓名、学号或证件后四位定位目标人员，只提取目标人员同一行的信息，不得默认选择第一行或相邻行。字段值必须能在 field_evidence 中找到页码、字段标签和原文依据；没有看到的字段填空字符串。student_id 只接收原文明确标注为“学号/学生编号/Student ID/Student No.”的号码；准考证号、考生号、报名号、身份证号、证件号、手机号、工号、档案号等必须按原标签放入 additional_fields，不得写入 student_id。
3. content_assessment 表示内容完整性，只列有证据的内容判断，区分结果性材料与过程性材料。evidence_points.assessment 必须表示该证据对核验要求的影响：核心身份、目标单位或录用/聘用结果缺失时使用 contradicts；不得因为“空白原文支持了材料未填写这句描述”而使用 supports。日期、薪酬和地址等辅助项缺失使用 unclear。印章是否存在、是否齐全只写入 seal_assessment，不得混入内容完整性风险。
4. destination_assessment 比较材料实际证明的类型与数据库“毕业去向”；业务含义相容即 consistent，不要求文字完全相同。
5. rule_results 覆盖上述规则：就业人身份、目标单位/院校、结果性事实和去向相容性标 essential；工作地点、日期、学校、学院、专业、单位性质、行业、地址、联系人、电话、岗位和薪酬等辅助字段标 supporting/advisory。未出现写 not_observed，不得把辅助差异标成 essential contradicted。
6. mismatch 只记录核心字段的明确冲突。地点在同一地级市、日期相差不超过 31 天、单位法律形式/合理简称/行政区括号不同、岗位专业与所学专业不同，以及其它辅助字段差异不得列入 mismatch。
7. decision_basis 用简短语句罗列最终结论直接依据；风险项必须给出具体可见证据，不能使用笼统的“可能伪造”。
8. 当前仅核查这组页面；多段材料会由程序再次汇总。按指定 JSON Schema 输出。"""

    def build_merge_prompt(
        self,
        route: PromptRoute,
        student_context: dict[str, Any],
        chunk_results: list[dict[str, Any]],
    ) -> str:
        return f"""请合并同一份材料的分段核查结果，并输出最终材料级结论。
模板：{route.title}
模板版本：{self.version}
学生上下文：
{json.dumps(student_context, ensure_ascii=False, indent=2)}
分段结果：
{json.dumps(chunk_results, ensure_ascii=False, indent=2)}

综合判定政策：
{json.dumps(self.decision_policy, ensure_ascii=False, indent=2)}

要求：
1. 不得创造分段结果中不存在的事实；保留可追溯的页码、原文和 visual_regions。每个实际可见印章保持独立 seal 区域，并保留分段结果中关于具体页码、栏位或位置的简短观察；不得把观察改写为“仅记录印章存在”等通用套话。印章文字不作为判定条件，不评价编号、主体对应关系、外观、拼接或真伪；实际印章区域使用 no_obvious_anomaly，无法确认是否为实际印迹时使用 unclear。
2. 页面字段可以互补。只有核心身份或核心单位字段的明确冲突才输出 mismatch。地点同市、日期在一个月容错内、单位合理简称和辅助字段差异不得升级为 mismatch。合并时不得把准考证号、考生号、报名号、身份证号、手机号或其他编号改写为 student_id；学号没有明确标签时保持为空。
3. 汇总 content_assessment、destination_assessment 和各类别规则的直接证据。统一按“是否满足核验要求”设置 assessment/result；“材料未体现身份/单位/结果”等否定描述必须是 contradicts/contradicted，不能因为原文支持该否定描述而标为 supports/supported。
4. 可选字段未出现、数据库为空、单页模糊但其他页面已有关键证据，均不得单独阻止 pass。
5. 关键身份与结果性事实有证据、去向相容且无核心身份/单位冲突或高风险时应输出 pass；辅助字段差异保留展示但不能单独阻止 pass。合并所有页面后重新形成一个 seal_assessment：只判断印章存在性及必需章位是否齐全。除明确必需章位缺失外，任何印章内容或外观特征都不得阻止 pass，也不得生成 seal 风险。
6. 只输出指定 JSON Schema。"""
