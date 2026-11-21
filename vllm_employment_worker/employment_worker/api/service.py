from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from typing import Any

from ..client import ModelOutputError
from .checker import MaterialChecker
from .database import JobRepository, SourceRepository
from .decision import (
    DECISION_POLICY_VERSION,
    empty_review_result,
    finalize_material_result,
)
from .models import CreateJobRequest
from .prompt_router import PromptRouter


class SelectionError(ValueError):
    pass


class EmploymentCheckService:
    def __init__(
        self,
        source: SourceRepository,
        jobs: JobRepository,
        router: PromptRouter,
        checker: MaterialChecker,
        max_parallel_items: int,
    ) -> None:
        self.source = source
        self.jobs = jobs
        self.router = router
        self.checker = checker
        self._item_gate = asyncio.Semaphore(max_parallel_items)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._job_tasks: dict[str, asyncio.Task[Any]] = {}
        self._verification_status_cache: tuple[float, dict[int, str]] = (0.0, {})

    async def start(self) -> dict[str, Any]:
        models = await self.checker.start()
        recovered = await asyncio.to_thread(self.jobs.recover_incomplete)
        for job_id in recovered:
            self._schedule(job_id)
        return {"models": models, "recovered_jobs": recovered}

    async def close(self) -> None:
        # 服务退出时不无限等待长耗时推理；未完成项会在下次启动时恢复。
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.checker.close()

    async def create_job(self, request: CreateJobRequest) -> dict[str, Any]:
        if request.options.model != self.checker.runtime.model:
            raise SelectionError(
                f"请求模型 {request.options.model!r} 与当前 vLLM 配置模型 "
                f"{self.checker.runtime.model!r} 不一致"
            )
        graduate_ids = request.graduate_ids
        evidence_ids = request.evidence_file_ids
        existing_graduates, existing_evidences = await asyncio.gather(
            asyncio.to_thread(self.source.existing_graduate_ids, graduate_ids),
            asyncio.to_thread(self.source.existing_evidence_ids, evidence_ids),
        )
        missing_graduates = sorted(set(graduate_ids) - existing_graduates)
        missing_evidences = sorted(set(evidence_ids) - existing_evidences)
        if missing_graduates or missing_evidences:
            raise SelectionError(
                f"不存在的ID：graduate_ids={missing_graduates}，"
                f"evidence_file_ids={missing_evidences}"
            )

        contexts = await asyncio.to_thread(
            self.source.selected_contexts,
            graduate_ids,
            evidence_ids,
        )
        if request.options.include_low_confidence_matches:
            candidate_contexts = await asyncio.to_thread(
                self.source.low_confidence_contexts,
                graduate_ids,
                evidence_ids,
            )
            confirmed_pairs = {
                (item["graduate_id"], item["evidence"]["evidence_file_id"])
                for item in contexts
            }
            contexts.extend(
                item
                for item in candidate_contexts
                if (item["graduate_id"], item["evidence"]["evidence_file_id"])
                not in confirmed_pairs
            )
        items = [self._item_for_context(context, request) for context in contexts]

        if graduate_ids and not evidence_ids:
            matched_graduates = {int(item["graduate_id"]) for item in items}
            missing_links = sorted(set(graduate_ids) - matched_graduates)
            graduate_contexts = await asyncio.to_thread(self.source.graduates, missing_links)
            items.extend(self._no_evidence_item(context) for context in graduate_contexts)

        if evidence_ids and not graduate_ids:
            matched_evidence = {int(item["evidence_file_id"]) for item in items}
            missing_links = sorted(set(evidence_ids) - matched_evidence)
            evidence_contexts = await asyncio.to_thread(self.source.evidences, missing_links)
            items.extend(self._unmatched_evidence_item(context) for context in evidence_contexts)

        request_payload = request.model_dump()
        job_id = await asyncio.to_thread(
            self.jobs.create_job,
            request_payload,
            request.options.model,
            self.router.version,
            items,
        )
        self._schedule(job_id)
        return self.jobs.job_status(job_id) or {}

    def status(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.job_status(job_id)

    async def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        value = await asyncio.to_thread(self.jobs.cancel_job, job_id)
        task = self._job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        return value

    def latest_verification_statuses(self) -> dict[int, str]:
        cached_at, cached = self._verification_status_cache
        if time.monotonic() - cached_at < 5:
            return dict(cached)
        summary = self.checked_student_summary(1, 1_000_000)
        statuses = {
            int(item["graduate_id"]): str(item.get("status") or "review")
            for item in summary.get("items", [])
        }
        self._verification_status_cache = (time.monotonic(), statuses)
        return dict(statuses)

    def results(self, job_id: str) -> dict[str, Any] | None:
        items = self.jobs.job_results(job_id)
        status = self.jobs.job_status(job_id)
        if items is None or status is None:
            return None
        items = [self._apply_current_decision_policy(item) for item in items]
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        unmatched = []
        for item in items:
            if item["graduate_id"] is None:
                unmatched.append(self._evidence_response(item))
            else:
                grouped[int(item["graduate_id"])].append(item)
        students = []
        for graduate_id, graduate_items in sorted(grouped.items()):
            context = graduate_items[0]["context"]
            student = context.get("student", {})
            student_result = self._aggregate_student(graduate_items)
            primary_ids = {
                int(value)
                for value in student_result.get("primary_evidence_file_ids", [])
                if value is not None
            }
            evidence_results = []
            for item in graduate_items:
                evidence_result = self._evidence_response(item)
                evidence_id = item.get("evidence_file_id")
                evidence_result["material_role"] = (
                    "primary"
                    if evidence_id is not None and int(evidence_id) in primary_ids
                    else "secondary"
                )
                evidence_result["dimension_outcomes"] = self._material_dimensions(
                    item.get("result") or {}
                )
                evidence_results.append(evidence_result)
            students.append(
                {
                    "graduate_id": graduate_id,
                    "xh": student.get("xh", ""),
                    "xm": student.get("xm", ""),
                    "xxmc": student.get("xxmc", ""),
                    "xymc": student.get("xymc", ""),
                    "zymc": student.get("zymc", ""),
                    "bynf": student.get("bynf", ""),
                    "province": student.get("province", "") or "江苏省",
                    "byqxlb": student.get("byqxlb", ""),
                    "student_result": student_result,
                    "evidence_results": evidence_results,
                }
            )
        return {
            "job_id": job_id,
            "task_name": status.get("task_name", job_id),
            "status": status["status"],
            "students": students,
            "unmatched_evidence_results": unmatched,
        }

    def checked_student_summary(
        self,
        page: int,
        page_size: int,
        result_status: str = "all",
        keyword: str = "",
        graduation_year: str = "",
        material_category: str = "",
    ) -> dict[str, Any]:
        """汇总全部任务中已完整核验的学生，并按学生保留最近一次结果。"""
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in self.jobs.checked_student_items():
            refreshed = self._apply_current_decision_policy(item)
            grouped[(str(item["job_id"]), int(item["graduate_id"]))].append(
                refreshed
            )

        completed_records = []
        for (job_id, graduate_id), items in grouped.items():
            evidence_items = [item for item in items if item.get("evidence_file_id")]
            if not evidence_items or any(
                item.get("status") != "completed" or not item.get("result")
                for item in evidence_items
            ):
                continue

            aggregate = self._aggregate_student(evidence_items)
            outcome = str(aggregate.get("status") or "review")
            if outcome not in {"pass", "review", "fail"}:
                # unsupported 等状态没有形成有效核验结论，不进入“已核验”汇总。
                continue

            context = items[0].get("context") or {}
            student = context.get("student") or {}
            destination_label = str(student.get("byqxlb") or "未明确毕业去向")
            categories = [destination_label]
            completed_at = max(
                (
                    str(item.get("completed_at") or item.get("job_updated_at") or "")
                    for item in evidence_items
                ),
                default="",
            )
            completed_records.append(
                {
                    "graduate_id": graduate_id,
                    "xh": str(student.get("xh") or ""),
                    "xm": str(student.get("xm") or ""),
                    "xxmc": str(student.get("xxmc") or ""),
                    "xymc": str(student.get("xymc") or ""),
                    "zymc": str(student.get("zymc") or ""),
                    "bynf": str(student.get("bynf") or ""),
                    "byqxlb": str(student.get("byqxlb") or ""),
                    "status": outcome,
                    "summary": str(aggregate.get("summary") or "已完成材料核验"),
                    "material_count": len(evidence_items),
                    "material_categories": categories,
                    "evidence_file_ids": [
                        int(item["evidence_file_id"]) for item in evidence_items
                    ],
                    "job_id": job_id,
                    "task_name": str(items[0].get("task_name") or job_id),
                    "completed_at": completed_at,
                }
            )

        completed_records.sort(
            key=lambda item: (str(item["completed_at"]), str(item["job_id"])),
            reverse=True,
        )
        latest_by_graduate: dict[int, dict[str, Any]] = {}
        for record in completed_records:
            latest_by_graduate.setdefault(int(record["graduate_id"]), record)
        all_records = list(latest_by_graduate.values())

        filter_options = {
            "graduation_years": sorted(
                {record["bynf"] for record in all_records if record["bynf"]},
                reverse=True,
            ),
            "material_categories": sorted(
                {
                    category
                    for record in all_records
                    for category in record["material_categories"]
                    if category
                }
            ),
        }
        metrics = {
            "checked_students": len(all_records),
            "pass_students": sum(record["status"] == "pass" for record in all_records),
            "fail_students": sum(record["status"] in {"review", "fail"} for record in all_records),
            "review_students": sum(record["status"] in {"review", "fail"} for record in all_records),
            "material_count": sum(record["material_count"] for record in all_records),
        }

        normalized_keyword = keyword.strip().casefold()
        filtered = [
            record
            for record in all_records
            if (
                result_status == "all"
                or record["status"] == result_status
                or (result_status == "review" and record["status"] == "fail")
            )
            and (not graduation_year or record["bynf"] == graduation_year)
            and (
                not material_category
                or material_category in record["material_categories"]
            )
            and (
                not normalized_keyword
                or normalized_keyword
                in " ".join(
                    str(record[field])
                    for field in (
                        "xm", "xh", "xxmc", "xymc", "zymc", "byqxlb", "task_name"
                    )
                ).casefold()
            )
        ]
        total = len(filtered)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        start = (page - 1) * page_size
        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "metrics": metrics,
            "filter_options": filter_options,
            "items": filtered[start : start + page_size],
        }

    def _apply_current_decision_policy(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """读取历史结果时应用当前确定性规则，不需要重新调用大模型。"""
        result = item.get("result")
        if not isinstance(result, dict):
            return item
        if result.get("decision_policy_version") == DECISION_POLICY_VERSION:
            return item
        context = item.get("context") or {}
        destination = str((context.get("student") or {}).get("byqxlb") or "")
        route = self.router.resolve(destination)
        if route is None:
            return item
        refreshed = dict(item)
        refreshed["result"] = finalize_material_result(result, context, route)
        return refreshed

    def _schedule(self, job_id: str) -> None:
        task = asyncio.create_task(self._process_job(job_id))
        self._tasks.add(task)
        self._job_tasks[job_id] = task

        def cleanup(completed: asyncio.Task[Any]) -> None:
            self._tasks.discard(completed)
            if self._job_tasks.get(job_id) is completed:
                self._job_tasks.pop(job_id, None)

        task.add_done_callback(cleanup)

    async def _process_job(self, job_id: str) -> None:
        items = await asyncio.to_thread(self.jobs.queued_items, job_id)

        async def process(item: dict[str, Any]) -> None:
            async with self._item_gate:
                started = await asyncio.to_thread(
                    self.jobs.mark_running, item["item_id"]
                )
                if not started:
                    return
                try:
                    destination = item["context"]["student"]["byqxlb"]
                    route = self.router.resolve(destination)
                    if route is None:
                        raise RuntimeError(f"提示词路由不存在：{destination}")
                    result, metrics = await self.checker.check(item["context"], route)
                    result = finalize_material_result(result, item["context"], route)
                    await asyncio.to_thread(
                        self.jobs.mark_completed,
                        item["item_id"],
                        result,
                        metrics,
                    )
                except ModelOutputError as exc:
                    # 业务设计要求：模型输出无法解析时进入人工复核，而不是丢失结果。
                    result = self._review_result(
                        "模型未生成可验证的结构化结果，请人工复核该材料"
                    )
                    result["risk_flags"] = [
                        {
                            "type": "other",
                            "severity": "medium",
                            "description": "模型输出解析或结构校验失败",
                            "evidence": "vLLM 返回内容未通过 JSON Schema 校验",
                        }
                    ]
                    await asyncio.to_thread(
                        self.jobs.mark_completed,
                        item["item_id"],
                        result,
                        {"fallback": True, "error_type": type(exc).__name__},
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    await asyncio.to_thread(
                        self.jobs.mark_failed,
                        item["item_id"],
                        type(exc).__name__,
                        str(exc),
                    )

        await asyncio.gather(*(process(item) for item in items))

    def _item_for_context(
        self,
        context: dict[str, Any],
        request: CreateJobRequest,
    ) -> dict[str, Any]:
        destination = str(context["student"].get("byqxlb", "")).strip()
        route = self.router.resolve(destination)
        if route is None:
            result = self._unsupported_result(destination)
            return {
                "graduate_id": context["graduate_id"],
                "evidence_file_id": context["evidence"]["evidence_file_id"],
                "route_id": None,
                "status": "unsupported",
                "context": context,
                "result": result,
            }
        cache_key = self._cache_key(context, route.route_id, request.options.model)
        if not request.force_recheck:
            cached = self.jobs.cached_result(cache_key)
            if cached is not None:
                return {
                    "graduate_id": context["graduate_id"],
                    "evidence_file_id": context["evidence"]["evidence_file_id"],
                    "route_id": route.route_id,
                    "status": "completed",
                    "cache_key": cache_key,
                    "context": context,
                    "result": cached,
                }
        return {
            "graduate_id": context["graduate_id"],
            "evidence_file_id": context["evidence"]["evidence_file_id"],
            "route_id": route.route_id,
            "status": "queued",
            "cache_key": cache_key,
            "context": context,
        }

    def _cache_key(self, context: dict[str, Any], route_id: str, model: str) -> str:
        payload = "|".join(
            [
                str(context["graduate_id"]),
                str(context["evidence"]["evidence_file_id"]),
                str(context["evidence"].get("md5", "")),
                str(context["student"].get("byqxlb", "")),
                route_id,
                self.router.version,
                model,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _unsupported_result(destination: str) -> dict[str, Any]:
        summary = f"毕业去向“{destination or '空'}”不属于当前自动核查范围"
        result = empty_review_result(summary, unsupported=True)
        result["destination_assessment"]["registered_destination"] = destination
        return result

    @staticmethod
    def _no_evidence_item(context: dict[str, Any]) -> dict[str, Any]:
        result = EmploymentCheckService._review_result("该学生没有已关联佐证材料")
        return {
            "graduate_id": context["graduate_id"], "evidence_file_id": None,
            "route_id": None, "status": "completed", "context": context,
            "result": result
        }

    @staticmethod
    def _unmatched_evidence_item(evidence: dict[str, Any]) -> dict[str, Any]:
        result = EmploymentCheckService._review_result("材料尚未关联毕业生，不能注入学生上下文")
        return {
            "graduate_id": None, "evidence_file_id": evidence["evidence_file_id"],
            "route_id": None, "status": "completed",
            "context": {"evidence": evidence}, "result": result
        }

    @staticmethod
    def _review_result(summary: str) -> dict[str, Any]:
        return empty_review_result(summary)

    @staticmethod
    def _evidence_response(item: dict[str, Any]) -> dict[str, Any]:
        evidence = item["context"].get("evidence", {})
        result = item.get("result") or EmploymentCheckService._review_result(
            item.get("error_message") or "核查尚未完成"
        )
        return {
            "evidence_file_id": item.get("evidence_file_id"),
            "file_name": evidence.get("file_name", ""),
            "category": evidence.get("category", ""),
            "extension": evidence.get("extension", ""),
            "item_status": item.get("status"),
            **result,
        }

    @staticmethod
    def _material_dimensions(result: dict[str, Any]) -> dict[str, str]:
        """计算单份材料自己的三项结果，禁止跨材料补齐。"""
        if not result or result.get("status") == "unsupported":
            return {
                "visual": "unverified",
                "content": "unverified",
                "destination": "unverified",
            }

        has_dimension_payload = bool(
            result.get("seal_assessment")
            or result.get("visual_regions")
            or result.get("content_assessment")
            or result.get("destination_assessment")
        )
        if result.get("status") == "pass" and not has_dimension_payload:
            # 兼容早期只保存综合状态、未保存三项结构的历史结果。
            return {
                "visual": "pass",
                "content": "pass",
                "destination": "pass",
            }

        seal_assessment = result.get("seal_assessment") or {}
        completeness = str(seal_assessment.get("completeness") or "")
        has_seal = any(
            isinstance(region, dict) and region.get("region_type") == "seal"
            for region in result.get("visual_regions", [])
        )
        if completeness in {"complete", "not_required"} or (
            not completeness and has_seal
        ):
            visual = "pass"
        elif completeness in {"incomplete", "unclear"}:
            visual = "suspicious"
        else:
            visual = "unverified"

        content_verdict = str(
            (result.get("content_assessment") or {}).get("verdict") or ""
        )
        if content_verdict == "supported":
            content = "pass"
        elif content_verdict in {"questionable", "contradicted", "insufficient"}:
            content = "suspicious"
        else:
            content = "unverified"

        destination_verdict = str(
            (result.get("destination_assessment") or {}).get("verdict") or ""
        )
        has_blocking_field_mismatch = any(
            isinstance(comparison, dict)
            and comparison.get("status") == "mismatch"
            and comparison.get("blocking", True)
            for comparison in result.get("field_comparisons", [])
        ) or bool(result.get("mismatches"))
        if has_blocking_field_mismatch:
            destination = "suspicious"
        elif destination_verdict == "consistent":
            destination = "pass"
        elif destination_verdict in {"inconsistent", "mismatch"}:
            destination = "suspicious"
        else:
            destination = "unverified"

        # 历史结果可能三项结构均为正向，但单材料综合规则仍未通过。
        # 将这种阻断落入内容项，避免出现“概览全绿、总结果不通过”。
        if (
            result.get("status") not in {"pass", "unsupported"}
            and visual == content == destination == "pass"
        ):
            content = "suspicious"

        return {
            "visual": visual,
            "content": content,
            "destination": destination,
        }

    @staticmethod
    def _aggregate_student(items: list[dict[str, Any]]) -> dict[str, Any]:
        """按单份材料闭环聚合：任一材料三项全通过则学生通过。"""
        is_complete = bool(items) and all(
            item.get("status") == "completed" for item in items
        )
        results = [item.get("result") for item in items if item.get("result")]
        statuses = [str(result.get("status") or "review") for result in results]
        if not results:
            return {
                "status": "review", "confidence": 0.0,
                "summary": "尚无可用材料结果", "main_risks": [],
                "decision_basis": [], "material_count": 0,
                "is_complete": is_complete,
                "dimensions": {
                    "visual": "unverified", "content": "unverified",
                    "destination": "unverified",
                },
            }
        if all(value == "unsupported" for value in statuses):
            return {
                "status": "unsupported", "confidence": 0.0,
                "summary": "现有材料均不在自动核验范围内", "main_risks": [],
                "decision_basis": [], "material_count": len(results),
                "is_complete": is_complete,
                "dimensions": {
                    "visual": "unverified", "content": "unverified",
                    "destination": "unverified",
                },
            }

        completed_materials: list[dict[str, Any]] = []
        for item in items:
            result = item.get("result") or {}
            if item.get("status") != "completed" or not result:
                continue
            dimensions = EmploymentCheckService._material_dimensions(result)
            passes_all = (
                result.get("status") == "pass"
                and all(value == "pass" for value in dimensions.values())
            )
            completed_materials.append({
                "item": item,
                "result": result,
                "dimensions": dimensions,
                "passes_all": passes_all,
                "score": sum(value == "pass" for value in dimensions.values()),
            })

        if not completed_materials:
            return {
                "status": "review", "confidence": 0.0,
                "summary": "材料仍在处理，尚未形成完整核验结果",
                "main_risks": [], "decision_basis": [],
                "review_reasons": [], "material_count": 0,
                "primary_evidence_file_ids": [],
                "secondary_evidence_file_ids": [],
                "material_contributions": [], "is_complete": is_complete,
                "dimensions": {
                    "visual": "unverified", "content": "unverified",
                    "destination": "unverified",
                },
            }

        primary_materials = [
            material for material in completed_materials if material["passes_all"]
        ]
        primary_materials.sort(
            key=lambda material: float(material["result"].get("confidence", 0)),
            reverse=True,
        )
        selected = (
            primary_materials[0]
            if primary_materials
            else max(
                completed_materials,
                key=lambda material: (
                    int(material["score"]),
                    float(material["result"].get("confidence", 0)),
                ),
            )
        )
        passing_primary_ids = [
            material["item"].get("evidence_file_id")
            for material in primary_materials
            if material["item"].get("evidence_file_id") is not None
        ]
        status = "pass" if primary_materials else "fail"
        all_completed_ids = [
            material["item"].get("evidence_file_id")
            for material in completed_materials
            if material["item"].get("evidence_file_id") is not None
        ]
        # 不通过时每份材料都可能包含阻断原因，全部作为主要复核材料。
        primary_ids = (
            passing_primary_ids if status == "pass" else all_completed_ids
        )
        secondary_ids = (
            [value for value in all_completed_ids if value not in primary_ids]
            if status == "pass" else []
        )
        selected_dimensions = dict(selected["dimensions"])

        def material_name(material: dict[str, Any]) -> str:
            return str(
                (material["item"].get("context") or {})
                .get("evidence", {})
                .get("file_name")
                or "未命名材料"
            )

        material_contributions = []
        for material in completed_materials:
            material_contributions.append({
                "evidence_file_id": material["item"].get("evidence_file_id"),
                "file_name": material_name(material),
                "material_status": str(material["result"].get("status") or "review"),
                "material_role": (
                    "primary"
                    if status == "fail" or material["passes_all"]
                    else "secondary"
                ),
                "passes_all_dimensions": bool(material["passes_all"]),
                "dimensions": dict(material["dimensions"]),
            })

        if primary_materials:
            primary_names = "、".join(material_name(value) for value in primary_materials)
            summary = (
                f"共核验 {len(completed_materials)} 份材料，其中 {len(primary_materials)} 份材料"
                "在印章、内容完整性和毕业去向一致性三项中均通过；"
                f"已将{primary_names}作为主要材料，学生核验结果为通过。"
            )
            decision_basis = [
                f"主要材料：{material_name(material)}三项核验均通过。"
                for material in primary_materials
            ]
            if secondary_ids:
                decision_basis.append(
                    f"其余 {len(secondary_ids)} 份材料未形成单份完整通过结论，"
                    "仅作为次要材料预览，不参与学生总结果拼接。"
                )
            review_reasons: list[str] = []
            main_risks = [
                risk
                for material in primary_materials
                for risk in material["result"].get("risk_flags", [])
                if isinstance(risk, dict) and risk.get("severity") in {"medium", "high"}
            ]
            confidence_values = [
                float(material["result"].get("confidence", 0))
                for material in primary_materials
            ]
        else:
            summary = (
                f"共核验 {len(completed_materials)} 份材料，没有任何一份材料在印章、"
                "内容完整性和毕业去向一致性三项中全部通过，学生核验结果为不通过；"
                "全部材料均作为主要复核材料展示。"
            )
            failed_dimension_labels = {
                "visual": "印章核验",
                "content": "内容完整性",
                "destination": "毕业去向一致性",
            }
            review_reasons = []
            for material in completed_materials:
                failed_labels = [
                    failed_dimension_labels[key]
                    for key, value in material["dimensions"].items()
                    if value != "pass"
                ]
                review_reasons.append(
                    f"{material_name(material)}未通过："
                    + "、".join(failed_labels or ["单份材料综合判定"])
                )
            decision_basis = list(review_reasons)
            main_risks = [
                risk
                for material in completed_materials
                for risk in material["result"].get("risk_flags", [])
                if isinstance(risk, dict) and risk.get("severity") in {"medium", "high"}
            ]
            confidence_values = [
                float(material["result"].get("confidence", 0))
                for material in completed_materials
            ]

        return {
            "status": status,
            "confidence": (
                round(sum(confidence_values) / len(confidence_values), 4)
                if confidence_values else 0.0
            ),
            "summary": summary,
            "main_risks": main_risks,
            "decision_basis": decision_basis,
            "review_reasons": review_reasons,
            "material_count": len(completed_materials),
            "material_names": [material_name(value) for value in completed_materials],
            "primary_evidence_file_ids": primary_ids,
            "secondary_evidence_file_ids": secondary_ids,
            "material_contributions": material_contributions,
            "is_complete": is_complete,
            "dimensions": selected_dimensions,
        }
