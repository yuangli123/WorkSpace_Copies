from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import (
    ModelOutputError,
    RequestBudget,
    RequestBudgetExceeded,
    VLLMClient,
)
from .config import RuntimeConfig, load_json_object
from .metadata import MetadataIndex
from .preprocessor import (
    DocumentPreprocessor,
    PreprocessOptions,
    dependency_status,
    discover_files,
)
from .prompts import PromptBuilder
from .reporter import ResultStore
from .schema import load_schema
from .types import DocumentOutcome, PageImage, PreparedDocument, ProcessingMetrics


def make_preprocess_options(config: RuntimeConfig) -> PreprocessOptions:
    return PreprocessOptions(
        cache_dir=Path(config.cache_dir),
        pdf_dpi=config.pdf_dpi,
        max_pages_per_document=config.max_pages_per_document,
        pdftoppm_bin=config.pdftoppm_bin,
        libreoffice_bin=config.libreoffice_bin,
        pdf_password=config.pdf_password,
        pdf_password_file=config.pdf_password_file,
        workers=config.preprocess_workers,
        show_progress=config.show_preprocess_progress,
    )


def check_dependencies(config: RuntimeConfig) -> dict[str, Any]:
    status = dependency_status(make_preprocess_options(config))
    try:
        import openai  # noqa: F401

        status["openai"] = True
    except ImportError:
        status["openai"] = False
    try:
        import jsonschema  # noqa: F401

        status["jsonschema"] = True
    except ImportError:
        status["jsonschema"] = False
    return status


async def run_pipeline(
    config: RuntimeConfig,
    input_path: Path,
    output_dir: Path,
    metadata: MetadataIndex,
    resume: bool = False,
    limit: int = 0,
    preprocess_only: bool = False,
    retry_model_failures: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    store = ResultStore(output_dir, save_raw=config.save_raw_responses)
    store.write_run_config(config.safe_dict())

    sources, skipped = discover_files(input_path, output_dir)
    if limit:
        sources = sources[:limit]
    store.write_skipped(skipped)
    print(f"发现支持文件 {len(sources)} 个，不支持文件 {len(skipped)} 个。")

    preprocessor = DocumentPreprocessor(make_preprocess_options(config))
    documents, preprocess_errors = await asyncio.to_thread(
        preprocessor.prepare_many,
        sources,
        input_path,
    )
    for error in preprocess_errors:
        store.append_error(error)
    archive_hits = sum(document.from_cache for document in documents)
    archive_created = len(documents) - archive_hits
    manifest_path = store.write_preprocess_manifest(documents)
    print(
        f"预处理完成：成功 {len(documents)}，存档命中 {archive_hits}，"
        f"新建存档 {archive_created}，失败 {len(preprocess_errors)}。"
    )

    if preprocess_only:
        return {
            "preprocess_only": True,
            "prepared_documents": len(documents),
            "preprocess_errors": len(preprocess_errors),
            "archive_hits": archive_hits,
            "archive_created": archive_created,
            "archive_dir": config.cache_dir,
            "manifest": str(manifest_path),
        }

    if resume:
        completed = store.completed_source_ids(
            include_model_failures=not retry_model_failures,
        )
        documents = [item for item in documents if item.source_id not in completed]
        print(f"续跑模式：剩余待处理文件 {len(documents)} 个。")

    business_rules = load_json_object(config.business_rules_file)
    schema = load_schema(config.output_schema_file)
    prompts = PromptBuilder(config.prompt_dir, business_rules)
    client = VLLMClient(config, schema)
    client.system_prompt = prompts.system_prompt
    try:
        available_models = await client.check_server()
        print(f"vLLM 服务可用，模型：{', '.join(available_models)}")
        if config.model not in available_models:
            raise RuntimeError(
                f"配置模型 {config.model!r} 不在 /v1/models 返回值中：{available_models}"
            )

        document_gate = asyncio.Semaphore(config.concurrency)

        async def guarded(document: PreparedDocument) -> DocumentOutcome:
            async with document_gate:
                return await _process_document(
                    client=client,
                    prompts=prompts,
                    document=document,
                    registered_fields=metadata.get(document.relative_path),
                    pages_per_request=config.pages_per_request,
                    max_tokens=config.max_tokens,
                    retry_max_tokens=config.retry_max_tokens,
                    max_inference_attempts=config.max_inference_attempts_per_document,
                )

        tasks = [asyncio.create_task(guarded(document)) for document in documents]
        completed_count = 0
        for task in asyncio.as_completed(tasks):
            outcome = await task
            completed_count += 1
            if outcome.record is not None:
                store.append_result(outcome.record)
                source_id = outcome.record["source"]["source_id"]
                store.write_chunks(source_id, outcome.chunk_payloads)
                relative_path = outcome.record["source"]["relative_path"]
                decision = outcome.record["result"]["decision"]["recommended_result"]
                label = (
                    "人工兜底"
                    if outcome.record.get("status") == "manual_review_fallback"
                    else "完成"
                )
                print(f"[{completed_count}/{len(tasks)}] {label}：{relative_path} -> {decision}")
            if outcome.error is not None:
                store.append_error(outcome.error)
                if outcome.record is None:
                    store.write_chunks(
                        str(outcome.error.get("source_id", "unknown")),
                        outcome.chunk_payloads,
                    )
                    print(
                        f"[{completed_count}/{len(tasks)}] 失败："
                        f"{outcome.error.get('relative_path', '')} -> "
                        f"{outcome.error.get('error', '')}"
                    )
    finally:
        await client.close()

    summary = store.finalize(
        preprocess_error_count=len(preprocess_errors),
        skipped_count=len(skipped),
        archive_hits=archive_hits,
        archive_created=archive_created,
        elapsed_seconds=time.perf_counter() - started,
    )
    return summary


async def _process_document(
    client: VLLMClient,
    prompts: PromptBuilder,
    document: PreparedDocument,
    registered_fields: dict[str, Any],
    pages_per_request: int,
    max_tokens: int,
    retry_max_tokens: int,
    max_inference_attempts: int,
) -> DocumentOutcome:
    started = time.perf_counter()
    metrics = ProcessingMetrics()
    budget = RequestBudget(max_inference_attempts)
    try:
        page_chunks = [
            list(document.pages[index : index + pages_per_request])
            for index in range(0, len(document.pages), pages_per_request)
        ]

        async def analyze_chunk(
            pages: list[PageImage],
            token_limit: int = max_tokens,
            compact: bool = False,
        ) -> list[tuple[list[PageImage], Any]]:
            prompt = prompts.build_review(
                relative_path=document.relative_path,
                page_numbers=[page.page_number for page in pages],
                registered_fields=registered_fields,
                compact=compact,
            )
            try:
                response = await client.analyze_pages(
                    pages,
                    prompt,
                    max_tokens=token_limit,
                    budget=budget,
                )
                return [(pages, response)]
            except ModelOutputError as exc:
                if len(pages) > 1:
                    page_range = ",".join(str(page.page_number) for page in pages)
                    print(
                        f"  结构化输出过长或无效，自动拆分页面："
                        f"{document.relative_path} 第{page_range}页"
                    )
                    groups = await asyncio.gather(
                        *(analyze_chunk([page], max_tokens) for page in pages)
                    )
                    return [item for group in groups for item in group]
                if token_limit < retry_max_tokens and not compact:
                    print(
                        f"  单页输出仍过长，启用紧凑模式并提升上限至 "
                        f"{retry_max_tokens} tokens："
                        f"{document.relative_path} 第{pages[0].page_number}页"
                    )
                    return await analyze_chunk(
                        pages,
                        retry_max_tokens,
                        compact=True,
                    )
                raise exc

        analyzed_groups = await asyncio.gather(
            *(analyze_chunk(chunk) for chunk in page_chunks)
        )
        analyzed_chunks = [item for group in analyzed_groups for item in group]
        page_chunks = [item[0] for item in analyzed_chunks]
        responses = [item[1] for item in analyzed_chunks]
        for response in responses:
            metrics.add(response)
        chunk_results = [response.parsed for response in responses]

        if len(chunk_results) == 1:
            final_result = copy.deepcopy(chunk_results[0])
        else:
            merge_prompt = prompts.build_merge(
                relative_path=document.relative_path,
                chunk_results=chunk_results,
                registered_fields=registered_fields,
            )
            try:
                merged = await client.synthesize(merge_prompt, budget=budget)
            except ModelOutputError:
                merged = await client.synthesize(
                    merge_prompt,
                    max_tokens=retry_max_tokens,
                    budget=budget,
                )
            metrics.add(merged)
            final_result = copy.deepcopy(merged.parsed)

        all_ocr_pages = []
        for chunk_result in chunk_results:
            all_ocr_pages.extend(chunk_result.get("ocr_pages", []))
        deduplicated = {
            int(page.get("page", 0)): page
            for page in all_ocr_pages
            if isinstance(page, dict) and int(page.get("page", 0)) > 0
        }
        final_result["ocr_pages"] = [deduplicated[key] for key in sorted(deduplicated)]

        record = {
            "schema_version": "1.0",
            "status": "success",
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "source": {
                "source_id": document.source_id,
                "source_file": str(document.source_file),
                "relative_path": document.relative_path,
                "format": document.source_format,
                "page_count": len(document.pages),
                "from_cache": document.from_cache,
            },
            "registered_fields": registered_fields,
            "result": final_result,
            "metrics": {
                **metrics.as_dict(),
                "inference_attempts_used": budget.used,
                "inference_attempts_limit": budget.maximum,
                "document_elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        }
        chunk_payloads = [
            {
                "pages": [page.page_number for page in pages],
                "parsed": response.parsed,
                "raw_text": response.raw_text,
            }
            for pages, response in zip(page_chunks, responses)
        ]
        return DocumentOutcome(record=record, chunk_payloads=chunk_payloads)
    except ModelOutputError as exc:
        return _manual_review_fallback(
            document=document,
            registered_fields=registered_fields,
            metrics=metrics,
            budget=budget,
            started=started,
            exc=exc,
            stage="structured_output",
            chunk_payloads=[
                {
                    "error": str(exc),
                    "finish_reason": exc.finish_reason,
                    "raw_text": exc.raw_text,
                }
            ],
            extra_error={
                "finish_reason": exc.finish_reason,
                "raw_output_characters": len(exc.raw_text),
                "parse_error": exc.parse_error,
            },
        )
    except RequestBudgetExceeded as exc:
        return _manual_review_fallback(
            document=document,
            registered_fields=registered_fields,
            metrics=metrics,
            budget=budget,
            started=started,
            exc=exc,
            stage="attempt_budget",
        )
    except Exception as exc:  # noqa: BLE001
        return _manual_review_fallback(
            document=document,
            registered_fields=registered_fields,
            metrics=metrics,
            budget=budget,
            started=started,
            exc=exc,
            stage="inference",
        )


def _manual_review_fallback(
    document: PreparedDocument,
    registered_fields: dict[str, Any],
    metrics: ProcessingMetrics,
    budget: RequestBudget,
    started: float,
    exc: Exception,
    stage: str,
    chunk_payloads: list[dict[str, Any]] | None = None,
    extra_error: dict[str, Any] | None = None,
) -> DocumentOutcome:
    """Persist a terminal model failure as a reviewable business result."""
    failure_reason = str(exc)
    if isinstance(exc, RequestBudgetExceeded):
        reason_code = "MODEL_ATTEMPT_LIMIT_EXCEEDED"
    else:
        reason_code = "MODEL_INFERENCE_FAILED"

    result = {
        "schema_version": "1.0",
        "document_classification": {
            "category": "无法判断",
            "subcategory": "无法判断",
            "document_name": "无法判断",
            "confidence": 0.0,
            "evidence": [],
        },
        "ocr_pages": [],
        "key_fields": [],
        "visual_evidence": {
            "seal_status": "无法判断",
            "seal_texts": [],
            "seal_completeness": "无法判断",
            "required_seals": [],
            "missing_required_seals": [],
            "signature_status": "无法判断",
            "suspicious_signs": [],
            "authenticity_assessment": "无法判断",
            "limitations": ["模型推理未能生成完整、可校验的结构化结果"],
        },
        "rule_checks": [],
        "registration_consistency": [],
        "decision": {
            "recommended_result": "人工复核",
            "reason_codes": [reason_code],
            "summary": "大模型推理失败，需要人工复核",
            "missing_items": [],
            "manual_review_items": [
                f"失败阶段：{stage}",
                f"失败原因：{failure_reason}",
                f"已发起模型请求 {budget.used}/{budget.maximum} 次",
            ],
        },
    }
    record = {
        "schema_version": "1.0",
        "status": "manual_review_fallback",
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "source_id": document.source_id,
            "source_file": str(document.source_file),
            "relative_path": document.relative_path,
            "format": document.source_format,
            "page_count": len(document.pages),
            "from_cache": document.from_cache,
        },
        "registered_fields": registered_fields,
        "result": result,
        "inference_failure": {
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": failure_reason,
        },
        "metrics": {
            **metrics.as_dict(),
            "inference_attempts_used": budget.used,
            "inference_attempts_limit": budget.maximum,
            "document_elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    error = {
        "source_id": document.source_id,
        "source_file": str(document.source_file),
        "relative_path": document.relative_path,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": failure_reason,
        "inference_attempts_used": budget.used,
        "inference_attempts_limit": budget.maximum,
        "resolved_by_manual_fallback": True,
        **(extra_error or {}),
    }
    return DocumentOutcome(
        record=record,
        error=error,
        chunk_payloads=chunk_payloads or [],
    )
