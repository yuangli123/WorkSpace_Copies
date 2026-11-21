from __future__ import annotations

import asyncio
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .checker import MaterialChecker
from .config import APISettings
from .database import JobRepository, SourceRepository
from .decision import DECISION_POLICY_VERSION
from .models import CreateJobRequest
from .prompt_router import PromptRouter
from .service import EmploymentCheckService, SelectionError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "api.example.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = Path(os.environ.get("EMPLOYMENT_API_CONFIG", DEFAULT_CONFIG))
    settings = APISettings.from_file(config_path)
    source = SourceRepository(settings.source_database, settings.evidence_root)
    jobs = JobRepository(settings.jobs_database)
    router = PromptRouter(
        settings.prompt_routes_file,
        settings.system_prompt_file,
    )
    checker = MaterialChecker(settings, router)
    service = EmploymentCheckService(
        source,
        jobs,
        router,
        checker,
        settings.max_parallel_items,
    )
    startup = await service.start()
    app.state.settings = settings
    app.state.service = service
    app.state.startup = startup
    try:
        yield
    finally:
        await service.close()


app = FastAPI(
    title="高校毕业生就业佐证材料核查 API",
    version="2.9.0",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")


def _service(request: Request) -> EmploymentCheckService:
    return request.app.state.service


@app.get("/health", tags=["system"])
async def health(request: Request) -> dict[str, Any]:
    service = _service(request)
    return {
        "status": "ok",
        "source_database": await asyncio.to_thread(service.source.verify_schema),
        "vllm_models": request.app.state.startup.get("models", []),
        "configured_model": service.checker.runtime.model,
        "recovered_jobs": request.app.state.startup.get("recovered_jobs", []),
    }


@app.get("/", include_in_schema=False)
async def frontend_index() -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/dashboard/overview", tags=["dashboard"])
async def dashboard_overview(request: Request) -> dict[str, Any]:
    service = _service(request)
    source, jobs = await asyncio.gather(
        asyncio.to_thread(service.source.dashboard_overview),
        asyncio.to_thread(service.jobs.dashboard_counts),
    )
    return {**source, **jobs}


@app.get("/api/v1/system/model-parameters", tags=["system"])
async def model_parameters(request: Request) -> dict[str, Any]:
    """返回当前服务实际加载的核心参数；不暴露密钥、密码和文件路径。"""
    service = _service(request)
    runtime = service.checker.runtime
    settings = request.app.state.settings
    startup = request.app.state.startup

    def item(
        key: str,
        label: str,
        value: Any,
        description: str,
        unit: str = "",
    ) -> dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "value": value,
            "unit": unit,
            "description": description,
        }

    return {
        "read_only": True,
        "model": runtime.model,
        "available_models": startup.get("models", []),
        "service_status": "online",
        "api_version": app.version,
        "prompt_version": service.router.version,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "groups": [
            {
                "key": "generation",
                "title": "模型与生成",
                "description": "当前多模态核验模型及结构化输出设置",
                "items": [
                    item("model", "模型名称", runtime.model, "当前 vLLM 服务使用的模型标识"),
                    item("temperature", "生成温度", runtime.temperature, "0 表示使用确定性输出"),
                    item("max_tokens", "单次输出上限", runtime.max_tokens, "常规材料分析的最大输出长度", "tokens"),
                    item("retry_max_tokens", "重试输出上限", runtime.retry_max_tokens, "输出截断后重试使用的长度", "tokens"),
                    item("merge_max_tokens", "合并输出上限", runtime.merge_max_tokens, "多段材料结果合并的最大输出长度", "tokens"),
                    item("enable_thinking", "思考模式", runtime.enable_thinking, "当前是否启用模型思考输出"),
                    item("use_json_schema", "结构化约束", runtime.use_json_schema, "是否使用 JSON Schema 约束模型结果"),
                ],
            },
            {
                "key": "scheduling",
                "title": "请求与调度",
                "description": "API 任务并发、超时和失败重试策略",
                "items": [
                    item("max_parallel_items", "任务并行数", settings.max_parallel_items, "同时进入材料核验流程的任务项数量", "项"),
                    item("concurrency", "模型请求并发", runtime.concurrency, "同时发往 vLLM 的最大请求数", "请求"),
                    item("timeout_seconds", "请求超时", runtime.timeout_seconds, "单次模型请求最长等待时间", "秒"),
                    item("max_retries", "网络重试次数", runtime.max_retries, "连接或请求异常时的最大重试次数", "次"),
                    item("max_inference_attempts", "单文件推理上限", runtime.max_inference_attempts_per_document, "每份材料允许的最大模型调用次数", "次"),
                ],
            },
            {
                "key": "document",
                "title": "材料预处理",
                "description": "PDF、图片和 Office 材料进入模型前的处理参数",
                "items": [
                    item("pages_per_request", "每次处理页数", runtime.pages_per_request, "每个多模态请求携带的页面数量", "页"),
                    item("pdf_dpi", "PDF 渲染精度", runtime.pdf_dpi, "PDF 页面转换为图像时使用的分辨率", "DPI"),
                    item("inference_max_pixels", "图像像素上限", runtime.inference_max_pixels, "送入模型前单张图像的最大像素数量", "像素"),
                    item("max_pages_per_document", "单文件页数上限", runtime.max_pages_per_document, "0 表示不限制文档页数", "页"),
                    item("preprocess_workers", "预处理线程数", runtime.preprocess_workers, "并行执行页面转换和图像预处理的线程数", "线程"),
                    item("image_transport", "图像传输方式", runtime.image_transport, "模型请求传递图像时使用的方式"),
                ],
            },
            {
                "key": "decision",
                "title": "核验与判定",
                "description": "当前提示词、决策规则和结果复核策略",
                "items": [
                    item("prompt_version", "提示词版本", service.router.version, "材料类别路由与业务提示词版本"),
                    item("decision_policy_version", "判定策略版本", DECISION_POLICY_VERSION, "后端确定性核验规则版本"),
                    item("seal_check", "印章核验", True, "定位材料中的实际印章；明确要求盖章时检查对应章位是否已盖"),
                    item("field_tolerance", "字段容错策略", "已启用", "核心字段严格核对，辅助字段采用业务容错"),
                    item("cache_enabled", "预处理存档", True, "复用已完成的页面预处理结果"),
                ],
            },
        ],
    }


@app.get("/api/v1/evidence-files", tags=["evidence"])
async def list_evidence_files(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=12, ge=1, le=100),
    keyword: str = Query(default="", max_length=100),
    category: str = Query(default="", max_length=100),
    link_status: str = Query(default="all", pattern="^(all|linked|unmatched)$"),
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _service(request).source.list_evidences,
        page,
        page_size,
        keyword,
        category,
        link_status,
    )


@app.get("/api/v1/graduates/filter-options", tags=["graduates"])
async def graduate_filter_options(request: Request) -> dict[str, Any]:
    return await asyncio.to_thread(
        _service(request).source.graduate_filter_options,
    )


@app.get("/api/v1/graduates", tags=["graduates"])
async def list_graduates(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    province: str = Query(default="", max_length=100),
    graduation_year: str = Query(default="", max_length=20),
    school: str = Query(default="", max_length=200),
    college: str = Query(default="", max_length=200),
    name: str = Query(default="", max_length=100),
    student_id: str = Query(default="", max_length=100),
    material_category: str = Query(default="", max_length=100),
    destination: str = Query(default="", max_length=200),
    evidence_status: str = Query(default="all", pattern="^(all|with|without)$"),
    verification_status: str = Query(
        default="all", pattern="^(all|unverified|pass|review|fail)$"
    ),
) -> dict[str, Any]:
    service = _service(request)
    statuses = await asyncio.to_thread(service.latest_verification_statuses)
    return await asyncio.to_thread(
        service.source.list_graduates,
        page,
        page_size,
        province,
        graduation_year,
        school,
        college,
        name,
        student_id,
        destination or material_category,
        evidence_status,
        verification_status,
        statuses,
    )


@app.get("/api/v1/graduates/selection", tags=["graduates"])
async def list_graduate_selection(
    request: Request,
    province: str = Query(default="", max_length=100),
    graduation_year: str = Query(default="", max_length=20),
    school: str = Query(default="", max_length=200),
    college: str = Query(default="", max_length=200),
    name: str = Query(default="", max_length=100),
    student_id: str = Query(default="", max_length=100),
    destination: str = Query(default="", max_length=200),
    evidence_status: str = Query(default="all", pattern="^(all|with|without)$"),
    verification_status: str = Query(
        default="all", pattern="^(all|unverified|pass|review|fail)$"
    ),
) -> dict[str, Any]:
    service = _service(request)
    statuses = await asyncio.to_thread(service.latest_verification_statuses)
    return await asyncio.to_thread(
        service.source.list_graduate_selection,
        province,
        graduation_year,
        school,
        college,
        name,
        student_id,
        destination,
        evidence_status,
        verification_status,
        statuses,
    )


@app.get("/api/v1/evidence-files/{evidence_id}/preview", tags=["evidence"])
async def preview_evidence(evidence_id: int, request: Request) -> FileResponse:
    value = await asyncio.to_thread(
        _service(request).source.evidence_file,
        evidence_id,
    )
    if value is None:
        raise HTTPException(status_code=404, detail="材料不存在")
    path = Path(value["file_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="材料文件不存在")
    if path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"
    }:
        raise HTTPException(status_code=415, detail="该格式暂不支持浏览器预览")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store, private",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get(
    "/api/v1/evidence-files/{evidence_id}/regions/{page_number}",
    tags=["evidence"],
)
async def preview_evidence_region(
    evidence_id: int,
    page_number: int,
    request: Request,
    x1: int = Query(ge=0, le=1000),
    y1: int = Query(ge=0, le=1000),
    x2: int = Query(ge=0, le=1000),
    y2: int = Query(ge=0, le=1000),
) -> Response:
    value = await asyncio.to_thread(
        _service(request).source.evidence_file,
        evidence_id,
    )
    if value is None:
        raise HTTPException(status_code=404, detail="材料不存在")
    path = Path(value["file_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="材料文件不存在")
    try:
        content = await asyncio.to_thread(
            _service(request).checker.render_region,
            path,
            page_number,
            (x1, y1, x2, y2),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post(
    "/api/v1/employment-evidence-check/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["employment-check"],
)
async def create_job(
    payload: CreateJobRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        return await _service(request).create_job(payload)
    except SelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/v1/employment-evidence-check/jobs",
    tags=["employment-check"],
)
async def list_jobs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    keyword: str = Query(default="", max_length=100),
) -> dict[str, Any]:
    items = await asyncio.to_thread(
        _service(request).jobs.list_jobs,
        limit,
        keyword,
    )
    return {"items": items, "total": len(items)}


@app.get(
    "/api/v1/employment-evidence-check/checked-students",
    tags=["employment-check"],
)
async def checked_students(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    result_status: str = Query(default="all", pattern="^(all|pass|review|fail)$"),
    keyword: str = Query(default="", max_length=100),
    graduation_year: str = Query(default="", max_length=20),
    material_category: str = Query(default="", max_length=100),
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _service(request).checked_student_summary,
        page,
        page_size,
        result_status,
        keyword,
        graduation_year,
        material_category,
    )


@app.get(
    "/api/v1/employment-evidence-check/jobs/{job_id}",
    tags=["employment-check"],
)
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    value = await asyncio.to_thread(_service(request).status, job_id)
    if value is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return value


@app.delete(
    "/api/v1/employment-evidence-check/jobs/{job_id}",
    tags=["employment-check"],
)
async def delete_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        value = await asyncio.to_thread(_service(request).jobs.delete_job, job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if value is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"deleted": True, **value}


@app.post(
    "/api/v1/employment-evidence-check/jobs/{job_id}/cancel",
    tags=["employment-check"],
)
async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        value = await _service(request).cancel_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if value is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return value


@app.get(
    "/api/v1/employment-evidence-check/jobs/{job_id}/results",
    tags=["employment-check"],
)
async def get_results(job_id: str, request: Request) -> dict[str, Any]:
    value = await asyncio.to_thread(_service(request).results, job_id)
    if value is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return value
