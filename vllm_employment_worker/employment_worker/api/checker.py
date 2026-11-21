from __future__ import annotations

import asyncio
import copy
import io
from pathlib import Path
from typing import Any

from PIL import Image

from ..client import ModelOutputError, RequestBudget, VLLMClient
from ..pipeline import make_preprocess_options
from ..preprocessor import DocumentPreprocessor
from ..schema import load_schema
from ..types import ModelResponse, PageImage, ProcessingMetrics
from .config import APISettings
from .prompt_router import PromptRoute, PromptRouter


class MaterialChecker:
    def __init__(
        self,
        settings: APISettings,
        router: PromptRouter,
    ) -> None:
        self.settings = settings
        self.router = router
        self.runtime = settings.load_runtime_config()
        options = make_preprocess_options(self.runtime)
        options = type(options)(
            **{**options.__dict__, "show_progress": False}
        )
        self.preprocessor = DocumentPreprocessor(options)
        self.client = VLLMClient(
            self.runtime,
            load_schema(settings.output_schema_file),
        )
        self.client.system_prompt = router.system_prompt

    async def start(self) -> list[str]:
        models = await self.client.check_server()
        if self.runtime.model not in models:
            raise RuntimeError(
                f"vLLM 未暴露配置模型 {self.runtime.model!r}；当前模型：{models}"
            )
        return models

    async def close(self) -> None:
        await self.client.close()

    def render_region(
        self,
        source: Path,
        page_number: int,
        bbox: tuple[int, int, int, int],
    ) -> bytes:
        """从预处理后的原始页面按模型的 0—1000 坐标裁剪可展示区域。"""
        document = self.preprocessor.prepare(source, self.settings.evidence_root)
        if page_number < 1 or page_number > len(document.pages):
            raise ValueError(f"页码超出范围：{page_number}/{len(document.pages)}")
        page = document.pages[page_number - 1]
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("区域坐标必须满足 0<=x1<x2<=1000 且 0<=y1<y2<=1000")
        with Image.open(page.image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            left = int(width * x1 / 1000)
            top = int(height * y1 / 1000)
            right = max(left + 1, int(width * x2 / 1000))
            bottom = max(top + 1, int(height * y2 / 1000))
            padding_x = max(8, int((right - left) * 0.12))
            padding_y = max(8, int((bottom - top) * 0.12))
            crop = image.crop(
                (
                    max(0, left - padding_x),
                    max(0, top - padding_y),
                    min(width, right + padding_x),
                    min(height, bottom + padding_y),
                )
            )
            output = io.BytesIO()
            crop.save(output, format="PNG", optimize=True)
            return output.getvalue()

    async def check(
        self,
        context: dict[str, Any],
        route: PromptRoute,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source = Path(context["evidence"]["file_path"])
        if not source.is_file():
            raise FileNotFoundError(f"材料文件不存在：{source}")
        document = await asyncio.to_thread(
            self.preprocessor.prepare,
            source,
            self.settings.evidence_root,
        )
        prompt_context = copy.deepcopy(context)
        prompt_context["evidence"].pop("file_path", None)
        metrics = ProcessingMetrics()
        budget = RequestBudget(self.runtime.max_inference_attempts_per_document)
        page_chunks = [
            list(document.pages[index : index + self.runtime.pages_per_request])
            for index in range(0, len(document.pages), self.runtime.pages_per_request)
        ]

        async def analyze(
            pages: list[PageImage],
            token_limit: int,
        ) -> list[tuple[list[PageImage], ModelResponse]]:
            prompt = self.router.build_material_prompt(
                route,
                prompt_context,
                [page.page_number for page in pages],
            )
            try:
                response = await self.client.analyze_pages(
                    pages,
                    prompt,
                    max_tokens=token_limit,
                    budget=budget,
                )
                return [(pages, response)]
            except ModelOutputError:
                if len(pages) > 1:
                    groups = await asyncio.gather(
                        *(analyze([page], self.runtime.max_tokens) for page in pages)
                    )
                    return [item for group in groups for item in group]
                if token_limit < self.runtime.retry_max_tokens:
                    return await analyze(pages, self.runtime.retry_max_tokens)
                raise

        groups = await asyncio.gather(
            *(analyze(chunk, self.runtime.max_tokens) for chunk in page_chunks)
        )
        analyzed = [item for group in groups for item in group]
        responses = [item[1] for item in analyzed]
        for response in responses:
            metrics.add(response)
        results = [response.parsed for response in responses]
        if len(results) == 1:
            final = results[0]
        else:
            merge_prompt = self.router.build_merge_prompt(
                route,
                prompt_context,
                results,
            )
            try:
                merged = await self.client.synthesize(merge_prompt, budget=budget)
            except ModelOutputError:
                merged = await self.client.synthesize(
                    merge_prompt,
                    max_tokens=self.runtime.retry_max_tokens,
                    budget=budget,
                )
            metrics.add(merged)
            final = merged.parsed
        output_metrics = {
            **metrics.as_dict(),
            "page_count": len(document.pages),
            "preprocess_archive_hit": document.from_cache,
            "template_version": self.router.version,
            "route_id": route.route_id,
            "inference_requests": budget.used,
        }
        return final, output_metrics
