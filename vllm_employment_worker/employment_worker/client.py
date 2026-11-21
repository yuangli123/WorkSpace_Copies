from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .schema import extract_json, response_format, validate_analysis
from .types import ModelResponse, PageImage


class ModelOutputError(RuntimeError):
    """请求已完成，但模型回答被截断或不符合结构化输出约束。"""

    def __init__(
        self,
        message: str,
        raw_text: str,
        finish_reason: str,
        parse_error: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.finish_reason = finish_reason
        self.parse_error = parse_error


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    """并发安全地限制一份材料触发的实际 HTTP 推理次数。"""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0
        self._lock = asyncio.Lock()

    async def consume(self) -> None:
        async with self._lock:
            if self.used >= self.maximum:
                raise RequestBudgetExceeded(
                    f"单份文件推理次数已达到上限 {self.maximum}"
                )
            self.used += 1


class VLLMClient:
    """调用 vLLM OpenAI 兼容接口，并由信号量控制连续批处理并发。"""

    def __init__(self, config: RuntimeConfig, schema: dict[str, Any]) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 包，请先安装 requirements.txt") from exc
        self.config = config
        self.schema = schema
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    async def close(self) -> None:
        await self._client.close()

    async def check_server(self) -> list[str]:
        models = await self._client.models.list()
        return [str(item.id) for item in models.data]

    async def analyze_pages(
        self,
        pages: list[PageImage],
        prompt: str,
        max_tokens: int | None = None,
        budget: RequestBudget | None = None,
    ) -> ModelResponse:
        image_paths = await asyncio.gather(
            *(asyncio.to_thread(self._prepare_inference_image, page.image_path) for page in pages)
        )
        content: list[dict[str, Any]] = []
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_url(image_path)},
                }
            )
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]
        return await self._request(
            messages,
            max_tokens or self.config.max_tokens,
            budget=budget,
        )

    async def synthesize(
        self,
        prompt: str,
        max_tokens: int | None = None,
        budget: RequestBudget | None = None,
    ) -> ModelResponse:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._request(
            messages,
            max_tokens or self.config.merge_max_tokens,
            budget=budget,
        )

    @property
    def _system_prompt(self) -> str:
        prompt = getattr(self, "system_prompt", "")
        if not prompt:
            raise RuntimeError("VLLMClient.system_prompt 尚未设置")
        return prompt

    async def _request(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        budget: RequestBudget | None = None,
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            try:
                if budget is not None:
                    await budget.consume()
                async with self._semaphore:
                    kwargs: dict[str, Any] = {
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": max_tokens,
                        "extra_body": {
                            "chat_template_kwargs": {
                                "enable_thinking": self.config.enable_thinking
                            }
                        },
                    }
                    if self.config.use_json_schema:
                        kwargs["response_format"] = response_format(self.schema)
                    completion = await self._client.chat.completions.create(**kwargs)
                raw_text = self._content_text(completion.choices[0].message.content)
                finish_reason = str(completion.choices[0].finish_reason or "unknown")
                if finish_reason == "length":
                    raise ModelOutputError(
                        message=(
                            f"模型输出达到 max_tokens={max_tokens}，JSON 被截断；"
                            f"已生成 {len(raw_text)} 个字符"
                        ),
                        raw_text=raw_text,
                        finish_reason=finish_reason,
                    )
                try:
                    parsed = extract_json(raw_text)
                    validate_analysis(parsed, self.schema)
                except Exception as exc:  # noqa: BLE001
                    raise ModelOutputError(
                        message=(
                            f"模型返回的结构化 JSON 无效；finish_reason={finish_reason}，"
                            f"字符数={len(raw_text)}，错误={exc}"
                        ),
                        raw_text=raw_text,
                        finish_reason=finish_reason,
                        parse_error=str(exc),
                    ) from exc
                usage = completion.usage
                return ModelResponse(
                    parsed=parsed,
                    raw_text=raw_text,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    latency_seconds=time.perf_counter() - started,
                )
            except ModelOutputError:
                # temperature=0 时重复同一截断/非法JSON没有意义，交给上层拆页或增大上限。
                raise
            except RequestBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int) and 400 <= status_code < 500:
                    raise RuntimeError(
                        f"vLLM 拒绝了请求（HTTP {status_code}），不再重复调用：{exc}"
                    ) from exc
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(self.config.retry_base_seconds * (2**attempt))
        assert last_error is not None
        raise RuntimeError(
            f"vLLM 请求连续失败 {self.config.max_retries + 1} 次：{last_error}"
        ) from last_error

    def _image_url(self, path: Path) -> str:
        path = path.resolve()
        if self.config.image_transport == "file":
            return path.as_uri()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

    def _prepare_inference_image(self, source: Path) -> Path:
        """生成受像素预算约束的推理副本，原始预处理存档保持不变。"""
        from PIL import Image, ImageOps

        source = source.resolve()
        with Image.open(source) as image:
            width, height = image.size
            if width * height <= self.config.inference_max_pixels:
                return source
            stat = source.stat()
            fingerprint = hashlib.sha256(
                (
                    f"{source}|{stat.st_size}|{stat.st_mtime_ns}|"
                    f"{self.config.inference_max_pixels}"
                ).encode("utf-8")
            ).hexdigest()[:20]
            output_dir = Path(self.config.cache_dir) / "_inference_media"
            output_dir.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                output_dir.chmod(0o700)
            output = output_dir / f"{fingerprint}.png"
            if output.exists():
                return output
            scale = (self.config.inference_max_pixels / (width * height)) ** 0.5
            target = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            resized = ImageOps.exif_transpose(image).convert("RGB").resize(
                target,
                Image.Resampling.LANCZOS,
            )
            temporary = output.with_suffix(".png.tmp")
            resized.save(temporary, format="PNG", optimize=True)
            temporary.replace(output)
            if os.name != "nt":
                output.chmod(0o600)
            return output

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif getattr(item, "text", None):
                    parts.append(str(item.text))
            return "".join(parts)
        return str(content or "")
