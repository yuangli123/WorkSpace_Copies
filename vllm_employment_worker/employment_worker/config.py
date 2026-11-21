from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeConfig:
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3-vl-32b"
    timeout_seconds: float = 600.0
    max_retries: int = 3
    retry_base_seconds: float = 2.0
    concurrency: int = 8
    max_inference_attempts_per_document: int = 12
    pages_per_request: int = 3
    max_tokens: int = 6144
    retry_max_tokens: int = 12288
    merge_max_tokens: int = 3072
    temperature: float = 0.0
    enable_thinking: bool = False
    image_transport: str = "file"
    inference_max_pixels: int = 1500000
    use_json_schema: bool = True
    pdf_dpi: int = 180
    max_pages_per_document: int = 0
    preprocess_workers: int = 4
    show_preprocess_progress: bool = True
    pdftoppm_bin: str = "pdftoppm"
    libreoffice_bin: str = ""
    pdf_password: str = ""
    pdf_password_file: str = ""
    cache_dir: str = "./page_cache"
    prompt_dir: str = "../prompts"
    business_rules_file: str = "./business_rules.json"
    output_schema_file: str = "../schemas/analysis_result.schema.json"
    save_raw_responses: bool = False

    @classmethod
    def from_file(cls, path: Path) -> "RuntimeConfig":
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("运行配置必须是 JSON 对象")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"运行配置包含未知字段：{', '.join(unknown)}")
        config = cls(**data)
        config.api_key = os.environ.get("VLLM_API_KEY", config.api_key)
        base = path.parent
        for name in (
            "pdf_password_file",
            "cache_dir",
            "prompt_dir",
            "business_rules_file",
            "output_schema_file",
        ):
            value = getattr(config, name)
            if value and not Path(value).is_absolute():
                setattr(config, name, str((base / value).resolve()))
        config.validate()
        return config

    def validate(self) -> None:
        if not self.api_base.endswith("/v1"):
            raise ValueError("api_base 应以 /v1 结尾")
        if self.concurrency < 1 or self.preprocess_workers < 1:
            raise ValueError("并发数必须大于 0")
        if self.max_inference_attempts_per_document < 1:
            raise ValueError("每份文件的最大推理次数必须大于 0")
        if self.pages_per_request < 1:
            raise ValueError("pages_per_request 必须大于 0")
        if (
            self.max_tokens < 256
            or self.retry_max_tokens < self.max_tokens
            or self.merge_max_tokens < 256
        ):
            raise ValueError("输出 token 上限过小")
        if self.pdf_dpi < 72:
            raise ValueError("pdf_dpi 不能小于 72")
        if self.max_pages_per_document < 0:
            raise ValueError("max_pages_per_document 不能小于 0")
        if self.image_transport not in {"file", "base64"}:
            raise ValueError("image_transport 只能是 file 或 base64")
        if self.inference_max_pixels < 250000:
            raise ValueError("inference_max_pixels 不能小于 250000")

    def safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if value.get("api_key") and value["api_key"] != "EMPTY":
            value["api_key"] = "***"
        if value.get("pdf_password"):
            value["pdf_password"] = "***"
        return value


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"文件必须包含 JSON 对象：{path}")
    return value
