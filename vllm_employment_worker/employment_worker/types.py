from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageImage:
    page_number: int
    image_path: Path


@dataclass(frozen=True)
class PreparedDocument:
    source_id: str
    source_file: Path
    relative_path: str
    source_format: str
    pages: tuple[PageImage, ...]
    from_cache: bool = False


@dataclass
class ModelResponse:
    parsed: dict[str, Any]
    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0


@dataclass
class ProcessingMetrics:
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0

    def add(self, response: ModelResponse) -> None:
        self.request_count += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.latency_seconds += response.latency_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_latency_seconds": round(self.latency_seconds, 3),
        }


@dataclass
class DocumentOutcome:
    record: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    chunk_payloads: list[dict[str, Any]] = field(default_factory=list)

