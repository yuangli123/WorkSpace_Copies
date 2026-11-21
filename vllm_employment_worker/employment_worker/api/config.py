from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import RuntimeConfig


@dataclass
class APISettings:
    source_database: Path
    jobs_database: Path
    evidence_root: Path
    archive_dir: Path
    runtime_config: Path
    prompt_routes_file: Path
    output_schema_file: Path
    system_prompt_file: Path
    max_parallel_items: int = 8

    @classmethod
    def from_file(cls, path: Path) -> "APISettings":
        path = path.resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("API 配置必须是 JSON 对象")
        base = path.parent

        def resolve(name: str) -> Path:
            raw = Path(str(value[name]))
            return raw if raw.is_absolute() else (base / raw).resolve()

        settings = cls(
            source_database=resolve("source_database"),
            jobs_database=resolve("jobs_database"),
            evidence_root=resolve("evidence_root"),
            archive_dir=resolve("archive_dir"),
            runtime_config=resolve("runtime_config"),
            prompt_routes_file=resolve("prompt_routes_file"),
            output_schema_file=resolve("output_schema_file"),
            system_prompt_file=resolve("system_prompt_file"),
            max_parallel_items=int(value.get("max_parallel_items", 8)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.source_database.is_file():
            raise FileNotFoundError(f"业务数据库不存在：{self.source_database}")
        if not self.evidence_root.is_dir():
            raise FileNotFoundError(f"材料根目录不存在：{self.evidence_root}")
        for path in (
            self.runtime_config,
            self.prompt_routes_file,
            self.output_schema_file,
            self.system_prompt_file,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"配置文件不存在：{path}")
        if self.max_parallel_items < 1:
            raise ValueError("max_parallel_items 必须大于0")
        self.jobs_database.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def load_runtime_config(self) -> RuntimeConfig:
        runtime = RuntimeConfig.from_file(self.runtime_config)
        runtime.cache_dir = str(self.archive_dir)
        runtime.validate()
        return runtime
