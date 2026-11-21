from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_low_confidence_matches: bool = False
    model: str = "qwen3-vl-32b"


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(default="", max_length=100)
    graduate_ids: list[int] = Field(default_factory=list)
    evidence_file_ids: list[int] = Field(default_factory=list)
    force_recheck: bool = False
    options: JobOptions = Field(default_factory=JobOptions)

    @model_validator(mode="after")
    def validate_selection(self) -> "CreateJobRequest":
        self.task_name = self.task_name.strip()
        self.graduate_ids = sorted(set(self.graduate_ids))
        self.evidence_file_ids = sorted(set(self.evidence_file_ids))
        if not self.graduate_ids and not self.evidence_file_ids:
            raise ValueError("graduate_ids 与 evidence_file_ids 不能同时为空")
        if any(value <= 0 for value in self.graduate_ids + self.evidence_file_ids):
            raise ValueError("ID 必须是正整数")
        return self
