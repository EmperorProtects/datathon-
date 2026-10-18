"""Схема ответа API (ТЗ, раздел 10).

Обязательные поля нарушения: type, timecode_sec, description, evidence_frames.
Дополнительные поля (rule, confidence, source) допускаются как расширение и
убираются при STRICT_SCHEMA=true.
"""
from __future__ import annotations

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JobStatus = Literal["processing", "done", "failed"]
Verdict = Literal["violation", "no_violation"]


class EvidenceFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=4)

    @field_validator("image_base64")
    @classmethod
    def _jpeg_base64(cls, v: str) -> str:
        if v.startswith("data:"):
            raise ValueError("image_base64 must not have a data: prefix")
        try:
            head = base64.b64decode(v[:16] + "=" * (-len(v[:16]) % 4), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError("image_base64 is not valid base64") from e
        if not head.startswith(b"\xff\xd8"):
            raise ValueError("image_base64 must be a JPEG")
        return v


class Violation(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class AnalysisResult(BaseModel):
    """То, что возвращает pipeline.run.analyze()."""

    verdict: Verdict
    violations: list[Violation]

    @model_validator(mode="after")
    def _verdict_consistent(self) -> AnalysisResult:
        if (self.verdict == "no_violation") != (len(self.violations) == 0):
            raise ValueError("verdict must be 'no_violation' iff violations is empty")
        return self


class AnalyzeResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Verdict | None = None
    violations: list[Violation] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _status_consistent(self) -> JobResponse:
        if self.status == "done":
            if self.verdict is None or self.violations is None:
                raise ValueError("done job must have verdict and violations")
            AnalysisResult(verdict=self.verdict, violations=self.violations)
        if self.status == "failed" and not self.error:
            raise ValueError("failed job must have error")
        return self


CORE_VIOLATION_FIELDS = set(Violation.model_fields)


def to_public(resp: JobResponse, strict: bool) -> dict:
    """Сериализация ответа; в строгом режиме — только поля из ТЗ."""
    data = resp.model_dump(exclude_none=True)
    if strict and "violations" in data:
        data["violations"] = [
            {k: v for k, v in viol.items() if k in CORE_VIOLATION_FIELDS} for viol in data["violations"]
        ]
    return data
