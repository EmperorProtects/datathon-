"""Реестр типов нарушений и нормативов (rules/registry.yaml).

Тип без нормативного источника не загружается: такое нарушение нельзя сообщить пользователю.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "rules" / "registry.yaml"

Relationship = Literal["direct", "conditional", "context_only", "exception"]


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str
    short_title: str
    title: str
    official_url: str = Field(pattern=r"^https://adilet\.zan\.kz/")
    verified_official_text: bool = False


class Rule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str
    source_id: str
    clause: str = Field(min_length=1)
    summary_ru: str = Field(min_length=1)
    applicability_limits_ru: str = ""
    verified_official_text: bool = False


class RuleRef(BaseModel):
    rule_id: str
    relationship: Relationship
    application_reason_ru: str = ""


class ViolationType(BaseModel):
    model_config = ConfigDict(extra="allow")

    type_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    name_ru: str = Field(min_length=1)
    kind: Literal["action", "omission"]
    rule_refs: list[RuleRef] = []
    primary_rule_id: str | None = None


class Source(BaseModel):
    """Нормативная ссылка, которая уходит в ответ API."""

    act: str
    clause: str
    url: str
    verified_official_text: bool


class Registry(BaseModel):
    schema_version: int
    source_documents: list[SourceDocument]
    rules: list[Rule]
    types: list[ViolationType]
    # Типы без нормативного источника: есть в разметке, но не сообщаются.
    skipped: list[ViolationType] = []

    @model_validator(mode="after")
    def _check_refs(self) -> Registry:
        docs = {d.source_id for d in self.source_documents}
        rules = {r.rule_id: r for r in self.rules}
        for r in self.rules:
            if r.source_id not in docs:
                raise ValueError(f"rule {r.rule_id}: unknown source_id {r.source_id}")
        seen: set[str] = set()
        for t in self.types + self.skipped:
            if t.type_id in seen:
                raise ValueError(f"duplicate type_id {t.type_id}")
            seen.add(t.type_id)
            for ref in t.rule_refs:
                if ref.rule_id not in rules:
                    raise ValueError(f"type {t.type_id}: unknown rule {ref.rule_id}")
        for t in self.types:
            if t.primary_rule_id is None:
                raise ValueError(f"type {t.type_id}: no normative source (primary_rule_id)")
            primary = [r for r in t.rule_refs if r.rule_id == t.primary_rule_id]
            if not primary or primary[0].relationship == "exception":
                raise ValueError(f"type {t.type_id}: primary_rule_id must be a non-exception rule_ref")
        return self

    def type_ids(self, include_skipped: bool = False) -> set[str]:
        ids = {t.type_id for t in self.types}
        if include_skipped:
            ids |= {t.type_id for t in self.skipped}
        return ids

    def get_type(self, type_id: str) -> ViolationType:
        for t in self.types:
            if t.type_id == type_id:
                return t
        raise KeyError(type_id)

    def source(self, type_id: str) -> Source:
        t = self.get_type(type_id)
        rule = next(r for r in self.rules if r.rule_id == t.primary_rule_id)
        doc = next(d for d in self.source_documents if d.source_id == rule.source_id)
        return Source(
            act=doc.short_title,
            clause=rule.clause,
            url=doc.official_url,
            verified_official_text=doc.verified_official_text and rule.verified_official_text,
        )


def load_registry(path: str | Path = REGISTRY_PATH) -> Registry:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    types, skipped = [], []
    for t in raw.get("types", []):
        (types if t.get("primary_rule_id") else skipped).append(t)
    return Registry(
        schema_version=raw["schema_version"],
        source_documents=raw["source_documents"],
        rules=raw["rules"],
        types=types,
        skipped=skipped,
    )


@lru_cache
def get_registry() -> Registry:
    return load_registry()
