"""Черновой rules/registry.yaml из командного реестра forte_annotations_v5/rules/registry.yaml.

Нормативы переносятся как есть (решение команды): тексты пунктов не сверены с официальной
редакцией, это отражено флагом verified_official_text: false. Новых нормативов скрипт не создаёт.

Основной норматив типа (primary_rule_id): первая ссылка direct, иначе conditional, иначе context_only.
Тип без ссылок остаётся без primary_rule_id и загрузчиком не сообщается (pipeline/registry.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.data.common import FORTE_DIR, ROOT

PRIORITY = ("direct", "conditional", "context_only")

HEADER = """\
# Реестр типов нарушений и нормативов.
# Сгенерирован training/data/build_registry.py из командного черновика (forte_annotations_v5),
# затем редактируется вручную. Нормативы НЕ сверены с официальной редакцией adilet.zan.kz
# (verified_official_text: false) — перенесены как есть по решению команды.
# Тип без primary_rule_id не загружается и не сообщается в ответе API.
"""


def _dedupe_refs(refs: list[dict]) -> list[dict]:
    out, seen = [], set()
    for r in refs:
        key = (r["rule_id"], r["relationship"])
        if key not in seen:
            seen.add(key)
            out.append({k: r[k] for k in ("rule_id", "relationship", "application_reason_ru") if k in r})
    return out


def _primary(refs: list[dict]) -> str | None:
    for rel in PRIORITY:
        for r in refs:
            if r["relationship"] == rel:
                return r["rule_id"]
    return None


def build(src: dict) -> dict:
    docs = [
        {
            "source_id": d["source_id"],
            "short_title": d["short_title"],
            "title": d["title"],
            "official_url": d["official_url"],
            "verified_official_text": False,
            "verification_note": d.get("previous_verification_as_reported_in_draft", {}).get("note", ""),
        }
        for d in src["source_documents"]
    ]
    rules = [
        {
            "rule_id": r["rule_id"],
            "source_id": r["source_id"],
            "clause": r["clause"],
            "summary_ru": r["summary_ru"],
            "applicability_limits_ru": r.get("applicability_limits_ru", ""),
            "verified_official_text": False,
        }
        for r in src["rules"]
    ]
    types = []
    for t in src["types"]:
        refs = _dedupe_refs(t.get("rule_refs", []))
        types.append({
            "type_id": t["type_id"],
            "name_ru": t["name_ru"],
            "kind": t["kind_default"],
            "primary_rule_id": _primary(refs),
            "rule_refs": refs,
            "status": "draft",
        })
    return {"schema_version": 1, "source_documents": docs, "rules": rules, "types": types}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=FORTE_DIR / "rules" / "registry.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "rules" / "registry.yaml")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий (ручные правки потеряются)")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; use --force to overwrite manual edits")
    with open(args.src, encoding="utf-8") as f:
        reg = build(yaml.safe_load(f))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(HEADER)
        yaml.safe_dump(reg, f, allow_unicode=True, sort_keys=False, width=110)
    missing = [t["type_id"] for t in reg["types"] if not t["primary_rule_id"]]
    print(f"wrote {args.out}: {len(reg['types'])} types, {len(reg['rules'])} rules; without source: {missing}")


if __name__ == "__main__":
    main()
