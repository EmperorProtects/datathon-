"""Общие утилиты подготовки данных: пути, имена видео, разбор xlsx организаторов."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "open"
VIDEOS_DIR = RAW_DIR / "videos"
LABELS_XLSX = RAW_DIR / "open_labels.xlsx"
FORTE_DIR = ROOT / "data" / "annotations" / "forte_v5" / "forte_annotations_v5"
UNIFIED_PATH = ROOT / "data" / "annotations" / "unified.json"
PROBE_PATH = ROOT / "data" / "annotations" / "probe.json"
REPORTS_DIR = ROOT / "eval" / "reports"

# В именах файлов организаторов смешаны латинская X и кириллическая Х.
_CYR_X = str.maketrans({"Х": "X", "х": "X"})


def normalize_video_id(name: str) -> str:
    return Path(name).stem.strip().translate(_CYR_X)


def video_sort_key(video_id: str) -> tuple[str, int]:
    prefix, _, num = video_id.partition("-")
    return prefix, int(num) if num.isdigit() else 0


def find_videos(videos_dir: Path = VIDEOS_DIR) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in videos_dir.glob("*.mov"):
        vid = normalize_video_id(p.name)
        if vid in out:
            raise ValueError(f"two files map to {vid}: {out[vid].name}, {p.name}")
        out[vid] = p
    return dict(sorted(out.items(), key=lambda kv: video_sort_key(kv[0])))


def rel_path(path: Path) -> str:
    """Путь относительно корня репозитория (POSIX), если файл внутри него."""
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_TIME_RE = re.compile(r"^\s*(?:(\d+)\s*мин\w*)?\s*(?:(\d+)\s*сек\w*)?\s*$", re.IGNORECASE)


def parse_organizer_time(raw: str | None) -> float | None:
    """'22 секунда' → 22.0; '1минута 8 секунда' → 68.0; пусто → None."""
    if raw is None or not str(raw).strip():
        return None
    m = _TIME_RE.match(str(raw))
    if not m or not (m.group(1) or m.group(2)):
        raise ValueError(f"unparseable organizer timecode: {raw!r}")
    return int(m.group(1) or 0) * 60.0 + int(m.group(2) or 0)


@dataclass(frozen=True)
class LabelRow:
    excel_row: int
    raw_name: str
    video_id: str
    raw_time: str | None
    time_sec: float | None
    raw_label: str
    has_violation: bool


def read_labels(path: Path = LABELS_XLSX) -> list[LabelRow]:
    import openpyxl

    ws = openpyxl.load_workbook(path, data_only=True, read_only=True).active
    rows: list[LabelRow] = []
    for i, (name, raw_time, raw_label) in enumerate(ws.iter_rows(min_row=2, max_col=3, values_only=True), start=2):
        if name is None:
            continue
        label = str(raw_label).strip().lower()
        if label == "нарушение":
            viol = True
        elif label == "без нарушения":
            viol = False
        else:
            raise ValueError(f"row {i}: unknown label {raw_label!r}")
        t = parse_organizer_time(raw_time)
        if viol != (t is not None):
            raise ValueError(f"row {i}: label {raw_label!r} inconsistent with timecode {raw_time!r}")
        rows.append(LabelRow(i, str(name), normalize_video_id(str(name)), raw_time, t, str(raw_label), viol))
    return rows


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
