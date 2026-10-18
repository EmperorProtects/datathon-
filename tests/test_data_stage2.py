import copy

import numpy as np
import pytest

from pipeline.registry import load_registry
from pipeline.types import VideoTimebase
from training.data.adapter_organizers import build
from training.data.common import LabelRow, normalize_video_id, parse_organizer_time
from training.data.validate import validate

REGISTRY = load_registry()


# --- время -------------------------------------------------------------------

def test_timebase_uses_pts_not_average_fps():
    # пропущенный кадр между 0.0667 и 0.1333
    tb = VideoTimebase(np.array([0.0, 1 / 30, 2 / 30, 4 / 30, 5 / 30]))
    assert tb.sec_to_frame(4 / 30) == 3
    assert tb.sec_to_frame(0.12) == 3
    assert tb.sec_to_frame(-1) == 0
    assert tb.sec_to_frame(99) == 4
    assert tb.frame_to_sec(3) == pytest.approx(4 / 30)


def test_sample_indices_no_duplicates():
    tb = VideoTimebase(np.arange(0, 3, 1 / 30))
    idx = tb.sample_indices(10)
    assert idx[:3] == [0, 3, 6]
    assert len(idx) == len(set(idx)) == 30


@pytest.mark.parametrize("raw,expected", [
    ("22 секунда", 22.0), ("1минута 8 секунда", 68.0), ("2 минуты", 120.0), (None, None), ("  ", None),
])
def test_parse_organizer_time(raw, expected):
    assert parse_organizer_time(raw) == expected


def test_parse_organizer_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_organizer_time("около полуминуты")


def test_normalize_cyrillic_x():
    assert normalize_video_id("Х-14.mov") == "X-14"
    assert normalize_video_id("X-1") == "X-1"


# --- реестр ------------------------------------------------------------------

def test_registry_every_loaded_type_has_real_source():
    assert REGISTRY.types
    for t in REGISTRY.types:
        src = REGISTRY.source(t.type_id)
        assert src.act and src.clause and src.url.startswith("https://adilet.zan.kz/")


def test_registry_type_without_source_is_skipped():
    assert "BEVERAGE_CONSUMPTION_AT_WORKSTATION" in {t.type_id for t in REGISTRY.skipped}
    with pytest.raises(KeyError):
        REGISTRY.source("BEVERAGE_CONSUMPTION_AT_WORKSTATION")


def test_registry_rejects_unknown_rule():
    raw = REGISTRY.model_dump()
    raw["types"][0]["primary_rule_id"] = "NO_SUCH_RULE"
    with pytest.raises(ValueError):
        type(REGISTRY).model_validate(raw)


# --- адаптер -----------------------------------------------------------------

TYPE_A, TYPE_B = REGISTRY.types[0].type_id, REGISTRY.types[1].type_id


def _event(eid, type_id, start, peak, end, kind="action", chk=None):
    return {"event_id": eid, "kind": kind, "type_id": type_id, "organizer_time_sec": start,
            "start_sec": start, "peak_sec": peak, "end_sec": end, "checkpoint_sec": chk,
            "visibility": "full", "rule_refs": []}


def _fixture():
    labels = [
        LabelRow(2, "X-1", "X-1", "10 секунда", 10.0, "нарушение", True),
        LabelRow(3, "Х-2", "X-2", None, None, "без нарушения", False),
        LabelRow(4, "Х-9", "X-9", "30 секунда", 30.0, "нарушение", True),  # дубль X-1
    ]
    forte = {"schema_version": "5", "videos": [
        {"video_id": "X-1", "events": [_event("a", TYPE_A, 9, 10, 11), _event("b", TYPE_B, 29, 30, 31)],
         "lookalikes": [{"start_sec": 1, "end_sec": 2, "what": "x"}]},
        {"video_id": "X-2", "events": [], "lookalikes": []},
        {"video_id": "X-9", "events": [_event("c", TYPE_A, 9, 10, 11)],
         "lookalikes": [{"start_sec": 1, "end_sec": 2, "what": "x"}]},
    ]}
    probe = {vid: {"path": f"v/{vid}.mov", "avg_fps": 30.0, "duration_sec": 40.0, "width": 464, "height": 848,
                   "n_frames": 1200, "sha256": sha}
             for vid, sha in (("X-1", "aaa"), ("X-2", "bbb"), ("X-9", "aaa"))}
    tbs = {vid: VideoTimebase(np.arange(0, 40, 1 / 30)) for vid in probe}
    return labels, forte, probe, tbs


def test_adapter_merges_byte_duplicates():
    u = build(*_fixture())
    assert [v["video_id"] for v in u["videos"]] == ["X-1", "X-2"]
    x1 = u["videos"][0]
    assert x1["duplicates"] == ["X-9"]
    assert x1["organizer_times_sec"] == [10.0, 30.0]
    assert sorted(e["type_id"] for e in x1["events"]) == sorted([TYPE_A, TYPE_B])
    # каждому событию — ближайший к пику таймкод организаторов
    assert {e["type_id"]: e["organizer_time_sec"] for e in x1["events"]} == {TYPE_A: 10.0, TYPE_B: 30.0}
    assert len(x1["lookalikes"]) == 1
    assert x1["events"][0]["peak_frame"] == 300


def test_adapter_rejects_conflicting_duplicate_labels():
    labels, forte, probe, tbs = _fixture()
    labels[2] = LabelRow(4, "Х-9", "X-9", None, None, "без нарушения", False)
    with pytest.raises(ValueError, match="conflicting"):
        build(labels, forte, probe, tbs)


# --- валидатор ---------------------------------------------------------------

def test_validator_accepts_adapter_output():
    assert validate(build(*_fixture()), REGISTRY, check_files=False) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda u: u["videos"][0]["events"][0].update(start_sec=12), "start ≤ peak ≤ end"),
    (lambda u: u["videos"][0]["events"][0].update(type_id="MADE_UP"), "not in registry"),
    (lambda u: u["videos"][0]["events"][0].update(type_id="BEVERAGE_CONSUMPTION_AT_WORKSTATION"), "no normative"),
    (lambda u: u["videos"][1].update(has_violation=True), "has_violation"),
    (lambda u: u["videos"][0]["events"][0].update(kind="omission"), "checkpoint"),
    (lambda u: u["videos"][0]["events"][0].update(end_sec=99), "beyond duration"),
])
def test_validator_catches_errors(mutate, needle):
    u = copy.deepcopy(build(*_fixture()))
    mutate(u)
    errors = validate(u, REGISTRY, check_files=False)
    assert any(needle in e for e in errors), errors
