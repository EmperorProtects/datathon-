import numpy as np
import pytest
import yaml

from pipeline.scene import analyze_scene
from pipeline.scene.anchors import compute_anchors
from pipeline.scene.zones import build_scene, side_of
from pipeline.types import Detection, FramePerception, Person
from training.data.common import ROOT

with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FPS = 10
WH = (464, 848)
COUNTER = (40, 365, 320, 470)   # центр по y = 417.5
SAFE = (300, 280, 400, 400)
DOOR = (0, 0, 140, 450)
NEAR_BOX = (150, 420, 280, 640)  # центр ниже стойки
FAR_BOX = (150, 250, 280, 380)   # центр выше стойки
GLASS_BOX = (10, 150, 50, 230)   # мелкий человек за стеклом двери


def person(box, tid):
    return Person(conf=0.9, box=box, keypoints=np.zeros((17, 3), dtype=np.float32), track_id=tid)


def statics():
    return [Detection("counter", 0.9, COUNTER, 1), Detection("safe", 0.9, SAFE, 2), Detection("door", 0.9, DOOR, 3)]


def make_frames(spec, n=200, with_objects=True, extra_objects=None):
    """spec: {track_id: (box, t_from, t_to)} в кадрах."""
    frames = []
    for i in range(n):
        ps = [person(box, tid) for tid, (box, a, b) in spec.items() if a <= i < b]
        objs = statics() if with_objects else []
        if extra_objects:
            objs = objs + extra_objects(i)
        frames.append(FramePerception(frame_idx=i * 3, t_sec=i / FPS, persons=ps, objects=objs))
    return frames


def test_anchors_are_median_boxes_with_presence():
    frames = make_frames({})
    frames[0] = FramePerception(0, 0.0, [], [Detection("counter", 0.9, (0, 0, 10, 10))])  # выброс
    a = compute_anchors(frames, CFG)
    assert a["counter"].box == COUNTER
    assert a["counter"].presence == 1.0
    assert "laptop" not in a


def test_side_of_with_dead_band():
    assert side_of(NEAR_BOX, COUNTER, 0.1) == "near"
    assert side_of(FAR_BOX, COUNTER, 0.1) == "far"
    assert side_of((100, 400, 200, 435), COUNTER, 0.1) is None


def test_cashier_near_side_client_far():
    scene, tracks = analyze_scene(make_frames({1: (NEAR_BOX, 0, 200), 2: (FAR_BOX, 50, 120)}), CFG, FPS, WH)
    assert scene.staff_side == "near"
    assert tracks[1].role == "cashier" and tracks[2].role == "client"
    assert scene.zones["staff_side"][1] == pytest.approx(417.5)


def test_staff_side_flips_with_cashier_position():
    scene, tracks = analyze_scene(make_frames({1: (FAR_BOX, 0, 200), 2: (NEAR_BOX, 30, 90)}), CFG, FPS, WH)
    assert scene.staff_side == "far"
    assert tracks[1].role == "cashier" and tracks[2].role == "client"


def test_fragmented_cashier_track_is_merged():
    spec = {1: (NEAR_BOX, 0, 120), 5: (NEAR_BOX, 125, 200), 2: (FAR_BOX, 20, 60)}
    _, tracks = analyze_scene(make_frames(spec), CFG, FPS, WH)
    assert tracks[1].role == tracks[5].role == "cashier"


def test_simultaneous_stranger_on_staff_side_is_other():
    spec = {1: (NEAR_BOX, 0, 200), 7: ((300, 430, 380, 620), 40, 140)}
    _, tracks = analyze_scene(make_frames(spec), CFG, FPS, WH)
    assert tracks[7].role == "other"


def test_person_behind_glass_door_is_outside():
    spec = {1: (NEAR_BOX, 0, 200), 9: (GLASS_BOX, 0, 200)}
    _, tracks = analyze_scene(make_frames(spec), CFG, FPS, WH)
    assert tracks[9].outside_room and tracks[9].role == "other"
    assert tracks[1].role == "cashier"


def test_short_track_is_noise():
    _, tracks = analyze_scene(make_frames({1: (NEAR_BOX, 0, 200), 3: (FAR_BOX, 10, 14)}), CFG, FPS, WH)
    assert tracks[3].role == "other"


def test_collector_by_bag_at_body():
    bag = lambda i: [Detection("collector_bag", 0.8, (300, 450, 360, 520))] if 40 <= i < 60 else []
    spec = {1: (FAR_BOX, 0, 200), 4: ((290, 420, 380, 640), 30, 80)}
    _, tracks = analyze_scene(make_frames(spec, extra_objects=bag), CFG, FPS, WH)
    assert tracks[4].role == "collector"


def test_missing_counter_flags_scene_and_does_not_crash():
    scene, tracks = analyze_scene(make_frames({1: (NEAR_BOX, 0, 200), 2: (FAR_BOX, 0, 50)}, with_objects=False),
                                  CFG, FPS, WH)
    assert scene.scene_confidence == 0.0
    assert any("missing_anchors" in f for f in scene.flags)
    assert scene.staff_side is None and "staff_side" not in scene.zones
    assert tracks[1].role == "cashier"


def test_empty_video():
    scene, tracks = analyze_scene([], CFG, FPS, WH)
    assert tracks == {} and scene.scene_confidence == 0.0


def test_build_scene_zones_relative_to_anchors():
    scene = build_scene(compute_anchors(make_frames({}), CFG), WH, CFG, staff_side="far")
    assert scene.zones["staff_side"][3] == pytest.approx(417.5)       # дальняя сторона — выше центра стойки
    assert scene.zones["client_side"][3] == WH[1]
    assert scene.scene_confidence == pytest.approx(3 / 5)


def test_full_size_person_far_behind_glass_is_outside():
    # рост как у людей в комнате, но ноги выше верха стойки (y1=365): стоит в коридоре за стеклом
    spec = {1: (NEAR_BOX, 0, 200), 2: (FAR_BOX, 0, 200), 9: ((20, 120, 100, 300), 0, 200)}
    _, tracks = analyze_scene(make_frames(spec), CFG, FPS, WH)
    assert tracks[9].outside_room


def test_person_entering_through_door_is_not_outside():
    # заходит через дверь: в зоне двери, но ноги на уровне пола комнаты
    spec = {1: (NEAR_BOX, 0, 200), 6: ((30, 180, 130, 380), 0, 200)}
    _, tracks = analyze_scene(make_frames(spec), CFG, FPS, WH)
    assert not tracks[6].outside_room


def test_duplicate_box_on_cashier_is_not_stranger():
    dup = (NEAR_BOX[0] + 5, NEAR_BOX[1] + 10, NEAR_BOX[2] - 5, NEAR_BOX[3])
    _, tracks = analyze_scene(make_frames({1: (NEAR_BOX, 0, 200), 8: (dup, 20, 80)}), CFG, FPS, WH)
    assert tracks[8].role == "cashier" and "дубль" in tracks[8].reason


def test_short_partial_box_touching_cashier_is_cashier_part():
    arm = (NEAR_BOX[2] - 15, NEAR_BOX[1] + 20, NEAR_BOX[2] + 15, NEAR_BOX[1] + 120)
    _, tracks = analyze_scene(make_frames({1: (NEAR_BOX, 0, 200), 7: (arm, 50, 63)}), CFG, FPS, WH)
    assert tracks[7].role == "cashier"
