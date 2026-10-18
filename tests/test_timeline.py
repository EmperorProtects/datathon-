import numpy as np
import pytest
import yaml

from pipeline.scene import analyze_scene
from pipeline.timeline import build_timeline
from pipeline.timeline.events import mask_to_intervals
from pipeline.types import Detection, FramePerception, Person
from training.data.common import ROOT

with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FPS = 10
WH = (464, 848)
COUNTER = (40, 365, 320, 470)
SAFE = (300, 280, 400, 400)
DOOR = (0, 0, 140, 450)
CASHIER = (150, 420, 280, 640)       # ближняя сторона
CLIENT = (150, 250, 280, 380)        # дальняя сторона
AWAY = (380, 600, 460, 840)          # кассир ушёл от стола (вне зон стойки/сейфа)


def person(box, tid, hands=(None, None)):
    k = np.zeros((17, 3), dtype=np.float32)
    k[7], k[9] = (box[0] + 10, box[1] + 40, 0.9), (box[0] + 10, box[1] + 70, 0.9)  # левый локоть/запястье
    k[8], k[10] = (box[2] - 10, box[1] + 40, 0.9), (box[2] - 10, box[1] + 70, 0.9)
    return Person(conf=0.9, box=box, keypoints=k, track_id=tid, hands=hands)


def statics():
    return [Detection("counter", 0.9, COUNTER, 1), Detection("safe", 0.9, SAFE, 2), Detection("door", 0.9, DOOR, 3)]


def run(build_frame, n=100):
    frames = [FramePerception(i * 3, i / FPS, *build_frame(i)) for i in range(n)]
    scene, tracks = analyze_scene(frames, CFG, FPS, WH)
    return build_timeline(frames, scene, tracks, CFG, FPS), tracks


def of(events, etype, **kw):
    return [e for e in events if e.type == etype and all(e.args.get(k) == v for k, v in kw.items())]


def test_mask_to_intervals_merges_gaps_and_drops_short():
    m = np.zeros(50, dtype=bool)
    m[5:15] = True
    m[17:25] = True      # разрыв 2 кадра < gap 0.5 с → склейка
    m[40:42] = True      # 0.2 с < min 0.3 с → отброшено
    assert mask_to_intervals(m, 10, 0.3, 0.5) == [(5, 24)]


def test_cash_left_unattended_after_cashier_leaves():
    cash = Detection("cash_bundle", 0.8, (180, 400, 220, 430), 50)

    def frame(i):
        cashier_box = CASHIER if i < 50 else AWAY
        return [person(cashier_box, 1)], statics() + [cash]

    events, _ = run(frame)
    left = of(events, "cashier_left_zone")
    assert left and left[0].t_start == pytest.approx(5.0)
    unattended = of(events, "object_unattended", cls="cash_bundle")
    assert unattended and unattended[0].t_start == pytest.approx(5.0)
    assert of(events, "cashier_left") and not of(events, "cashier_returned")  # конец видео — не возвращение


def test_hold_cash_and_handover_direction():
    cash_box = (200, 380, 240, 410)

    def frame(i):
        c_hands = ((210, 395), None) if i < 60 else (None, None)
        k_hands = ((220, 400), None) if 40 <= i else (None, None)
        objs = statics() + [Detection("cash", 0.8, cash_box, 60)]
        return [person(CASHIER, 1, c_hands), person(CLIENT, 2, k_hands)], objs

    events, tracks = run(frame)
    assert tracks[1].role == "cashier" and tracks[2].role == "client"
    assert of(events, "hold_cash") and {e.actor for e in of(events, "hold_cash")} == {1, 2}
    ho = of(events, "cash_handover")
    assert ho and ho[0].args["from_role"] == "cashier" and ho[0].args["to_role"] == "client"
    assert ho[0].t_start == pytest.approx(4.0)


def test_multiple_clients_and_client_edges():
    def frame(i):
        ps = [person(CASHIER, 1)]
        if 20 <= i < 80:
            ps.append(person(CLIENT, 2))
        if 40 <= i < 60:
            ps.append(person((60, 250, 140, 380), 3))
        return ps, statics()

    events, _ = run(frame)
    mc = of(events, "multiple_clients")
    assert mc and mc[0].t_start == pytest.approx(4.0) and mc[0].t_end == pytest.approx(5.9)
    assert of(events, "client_arrived")[0].t_start == pytest.approx(2.0)
    assert of(events, "client_left_window")[0].t_start == pytest.approx(7.9)


def test_cash_disappeared_into_body_zone():
    def frame(i):
        k = np.zeros((17, 3), dtype=np.float32)
        k[5], k[6], k[11], k[12] = (170, 450, .9), (260, 450, .9), (175, 560, .9), (255, 560, .9)
        p = Person(0.9, CASHIER, k, 1, ((215, 500), None))
        objs = statics() + ([Detection("cash", 0.8, (200, 480, 235, 520), 70)] if i < 50 else [])
        return [p], objs

    events, _ = run(frame)
    d = of(events, "cash_disappeared")
    assert d and d[0].args["last_zone"] == "body" and d[0].actor == 1
    assert of(events, "cash_near_body")


def test_recount_visible_requires_hand_motion_over_cash():
    cash_box = (180, 400, 260, 440)

    def frame(i):
        x = 200 + (25 if i % 2 else 0) if i >= 30 else 200   # кисть «перебирает» купюры с 3 с
        return [person(CASHIER, 1, ((x, 420), None))], statics() + [Detection("cash", 0.8, cash_box, 80)]

    events, _ = run(frame)
    rc = of(events, "recount_visible")
    assert rc and rc[0].t_start >= 2.5
    assert of(events, "hold_cash")[0].t_start == pytest.approx(0.0)


def test_no_anchors_no_left_zone_events():
    events, _ = run(lambda i: ([person(CASHIER, 1)], []))
    assert not of(events, "cashier_left_zone")


def test_people_behind_glass_do_not_trigger_events():
    glass = (20, 120, 100, 300)  # ноги выше верха стойки, внутри зоны двери
    events, tracks = run(lambda i: ([person(CASHIER, 1), person(glass, 9)], statics()))
    assert tracks[9].outside_room
    assert not of(events, "client_at_window") and not of(events, "outsider_in_staff_zone")
