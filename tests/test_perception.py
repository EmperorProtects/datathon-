import numpy as np
import pytest

from pipeline.perception.pose import L_ELBOW, L_WRIST, hand_point
from pipeline.perception.privacy import blur_heads, head_box
from pipeline.types import Person


def person(box=(100, 100, 300, 500), head=True, conf=0.9):
    k = np.zeros((17, 3), dtype=np.float32)
    if head:
        k[0] = (200, 150, conf)   # нос
        k[1] = (190, 140, conf)   # глаза
        k[2] = (210, 140, conf)
        k[3] = (180, 145, conf)   # уши
        k[4] = (220, 145, conf)
    k[5] = (150, 200, 0.9)        # плечи
    k[6] = (250, 200, 0.9)
    return Person(conf=0.9, box=box, keypoints=k)


def test_hand_extrapolated_from_forearm():
    k = np.zeros((17, 3), dtype=np.float32)
    k[L_ELBOW] = (100, 100, 0.9)
    k[L_WRIST] = (100, 200, 0.9)
    assert hand_point(k, L_ELBOW, L_WRIST, 0.3) == pytest.approx((100, 230))


def test_hand_without_elbow_uses_wrist_and_invisible_wrist_is_none():
    k = np.zeros((17, 3), dtype=np.float32)
    k[L_WRIST] = (50, 60, 0.9)
    assert hand_point(k, L_ELBOW, L_WRIST, 0.3) == (50, 60)
    k[L_WRIST, 2] = 0.1
    assert hand_point(k, L_ELBOW, L_WRIST, 0.3) is None


def test_head_box_covers_head_points_with_margin():
    x1, y1, x2, y2 = head_box(person())
    assert x1 < 180 and x2 > 220 and y1 < 140 and y2 > 150
    # голова не меньше доли ширины плеч
    assert x2 - x1 >= 0.35 * 200


def test_head_box_fallback_is_top_of_person():
    assert head_box(person(head=False)) == pytest.approx((100, 100, 300, 100 + 0.35 * 400))


def test_blur_changes_only_head_region():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (600, 400, 3), dtype=np.uint8)
    out = blur_heads(img, [person()])
    x1, y1, x2, y2 = map(int, head_box(person()))
    assert out.shape == img.shape
    assert not np.array_equal(out[y1 + 2:y2 - 2, x1 + 2:x2 - 2], img[y1 + 2:y2 - 2, x1 + 2:x2 - 2])
    # высокочастотная текстура в области головы уничтожена
    assert out[y1 + 2:y2 - 2, x1 + 2:x2 - 2].std() < img.std() / 3
    assert np.array_equal(out[550:, :], img[550:, :])


def test_blur_handles_boxes_outside_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = blur_heads(img, [person(box=(-50, -50, 20, 20))])
    assert out.shape == img.shape
