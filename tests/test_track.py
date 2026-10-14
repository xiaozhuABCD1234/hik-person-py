# SPDX-License-Identifier: AGPL-3.0-or-later
"""3D 关键点卡尔曼滤波 + IoU 跟踪的单元测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.pose import NUM_KEYPOINTS
from hik_person.track import Pose3DTracker, ScalarKalman, iou

CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def points(offset: float = 0.0, depth: float = 2.0) -> np.ndarray:
    """17 个假关键点：X 依次拉开，Z（深度）固定。"""
    pts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    pts[:, 0] = np.arange(NUM_KEYPOINTS) * 0.01
    pts[:, 1] = 0.5
    pts[:, 2] = depth
    pts[:, 0] += offset
    return pts


def empty_batch() -> tuple[list, np.ndarray, np.ndarray]:
    return [], np.zeros((0, NUM_KEYPOINTS, 3), np.float32), np.zeros((0, NUM_KEYPOINTS), np.float32)


# ------------------------------------------------------------------------ IoU
def test_iou_identical_is_one_and_disjoint_is_zero() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --------------------------------------------------------------------- Kalman
def test_kalman_holds_a_constant_signal() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.01)
    kf.init(5.0)
    for _ in range(10):
        kf.predict(0.1)
        kf.update(5.0)
    assert kf.position == pytest.approx(5.0, abs=1e-3)


def test_kalman_follows_a_ramp_with_lag() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.05)
    kf.init(0.0)
    for step in range(1, 30):
        kf.predict(1.0)
        kf.update(float(step))
    # 跟得上但不会瞬间跳到位
    assert 25.0 < kf.position < 29.5


def test_kalman_predict_moves_along_velocity() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.01)
    kf.init(0.0)
    kf.predict(1.0)
    kf.update(1.0)  # 学到速度
    before = kf.position
    kf.predict(1.0)  # 纯预测
    assert kf.position > before


# -------------------------------------------------------------------- Tracker
def test_tracker_keeps_one_track_for_overlapping_boxes() -> None:
    tracker = Pose3DTracker(process_noise=1.0, measurement_noise=0.01)
    first = tracker.update([(0, 0, 100, 200)], points()[None], CONF[None], threshold=0.5, t=0.0)
    assert len(first) == 1
    assert len(first[0][0]) == NUM_KEYPOINTS

    second = tracker.update(
        [(2, 0, 102, 200)], points(0.05)[None], CONF[None], threshold=0.5, t=0.1
    )
    assert len(tracker.tracks) == 1
    # 观测从 0 跳到 0.05，滤波后不该一步到位（被平滑了）
    assert 0.0 < second[0][0][0, 0] < 0.05


def test_tracker_creates_a_new_track_for_a_far_box() -> None:
    tracker = Pose3DTracker(process_noise=1.0, measurement_noise=0.01)
    tracker.update([(0, 0, 100, 200)], points()[None], CONF[None], threshold=0.5, t=0.0)
    tracker.update([(500, 500, 600, 700)], points(1.0)[None], CONF[None], threshold=0.5, t=0.1)
    assert len(tracker.tracks) == 2


def test_tracker_drops_stale_tracks() -> None:
    tracker = Pose3DTracker(process_noise=1.0, measurement_noise=0.01, max_missed=2)
    tracker.update([(0, 0, 100, 200)], points()[None], CONF[None], threshold=0.5, t=0.0)
    for step in range(1, 4):
        boxes, pts, conf = empty_batch()
        tracker.update(boxes, pts, conf, threshold=0.5, t=0.1 * step)
    assert tracker.tracks == []


def test_tracker_predicts_for_a_few_frames_without_detections() -> None:
    tracker = Pose3DTracker(process_noise=1.0, measurement_noise=0.01, max_missed=5)
    tracker.update([(0, 0, 100, 200)], points()[None], CONF[None], threshold=0.5, t=0.0)
    boxes, pts, conf = empty_batch()
    items = tracker.update(boxes, pts, conf, threshold=0.5, t=0.1)
    assert len(items) == 1  # 还没删，先用预测顶着


def test_tracker_hides_keypoints_never_seen() -> None:
    tracker = Pose3DTracker(process_noise=1.0, measurement_noise=0.01)
    pts = points()
    pts[5] = np.nan  # 左肩没有有效深度
    items = tracker.update([(0, 0, 100, 200)], pts[None], CONF[None], threshold=0.5, t=0.0)
    _, names, _ = items[0]
    assert "left_shoulder" not in names
    assert len(names) == NUM_KEYPOINTS - 1
