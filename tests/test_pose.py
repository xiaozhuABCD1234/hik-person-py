# SPDX-License-Identifier: AGPL-3.0-or-later
"""姿态关键点工具的单元测试 —— 纯函数，不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.pose import (
    KEYPOINT_NAMES_17,
    NUM_KEYPOINTS,
    SKELETON_17,
    bones,
    visible_keypoints,
)


def fake_keypoints(conf: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """17 个互不相同的关键点坐标 + 统一置信度。"""
    points = np.stack(
        [np.arange(NUM_KEYPOINTS) * 10.0, np.arange(NUM_KEYPOINTS) * 5.0], axis=1
    ).astype(np.float32)
    return points, np.full(NUM_KEYPOINTS, conf, dtype=np.float32)


# ------------------------------------------------------------------ 骨架常量
def test_skeleton_is_19_bones_with_valid_indices() -> None:
    assert len(SKELETON_17) == 19
    for start, end in SKELETON_17:
        assert 0 <= start < NUM_KEYPOINTS
        assert 0 <= end < NUM_KEYPOINTS
        assert start != end


def test_every_keypoint_appears_in_some_bone() -> None:
    """漏一个索引说明表抄错了（比如 ultralytics 那份是 1-based）。"""
    used = {index for pair in SKELETON_17 for index in pair}
    assert used == set(range(NUM_KEYPOINTS))


def test_keypoint_names_match_count() -> None:
    assert NUM_KEYPOINTS == 17
    assert len(KEYPOINT_NAMES_17) == NUM_KEYPOINTS
    assert len(set(KEYPOINT_NAMES_17)) == NUM_KEYPOINTS


def test_coco_landmark_positions_are_where_they_should_be() -> None:
    """抽查几个位置：0=nose、5/6=左右肩、15/16=左右踝（COCO 顺序）。"""
    assert KEYPOINT_NAMES_17[0] == "nose"
    assert KEYPOINT_NAMES_17[5] == "left_shoulder"
    assert KEYPOINT_NAMES_17[6] == "right_shoulder"
    assert KEYPOINT_NAMES_17[15] == "left_ankle"
    assert KEYPOINT_NAMES_17[16] == "right_ankle"


# ---------------------------------------------------------------------- 骨头
def test_all_visible_yields_every_bone() -> None:
    points, conf = fake_keypoints()
    strips = bones(points, conf, threshold=0.5)
    assert len(strips) == len(SKELETON_17)
    assert all(strip.shape == (2, 2) for strip in strips)


def test_bone_geometry_follows_the_keypoints() -> None:
    points, conf = fake_keypoints()
    strips = bones(points, conf, threshold=0.5)
    start, end = SKELETON_17[0]
    assert np.array_equal(strips[0], np.array([points[start], points[end]], np.float32))


def test_low_confidence_endpoint_drops_only_its_own_bones() -> None:
    """左肩(5) 连着 4 根骨头，置信度掉了就只该少这 4 根。"""
    points, conf = fake_keypoints()
    conf[5] = 0.1
    strips = bones(points, conf, threshold=0.5)
    expected_missing = sum(1 for pair in SKELETON_17 if 5 in pair)
    assert expected_missing == 4
    assert len(strips) == len(SKELETON_17) - expected_missing
    # 剩下的骨头里不该再出现左肩那个点
    assert not any(
        np.array_equal(strip[0], points[5]) or np.array_equal(strip[1], points[5])
        for strip in strips
    )


def test_all_below_threshold_yields_no_bones() -> None:
    points, conf = fake_keypoints(conf=0.2)
    assert bones(points, conf, threshold=0.5) == []


def test_threshold_is_inclusive() -> None:
    """恰好等于阈值算"看见"。"""
    points, conf = fake_keypoints(conf=0.5)
    assert len(bones(points, conf, threshold=0.5)) == len(SKELETON_17)


# -------------------------------------------------------------------- 可见点
def test_visible_keypoints_filters_and_keeps_names() -> None:
    points, conf = fake_keypoints()
    conf[0] = 0.1  # nose 丢了
    visible, names = visible_keypoints(points, conf, threshold=0.5)
    assert visible.shape == (NUM_KEYPOINTS - 1, 2)
    assert "nose" not in names
    assert names[0] == "left_eye"
    assert len(names) == len(visible)


def test_visible_keypoints_coordinates_match_the_filter() -> None:
    points, conf = fake_keypoints()
    conf[15] = 0.0
    visible, _ = visible_keypoints(points, conf, threshold=0.5)
    assert not any(np.array_equal(point, points[15]) for point in visible)


def test_visible_keypoints_none_visible() -> None:
    points, conf = fake_keypoints(conf=0.0)
    visible, names = visible_keypoints(points, conf, threshold=0.5)
    assert visible.shape == (0, 2)
    assert names == []


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.99])
def test_visible_keypoints_never_exceeds_all(threshold: float) -> None:
    points, conf = fake_keypoints(conf=0.99)
    visible, names = visible_keypoints(points, conf, threshold=threshold)
    assert len(visible) == len(names) <= NUM_KEYPOINTS
