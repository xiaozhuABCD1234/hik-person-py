# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D 关键点 + 深度图 -> 3D 骨架/位置的纯函数测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.geometry import Intrinsics, lift_keypoints
from hik_person.pose import NUM_KEYPOINTS, SKELETON_17, PoseInstance
from hik_person.pose3d import Person3D, person_position, pose_to_3d

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def test_pose3d_reexports_geometry_names() -> None:
    """旧调用点导入 ``pose3d.Intrinsics`` 等名字仍然可用（几何已搬到 geometry）。"""
    from hik_person import pose3d

    assert pose3d.Intrinsics is Intrinsics
    assert pose3d.lift_keypoints is lift_keypoints
    assert pose3d.valid_mask is pose3d.depth_valid_mask
    assert pose3d.DEPTH_PATCH == 5


def flat_depth(value: float = 2.0) -> np.ndarray:
    return np.full((480, 640), value, dtype=np.float32)


def make_pose(points: np.ndarray | None = None, conf: np.ndarray | None = None) -> PoseInstance:
    """默认关键点：X 依次拉开、Y 固定，全都在画面内。"""
    if points is None:
        points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        points[:, 0] = 300.0 + np.arange(NUM_KEYPOINTS)
        points[:, 1] = 240.0
    return PoseInstance(
        conf=0.77,
        xyxy=(300.0, 100.0, 340.0, 400.0),
        keypoints=points,
        kpt_conf=CONF if conf is None else conf,
    )


# ------------------------------------------------------------------- pose_to_3d
def test_pose_to_3d_keeps_all_points_and_bones() -> None:
    person = pose_to_3d(make_pose(), flat_depth(2.5), K, threshold=0.5)
    assert isinstance(person, Person3D)
    assert person.joints.shape == (NUM_KEYPOINTS, 3)
    assert len(person.names) == NUM_KEYPOINTS
    assert len(person.strips) == len(SKELETON_17)
    assert all(strip.shape == (2, 3) for strip in person.strips)
    assert person.valid_count == NUM_KEYPOINTS


def test_pose_to_3d_drops_low_confidence_points() -> None:
    pose = make_pose()
    pose.kpt_conf[5] = 0.1  # 左肩掉了
    person = pose_to_3d(pose, flat_depth(2.5), K, threshold=0.5)
    assert len(person.joints) == NUM_KEYPOINTS - 1
    assert len(person.strips) == len(SKELETON_17) - 4  # 左肩连的 4 根也断


def test_pose_to_3d_position_uses_hip_midpoint() -> None:
    """位置优先取双髋中点（髋比肩稳），不是"所有点的平均"。"""
    person = pose_to_3d(make_pose(), flat_depth(3.0), K, threshold=0.5)
    left_hip = person.joints[person.names.index("left_hip")]
    right_hip = person.joints[person.names.index("right_hip")]
    assert person.position == pytest.approx((left_hip + right_hip) * 0.5, abs=1e-6)
    assert person.distance == pytest.approx(np.linalg.norm(person.position))


def test_pose_to_3d_without_depth_has_no_position() -> None:
    person = pose_to_3d(make_pose(), np.full((480, 640), np.nan, np.float32), K, threshold=0.5)
    assert person.valid_count == 0
    assert np.isnan(person.position).all()
    assert np.isnan(person.distance)


# ------------------------------------------------------------- person_position
def test_person_position_falls_back_to_shoulders() -> None:
    """髋不可用就退到肩中点，而不是给个 nan。"""
    lifted = np.full((NUM_KEYPOINTS, 3), np.nan, np.float32)
    lifted[5] = (0.0, 0.0, 2.0)  # left_shoulder
    lifted[6] = (0.2, 0.0, 2.0)  # right_shoulder
    valid = np.isfinite(lifted).all(axis=1)
    position = person_position(lifted, valid, (0.0, 0.0, 10.0, 10.0), flat_depth(2.0), K)
    assert position == pytest.approx((0.1, 0.0, 2.0), abs=1e-5)


def test_person_position_falls_back_to_valid_joint_median() -> None:
    lifted = np.full((NUM_KEYPOINTS, 3), np.nan, np.float32)
    lifted[0] = (0.0, 0.0, 2.0)
    lifted[1] = (0.0, 0.0, 4.0)
    valid = np.isfinite(lifted).all(axis=1)
    position = person_position(lifted, valid, (0.0, 0.0, 10.0, 10.0), flat_depth(2.0), K)
    assert position[2] == pytest.approx(3.0)  # 两个点的中位数


def test_person_position_falls_back_to_box_center_depth() -> None:
    """关键点全丢（人背对镜头/被挡）时，用框中心附近的深度兜底。"""
    lifted = np.full((NUM_KEYPOINTS, 3), np.nan, np.float32)
    valid = np.zeros(NUM_KEYPOINTS, bool)
    box = (300.0, 100.0, 340.0, 400.0)
    position = person_position(lifted, valid, box, flat_depth(2.0), K)
    # 先试框中心（320, 250），命中 2 m 平面
    assert position[2] == pytest.approx(2.0)
    assert position[0] == pytest.approx((320.0 - 320.0) * 2.0 / 500.0, abs=1e-5)


def test_person_position_is_nan_when_nothing_works() -> None:
    lifted = np.full((NUM_KEYPOINTS, 3), np.nan, np.float32)
    valid = np.zeros(NUM_KEYPOINTS, bool)
    empty = np.full((480, 640), np.nan, np.float32)
    position = person_position(lifted, valid, (300.0, 100.0, 340.0, 400.0), empty, K)
    assert np.isnan(position).all()
