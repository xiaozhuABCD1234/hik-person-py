# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D 关键点 + 深度图 -> 3D 骨架的纯函数测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.pose import NUM_KEYPOINTS, SKELETON_17, PoseInstance
from hik_person.pose3d import Intrinsics, lift_keypoints, pose_to_3d, sample_depth


def test_from_fov_90_puts_focal_length_at_half_width() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    assert intr.fx == pytest.approx(320.0)
    assert intr.fy == pytest.approx(320.0)
    assert (intr.cx, intr.cy) == (320.0, 240.0)


def test_from_fov_wider_lens_shorter_focal() -> None:
    assert Intrinsics.from_fov(640, 480, 120.0).fx < Intrinsics.from_fov(640, 480, 60.0).fx


def test_sample_depth_takes_window_median() -> None:
    depth = np.full((10, 10), 2.0, np.float32)
    depth[5, 5] = 100.0  # 单像素野值
    assert sample_depth(depth, 5, 5, patch=3) == pytest.approx(2.0)


def test_sample_depth_out_of_frame_is_nan() -> None:
    depth = np.ones((4, 4), np.float32)
    assert np.isnan(sample_depth(depth, -10, 2))
    assert np.isnan(sample_depth(depth, 2, -10))


def test_lift_center_pixel_is_straight_ahead() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    depth = np.full((480, 640), 3.0, np.float32)
    points = lift_keypoints(np.array([[320.0, 240.0]]), depth, intr)
    # 相机坐标：画面中心 -> X=Y=0，Z 就是深度
    assert points[0] == pytest.approx([0.0, 0.0, 3.0], abs=1e-5)


def test_lift_direction_signs() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    depth = np.full((480, 640), 2.0, np.float32)
    right = lift_keypoints(np.array([[400.0, 240.0]]), depth, intr)[0]
    down = lift_keypoints(np.array([[320.0, 300.0]]), depth, intr)[0]
    assert right[0] > 0  # 像素往右 -> X 为正
    assert down[1] > 0  # 像素往下（v 变大）-> Y 为正（Y 向下）
    assert right[2] == pytest.approx(2.0)  # Z 就是深度（朝前）


def test_lift_invalid_depth_stays_nan() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    depth = np.zeros((480, 640), np.float32)  # 0 不是有效深度
    assert np.isnan(lift_keypoints(np.array([[320.0, 240.0]]), depth, intr)).all()


def make_pose(kpt_conf: float = 0.9) -> PoseInstance:
    points = np.stack(
        [np.arange(NUM_KEYPOINTS) * 20.0 + 100.0, np.arange(NUM_KEYPOINTS) * 10.0 + 100.0], axis=1
    ).astype(np.float32)
    return PoseInstance(
        conf=0.9,
        xyxy=(0.0, 0.0, 640.0, 480.0),
        keypoints=points,
        kpt_conf=np.full(NUM_KEYPOINTS, kpt_conf, dtype=np.float32),
    )


def test_pose_to_3d_keeps_all_points_and_bones() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    depth = np.full((480, 640), 2.5, np.float32)
    joints, names, strips = pose_to_3d(make_pose(), depth, intr, threshold=0.5)
    assert joints.shape == (NUM_KEYPOINTS, 3)
    assert len(names) == NUM_KEYPOINTS
    assert len(strips) == len(SKELETON_17)
    assert all(strip.shape == (2, 3) for strip in strips)


def test_pose_to_3d_drops_low_confidence_points() -> None:
    intr = Intrinsics.from_fov(640, 480, 90.0)
    depth = np.full((480, 640), 2.5, np.float32)
    pose = make_pose()
    pose.kpt_conf[5] = 0.1  # 左肩掉了
    joints, _, strips = pose_to_3d(pose, depth, intr, threshold=0.5)
    assert len(joints) == NUM_KEYPOINTS - 1
    assert len(strips) == len(SKELETON_17) - 4  # 左肩连的 4 根也断
