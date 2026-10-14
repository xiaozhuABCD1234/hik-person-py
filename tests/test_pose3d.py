# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D 关键点 + 深度图 -> 3D 骨架的纯函数测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np

from hik_person.geometry import Intrinsics, lift_keypoints
from hik_person.pose import NUM_KEYPOINTS, SKELETON_17, PoseInstance
from hik_person.pose3d import pose_to_3d


def test_pose3d_reexports_geometry_names() -> None:
    """旧调用点导入 ``pose3d.Intrinsics`` 等名字仍然可用（几何已搬到 geometry）。"""
    from hik_person import pose3d

    assert pose3d.Intrinsics is Intrinsics
    assert pose3d.lift_keypoints is lift_keypoints
    assert pose3d.valid_mask is pose3d.depth_valid_mask
    assert pose3d.DEPTH_PATCH == 5


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
