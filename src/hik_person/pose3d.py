# SPDX-License-Identifier: AGPL-3.0-or-later
"""把 2D 关键点 + 单目深度图抬成 3D 骨架。

没有深度相机，做法是：`yolo26*-pose` 出 17 个像素关键点，`yolo26*-depth` 出同一帧的
公制深度图，在每个关键点像素处取深度做 z，再用针孔模型还原成**相机坐标系**里的 3D 点。

几何计算（内参、取样、反投影）都在 ``geometry`` 里，本模块只负责“关键点 -> 可显示的
骨架”这一段；命名沿用 d435i-person-py，方便两边对照。

坐标系用 rerun 的相机约定 ``ViewCoordinates.RIGHT_HAND_Y_DOWN``（等价 ``RDF``）：

* ``X`` 向右，``Y`` 向下，``Z`` 向前（等于深度）

内参没标定，只能按一个假定的水平视场角估（``Intrinsics.from_fov``）。
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    DEPTH_PATCH,
    Intrinsics,
    depth_valid_mask,
    lift_keypoints,
    sample_depth,
)
from .pose import KEYPOINT_NAMES_17, SKELETON_17, PoseInstance

# 兼容旧调用点：``valid_mask`` 是 ``depth_valid_mask`` 的旧名字。
valid_mask = depth_valid_mask

__all__ = [
    "DEPTH_PATCH",
    "Intrinsics",
    "depth_valid_mask",
    "lift_keypoints",
    "pose_to_3d",
    "sample_depth",
    "to_display",
    "valid_mask",
]


def to_display(points3d: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str], list]:
    """``(17,3)`` + 有效掩码 -> ``(关节 (M,3), 名字, 骨头线段列表)``。

    缺一个端点的骨头不连（宁可少画，也不把点甩到 nan 位置）。
    """
    points3d = np.asarray(points3d)
    joints = points3d[valid]
    names = [KEYPOINT_NAMES_17[i] for i in range(len(valid)) if valid[i]]
    strips = [
        np.stack([points3d[start], points3d[end]]).astype(np.float32)
        for start, end in SKELETON_17
        if valid[start] and valid[end]
    ]
    return joints, names, strips


def pose_to_3d(
    pose: PoseInstance,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    threshold: float,
    patch: int = DEPTH_PATCH,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """一个人 -> ``(关节 (M,3), 名字, 骨头线段列表)``；不做滤波，滤波见 ``track``。"""
    points3d = lift_keypoints(pose.keypoints, depth, intrinsics, patch=patch)
    return to_display(points3d, depth_valid_mask(points3d, pose.kpt_conf, threshold))
