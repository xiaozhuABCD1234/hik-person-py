# SPDX-License-Identifier: AGPL-3.0-or-later
"""把 2D 关键点 + 单目深度图抬成 3D 骨架，并估出每个人的 3D 位置。

没有深度相机，做法是：`yolo26*-pose` 出 17 个像素关键点，`yolo26*-depth` 出同一帧的
公制深度图，在每个关键点像素处取深度做 z，再用针孔模型还原成**相机坐标系**里的 3D 点。

和 d435i-person-py 的差别只有深度来源（那边是真深度，内参是设备标定值）：绝对尺度会飘，
但骨架形状与逐帧相对变化是同源的。所以除了"距离/速度"这类相对量，本模块不输出
身高、肩宽那样的绝对长度指标。

几何计算（内参、取样、反投影）都在 ``geometry`` 里，本模块负责"关键点 -> 3D 人"。

坐标系用 rerun 的相机约定 ``ViewCoordinates.RIGHT_HAND_Y_DOWN``（等价 ``RDF``）：

* ``X`` 向右，``Y`` 向下，``Z`` 向前（等于深度）

内参没标定，只能按一个假定的水平视场角估（``Intrinsics.from_fov``）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    DEPTH_PATCH,
    Intrinsics,
    deproject,
    depth_valid_mask,
    lift_keypoints,
    sample_depth,
)
from .pose import KEYPOINT_NAMES_17, SKELETON_17, PoseInstance

# 兼容旧调用点：``valid_mask`` 是 ``depth_valid_mask`` 的旧名字。
valid_mask = depth_valid_mask

# 用于估"这个人在哪"的对称关节对，按可靠性排序：髋比肩更稳（肩会被背包/抬手带偏）。
POSITION_PAIRS: tuple[tuple[int, int], ...] = ((11, 12), (5, 6))
# 框中心取深度失败时，再沿框的竖直线扫这几个相对高度（0=上边缘，1=下边缘）。
POSITION_SCAN = (0.5, 0.35, 0.65, 0.25, 0.75)

__all__ = [
    "DEPTH_PATCH",
    "POSITION_PAIRS",
    "POSITION_SCAN",
    "Intrinsics",
    "Person3D",
    "depth_valid_mask",
    "lift_keypoints",
    "person_position",
    "pose_to_3d",
    "sample_depth",
    "to_display",
    "valid_mask",
]


@dataclass
class Person3D:
    """一个人的 3D 表示：关节/骨头 + 位置（相机坐标系，米）。"""

    joints: np.ndarray  # (M, 3)
    names: list[str]
    strips: list[np.ndarray]
    position: np.ndarray  # (3,) 相机坐标；测不到就是 nan
    valid: np.ndarray  # (17,) 哪些关键点这一次是有效的

    @property
    def distance(self) -> float:
        """到相机光心的距离（米），测不到返回 nan。"""
        return (
            float(np.linalg.norm(self.position))
            if np.isfinite(self.position).all()
            else float("nan")
        )

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))


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


def person_position(
    lifted: np.ndarray,
    valid: np.ndarray,
    box: tuple[float, float, float, float],
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    patch: int = DEPTH_PATCH,
) -> np.ndarray:
    """估一个人的 3D 位置（相机坐标，米），按可靠性依次退化：

    1. 双侧髋中点（最稳，靠近人体质心）
    2. 双侧肩中点
    3. 所有有效关节的中位数（只露半身、侧面时用）
    4. 框中心附近的深度（关键点全丢时兜底；沿框竖线扫几个高度取第一个有效值）

    全失败返回 (nan, nan, nan)。纯函数。
    """
    for left, right in POSITION_PAIRS:
        if valid[left] and valid[right]:
            return np.asarray((lifted[left] + lifted[right]) * 0.5, dtype=np.float32)

    if valid.any():
        return np.asarray(np.median(lifted[valid], axis=0), dtype=np.float32)

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) * 0.5
    for ratio in POSITION_SCAN:
        v = y1 + (y2 - y1) * ratio
        d = sample_depth(depth, center_x, v, patch)
        if np.isfinite(d) and d > 0:
            return np.asarray(deproject(center_x, v, d, intrinsics), dtype=np.float32)
    return np.full(3, np.nan, dtype=np.float32)


def pose_to_3d(
    pose: PoseInstance,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    threshold: float,
    patch: int = DEPTH_PATCH,
) -> Person3D:
    """一个人 -> ``Person3D``；不做滤波（滤波见 ``track``）。"""
    lifted = lift_keypoints(pose.keypoints, depth, intrinsics, patch=patch)
    valid = depth_valid_mask(lifted, pose.kpt_conf, threshold)
    joints, names, strips = to_display(lifted, valid)
    return Person3D(
        joints=joints,
        names=names,
        strips=strips,
        position=person_position(lifted, valid, pose.xyxy, depth, intrinsics, patch=patch),
        valid=valid,
    )
