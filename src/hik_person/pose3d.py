# SPDX-License-Identifier: AGPL-3.0-or-later
"""把 2D 关键点 + 单目深度图抬成 3D 骨架。

没有深度相机，做法是：`yolo26n-pose` 出 17 个像素关键点，`yolo26n-depth` 出同一帧的
公制深度图，在每个关键点像素处取深度做 z，再用针孔模型还原成**相机坐标系**里的 3D 点。

坐标系用 rerun 的相机约定 ``ViewCoordinates.RIGHT_HAND_Y_DOWN``（等价 ``RDF``）：

* ``X`` 向右，``Y`` 向下，``Z`` 向前（等于深度）

内参没标定，只能按一个假定的水平视场角估（``Intrinsics.from_fov``）。纯函数，便于单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pose import KEYPOINT_NAMES_17, SKELETON_17, PoseInstance

# 取深度时的小窗口边长（奇数）：单像素容易被前景/背景边界带偏，取窗口中位数更稳。
DEPTH_PATCH = 5


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> Intrinsics:
        """按水平视场角估内参（像素假设是方的，主点在图像中心）。"""
        fx = (width * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
        return cls(fx=float(fx), fy=float(fx), cx=width * 0.5, cy=height * 0.5)


def sample_depth(depth: np.ndarray, u: float, v: float, patch: int = DEPTH_PATCH) -> float:
    """取关键点附近 ``patch``×``patch`` 窗口的深度中位数；越界/无有效值返回 nan。"""
    height, width = depth.shape[:2]
    x, y = math.floor(u + 0.5), math.floor(v + 0.5)
    half = patch // 2
    x0, x1 = max(x - half, 0), min(x + half + 1, width)
    y0, y1 = max(y - half, 0), min(y + half + 1, height)
    if x0 >= x1 or y0 >= y1:
        return float("nan")
    window = depth[y0:y1, x0:x1]
    window = window[np.isfinite(window)]
    return float(np.median(window)) if window.size else float("nan")


def lift_keypoints(
    keypoints: np.ndarray,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    patch: int = DEPTH_PATCH,
) -> np.ndarray:
    """``(N, 2)`` 像素 + 深度图 -> ``(N, 3)`` 相机坐标（米，X 右 / Y 下 / Z 前）。

    深度无效（越界、0、非有限）的点是 nan，调用方按 ``valid_mask`` 过滤。
    """
    out = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    for i, (u, v) in enumerate(np.asarray(keypoints, dtype=np.float64)):
        d = sample_depth(depth, u, v, patch)
        if not np.isfinite(d) or d <= 0:
            continue
        out[i] = (
            (u - intrinsics.cx) * d / intrinsics.fx,
            (v - intrinsics.cy) * d / intrinsics.fy,
            d,
        )
    return out


def valid_mask(points3d: np.ndarray, kpt_conf: np.ndarray, threshold: float) -> np.ndarray:
    """关键点要“置信度过阈值 **且** 深度有效”才算数。"""
    return np.isfinite(points3d).all(axis=1) & (np.asarray(kpt_conf) >= threshold)


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
    return to_display(points3d, valid_mask(points3d, pose.kpt_conf, threshold))
