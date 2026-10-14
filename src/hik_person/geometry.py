# SPDX-License-Identifier: AGPL-3.0-or-later
"""针孔模型的几何工具：内参、深度取样、反投影。

本包内部统一用**相机坐标系 RDF**（rerun 的 ``RIGHT_HAND_Y_DOWN``）：

* ``X`` 向右，``Y`` 向下，``Z`` 向前（等于深度，单位米）

深度来自单目估计模型（``yolo26*-depth``），给的是沿光轴的 Z（不是斜距），所以针孔模型
可以直接用：

    X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = depth

和 d435i-person-py 的 ``geometry`` 是同一套函数，区别只有：那边多了 z16 的
``depth_to_meters`` 与点云 ``pointcloud``（真深度相机才有），这边不需要。

纯净函数 + ``numpy``，不做任何相机/SDK 调用，便于单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 取深度时的小窗口边长（奇数）：单像素容易被前景/背景边界带偏，取窗口中位数更稳。
DEPTH_PATCH = 5


@dataclass(frozen=True)
class Intrinsics:
    """相机内参（单位像素）。深度图与彩色图同分辨率时用同一份。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> Intrinsics:
        """按水平视场角估内参（像素假设是方的，主点在图像中心）。

        单目方案没有标定值，只能这样估；写错的视场角会让 3D 整体缩放不对
        （X/Y 与 Z 的比例变了），但不影响骨架形状。
        """
        fx = (width * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
        return cls(
            fx=float(fx),
            fy=float(fx),
            cx=width * 0.5,
            cy=height * 0.5,
            width=int(width),
            height=int(height),
        )

    def scaled_to(self, width: int, height: int) -> Intrinsics:
        """把内参按比例缩放到另一个分辨率（fx/fy/cx/cy 同比例缩放）。

        深度模型输出会自动缩回原分辨率，正常情况下这一步是恒等变换；留着是为了
        万一尺寸对不上（改 imgsz、加抽稀）时不会静默算错 3D。
        """
        if (width, height) == (self.width, self.height):
            return self
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=int(width),
            height=int(height),
        )

    def matrix(self) -> np.ndarray:
        """3×3 内参矩阵（纯看/调试用）。"""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def fov_deg(self) -> tuple[float, float]:
        """(水平, 垂直) 视场角，度。"""
        horizontal = 2.0 * math.degrees(math.atan(self.width * 0.5 / self.fx))
        vertical = 2.0 * math.degrees(math.atan(self.height * 0.5 / self.fy))
        return horizontal, vertical


def sample_depth(depth: np.ndarray, u: float, v: float, patch: int = DEPTH_PATCH) -> float:
    """取关键点附近 ``patch``×``patch`` 窗口的深度中位数；越界/无有效值返回 nan。"""
    height, width = depth.shape[:2]
    if not (math.isfinite(u) and math.isfinite(v)):
        return float("nan")
    x, y = math.floor(u + 0.5), math.floor(v + 0.5)
    half = patch // 2
    x0, x1 = max(x - half, 0), min(x + half + 1, width)
    y0, y1 = max(y - half, 0), min(y + half + 1, height)
    if x0 >= x1 or y0 >= y1:
        return float("nan")
    window = depth[y0:y1, x0:x1]
    window = window[np.isfinite(window)]
    return float(np.median(window)) if window.size else float("nan")


def deproject(u: float, v: float, d: float, intrinsics: Intrinsics) -> tuple[float, float, float]:
    """像素 + 米 -> 相机坐标 (X 右 / Y 下 / Z 前)。非有限输入返回 (nan, nan, nan)。"""
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(d)) or d <= 0:
        return (float("nan"),) * 3
    return (
        (u - intrinsics.cx) * d / intrinsics.fx,
        (v - intrinsics.cy) * d / intrinsics.fy,
        d,
    )


def lift_keypoints(
    keypoints: np.ndarray,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    patch: int = DEPTH_PATCH,
) -> np.ndarray:
    """``(N, 2)`` 像素 + 深度图（米）-> ``(N, 3)`` 相机坐标（米）。

    深度无效（越界、0、非有限）的点是 nan，调用方按 ``depth_valid_mask`` 过滤。
    """
    out = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    for i, (u, v) in enumerate(np.asarray(keypoints, dtype=np.float64)):
        out[i] = deproject(u, v, sample_depth(depth, u, v, patch), intrinsics)
    return out


def depth_valid_mask(points3d: np.ndarray, kpt_conf: np.ndarray, threshold: float) -> np.ndarray:
    """关键点要“置信度过阈值 **且** 深度有效”才算数。"""
    return np.isfinite(points3d).all(axis=1) & (np.asarray(kpt_conf) >= threshold)
