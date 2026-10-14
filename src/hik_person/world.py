# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机坐标系 -> 世界坐标系（地面系）的固定刚体变换。

相机装好之后相对地面就不动了，所以"人在世界里的位置"只需要一个静态外参：
**光心离地高度** + **镜头俯仰角**（再可选 yaw/roll）。本模块把这个外参写成一个
纯函数式的刚体变换，不需要标定板，也不需要 ROS 的 TF。

坐标系约定（照着 ROS REP-103 的习惯）：

* 相机 ``RDF``：X 右 / Y 下 / Z 前（前 = 光轴 = 深度方向）
* 世界 ``FLU``：X 前 / Y 左 / Z 上，原点在**相机正下方的地面点**

于是水平放置（pitch=0）时：相机前方 2 m 的点 -> 世界 ``(2, 0, 高度)``；
镜头垂直向下（pitch=90）时同一个点 -> 世界 ``(0, 0, 高度-2)``，也就是脚下 2 m。

外参不准只会让人的**绝对位置**整体偏移，不影响 3D 骨架的形状和相对距离。

注意（单目方案的前提）：这里的输入来自 `yolo26*-depth` 估的深度，**绝对尺度本身会飘**。
所以世界坐标的用途是"同一场景里各个人的相对位置 / 谁在谁左边"这类比较，
不能当成公制真值；要多相机拼接或者给底盘发坐标之前，先用已知身高或已知距离校一遍尺度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 姿态为零时的基变换：相机 (X 右, Y 下, Z 前) -> 世界 (X 前, Y 左, Z 上)
_BASE = np.array(
    [
        [0.0, 0.0, 1.0],  # 世界 X（前） = 相机 Z
        [-1.0, 0.0, 0.0],  # 世界 Y（左） = -相机 X
        [0.0, -1.0, 0.0],  # 世界 Z（上） = -相机 Y
    ],
    dtype=np.float64,
)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    )


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


@dataclass(frozen=True)
class CameraToWorld:
    """静态外参：``world = R @ camera + t``，``t = (0, 0, height)``。

    * ``height``：光心离地高度（米）
    * ``pitch_deg``：镜头向下俯的角度，**正 = 向下**（0 = 完全水平）
    * ``yaw_deg``：绕世界 Z 轴左转为正（相机朝向和世界 X 的夹角）
    * ``roll_deg``：绕光轴旋转
    """

    height: float = 1.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    def rotation(self) -> np.ndarray:
        """相机 -> 世界的旋转矩阵（顺序：先绕光轴 roll、再俯仰、再基变换、最后 yaw）。"""
        return rot_z(self.yaw_deg) @ _BASE @ rot_x(-self.pitch_deg) @ rot_z(self.roll_deg)

    def translation(self) -> np.ndarray:
        return np.array([0.0, 0.0, float(self.height)])

    def apply(self, points: np.ndarray) -> np.ndarray:
        """``(3,)`` 或 ``(N,3)`` 相机坐标 -> 世界坐标；nan 原样保留。"""
        arr = np.asarray(points, dtype=np.float64)
        flat = arr.reshape(-1, 3)
        out = (self.rotation() @ flat.T).T + self.translation()
        return out.reshape(arr.shape).astype(np.float32)

    def up_in_camera(self) -> np.ndarray:
        """世界系的"上"方向在**相机坐标系**里的表示（= Rᵀ·(0,0,1)_world）。

        算"地面速度"这类沿上方向的量时必须用这个，而不是相机 Y：
        镜头一旦俯仰，相机 Y 就不再是竖直方向（默认 0° 时两者一致）。
        """
        return (self.rotation().T @ np.array([0.0, 0.0, 1.0])).astype(np.float64)

    def describe(self) -> str:
        return (
            f"高度 {self.height:g} m, 俯仰 {self.pitch_deg:g}°, "
            f"yaw {self.yaw_deg:g}°, roll {self.roll_deg:g}°"
        )
