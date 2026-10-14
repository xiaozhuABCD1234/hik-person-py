# SPDX-License-Identifier: AGPL-3.0-or-later
"""几何工具（内参/取样/反投影）的单元测试 —— 不需要相机、不加载模型。

和 d435i-person-py 的 ``test_geometry.py`` 是同一套用例，去掉真深度相机才有的
``depth_to_meters`` 与点云两节。
"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.geometry import (
    Intrinsics,
    deproject,
    depth_valid_mask,
    lift_keypoints,
    sample_depth,
)

# 640×480、fx=fy=500 的假内参：手算坐标方便
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def flat_depth(value: float = 2.0, *, width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


# ----------------------------------------------------------------------- 内参
def test_intrinsics_from_fov_90_degrees() -> None:
    intrinsics = Intrinsics.from_fov(640, 480, 90.0)
    assert intrinsics.fx == pytest.approx(320.0)
    assert intrinsics.fy == pytest.approx(320.0)
    assert intrinsics.cx == pytest.approx(320.0)
    assert intrinsics.cy == pytest.approx(240.0)
    assert (intrinsics.width, intrinsics.height) == (640, 480)


def test_intrinsics_from_fov_wider_lens_has_shorter_focal() -> None:
    assert Intrinsics.from_fov(640, 480, 120.0).fx < Intrinsics.from_fov(640, 480, 60.0).fx


def test_intrinsics_scaled_to_doubles_pixels() -> None:
    scaled = K.scaled_to(1280, 960)
    assert scaled.fx == pytest.approx(1000.0)
    assert scaled.cy == pytest.approx(480.0)
    # 同尺寸时是恒等变换（返回自己，省一次分配）
    assert K.scaled_to(640, 480) is K


def test_intrinsics_matrix_and_fov_are_consistent() -> None:
    assert K.matrix().tolist() == [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    # fx=500、宽 640 -> 水平 FOV = 2*atan(320/500) ≈ 65.24°，垂直 ≈ 51.28°
    horizontal, vertical = K.fov_deg()
    assert horizontal == pytest.approx(65.24, abs=0.05)
    assert vertical == pytest.approx(51.28, abs=0.05)


# ------------------------------------------------------------------- 反投影
def test_deproject_center_is_on_the_optical_axis() -> None:
    assert deproject(320.0, 240.0, 2.0, K) == pytest.approx((0.0, 0.0, 2.0))


def test_deproject_offsets_scale_with_depth() -> None:
    # 像素往右 100 -> X 为正；像素往下 100 -> Y 为正（Y 向下）
    assert deproject(420.0, 340.0, 2.0, K) == pytest.approx((0.4, 0.4, 2.0))


@pytest.mark.parametrize("bad", [(np.nan, 240.0, 2.0), (320.0, np.inf, 2.0), (320.0, 240.0, 0.0)])
def test_deproject_rejects_invalid(bad) -> None:
    assert all(np.isnan(v) for v in deproject(*bad, K))


# ------------------------------------------------------------------- 取深度
def test_sample_depth_takes_median_of_patch() -> None:
    depth = flat_depth(2.0, width=11, height=11)
    depth[5, 5] = 100.0  # 前景/背景边界上的单个离群值
    assert sample_depth(depth, 5.0, 5.0, patch=3) == pytest.approx(2.0)
    # 窗口里大多数像素都变了 -> 中位数跟着走
    depth[4:7, 4:7] = 5.0
    assert sample_depth(depth, 5.0, 5.0, patch=3) == pytest.approx(5.0)


def test_sample_depth_ignores_nan_and_out_of_range() -> None:
    depth = flat_depth(np.nan, width=4, height=4)
    assert np.isnan(sample_depth(depth, 2.0, 2.0))
    assert np.isnan(sample_depth(depth, -5.0, 2.0))
    assert np.isnan(sample_depth(depth, np.nan, 2.0))


# ------------------------------------------------------------------- 抬 3D
def test_lift_keypoints_maps_pixels_to_camera_frame() -> None:
    keypoints = np.array([[320.0, 240.0], [420.0, 140.0]], dtype=np.float32)
    lifted = lift_keypoints(keypoints, flat_depth(2.0), K)
    assert lifted[0] == pytest.approx((0.0, 0.0, 2.0), abs=1e-5)
    assert lifted[1] == pytest.approx((0.4, -0.4, 2.0), abs=1e-5)


def test_lift_keypoints_nans_where_depth_missing() -> None:
    depth = flat_depth(2.0)
    depth[238:243, 318:323] = np.nan  # 把 5×5 的取样窗口整块挖掉
    lifted = lift_keypoints(np.array([[320.0, 240.0], [420.0, 140.0]]), depth, K)
    assert np.isnan(lifted[0]).all()
    assert np.isfinite(lifted[1]).all()


def test_depth_valid_mask_needs_both_confidence_and_depth() -> None:
    lifted = np.array([[0.0, 0.0, 2.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    conf = np.array([0.4, 0.9], dtype=np.float32)
    assert list(depth_valid_mask(lifted, conf, 0.5)) == [False, False]
    conf = np.array([0.9, 0.9], dtype=np.float32)
    assert list(depth_valid_mask(lifted, conf, 0.5)) == [True, False]
