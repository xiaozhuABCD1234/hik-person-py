# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机->世界（地面系）外参的单元测试 —— 不需要相机。"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.world import CameraToWorld


def test_level_camera_maps_forward_to_world_x() -> None:
    transform = CameraToWorld(height=1.5)
    # 相机前方 2 m -> 世界 (前 2, 左 0, 高 1.5)
    assert transform.apply(np.array([0.0, 0.0, 2.0])) == pytest.approx((2.0, 0.0, 1.5))


def test_level_camera_maps_right_and_down() -> None:
    transform = CameraToWorld(height=1.0)
    assert transform.apply(np.array([1.0, 0.0, 0.0])) == pytest.approx((0.0, -1.0, 1.0))
    assert transform.apply(np.array([0.0, 1.0, 0.0])) == pytest.approx((0.0, 0.0, 0.0))


def test_camera_pitched_down_90_degrees_looks_at_the_floor() -> None:
    transform = CameraToWorld(height=2.0, pitch_deg=90.0)
    # 光轴朝下：光轴上 2 m 的点就在正下方地面
    assert transform.apply(np.array([0.0, 0.0, 2.0])) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_pitch_down_30_degrees_hits_the_ground_at_expected_range() -> None:
    height = 2.0
    transform = CameraToWorld(height=height, pitch_deg=30.0)
    # 沿光轴走到地面的距离 = 高度 / sin(俯角)
    distance = height / np.sin(np.radians(30.0))
    world = transform.apply(np.array([0.0, 0.0, distance]))
    assert world[2] == pytest.approx(0.0, abs=1e-6)
    assert world[0] == pytest.approx(distance * np.cos(np.radians(30.0)))


def test_yaw_rotates_around_world_z() -> None:
    transform = CameraToWorld(height=1.0, yaw_deg=90.0)
    assert transform.apply(np.array([0.0, 0.0, 2.0])) == pytest.approx((0.0, 2.0, 1.0))


def test_roll_rotates_around_the_optical_axis() -> None:
    transform = CameraToWorld(height=1.0, roll_deg=90.0)
    assert transform.apply(np.array([1.0, 0.0, 0.0])) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_rotation_is_orthonormal() -> None:
    transform = CameraToWorld(height=1.2, pitch_deg=17.0, yaw_deg=-40.0, roll_deg=5.0)
    rotation = transform.rotation()
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_apply_keeps_shape_and_nan() -> None:
    transform = CameraToWorld(height=1.0)
    points = np.array([[0.0, 0.0, 2.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    out = transform.apply(points)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_distance_from_camera_is_preserved() -> None:
    transform = CameraToWorld(height=1.3, pitch_deg=12.0, yaw_deg=33.0)
    first = np.array([0.4, 0.2, 2.5])
    second = np.array([-0.6, 0.9, 1.1])
    # 刚体变换：点与点之间的距离不变（到世界原点的距离会变，因为平移把原点挑高了）
    before = np.linalg.norm(first - second)
    after = np.linalg.norm(transform.apply(first) - transform.apply(second))
    assert after == pytest.approx(before, abs=1e-5)


def test_describe_mentions_the_extrinsics() -> None:
    text = CameraToWorld(height=1.1, pitch_deg=20.0).describe()
    assert "1.1" in text and "20" in text
