# SPDX-License-Identifier: AGPL-3.0-or-later
"""测光步进算法的单元测试 —— 纯函数，不需要相机。"""

from __future__ import annotations

import pytest

from hik_person.mvs import next_exposure_step

RANGE = {
    "exposure_min_us": 15.0,
    "exposure_max_us": 60000.0,
    "gain_min_db": 0.0,
    "gain_max_db": 16.98,
}


def test_brighten_by_exposure_alone() -> None:
    """亮度差 2 倍 -> 曝光翻倍，增益不动。"""
    exposure, gain = next_exposure_step(50.0, 100.0, 8000.0, 0.0, **RANGE)
    assert exposure == pytest.approx(16000.0)
    assert gain == 0.0


def test_brighten_clamps_exposure_then_uses_gain() -> None:
    """要 4 倍但曝光只能到 2 倍 -> 剩下的 2 倍换成 +6.02 dB。"""
    exposure, gain = next_exposure_step(25.0, 100.0, 30000.0, 0.0, **RANGE)
    assert exposure == pytest.approx(60000.0)
    assert gain == pytest.approx(6.02, abs=0.02)


def test_gain_never_exceeds_ceiling() -> None:
    exposure, gain = next_exposure_step(1.0, 255.0, 60000.0, 16.98, **RANGE)
    assert exposure == pytest.approx(60000.0)
    assert gain == pytest.approx(16.98)


def test_darken_reduces_exposure_first() -> None:
    """太亮时先降曝光，增益不动。"""
    exposure, gain = next_exposure_step(200.0, 100.0, 20000.0, 8.0, **RANGE)
    assert exposure == pytest.approx(10000.0)
    assert gain == pytest.approx(8.0)


def test_darken_below_exposure_floor_reduces_gain() -> None:
    """要降到 1/4，但曝光下限只允许 1/2 -> 增益降 6.02 dB。"""
    exposure, gain = next_exposure_step(400.0, 100.0, 30.0, 10.0, **RANGE)
    assert exposure == pytest.approx(15.0)
    assert gain == pytest.approx(3.98, abs=0.02)


def test_gain_never_below_floor() -> None:
    exposure, gain = next_exposure_step(400.0, 100.0, 30.0, 2.0, **RANGE)
    assert exposure == pytest.approx(15.0)
    assert gain == pytest.approx(0.0)


def test_black_frame_does_not_divide_by_zero() -> None:
    """全黑帧（mean=0）也要能算出有限的一步，而不是崩掉。"""
    exposure, gain = next_exposure_step(0.0, 110.0, 8000.0, 0.0, **RANGE)
    assert exposure > 0
    assert 0.0 <= gain <= RANGE["gain_max_db"]
