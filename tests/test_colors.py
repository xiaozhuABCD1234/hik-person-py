# SPDX-License-Identifier: AGPL-3.0-or-later
"""颜色转换的单元测试 —— 纯函数，不需要相机。

这里把「MVS BayerRG8 必须用 cv2.COLOR_BayerBG2BGR」这条约定钉死：
如果以后有人“顺手修正”成 COLOR_BayerRG2BGR，下面的测试会立刻失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.colors import (
    BAYER_RG8,
    BGR8_PACKED,
    MONO8,
    RGB8_PACKED,
    bayer_rg8_to_bgr,
    bgr_to_rgb,
    pixel_type_name,
)


def synthetic_bayer_rg8(size: int = 8) -> np.ndarray:
    """合成一张 MVS 语义的 BayerRG8 马赛克：R 在 (偶,偶)，B 在 (奇,奇)，G 其余。

    取值刻意拉开：R=200, B=100, G=50，这样通道放错位置一眼就能看出来。
    """
    mosaic = np.full((size, size), 50, dtype=np.uint8)
    mosaic[0::2, 0::2] = 200
    mosaic[1::2, 1::2] = 100
    return mosaic


# --------------------------------------------------------------- 像素格式映射
@pytest.mark.parametrize(
    ("value", "name"),
    [
        (MONO8, "Mono8"),
        (BAYER_RG8, "BayerRG8"),
        (RGB8_PACKED, "RGB8_Packed"),
        (BGR8_PACKED, "BGR8_Packed"),
    ],
)
def test_pixel_type_name_known(value: int, name: str) -> None:
    assert pixel_type_name(value) == name


def test_pixel_type_name_unknown_is_hex() -> None:
    assert pixel_type_name(0xDEADBEEF) == "0xdeadbeef"


def test_pixel_type_constants_match_sdk() -> None:
    """数值必须与 /opt/MVS/Samples/64/Python/MvImport/PixelType_header.py 一致。"""
    assert (MONO8, BAYER_RG8, RGB8_PACKED, BGR8_PACKED) == (
        17301505,
        17301513,
        35127316,
        35127317,
    )


# ------------------------------------------------------------- Bayer -> BGR
def test_bayer_rg8_to_bgr_shape_and_dtype() -> None:
    bgr = bayer_rg8_to_bgr(synthetic_bayer_rg8(8), 8, 8)
    assert bgr.shape == (8, 8, 3)
    assert bgr.dtype == np.uint8


def test_bayer_rg8_to_bgr_puts_red_in_red_channel() -> None:
    """MVS BayerRG8（R 在 (偶,偶)）用 COLOR_BayerBG2BGR 后，R 必须落在 R 通道。"""
    bgr = bayer_rg8_to_bgr(synthetic_bayer_rg8(8), 8, 8)
    blue, green, red = (int(v) for v in bgr[4, 4])  # (4,4) 是 R 位置
    assert (blue, green, red) == (100, 50, 200)


def test_bayer_rg8_to_bgr_ignores_extra_padding() -> None:
    """只吃前 width*height 字节，多出来的（行填充等）不参与。"""
    raw = np.concatenate([synthetic_bayer_rg8(8).ravel(), np.zeros(64, np.uint8)])
    assert bayer_rg8_to_bgr(raw, 8, 8).shape == (8, 8, 3)


def test_wrong_bayer_convention_would_swap_red_and_blue() -> None:
    """反向断言：COLOR_BayerRG2BGR 会把 R/B 对调（README 实测 corr 0.746）。"""
    import cv2

    mosaic = synthetic_bayer_rg8(8)
    wrong = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2BGR)
    right = bayer_rg8_to_bgr(mosaic, 8, 8)
    assert [int(v) for v in wrong[4, 4]] == [200, 50, 100]  # 对调了
    assert not np.array_equal(wrong, right)


# ------------------------------------------------------------------ BGR <-> RGB
def test_bgr_to_rgb_swaps_first_and_last_channel() -> None:
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)  # B=10, G=20, R=30
    assert [int(v) for v in bgr_to_rgb(bgr)[0, 0]] == [30, 20, 10]


def test_bgr_to_rgb_roundtrip() -> None:
    import cv2

    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    assert np.array_equal(cv2.cvtColor(bgr_to_rgb(bgr), cv2.COLOR_RGB2BGR), bgr)
