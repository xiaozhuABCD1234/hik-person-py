# SPDX-License-Identifier: AGPL-3.0-or-later
"""人数横幅的单元测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np

from hik_person.annotate import draw_count_banner


def frame(height: int = 120, width: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_draw_count_keeps_shape_and_dtype() -> None:
    out = draw_count_banner(frame(), 2)
    assert out.shape == (120, 480, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_draw_count_does_not_touch_the_input() -> None:
    """横幅画在副本上，不把调用方的帧涂黑。"""
    original = frame()
    draw_count_banner(original, 1)
    assert original.min() == 255


def test_draw_count_paints_a_banner_in_the_top_left() -> None:
    out = draw_count_banner(frame(), 1)
    assert out[4, 4].tolist() == [0, 0, 0]  # 横幅底色
    assert out[110, 470].tolist() == [255, 255, 255]  # 右下角不受影响


def test_draw_count_changes_text_colour_with_count() -> None:
    """有人是绿色，没人（0 人）是灰色 —— 一眼能看出当前有没有人。"""
    someone = draw_count_banner(frame(), 1)
    nobody = draw_count_banner(frame(), 0)
    assert not np.array_equal(someone, nobody)


def test_draw_count_handles_non_contiguous_input() -> None:
    flipped = frame()[:, ::-1]
    out = draw_count_banner(flipped, 1)
    assert out.shape == flipped.shape
    assert out.flags["C_CONTIGUOUS"]
