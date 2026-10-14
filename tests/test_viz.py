# SPDX-License-Identifier: AGPL-3.0-or-later
"""RerunView 的单元测试 —— spawn=False、不写文件。

只留两个真出过问题、以后也可能再犯的点：

1. 2D 视图取景必须钉在图像尺寸上。不设 visual_bounds 时 rerun 按"视图内数据的包围盒"
   自动取景，检测框/关键点一伸出画面（人半出画就是这样）画面就跟着缩放。
2. 默认不写磁盘：只有显式给 save_path 才落盘。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr

from hik_person.viz import RerunView


def blank(height: int = 1080, width: int = 1440) -> np.ndarray:
    return np.zeros((height, width, 3), np.uint8)


def frame(number: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame_number=number)


def test_blueprint_pins_visual_bounds_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), blank())

    spatial = sent[0].root_container.contents[0]
    bounds = spatial.properties["VisualBounds2D"].range.as_arrow_array().to_pylist()
    assert bounds == [{"x_range": [0.0, 1440.0], "y_range": [0.0, 1080.0]}]


def test_default_run_records_nothing_to_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list = []
    monkeypatch.setattr(rr, "save", lambda path: saved.append(path))
    monkeypatch.setattr(rr, "spawn", lambda **kw: None)
    RerunView(spawn=True)
    assert saved == []
