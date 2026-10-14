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


def test_default_logs_static_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认不留历史：图像/框必须 static，否则 viewer 内存会随帧数一直涨。"""
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False)
    v.log_frame(frame(), blank())

    heavy = [static for entity, static in calls if entity.startswith("camera/")]
    assert heavy and all(heavy), f"图像/框没走 static：{calls}"
    entities = [entity for entity, _ in calls]
    # 送的是相机原图，不再送模型 plot 出来的标注图（框/骨架由结构化实体画）
    assert "camera/image" in entities
    assert not any("annotated" in entity for entity in entities), entities
    # 人数曲线需要时间戳（它只有几十字节/帧）
    assert ("person/count", False) in calls


def test_blueprint_has_3d_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_pose3d=True 时右边那列要拆成「人数 + 3D 骨架」上下两块。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), blank())

    right = sent[0].root_container.contents[1]
    names = [type(child).__name__ for child in right.contents]
    assert any("TimeSeriesView" in n for n in names), names
    assert any("Spatial3DView" in n for n in names), names


def test_blueprint_has_no_3d_panel_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), blank())

    right = sent[0].root_container.contents[1]
    children = getattr(right, "contents", [])
    assert not any("3D" in type(child).__name__ for child in children)


def test_log_poses3d_logs_static_and_sets_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_pose3d=True)
    strips = [np.zeros((2, 3), np.float32)]
    v.log_poses3d([(np.zeros((3, 3), np.float32), ["a", "b", "c"], strips)])

    entities = {entity for entity, _ in calls}
    assert {"camera/pose3d", "camera/pose3d/joints", "camera/pose3d/skeleton"} <= entities
    assert all(static for entity, static in calls if entity.startswith("camera/"))


def test_blueprint_has_depth_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_depth=True 时左列拆成「原图 + 深度图」上下两块。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_depth=True)
    v.log_frame(frame(), blank())

    left = sent[0].root_container.contents[0]
    names = [type(child).__name__ for child in left.contents]
    assert sum("Spatial2DView" in n for n in names) == 2, names


def test_3d_view_pins_the_eye_so_it_does_not_auto_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), blank())

    right = sent[0].root_container.contents[1]
    view3d = next(c for c in right.contents if "Spatial3DView" in type(c).__name__)
    assert "EyeControls3D" in view3d.properties


def test_log_depth_logs_a_static_depth_image(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_depth=True)
    v.log_depth(np.full((1080, 1440), 2.0, np.float32))
    assert ("camera/depth", True) in calls
