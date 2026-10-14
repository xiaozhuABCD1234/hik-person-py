# SPDX-License-Identifier: AGPL-3.0-or-later
"""RerunView 的单元测试 —— spawn=False、不写文件。

只留真出过问题、以后也可能再犯的点：

1. 2D 视图取景必须钉在图像尺寸上。不设 visual_bounds 时 rerun 按"视图内数据的包围盒"
   自动取景，检测框/关键点一伸出画面（人半出画就是这样）画面就跟着缩放。
2. 默认不写磁盘：只有显式给 save_path 才落盘。
3. 同一个人在任何视图里必须是同一个颜色（颜色由 id 决定，不是 rerun 的路径哈希）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr

from hik_person.track import TrackedPerson
from hik_person.viz import RerunView, color_for, value_line


def blank(height: int = 1080, width: int = 1440) -> np.ndarray:
    return np.zeros((height, width, 3), np.uint8)


def frame(number: int = 0, size: tuple[int, int] = (1080, 1440)) -> SimpleNamespace:
    return SimpleNamespace(frame_number=number, bgr=blank(*size))


def person(
    person_id: int = 1,
    *,
    x: float = 0.0,
    depth: float = 2.5,
    missed: int = 0,
    valid: int = 17,
) -> TrackedPerson:
    """一个够画 2D/3D 的假跟踪结果（不带滤波器，只有显示需要的字段）。"""
    joints = np.zeros((valid, 3), np.float32)
    joints[:, 2] = depth
    return TrackedPerson(
        id=person_id,
        conf=0.87,
        box=(300.0, 100.0, 340.0, 400.0),
        keypoints2d=np.full((17, 2), 320.0, np.float32),
        kpt_conf=np.full(17, 0.9, np.float32),
        joints=joints,
        names=["nose"] * valid,
        strips=[np.zeros((2, 3), np.float32)],
        position=np.array([x, 0.0, depth], np.float32),
        velocity=np.array([0.2, 0.0, 0.0], np.float32),
        speed=0.2,
        speed_ground=0.2,
        trail=np.zeros((3, 3), np.float32),
        age=3,
        missed=missed,
        valid_count=valid,
    )


# ------------------------------------------------------------------ 颜色与文本
def test_color_is_a_function_of_id_only() -> None:
    assert color_for(1) == color_for(1)
    assert color_for(1) != color_for(2)
    assert color_for(1) == color_for(1 + 8)  # 轮转，但同一个 id 永远同一个色


def test_value_line_lists_only_what_is_present() -> None:
    line = value_line(person(3))
    assert line.startswith("#3")
    assert "距 2.50 m" in line
    assert "速 0.20 m/s" in line
    assert "关节 17/17" in line


def test_value_line_hides_nan_instead_of_printing_it() -> None:
    """测不到距离/速度时不能出现 nan。"""
    item = person()
    item.position = np.full(3, np.nan, np.float32)
    item.speed_ground = float("nan")
    item.valid_count = 0
    line = value_line(item)
    # 什么都没有的时候至少留个置信度，仍然不能出现 nan
    assert line == "#1 · conf 0.87"
    assert "nan" not in line


# -------------------------------------------------------------------- 布局
def test_blueprint_pins_visual_bounds_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), [])

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
    v.log_frame(frame(), [person()])

    heavy = [static for entity, static in calls if entity.startswith("camera/")]
    assert heavy and all(heavy), f"图像/框没走 static：{calls}"
    entities = [entity for entity, _ in calls]
    # 送的是相机原图，不再送模型 plot 出来的标注图（框/骨架由结构化实体画）
    assert "camera/image" in entities
    assert not any("annotated" in entity for entity in entities), entities
    # 曲线需要时间戳（它们只有几十字节/帧）
    assert ("person/count", False) in calls


def test_right_column_has_values_and_coords_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """右列默认是「数值行 + 坐标曲线」；人数默认不上（要就 --show-count）。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), [])

    right = sent[0].root_container.contents[1]
    names = [type(child).__name__ for child in right.contents]
    assert any("TextDocumentView" in n for n in names), names
    assert any("TimeSeriesView" in n for n in names), names


def test_count_panel_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_coords=False, show_count=True)
    v.log_frame(frame(), [])
    right = sent[0].root_container.contents[1]
    origins = [child.origin for child in right.contents if "TimeSeriesView" in type(child).__name__]
    assert "person/count" in origins, origins


def test_blueprint_has_3d_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_pose3d=True 时右边那列下面要挂 3D 骨架视图。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), [person()])

    right = sent[0].root_container.contents[1]
    names = [type(child).__name__ for child in right.contents]
    assert any("TextDocumentView" in n for n in names), names
    assert any("Spatial3DView" in n for n in names), names


def test_blueprint_has_no_3d_panel_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), [])

    right = sent[0].root_container.contents[1]
    children = getattr(right, "contents", [])
    assert not any("3D" in type(child).__name__ for child in children)


def test_blueprint_has_depth_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_depth=True 时左列拆成「原图 + 深度图」上下两块。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_depth=True)
    v.log_frame(frame(), [])

    left = sent[0].root_container.contents[0]
    names = [type(child).__name__ for child in left.contents]
    assert sum("Spatial2DView" in n for n in names) == 2, names


def test_3d_view_pins_the_eye_so_it_does_not_auto_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), [])

    right = sent[0].root_container.contents[1]
    view3d = next(c for c in right.contents if "Spatial3DView" in type(c).__name__)
    assert "EyeControls3D" in view3d.properties


# ------------------------------------------------------------------ 3D 与深度
def test_log_people3d_logs_static_and_sets_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_people3d([person()])

    entities = {entity for entity, _ in calls}
    assert {
        "camera/pose3d",
        "camera/pose3d/joints",
        "camera/pose3d/skeleton",
        "camera/pose3d/trails",
        "camera/pose3d/positions",
    } <= entities
    assert all(static for entity, static in calls if entity.startswith("camera/"))


def test_log_people3d_with_nobody_clears_the_view(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_people3d([])
    entities = {entity for entity, _ in calls}
    assert "camera/pose3d/joints" in entities  # 记一份空的，旧骨架才会消失


def test_log_depth_logs_a_static_depth_image(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_depth=True)
    v.log_depth(np.full((1080, 1440), 2.0, np.float32))
    assert ("camera/depth", True) in calls


def test_log_depth_replaces_nan_so_colormap_has_no_fake_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nan 先换成 0：否则 rerun 侧上色时会出现一片“贴到镜头前”的假点。"""
    captured: dict = {}

    def fake_depth_image(array, **kw):
        captured["array"] = np.asarray(array)
        return object()

    monkeypatch.setattr(rr, "DepthImage", fake_depth_image)
    monkeypatch.setattr(rr, "log", lambda entity, data, **kw: None)

    v = RerunView(spawn=False, show_depth=True)
    v.log_depth(np.full((8, 8), np.nan, np.float32), max_width=8)
    assert np.isfinite(captured["array"]).all()
