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

from hik_person.fall import FALLEN, SUSPECT
from hik_person.track import TrackedPerson
from hik_person.viz import (
    ALARM_COLOR,
    LIGHT_COLORS,
    LIGHT_IDLE_COLOR,
    RerunView,
    color_for,
    fall_alert,
    fall_alert_panel,
    fall_light,
    person_color,
    value_line,
)
from hik_person.world import CameraToWorld


def views(node) -> list[tuple[str, str]]:
    """递归收集 blueprint 里的视图：(类型名, origin)。别猜层级——rerun 的容器会套好几层。"""
    stack = [getattr(node, "root_container", node)]
    found: list[tuple[str, str]] = []
    while stack:
        current = stack.pop()
        if hasattr(current, "origin"):
            found.append((type(current).__name__, current.origin))
        for child in getattr(current, "contents", None) or []:
            if child is None or isinstance(child, (str, bytes)):
                continue
            stack.append(child)
    return found


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
        metrics={"speed": 0.2, "speed_ground": 0.2, "fall_level": 0.0},
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


def test_right_column_has_values_and_metric_curves_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """右列默认是「数值行 + 指标曲线格」；人数默认不上（要就 --show-count）。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), [])

    found = views(sent[0])
    assert ("TextDocumentView", "person/values") in found, found
    assert ("TimeSeriesView", "person/metrics/speed") in found, found
    assert ("TimeSeriesView", "person/metrics/fall") in found, found
    # 没开摔倒检测就没有报警灯那一格
    assert not [item for item in found if item[1].startswith("person/fall_light")], found


def test_count_panel_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_metrics=False, show_count=True)
    v.log_frame(frame(), [])
    found = views(sent[0])
    assert ("TimeSeriesView", "person/count") in found, found
    assert ("TimeSeriesView", "person/metrics/speed") not in found, found


def test_blueprint_has_3d_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_pose3d=True 时右边那列下面要挂 3D 骨架视图。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), [person()])

    found = views(sent[0])
    assert ("Spatial3DView", "camera/pose3d") in found, found


def test_blueprint_has_no_3d_panel_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False)
    v.log_frame(frame(), [])

    found = views(sent[0])
    assert not [item for item in found if "3D" in item[0]], found


def test_blueprint_has_depth_panel_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """show_depth=True 时左列拆成「原图 + 深度图」上下两块。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_depth=True)
    v.log_frame(frame(), [])

    found = views(sent[0])
    origins = {origin for name, origin in found if name == "Spatial2DView"}
    assert {"camera/image", "camera/depth"} <= origins, found


def test_3d_view_pins_the_eye_so_it_does_not_auto_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_frame(frame(), [])

    stack = [sent[0].root_container]
    view3d = None
    while stack:
        current = stack.pop()
        for child in getattr(current, "contents", None) or []:
            if child is None or isinstance(child, (str, bytes)):
                continue
            if type(child).__name__ == "Spatial3DView":
                view3d = child
            stack.append(child)
    assert view3d is not None and "EyeControls3D" in view3d.properties


# ------------------------------------------------------------------ 3D 与深度
def test_log_people3d_logs_static_and_sets_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_people3d([person()])

    entities = {entity for entity, _ in calls}
    assert {"camera/pose3d", "camera/pose3d/skeleton", "camera/pose3d/positions"} <= entities
    # 逐个关节的点与名字不进 3D 视图（只留骨架与每人一个位置球）
    assert not [entity for entity in entities if entity.endswith("/joints")], entities
    assert all(static for entity, static in calls if entity.startswith("camera/"))


def test_speed_curve_is_per_person_and_coords_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """右列那条曲线是"每人一条速度"，不是 x/y/z 坐标。"""
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False)
    v.log_frame(frame(), [person(2), person(3)])

    entities = [entity for entity, _ in calls]
    # 路径按组分层（与 d435i 的指标命名一致）：person/metrics/<组>/<指标>/<id>
    assert "person/metrics/speed/speed_ground/2" in entities
    assert "person/metrics/speed/speed_ground/3" in entities
    # 三维速度模长 speed 不画曲线（单目下竖直抖动太大），只在 JSONL 里
    assert not [e for e in entities if e.endswith("/speed/2")], entities
    assert not [entity for entity in entities if entity.startswith("person/coords")], entities
    # 曲线要带时间戳（static 的话画不成曲线）
    assert ("person/metrics/speed/speed_ground/2", False) in calls


def test_metric_curves_respect_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rr, "log", lambda entity, data, **kw: calls.append(entity))

    RerunView(spawn=False, show_metrics=False).log_frame(frame(), [person()])
    assert not [entity for entity in calls if entity.startswith("person/metrics")], calls


# ------------------------------------------------------------------ 摔倒显示
def test_fallen_person_turns_red_and_is_flagged() -> None:
    """摔倒：整个人换警报色（比"去曲线框里找线"快得多），数值行带 ⚠ 与躯干角。"""
    item = person()
    assert person_color(item) == color_for(item.id)
    assert fall_alert(item) == ""

    item.fall_level, item.lying = SUSPECT, True  # 倒地中（还没确认）
    assert fall_alert(item) == "倒地中"
    assert person_color(item) == color_for(item.id)

    item.fall_level, item.lying = SUSPECT, False  # 只是挨了一下冲击（跳、快速蹲下）
    assert fall_alert(item) == ""

    item.fall_level = FALLEN
    item.metrics["torso_angle"] = 88.6
    assert fall_alert(item) == "⚠ 摔倒 89°"
    assert person_color(item) == ALARM_COLOR
    assert "⚠ 摔倒" in value_line(item)


def test_fall_light_colors_by_level() -> None:
    """报警灯四档颜色：红=摔倒 / 琥珀=疑似倒地 / 绿=正常 / 暗蓝灰=无人。"""
    down = person()
    down.fall_level, down.lying = SUSPECT, True
    fallen = person(2)
    fallen.fall_level = FALLEN
    fallen.fall_state = "摔倒"

    assert fall_light([person()])[0] == LIGHT_COLORS[0]  # 正常：绿
    assert fall_light([down])[0] == LIGHT_COLORS[SUSPECT]  # 疑似倒地：琥珀
    assert fall_light([])[0] == LIGHT_IDLE_COLOR  # 无人：暗蓝灰（故意不用绿）
    color, lines = fall_light([person(3), fallen], now=0.0)
    assert color == LIGHT_COLORS[FALLEN]  # 有人摔倒就红，与别人是否正常无关
    assert lines[0].startswith("摔倒 #2")
    assert "#2" in lines[1]


def test_fall_light_flashes_only_when_fallen() -> None:
    """红灯按 fall_light_flash 明暗交替；正常/疑似不闪（闪的是警报）。"""
    fallen = person()
    fallen.fall_level = FALLEN
    bright, _ = fall_light([fallen], now=0.0)
    dim, _ = fall_light([fallen], now=0.6)  # 2 Hz -> 亮 0.5 s、暗 0.5 s
    assert bright == LIGHT_COLORS[FALLEN]
    assert dim != bright and dim[0] < bright[0]
    assert fall_light([fallen], now=0.6, flash_hz=0.0)[0] == bright  # 0 = 常亮

    down = person()
    down.fall_level, down.lying = SUSPECT, True
    assert fall_light([down], now=0.0)[0] == fall_light([down], now=0.25)[0]


def test_fall_alert_panel_states() -> None:
    """纯文本副本：正常/无人是安静的小字；报警时带躯干角、距离、已多久。"""
    assert fall_alert_panel([]) == "画面内无人"
    assert fall_alert_panel([person()]) == "正常"

    item = person()
    item.metrics["torso_angle"] = 88.6
    item.fall_level, item.lying, item.fall_since = SUSPECT, True, 10.0
    down = fall_alert_panel([item], now=12.5)
    assert down.startswith("# 疑似倒地 #")
    assert "躯干 89°" in down and "已 2.5 s" in down

    item.fall_level, item.fall_state = FALLEN, "摔倒"
    alarm = fall_alert_panel(
        [item], history=[(10.2, "#1 摔倒", "#1 检测到摔倒（躯干 90°）")], now=12.5
    )
    assert alarm.startswith("# 摔倒 #")
    assert "最近：10.2s #1 检测到摔倒（躯干 90°）" in alarm
    assert "倒地中" not in alarm.split("\n")[1]  # 确认后不再写"倒地中"


def test_fall_alarm_cell_is_on_when_fall_detection_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """开了摔倒检测：右列格子里有报警灯那一格（2D 视图，锁在色块的 [0,2]² 上）。"""
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_fall_alarm=True)
    v.log_frame(frame(), [person()])
    found = views(sent[0])
    assert ("Spatial2DView", "person/fall_light") in found, found
    assert ("TimeSeriesView", "person/metrics/speed") in found, found
    assert ("TimeSeriesView", "person/metrics/fall") in found, found


def test_alarm_light_entities_are_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_fall_alarm=True)
    v.log_frame(frame(), [person()])
    entities = {entity for entity, _ in calls}
    assert {"person/fall_light/state", "person/fall_light/text"} <= entities
    # 报警灯的色块是 static 的（每帧重记一次颜色来闪），纯文本副本也是
    assert all(static for entity, static in calls if entity.startswith("person/fall"))


def test_fall_events_are_logged_only_on_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """摔倒状态一变记一条 person/fall 日志，不能每帧都刷。"""
    calls: list[str] = []
    monkeypatch.setattr(rr, "log", lambda entity, data, **kw: calls.append(entity))

    v = RerunView(spawn=False, show_fall_alarm=True)
    item = person()
    v.log_frame(frame(1), [item])
    v.log_frame(frame(2), [item])
    assert calls.count("person/fall") == 0  # 一直正常就不用记

    item.fall_level = FALLEN
    item.fall_state = "摔倒"
    v.log_frame(frame(3), [item])
    v.log_frame(frame(4), [item])
    assert calls.count("person/fall") == 1  # 跳变那一次

    item.fall_level = 0
    item.fall_state = ""
    v.log_frame(frame(5), [item])
    assert calls.count("person/fall") == 2  # 恢复也记一次


def test_log_people3d_with_nobody_clears_the_view(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        rr, "log", lambda entity, data, **kw: calls.append((entity, kw.get("static", False)))
    )

    v = RerunView(spawn=False, show_pose3d=True)
    v.log_people3d([])
    entities = {entity for entity, _ in calls}
    assert "camera/pose3d/skeleton" in entities  # 记一份空的，旧骨架才会消失


def test_trails_are_not_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """轨迹不再画到画面上（tracker 仍记录，见 [track].trail_*）。"""
    calls: list[str] = []
    monkeypatch.setattr(rr, "log", lambda entity, data, **kw: calls.append(entity))

    RerunView(spawn=False, show_pose3d=True, world=CameraToWorld(height=1.5)).log_people3d(
        [person()]
    )
    assert not [entity for entity in calls if entity.endswith("/trails")], calls


def test_world_view_logs_a_transformed_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """开了 [world]：世界系那份必须是变换后的坐标，而不是相机系原样。"""
    captured: dict = {}

    def fake_points3d(positions, **kw):
        captured.setdefault("positions", []).append(np.asarray(positions))
        return object()

    monkeypatch.setattr(rr, "Points3D", fake_points3d)
    monkeypatch.setattr(rr, "log", lambda entity, data, **kw: None)

    world = CameraToWorld(height=1.5)
    v = RerunView(spawn=False, show_pose3d=True, world=world, show_world_view=True)
    v.log_people3d([person()])

    # Points3D 只被调两次：相机系位置、世界系位置（关节不再单独记）
    calls = captured["positions"]
    assert len(calls) == 2
    camera_position = calls[0]
    world_position = calls[1]
    assert np.allclose(camera_position, [[0.0, 0.0, 2.5]], atol=1e-5)
    assert np.allclose(
        world_position, [world.apply(np.array([0.0, 0.0, 2.5], np.float32))], atol=1e-5
    )


def test_world_view_is_off_when_world_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list = []
    monkeypatch.setattr(rr, "send_blueprint", lambda bp: sent.append(bp))

    v = RerunView(spawn=False, show_pose3d=True, show_world_view=True)
    v.log_frame(frame(), [])
    right = sent[0].root_container.contents[1]
    origins = [c.origin for c in right.contents if hasattr(c, "origin")]
    assert "world" not in origins, origins


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
