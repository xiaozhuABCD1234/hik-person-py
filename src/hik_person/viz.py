# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化：2D 原图/框/骨架、估计深度、相机系 3D 骨架/轨迹/位置、数值行。

记录到这些实体：

===================================  =======================================
``camera/image``                     相机原图（框/骨架由下面的结构化实体画）
``camera/image/boxes``               框的 Boxes2D（标签带 id 与置信度）
``camera/image/keypoints``           2D 关键点（带名称与 id）
``camera/image/skeleton``            2D 骨架连线
``camera/depth``                     估计深度（DepthImage，悬停看米数）
``camera/pose3d``                    相机系 3D 视图根（RIGHT_HAND_Y_DOWN）
``camera/pose3d/skeleton``           3D 骨架连线（单目深度）
``camera/pose3d/positions``          每个人的 3D 位置（标签 = id + 距离 + 速度）
``person/values``                    极简数值行（每人一行短文本，报警时带 ⚠）
``person/fall_light``                摔倒报警灯（一格 2D：整格变色 + 格内文字；红=摔倒会闪）
``person/fall_alert``                报警灯的纯文本副本（markdown，可选中复制）
``person/fall``                      摔倒状态变化日志（只在跳变时记一条）
``person/metrics/speed/<id>``        每个人的速度曲线（右侧，一人一条）
``person/count``                     人数标量
``person/events``                    人数变化日志
``world/...``                        世界系 3D 视图根（RIGHT_HAND_Z_UP，可选）
``world/{skeleton,positions}``       世界系副本（刚体变换后）
===================================  =======================================

每个人的颜色由 id 决定（``color_for``），所以同一个 id 在 2D、3D 与坐标曲线里都是
同一种颜色；曲线颜色是显式声明的，图例直接显示 ``#1`` 而不是一长串实体路径。

进 rerun 前统一 `bgr_to_rgb()`：`rr.Image` 默认按 RGB 解读三通道数据，直接喂 BGR 会红蓝互换。
"""

from __future__ import annotations

import contextlib
from collections import deque

import numpy as np

from .colors import bgr_to_rgb
from .fall import FALLEN, NORMAL, SUSPECT
from .pose import bones, visible_keypoints

# 按 id 轮转的颜色（RGB）。挑的是互相区分度高、在深色背景上也看得清的几种。
PALETTE: tuple[tuple[int, int, int], ...] = (
    (90, 220, 120),
    (255, 180, 60),
    (120, 180, 255),
    (255, 120, 160),
    (180, 140, 255),
    (120, 230, 230),
    (240, 220, 90),
    (150, 255, 140),
)
POSITION_RADIUS = 0.04  # 米：3D 位置球半径
ALARM_COLOR = (255, 64, 64)  # 摔倒报警色（2D/3D 都换成它，比"去曲线里找线"快得多）

# 摔倒报警灯：占「摔倒警报」那一格（2D 视图：整格一块颜色 + 白字）。
# 用颜色而不是表情符号：颜色是屏幕上最先被看见的东西，也不会有字体缺字的问题。
LIGHT_ORIGIN = "person/fall_light"
LIGHT_STATE = f"{LIGHT_ORIGIN}/state"  # 铺满整格的色块（2×2 纯色图）
LIGHT_TEXT = f"{LIGHT_ORIGIN}/text"  # 格内文字（2D 标签：屏幕像素大小，格子小也看得清）
LIGHT_COLORS: dict[int, tuple[int, int, int]] = {
    NORMAL: (48, 200, 96),  # 绿：正常
    SUSPECT: (255, 176, 32),  # 琥珀：疑似倒地
    FALLEN: (255, 64, 64),  # 红：摔倒（跟"整个人变红"同一个色）
}
LIGHT_IDLE_COLOR = (90, 100, 120)  # 画面内无人：暗蓝灰（故意不用绿——没人在 ≠ 一切正常）
LIGHT_TEXT_COLOR = (255, 255, 255)
LIGHT_FLASH_DIM = 0.35  # 闪烁的暗相位保留多少亮度（还看得出是红灯）
# 色块怎么画：这个版本的 rr.Boxes2D 只画边框（没有 fill_mode），Points2D 只能画圆盘，
# 真正"铺满整格的矩形"只能用纯色小图（试出来的：1×1 的图 rerun 会画成白屏，2×2 正常）。
LIGHT_FILL_SIZE = 2
# 文字锚点（图像坐标）：rerun 的 2D 标签以锚点为中心、向上展开，所以锚点给图中心稍下一点，
# 几行字才落在色块中间；x 给图中心才是左右居中（靠边会被格子裁掉）。
LIGHT_TEXT_ANCHOR = (LIGHT_FILL_SIZE / 2, 0.935)
ALERT_TEXT_ORIGIN = "person/fall_alert"  # 纯文本副本（可选中/可复制）


def color_for(person_id: int) -> tuple[int, int, int]:
    """同一个 id 永远同一个颜色（id 从 1 开始，单调递增）。"""
    return PALETTE[(int(person_id) - 1) % len(PALETTE)]


def person_color(person) -> tuple[int, int, int]:
    """报警时整个人变红：视觉上比"去某个框里找条线"快得多。"""
    return ALARM_COLOR if getattr(person, "fall_level", 0) >= FALLEN else color_for(person.id)


def fall_alert(person) -> str:
    """数值行里的摔倒标注：确认了才写"⚠ 摔倒"（带躯干角，方便一眼判断是不是真躺下了），
    只是倒地中的写"倒地中"，仅挨一下冲击（跳、快速蹲下）不写——别把正常动作显示成警报。"""
    level = int(getattr(person, "fall_level", 0))
    if level >= FALLEN:
        angle = person.metrics.get("torso_angle")
        if angle is not None and np.isfinite(angle):
            return f"⚠ 摔倒 {angle:.0f}°"
        return "⚠ 摔倒"
    if level == SUSPECT and getattr(person, "lying", False):
        return "倒地中"
    return ""


def _fall_detail(person, now: float | None = None, *, with_state: bool = True) -> str:
    """纯文本副本里的一行（长写法）：谁、什么状态、躯干多横、多远、已经多久了。

    ``with_state=False`` 时省掉状态名（标题里已经写了，明细行重复就啰嗦）。
    """
    level = int(getattr(person, "fall_level", 0))
    parts = [f"#{person.id}"]
    if with_state:
        parts.append(person.fall_state if level >= FALLEN else "倒地中")
    angle = person.metrics.get("torso_angle")
    if angle is not None and np.isfinite(angle):
        parts.append(f"躯干 {angle:.0f}°")
    if np.isfinite(person.distance):
        parts.append(f"距 {person.distance:.1f} m")
    since = getattr(person, "fall_since", None)
    if since is not None and now is not None:
        parts.append(f"已 {max(now - since, 0.0):.1f} s")
    return " · ".join(parts)


def _light_detail(person, now: float | None = None) -> str:
    """报警灯格内的一行（短写法）：``#1 · 躯干 90° · 4.1 m · 0.7 s``。

    格内文字是居中画的、不会自动折行，太长就左右被格子裁掉，所以这里去掉"距/已"这类字，
    只留数字与单位。
    """
    parts = [f"#{person.id}"]
    angle = person.metrics.get("torso_angle")
    if angle is not None and np.isfinite(angle):
        parts.append(f"躯干 {angle:.0f}°")
    if np.isfinite(person.distance):
        parts.append(f"{person.distance:.1f} m")
    since = getattr(person, "fall_since", None)
    if since is not None and now is not None:
        parts.append(f"{max(now - since, 0.0):.1f} s")
    return " · ".join(parts)


def fall_light(
    people, history=(), now: float | None = None, *, flash_hz: float = 2.0
) -> tuple[tuple[int, int, int], list[str]]:
    """报警灯：返回 (颜色, 文字行)。纯函数，不依赖 rerun，方便单测。

    四档颜色：红（摔倒）/ 琥珀（疑似倒地）/ 绿（正常）/ 暗蓝灰（画面内无人）。
    ``flash_hz`` > 0 时红灯按这个频率明暗交替——报警灯就该闪；其余档位常亮。
    """
    alive = list(people)
    alarmed = [p for p in alive if int(getattr(p, "fall_level", 0)) >= FALLEN]
    down = [
        p
        for p in alive
        if int(getattr(p, "fall_level", 0)) == SUSPECT and getattr(p, "lying", False)
    ]
    if alarmed:
        who = "、".join(f"#{p.id}" for p in alarmed)
        head = f"摔倒 {who}" + ("（多人）" if len(alarmed) > 1 else "")
        color = LIGHT_COLORS[FALLEN]
    elif down:
        head = "疑似倒地 " + "、".join(f"#{p.id}" for p in down)
        color = LIGHT_COLORS[SUSPECT]
    elif alive:
        head, color = "正常", LIGHT_COLORS[NORMAL]
    else:
        head, color = "画面内无人", LIGHT_IDLE_COLOR
    if (
        color is LIGHT_COLORS[FALLEN]
        and flash_hz > 0
        and now is not None
        and int(now * flash_hz) % 2
    ):
        color = tuple(int(c * LIGHT_FLASH_DIM) for c in color)  # 暗相位：暗下去但还看得出是红的
    lines = [head, *(_light_detail(p, now) for p in [*alarmed, *down])]
    # 格内只放最近一次状态变化（多了会把格子占满，完整的在纯文本副本里）
    lines += [f"最近 {t:.1f}s {short}" for t, short, _ in list(history)[-1:]]
    return color, lines


def fall_alert_panel(people, history=(), now: float | None = None) -> str:
    """报警灯的纯文本副本：标题 + 报警的人各一行 + 最近几条状态变化。

    报警灯（2D 色块）是给眼睛看的，这份文本是给"选中/复制/搜索"用的——两者内容同源。
    单独拎出来也是为了可测（不依赖 rerun）。
    """
    alive = list(people)
    alarmed = [p for p in alive if int(getattr(p, "fall_level", 0)) >= FALLEN]
    down = [
        p
        for p in alive
        if int(getattr(p, "fall_level", 0)) == SUSPECT and getattr(p, "lying", False)
    ]
    if alarmed:
        who = "、".join(f"#{p.id}" for p in alarmed)
        head = f"# 摔倒 {who}" + ("（多人）" if len(alarmed) > 1 else "")
    elif down:
        head = "# 疑似倒地 " + "、".join(f"#{p.id}" for p in down)
    elif alive:
        head = "正常"
    else:
        head = "画面内无人"
    lines = [head, *(f"- {_fall_detail(person, now)}" for person in [*alarmed, *down])]
    if history:
        lines += ["", "最近：" + "；".join(f"{t:.1f}s {long}" for t, _, long in history)]
    return "\n".join(lines)


# 指标曲线按"什么时候会一起看"分组：框里的实体路径是 person/metrics/<组>/<指标>/<id>，
# 组名用来开一个 TimeSeriesView（一人一条曲线，颜色与图例名由 id 决定）。
METRIC_GROUP: dict[str, str] = {
    "speed": "speed",
    "speed_ground": "speed",
    "fall_level": "fall",
    "torso_angle": "fall",
}


def metric_group_path(group: str) -> str:
    """一个指标组的视图根（给 TimeSeriesView 的 origin 用）。"""
    return f"person/metrics/{group}"


def metric_path(metric: str, person_id: int) -> str:
    """一个指标的实体路径（按组分层）。"""
    return f"{metric_group_path(METRIC_GROUP.get(metric, 'other'))}/{metric}/{person_id}"


def value_line(person) -> str:
    """一个人一行短文本：只列本帧真有的项，不占地方也看不到 nan。

    没有深度的那几帧（或单目深度全丢）就只有 id 与置信度，不会凭空写一个距离。
    """
    parts = [f"#{person.id}"]
    alert = fall_alert(person)
    if alert:
        parts.append(alert)
    if np.isfinite(person.distance):
        parts.append(f"距 {person.distance:.2f} m")
    if np.isfinite(person.speed_ground):
        parts.append(f"速 {person.speed_ground:.2f} m/s")
        parts.append(person.state)
    if person.valid_count:
        parts.append(f"关节 {person.valid_count}/{len(person.kpt_conf)}")
    if len(parts) == 1:
        parts.append(f"conf {person.conf:.2f}")
    return " · ".join(parts)


def _blueprint_module():
    """blueprint API 的位置在 rerun 0.23 前后变过：优先 rerun.blueprint，退回 rerun。"""
    try:
        import rerun.blueprint as rrb
    except ImportError:  # pragma: no cover - 老版本
        import rerun as rrb

    return rrb


class RerunView:
    def __init__(
        self,
        app_id: str = "hik-person",
        *,
        spawn: bool = True,
        save_path: str | None = None,
        kpt_conf: float = 0.5,
        keep_history: bool = False,
        memory_limit: str = "512MB",
        server_memory_limit: str = "512MB",
        jpeg_quality: int = 75,
        port: int = 9876,
        show_pose3d: bool = False,
        show_depth: bool = False,
        show_values: bool = True,
        show_metrics: bool = True,
        show_count: bool = False,
        show_fall_alarm: bool = False,
        fall_light_flash: float = 2.0,
        world=None,
        show_world_view: bool = False,
        eye: tuple[float, float, float] = (0.0, 0.0, -1.5),
        look_at: tuple[float, float, float] = (0.0, 0.0, 2.5),
        eye_up: tuple[float, float, float] = (0.0, -1.0, 0.0),
    ):
        """跑起来就弹 viewer（除非给了 save_path）。

        viewer 只当显示器：不给 save_path 就不往磁盘写任何东西。
        keep_history=False（默认）时图像/框/关键点用 static 记录，viewer 只保留最新一份，
        没有时间轴可回拖，内存不随帧数涨；memory_limit 是兜底硬上限。
        """
        import rerun as rr

        self.rr = rr
        self.kpt_conf = kpt_conf
        self.keep_history = keep_history
        self.jpeg_quality = int(jpeg_quality)
        self.show_pose3d = show_pose3d
        self.show_depth = show_depth
        self.show_values = bool(show_values)
        # 指标曲线（速度 + 摔倒两组）占右列中间的格子；报警灯那一格只在开了摔倒检测时才上
        self.show_metrics = bool(show_metrics)
        self.show_fall_alarm = bool(show_fall_alarm)
        # 报警灯闪烁频率（Hz）：红灯明暗交替；0 = 常亮
        self.fall_light_flash = max(float(fall_light_flash), 0.0)
        self.show_count = bool(show_count)
        self.world = world
        self.show_world_view = bool(show_world_view and world is not None)
        self.eye = tuple(eye)
        self.look_at = tuple(look_at)
        self.eye_up = tuple(eye_up)
        self._last_count: int | None = None  # None = 还没记录过人数
        self._image_size: tuple[int, int] | None = None  # (宽, 高)，用于固定取景
        self._coords_logged: set[str] = set()  # 3D 视图的坐标系（每个根只发一次）
        self._styled_series: set[str] = set()  # 已经声明过颜色的曲线（只声明一次）
        self._fall_levels: dict[int, int] = {}  # 每个人的摔倒等级，用于只在跳变时记事件
        self._fall_history: deque = deque(maxlen=3)  # 报警灯里附的"最近"几条 (时刻, 短, 长)

        rr.init(app_id, spawn=False)
        if save_path:
            rr.save(str(save_path))
        elif spawn:
            # rerun 默认 memory_limit="75%"（本机约 23 GB），这里显式压住
            rr.spawn(
                port=port,
                memory_limit=memory_limit,
                server_memory_limit=server_memory_limit,
                hide_welcome_screen=True,
            )

    def _send_blueprint(self, width: int, height: int) -> None:
        """布局 + 把 2D 视图的取景钉死在图像范围内。

        不设 visual_bounds 时，rerun 按"视图内数据的包围盒"自动取景：检测框/关键点
        一旦伸出画面（人半出画就是这样），取景就跟着放大缩小，看起来像画面在自动缩放。
        """
        rrb = _blueprint_module()
        view_args: dict = {"origin": "camera/image", "name": "彩色 + 骨架"}
        # 老版本没有 VisualBounds2D：只能退回自动取景
        with contextlib.suppress(AttributeError, TypeError):
            view_args["visual_bounds"] = rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height])
        try:
            left = rrb.Spatial2DView(**view_args)
            if self.show_depth:
                # 左列：上面原图、下面估计深度图
                left = rrb.Vertical(
                    left,
                    rrb.Spatial2DView(origin="camera/depth", name="估计深度"),
                    row_shares=[1, 1],
                )

            panels: list = []
            shares: list[int] = []
            top: list = []
            top_shares: list[int] = []
            if self.show_values:
                top.append(rrb.TextDocumentView(origin="person/values", name="数值"))
                top_shares.append(1)
            if self.show_metrics:
                # 指标曲线按组拆格 + 摔倒警报那一格（整块颜色，不是曲线）
                top.append(self._metrics_panel(rrb))
                top_shares.append(5)
            if top:
                panels.append(
                    top[0] if len(top) == 1 else rrb.Vertical(*top, row_shares=top_shares)
                )
                shares.append(3)
            if self.show_count:
                panels.append(rrb.TimeSeriesView(origin="person/count", name="人数"))
                shares.append(1)
            if self.show_pose3d:
                panels.append(self._make_3d_view(rrb, "camera/pose3d", "3D 姿态"))
                shares.append(3)
            if self.show_world_view:
                panels.append(self._make_3d_view(rrb, "world", "世界系（地面）"))
                shares.append(2)

            right = None
            if len(panels) == 1:
                right = panels[0]
            elif panels:
                right = rrb.Vertical(*panels, row_shares=shares)
            library = rrb.Horizontal(left, right, column_shares=[3, 1]) if right else left
            blueprint = rrb.Blueprint(library, rrb.TimePanel(state="collapsed"))
        except AttributeError:
            # blueprint API 不可用就用 rerun 的默认布局，不影响记录本身
            return
        self.rr.send_blueprint(blueprint)

    def _metrics_panel(self, rrb):
        """指标曲线 + 报警灯的格子：速度、摔倒两组曲线（+ 摔倒警报那一格）。

        一个框里塞几条线根本看不清，按"什么时候会一起看"拆格子；摔倒警报不是曲线
        （曲线得去找），而是一格整块颜色 + 格内文字。
        """
        views = [rrb.TimeSeriesView(origin=metric_group_path("speed"), name="速度")]
        if self.show_fall_alarm:
            views.append(rrb.TimeSeriesView(origin=metric_group_path("fall"), name="摔倒"))
            views.append(self._alarm_view(rrb, LIGHT_ORIGIN, "摔倒警报"))
        else:
            views.append(rrb.TimeSeriesView(origin=metric_group_path("fall"), name="摔倒"))
        grid = getattr(rrb, "Grid", None)
        if grid is None:  # pragma: no cover - 老版本没有 Grid
            return rrb.Vertical(*views)
        return grid(*views)

    def _alarm_view(self, rrb, origin: str, name: str):
        """报警灯那一格：2D 视图，坐标系锁在色块的 [0,2]² 上。

        不锁的话 rerun 会按内容自动取景；锁住后色块就是"整格一块颜色"，与报警灯的样子一致
        （内容就是 [0,2]² 的色块，所以两种情况下画面一样）。
        """
        bounds = getattr(rrb, "VisualBounds2D", None)
        if bounds is not None:
            with contextlib.suppress(Exception):  # 老版本没有 visual_bounds
                return rrb.Spatial2DView(
                    origin=origin,
                    name=name,
                    visual_bounds=bounds(
                        x_range=[0.0, LIGHT_FILL_SIZE], y_range=[0.0, LIGHT_FILL_SIZE]
                    ),
                )
        return rrb.Spatial2DView(origin=origin, name=name)

    def _make_3d_view(self, rrb, origin: str, name: str):
        """3D 视图，相机钉在默认位置。

        不设 ``eye_controls`` 时 rerun 会按数据的包围盒自动取景，而单目深度逐帧抖动，
        骨架大小一直变，画面就跟着"自动缩放"。这里把眼位/视点/上方向固定下来，
        打开后就用一个固定视角看；想转视角在 viewer 里鼠标拖即可。
        """
        eye_cls = getattr(rrb, "EyeControls3D", None)
        controls = None
        if eye_cls is not None:
            with contextlib.suppress(Exception):  # 老版本没有 kind/这些字段
                controls = eye_cls(
                    kind="Orbital",
                    position=list(self.eye),
                    look_target=list(self.look_at),
                    eye_up=list(self.eye_up),
                )
        if controls is None:
            return rrb.Spatial3DView(origin=origin, name=name)
        return rrb.Spatial3DView(origin=origin, name=name, eye_controls=controls)

    def _set_time(self, frame_number: int) -> None:
        # 0.23+ 用 set_time(...)，更早的版本用 set_time_sequence
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_number)
        else:  # pragma: no cover - 老版本
            rr.set_time_sequence("frame", frame_number)

    def _log(self, entity: str, data) -> None:
        """默认 static：不带时间戳，viewer 只保留最新一份，没有历史可回拖。

        图像/框/关键点是内存大头（原图 4.67 MB/帧），它们不留历史后 viewer 的内存
        就不随帧数涨；曲线需要时间戳，在 ``log_frame`` 里单独 set_time 后写。
        """
        if self.keep_history:
            self.rr.log(entity, data)
        else:
            self.rr.log(entity, data, static=True)

    def _log_scalar(self, entity: str, value: float, *, person_id: int | None = None) -> None:
        """标量曲线：**带时间戳**（跟 ``_log`` 的 static 相反），否则画不成曲线。

        样式（颜色 + 图例名）只在第一次记：不写的话 rerun 会按路径哈希分配颜色，
        同一个人在不同曲线里可能是不同颜色，图例还会显示完整实体路径。
        """
        rr = self.rr
        scalar = getattr(rr, "Scalars", None) or rr.Scalar
        rr.log(entity, scalar(float(value)))
        if entity in self._styled_series:
            return
        self._styled_series.add(entity)
        lines = getattr(rr, "SeriesLines", None)
        if lines is None or person_id is None:  # pragma: no cover - 老版本没有 SeriesLines
            return
        with contextlib.suppress(Exception):
            rr.log(
                entity,
                lines(colors=[color_for(person_id)], names=[f"#{person_id}"]),
                static=True,
            )

    def _encode(self, bgr: np.ndarray):
        """按 jpeg_quality 压过再送。

        原图（1440×1080）每帧 4.67 MB，viewer 得把它整份留着供回拖，内存实测 +70 MB/s；
        压成 JPEG 后每帧约 0.13 MB，+4 MB/s。压过的只是一份显示副本，
        `--save-dir` 存的图仍是原分辨率。
        """
        rr = self.rr
        image = rr.Image(bgr_to_rgb(bgr))
        if self.jpeg_quality > 0 and hasattr(image, "compress"):
            return image.compress(jpeg_quality=self.jpeg_quality)
        return image

    # ---------------------------------------------------------------- 2D
    def log_frame(self, frame, people) -> None:
        """记一帧的 2D 内容：原图 + 结构化框/关键点/骨架 + 数值行。

        ``frame.bgr`` 必须是**相机原图**，不要传 ``result.plot()`` 画过的图：框、
        关键点、骨架都由 rerun 用结构化数据在这里画，传标注图会让同一份信息画两遍。
        所有叠加信息都从 ``people``（带 id 的跟踪结果）来，所以 2D 上也看得到 id。
        """
        rr = self.rr
        number = frame.frame_number
        if self.keep_history:
            self._set_time(number)

        # 分辨率变了（换相机/换模式）才重发 blueprint，把取景钉到新尺寸
        height, width = frame.bgr.shape[:2]
        if (width, height) != self._image_size:
            self._image_size = (width, height)
            self._send_blueprint(width, height)

        self._log("camera/image", self._encode(frame.bgr))
        self._log_boxes(people)
        self._log_poses2d(people)

        live = [person for person in people if person.missed == 0]
        count = len(live)
        if count != self._last_count:
            self._last_count = count
            self._log(
                "person/events",
                rr.TextLog(
                    f"{count} person(s): " + ", ".join(f"#{d.id} conf={d.conf:.2f}" for d in live)
                ),
            )

        self._log_fall_events(live, number)
        self._log_fall_light(live, number)
        self._log_fall_alert(live, number)

        # 曲线一直带时间戳（rerun 靠它画曲线）：人数与每个人的指标（速度/摔倒）
        self._set_time(number)
        self._log_scalar("person/count", count)
        if self.show_metrics:
            for person in live:
                for key, value in person.metrics.items():
                    if np.isfinite(value):
                        self._log_scalar(
                            metric_path(key, person.id), float(value), person_id=person.id
                        )
        if self.show_values:
            self._log_values(live)

    def _log_boxes(self, people) -> None:
        rr = self.rr
        cls = getattr(rr, "Boxes2D", None) or rr.Box2D  # 0.14 叫 Box2D
        # 图像在 camera/image，框挂到它的子路径上就会叠加显示
        if not people:
            empty = np.zeros((0, 2), dtype=np.float32)
            self._log("camera/image/boxes", cls(centers=empty, sizes=empty))
            return
        xyxy = np.array([d.box for d in people], dtype=np.float32)
        centers = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1
        )
        sizes = np.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        self._log(
            "camera/image/boxes",
            cls(
                centers=centers,
                sizes=sizes,
                labels=[person.label for person in people],
                colors=[person_color(person) for person in people],
            ),
        )

    def _log_poses2d(self, people) -> None:
        """2D 关键点用 Points2D（带名字），骨头用 LineStrips2D；颜色跟着 id。"""
        rr = self.rr
        points: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[tuple[int, int, int]] = []
        strips: list[np.ndarray] = []
        strip_colors: list[tuple[int, int, int]] = []
        for person in people:
            color = person_color(person)
            visible, names = visible_keypoints(
                person.keypoints2d, person.kpt_conf, threshold=self.kpt_conf
            )
            if len(visible):
                points.append(visible)
                labels.extend(f"#{person.id} {name}" for name in names)
                colors.extend([color] * len(visible))
            for strip in bones(person.keypoints2d, person.kpt_conf, threshold=self.kpt_conf):
                strips.append(strip)
                strip_colors.append(color)

        self._log(
            "camera/image/keypoints",
            rr.Points2D(
                np.concatenate(points) if points else np.zeros((0, 2), dtype=np.float32),
                labels=labels,
                colors=colors or None,
                radii=3.0,
            ),
        )
        self._log("camera/image/skeleton", rr.LineStrips2D(strips, colors=strip_colors))

    def _log_values(self, live) -> None:
        """极简数值行：每人一行短文本（距/速/状态/有效关节）。

        刻意不用表格：一是表格在窄窗口里要横向拖，二是"大段文字"把图像挤没了。
        每行只列本帧真有的项，所以看不到 nan；摔倒报警的人行首带 ⚠。
        """
        rr = self.rr
        document = getattr(rr, "TextDocument", None)
        if document is None:  # pragma: no cover - 老版本没有 TextDocument
            return
        lines = [f"- {value_line(person)}" for person in live] or ["- 无人"]
        media_type = getattr(rr, "MediaType", None)
        kwargs = {"media_type": media_type.MARKDOWN} if media_type is not None else {}
        self._log("person/values", document("\n".join(lines), **kwargs))

    def _log_fall_events(self, live, t: float) -> None:
        """摔倒状态一变就记一条 TextLog（rerun 的文本日志面板里能看到整串时间线）。

        只在跳变时记，不是每帧都记：报警从"倒地中"到"摔倒"再到"已恢复"一共就几条。
        历史存三元组 (时刻, 短写法, 长写法)：报警灯那格用短写法（格子窄，长了会被裁），
        纯文本副本用长写法。
        """
        rr = self.rr
        for person in live:
            level = int(getattr(person, "fall_level", 0))
            if level == self._fall_levels.get(person.id, 0):
                continue
            self._fall_levels[person.id] = level
            angle = person.metrics.get("torso_angle")
            detail = f"（躯干 {angle:.0f}°）" if angle is not None and np.isfinite(angle) else ""
            if level >= FALLEN:
                text, short = f"#{person.id} 检测到摔倒{detail}", f"#{person.id} 摔倒"
            elif level == SUSPECT:
                text, short = f"#{person.id} 疑似摔倒 / 倒地中{detail}", f"#{person.id} 疑似"
            else:
                text, short = f"#{person.id} 已恢复正常", f"#{person.id} 恢复"
            self._fall_history.append((t, short, text))
            self._log("person/fall", rr.TextLog(text))

    def _log_fall_light(self, live, t: float) -> None:
        """摔倒报警灯：一格 2D 视图里画一块颜色 + 白字（不用表情符号）。

        红=摔倒（还会闪）/ 琥珀=疑似倒地 / 绿=正常 / 暗蓝灰=画面内无人。闪烁靠"每帧重新
        记一次颜色"实现，所以 keep_history=False 的 static 记录也照样闪。
        """
        if not self.show_fall_alarm:
            return
        rr = self.rr
        points = getattr(rr, "Points2D", None)
        if points is None:  # pragma: no cover - 老版本
            return
        color, lines = fall_light(live, self._fall_history, t, flash_hz=self.fall_light_flash)
        # 纯色小图铺满整格：那一格本身就是"色块"，文字画在它里面（居中）
        self._log(
            LIGHT_STATE,
            rr.Image(np.full((LIGHT_FILL_SIZE, LIGHT_FILL_SIZE, 3), color, dtype=np.uint8)),
        )
        # 2D 标签的文字是屏幕像素大小：格子缩小也不会看不见（不像图像里的字会跟着缩）
        self._log(
            LIGHT_TEXT,
            points(
                positions=[list(LIGHT_TEXT_ANCHOR)],
                labels=["\n".join(lines)],
                colors=[LIGHT_TEXT_COLOR],
                radii=[0.0],
            ),
        )

    def _log_fall_alert(self, live, t: float) -> None:
        """报警灯的纯文本副本（markdown）：报警灯是给眼睛看的，这份给"选中/复制"用。"""
        if not self.show_fall_alarm:
            return
        document = getattr(self.rr, "TextDocument", None)
        if document is None:  # pragma: no cover - 老版本没有 TextDocument
            return
        media_type = getattr(self.rr, "MediaType", None)
        kwargs = {"media_type": media_type.MARKDOWN} if media_type is not None else {}
        text = fall_alert_panel(live, self._fall_history, t)
        self._log(ALERT_TEXT_ORIGIN, document(text, **kwargs))

    # ---------------------------------------------------------------- 深度
    def log_depth(self, depth: np.ndarray, *, max_width: int = 480) -> None:
        """把估计深度图送进 viewer（Turbo 伪彩）。

        深度是 float32（1440×1080 每帧 6 MB），送之前先整倍数降采样，别为一个小窗
        白白撑带宽。nan（无有效深度）先换成 0，否则 rerun 侧上色时会出现一片
        "贴到镜头前"的假点。``rr.DepthImage`` 在 viewer 侧上色，悬停能看到米数。
        """
        rr = self.rr
        step = max(1, -(-depth.shape[1] // max_width))  # 向上取整的整除
        downsampled = np.nan_to_num(depth[::step, ::step], nan=0.0)
        downsampled = np.ascontiguousarray(downsampled, dtype=np.float32)
        kwargs: dict = {"meter": 1.0}
        colormap = getattr(getattr(rr, "components", None), "Colormap", None)
        if colormap is not None:
            kwargs["colormap"] = colormap.Turbo
        self._log("camera/depth", rr.DepthImage(downsampled, **kwargs))

    # ---------------------------------------------------------------- 3D
    def log_people3d(self, people) -> None:
        """记 3D：骨架连线 + 每人一个位置球（相机坐标系，米）；开了 ``[world]`` 再记一份世界系。

        **不记逐个关节**：17 个带名字的关节点会盖住骨架本身，鼠标扫过去还全是
        坐标读数；关节数据仍在（``person.joints`` / JSONL），只是不进 3D 视图。
        想逐个关节看坐标就看 2D 关键点，或把 joints 那段加回来。

        与 2D 一样走 static：viewer 只留最新一份，``people`` 为空就把 3D 视图清空。
        世界系那份只是同一个刚体变换套一遍，方便直接看“人在地面上的位置”。
        """
        rr = self.rr
        self._log_coords("camera/pose3d", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, name="3D 姿态")
        if self.world is not None:
            self._log_coords("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, name="世界系")

        strips: list[np.ndarray] = []
        strip_colors: list[tuple[int, int, int]] = []
        positions: list[np.ndarray] = []
        position_labels: list[str] = []
        position_colors: list[tuple[int, int, int]] = []

        for person in people:
            color = person_color(person)
            for strip in person.strips:
                strips.append(np.asarray(strip, dtype=np.float32))
                strip_colors.append(color)
            if np.isfinite(person.position).all():
                positions.append(np.asarray(person.position, dtype=np.float32))
                label = (
                    f"#{person.id} {person.distance:.2f} m · {person.speed_ground:.2f} m/s"
                    if np.isfinite(person.speed_ground)
                    else f"#{person.id} {person.distance:.2f} m"
                )
                alert = fall_alert(person)
                position_labels.append(f"{label} · {alert}" if alert else label)
                position_colors.append(color)

        self._log("camera/pose3d/skeleton", rr.LineStrips3D(strips, colors=strip_colors))
        self._log(
            "camera/pose3d/positions",
            rr.Points3D(
                np.stack(positions) if positions else np.zeros((0, 3), dtype=np.float32),
                labels=position_labels,
                colors=position_colors or None,
                radii=POSITION_RADIUS,
            ),
        )

        if self.world is None:
            return
        # 世界系视图：同一个刚体变换套一遍
        world_strips = [self.world.apply(strip) for strip in strips]
        world_positions = [self.world.apply(point) for point in positions]
        self._log("world/skeleton", rr.LineStrips3D(world_strips, colors=strip_colors))
        self._log(
            "world/positions",
            rr.Points3D(
                np.stack(world_positions)
                if world_positions
                else np.zeros((0, 3), dtype=np.float32),
                labels=position_labels,
                colors=position_colors or None,
                radii=POSITION_RADIUS,
            ),
        )

    def _log_coords(self, entity: str, coordinates, *, name: str) -> None:
        """坐标系只发一次：不发的话 3D 视图可能倒着/转着显示。"""
        if entity in self._coords_logged:
            return
        self._coords_logged.add(entity)
        self._log(entity, coordinates)
