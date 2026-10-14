# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化：关键点/骨架 + 人数。

记录到这些实体：

==================================  ========================================
``camera/image``                    相机原图（框/骨架由 rerun 按下面的结构化实体画）
``camera/image/boxes``              框的 Boxes2D（可悬停看标签与置信度）
``camera/image/keypoints``          姿态关键点（带名称）
``camera/image/skeleton``           骨架连线
``person/count``                    人数（TimeSeriesView 里的曲线）
``person/events``                   人数变化日志（带各框置信度）
==================================  ========================================

布局由 blueprint 显式指定（左边标注图、右边人数曲线），不依赖 rerun 的自动推断。

进 rerun 前统一 `bgr_to_rgb()`：`rr.Image` 默认按 RGB 解读三通道数据，直接喂 BGR 会红蓝互换。
"""

from __future__ import annotations

import contextlib

import numpy as np

from .colors import bgr_to_rgb
from .pose import bones, visible_keypoints

BOX_COLOR = (90, 220, 120)  # RGB
KPT_COLOR = (255, 180, 60)
SKELETON_COLOR = (120, 180, 255)


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
        eye: tuple[float, float, float] = (0.0, 0.0, -1.5),
        look_at: tuple[float, float, float] = (0.0, 0.0, 2.5),
        eye_up: tuple[float, float, float] = (0.0, -1.0, 0.0),
    ):
        """跑起来就弹 viewer（除非给了 save_path）。

        viewer 只当显示器：不给 save_path 就不往磁盘写任何东西。
        keep_history=False（默认）时图像/框/关键点用 static 记录，viewer 只保留最新一份，
        没有时间轴可回拖，内存不随帧数涨；memory_limit 是兼底硬上限。
        """
        import rerun as rr

        self.rr = rr
        self.kpt_conf = kpt_conf
        self.keep_history = keep_history
        self.jpeg_quality = int(jpeg_quality)
        self.show_pose3d = show_pose3d
        self.show_depth = show_depth
        self.eye = tuple(eye)
        self.look_at = tuple(look_at)
        self.eye_up = tuple(eye_up)
        self._last_count: int | None = None  # None = 还没记录过人数
        self._image_size: tuple[int, int] | None = None  # (宽, 高)，用于固定取景
        self._coords_logged = False  # 3D 视图的坐标系只发一次

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

        不设 visual_bounds 时，rerun 按“视图内数据的包围盒”自动取景：检测框/关键点
        一旦伸出画面（人半出画就是这样），取景就跟着放大缩小，看起来像画面在自动缩放。
        """
        rrb = _blueprint_module()
        view_args: dict = {"origin": "camera/image", "name": "检测框"}
        # 老版本没有 VisualBounds2D：只能退回自动取景
        with contextlib.suppress(AttributeError, TypeError):
            view_args["visual_bounds"] = rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height])
        try:
            left = rrb.Spatial2DView(**view_args)
            if self.show_depth:
                # 左列：上面原图、下面深度图
                left = rrb.Vertical(
                    left,
                    rrb.Spatial2DView(origin="camera/depth", name="深度"),
                    row_shares=[1, 1],
                )
            right = rrb.TimeSeriesView(origin="person/count", name="人数")
            if self.show_pose3d:
                # 右下角小面板：上面人数曲线、下面 3D 骨架
                right = rrb.Vertical(
                    right,
                    self._make_3d_view(rrb),
                    row_shares=[1, 1],
                )
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    left,
                    right,
                    column_shares=[3, 1],
                ),
                rrb.TimePanel(state="collapsed"),
            )
        except AttributeError:
            # blueprint API 不可用就用 rerun 的默认布局，不影响记录本身
            return
        self.rr.send_blueprint(blueprint)

    def _make_3d_view(self, rrb):
        """3D 骨架视图，相机钉在相机坐标系的默认位置。

        不设 ``eye_controls`` 时 rerun 会按数据的包围盒自动取景，而单目深度逐帧抖动，
        骨架大小一直变，画面就跟着“自动缩放”。这里把眼位/视点/上方向固定下来，
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
            return rrb.Spatial3DView(origin="camera/pose3d", name="3D 姿态")
        return rrb.Spatial3DView(origin="camera/pose3d", name="3D 姿态", eye_controls=controls)

    def _set_time(self, frame_number: int) -> None:
        # 0.23+ 用 set_time(...)，更早的版本用 set_time_sequence
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_number)
        else:  # pragma: no cover - 老版本
            rr.set_time_sequence("frame", frame_number)

    def log_frame(self, frame, image_bgr, *, poses=()) -> None:
        """记一帧：原图 + 结构化框/骨架 + 人数。

        ``image_bgr`` 必须是**相机原图**，不要传 ``result.plot()`` 画过的图：框、
        关键点、骨架都由 rerun 用结构化数据在下面画，传标注图会让同一份信息画两遍。
        """
        rr = self.rr
        number = frame.frame_number
        if self.keep_history:
            self._set_time(number)

        # 分辨率变了（换相机/换模式）才重发 blueprint，把取景钉到新尺寸
        height, width = image_bgr.shape[:2]
        if (width, height) != self._image_size:
            self._image_size = (width, height)
            self._send_blueprint(width, height)

        self._log("camera/image", self._encode(image_bgr))
        instances = list(poses)
        self._log_boxes(instances)
        if poses:
            self._log_poses(poses)

        count = len(instances)
        if count != self._last_count:
            self._last_count = count
            self._log(
                "person/events",
                rr.TextLog(
                    f"{count} person(s): " + ", ".join(f"conf={d.conf:.2f}" for d in instances)
                ),
            )

        # 人数只是几十字节的标量，一直带时间戳：右边那条曲线靠它画
        self._set_time(number)
        scalar = getattr(rr, "Scalars", None) or rr.Scalar
        rr.log("person/count", scalar(count))

    def _log(self, entity: str, data) -> None:
        """默认 static：不带时间戳，viewer 只保留最新一份，没有历史可回拖。

        图像/框/关键点是内存大头（原图 4.67 MB/帧），它们不留历史后 viewer 的内存
        就不随帧数涨；人数曲线需要时间戳，在 log_frame 里单独 set_time 后写。
        """
        if self.keep_history:
            self.rr.log(entity, data)
        else:
            self.rr.log(entity, data, static=True)

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

    def _log_boxes(self, instances) -> None:
        rr = self.rr
        cls = getattr(rr, "Boxes2D", None) or rr.Box2D  # 0.14 叫 Box2D
        # 图像在 camera/image，框挂到它的子路径上就会叠加显示
        if not instances:
            empty = np.zeros((0, 2), dtype=np.float32)
            self._log("camera/image/boxes", cls(centers=empty, sizes=empty))
            return
        xyxy = np.array([d.xyxy for d in instances], dtype=np.float32)
        centers = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1
        )
        sizes = np.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        self._log(
            "camera/image/boxes",
            cls(
                centers=centers,
                sizes=sizes,
                labels=[f"{getattr(d, 'label', 'person')} {d.conf:.2f}" for d in instances],
                colors=[BOX_COLOR] * len(instances),
            ),
        )

    def _log_poses(self, poses) -> None:
        """关键点用 Points2D（带名字），骨头用 LineStrips2D。"""
        rr = self.rr
        points: list[np.ndarray] = []
        labels: list[str] = []
        strips: list[np.ndarray] = []
        for pose in poses:
            visible, names = visible_keypoints(
                pose.keypoints, pose.kpt_conf, threshold=self.kpt_conf
            )
            if len(visible):
                points.append(visible)
                labels.extend(names)
            strips.extend(bones(pose.keypoints, pose.kpt_conf, threshold=self.kpt_conf))

        self._log(
            "camera/image/keypoints",
            rr.Points2D(
                np.concatenate(points) if points else np.zeros((0, 2), dtype=np.float32),
                labels=labels,
                colors=KPT_COLOR,
                radii=3.0,
            ),
        )
        self._log("camera/image/skeleton", rr.LineStrips2D(strips, colors=SKELETON_COLOR))

    def log_depth(self, depth: np.ndarray, *, max_width: int = 480) -> None:
        """把估计深度图送进 viewer（Turbo 伪彩）。

        深度是 float32（1440×1080 每帧 6 MB），送之前先整倍数降采样，别为一个小窗
        白白撑带宽。``rr.DepthImage`` 在 viewer 侧上色，悬停能看到米数。
        """
        rr = self.rr
        step = max(1, -(-depth.shape[1] // max_width))  # 向上取整的整除
        downsampled = np.ascontiguousarray(depth[::step, ::step])
        kwargs: dict = {"meter": 1.0}
        colormap = getattr(getattr(rr, "components", None), "Colormap", None)
        if colormap is not None:
            kwargs["colormap"] = colormap.Turbo
        self._log("camera/depth", rr.DepthImage(downsampled, **kwargs))

    def log_poses3d(self, poses3d) -> None:
        """记 3D 骨架（rerun 右下角小窗）。

        ``poses3d`` 是 ``(joints (M,3), names, strips)`` 的列表（见
        ``pose3d.to_display`` / ``track.Pose3DTracker``）；传空列表就把小窗清空。
        与 2D 一样走 static：viewer 只留最新一份。
        """
        rr = self.rr
        if not self._coords_logged:
            # 相机坐标系（X 右 / Y 下 / Z 前）；不发的话 3D 视图可能倒着显示
            self._log("camera/pose3d", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
            self._coords_logged = True

        points: list[np.ndarray] = []
        labels: list[str] = []
        strips: list[np.ndarray] = []
        for joints, names, person_strips in poses3d:
            if len(joints):
                points.append(np.asarray(joints, dtype=np.float32))
                labels.extend(names)
            strips.extend(person_strips)

        self._log(
            "camera/pose3d/joints",
            rr.Points3D(
                np.concatenate(points) if points else np.zeros((0, 3), dtype=np.float32),
                labels=labels,
                colors=KPT_COLOR,
                radii=0.02,  # 米：关节小球半径 2 cm
            ),
        )
        self._log("camera/pose3d/skeleton", rr.LineStrips3D(strips, colors=SKELETON_COLOR))
