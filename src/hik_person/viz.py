# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化：检测框 + 人数。

记录到这些实体：

===========================  ==========================================
``camera/annotated``         标注图（框 + 人数横幅，由 detect.py 画好）
``camera/annotated/boxes``   同一批框的 Boxes2D（可悬停看标签与置信度）
``person/count``             人数（TimeSeriesView 里的曲线）
``person/events``            人数变化日志（带各框置信度）
===========================  ==========================================

布局由 blueprint 显式指定（左边标注图、右边人数曲线），不依赖 rerun 的自动推断。

进 rerun 前统一 `bgr_to_rgb()`：`rr.Image` 默认按 RGB 解读三通道数据，直接喂 BGR 会红蓝互换。
"""

from __future__ import annotations

import numpy as np

from .colors import bgr_to_rgb

BOX_COLOR = (90, 220, 120)  # RGB


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
    ):
        """spawn=True 弹 viewer；save_path 则把记录写成 .rrd（不弹窗口）。"""
        import rerun as rr

        self.rr = rr
        self._last_count: int | None = None  # None = 还没记录过人数
        rr.init(app_id, spawn=spawn)
        if save_path:
            rr.save(str(save_path))
        self._send_blueprint()

    def _send_blueprint(self) -> None:
        rrb = _blueprint_module()
        try:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="camera/annotated", name="检测框"),
                    rrb.TimeSeriesView(origin="person/count", name="人数"),
                    column_shares=[3, 1],
                ),
                rrb.TimePanel(state="collapsed"),
            )
        except AttributeError:
            # blueprint API 不可用就用 rerun 的默认布局，不影响记录本身
            return
        self.rr.send_blueprint(blueprint)

    def _set_time(self, frame_number: int) -> None:
        # 0.23+ 用 set_time(...)，更早的版本用 set_time_sequence
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_number)
        else:  # pragma: no cover - 老版本
            rr.set_time_sequence("frame", frame_number)

    def log_frame(self, frame, detections, annotated_bgr) -> None:
        rr = self.rr
        self._set_time(frame.frame_number)

        rr.log("camera/annotated", rr.Image(bgr_to_rgb(annotated_bgr)))
        self._log_boxes(detections)

        scalar = getattr(rr, "Scalars", None) or rr.Scalar
        rr.log("person/count", scalar(len(detections)))

        count = len(detections)
        if count != self._last_count:
            self._last_count = count
            rr.log(
                "person/events",
                rr.TextLog(
                    f"{count} person(s): " + ", ".join(f"conf={d.conf:.2f}" for d in detections)
                ),
            )

    def _log_boxes(self, detections) -> None:
        rr = self.rr
        cls = getattr(rr, "Boxes2D", None) or rr.Box2D  # 0.14 叫 Box2D
        # 图像在 camera/annotated，框挂到它的子路径上就会叠加显示
        if not detections:
            empty = np.zeros((0, 2), dtype=np.float32)
            rr.log("camera/annotated/boxes", cls(centers=empty, sizes=empty))
            return
        xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
        centers = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1
        )
        sizes = np.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        rr.log(
            "camera/annotated/boxes",
            cls(
                centers=centers,
                sizes=sizes,
                labels=[f"{d.label} {d.conf:.2f}" for d in detections],
                colors=[BOX_COLOR] * len(detections),
            ),
        )
