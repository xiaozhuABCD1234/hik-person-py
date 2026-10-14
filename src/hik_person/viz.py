# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化（对 rerun 版本做最小兼容）。

进 rerun 前统一 bgr_to_rgb()：rr.Image 默认按 RGB 解读三通道数据，
直接喂驱动的 bgr8 就会出现红蓝互换。
"""

from __future__ import annotations

import numpy as np

from .colors import bgr_to_rgb

BOX_COLOR = (90, 220, 120)  # RGB


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
        self._last_count = -1
        rr.init(app_id, spawn=spawn)
        if save_path:
            rr.save(str(save_path))

    def _set_time(self, frame_number: int) -> None:
        # 0.23+ 用 set_time(...)，更早的版本用 set_time_sequence
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_number)
        else:
            rr.set_time_sequence("frame", frame_number)

    def log_frame(self, frame, detections, annotated_bgr) -> None:
        rr = self.rr
        self._set_time(frame.frame_number)

        rr.log("camera/image", rr.Image(bgr_to_rgb(frame.bgr)))
        rr.log("camera/annotated", rr.Image(bgr_to_rgb(annotated_bgr)))
        self._log_boxes(detections)

        scalars = getattr(rr, "Scalars", None) or rr.Scalar
        rr.log("person/count", scalars(len(detections)))

        if len(detections) != self._last_count:
            self._last_count = len(detections)
            rr.log(
                "person/events",
                rr.TextLog(
                    f"{len(detections)} person(s): "
                    + ", ".join(f"conf={d.conf:.2f}" for d in detections)
                ),
            )

    def _log_boxes(self, detections) -> None:
        rr = self.rr
        cls = getattr(rr, "Boxes2D", None) or rr.Box2D  # 0.14 叫 Box2D
        # 图像在 camera/annotated，框挂到它的子路径上就会被叠加显示
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
