# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO 人体检测（用训练好的 models/best.pt）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PERSON_ALIASES = {"person", "people", "pedestrian", "human"}


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]


class PersonDetector:
    def __init__(
        self,
        weights,
        *,
        device: str | None = None,
        conf: float = 0.35,
        iou: float = 0.50,
        imgsz: int = 640,
    ):
        import torch
        from ultralytics import YOLO

        self.weights = str(weights)
        if not Path(self.weights).is_file():
            raise FileNotFoundError(f"找不到权重: {self.weights}")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.model = YOLO(self.weights)
        self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}
        # 训练集 nc=1 / names: {0: person}；多类模型则自动只保留“人”
        self.classes = [i for i, n in self.names.items() if n.lower() in PERSON_ALIASES] or list(
            self.names
        )

    def __call__(self, bgr: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """输入 BGR uint8（与驱动 bgr8 一致），返回 (检测列表, 画好框的 BGR 图)。"""
        result = self.model.predict(
            bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]
        boxes = result.boxes
        detections = [
            Detection(
                self.names.get(int(c), str(int(c))),
                float(p),
                (float(x1), float(y1), float(x2), float(y2)),
            )
            for (x1, y1, x2, y2), p, c in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            )
        ]
        return detections, np.ascontiguousarray(result.plot())
