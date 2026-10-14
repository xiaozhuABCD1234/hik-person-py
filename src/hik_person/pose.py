# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO 姿态估计（骨架关键点）。

骨架与关键点名字取自 ultralytics：
`ultralytics/utils/plotting.py` 的 `Annotator.skeleton` 是 **1-based** 的
（那边用 `sk[0]-1` 取），这里统一转成 0-based 并落地成常量，避免各处再减一。

COCO-17 关键点顺序（0-based）：
0 nose, 1/2 left/right eye, 3/4 left/right ear, 5/6 left/right shoulder,
7/8 left/right elbow, 9/10 left/right wrist, 11/12 left/right hip,
13/14 left/right knee, 15/16 left/right ankle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 19 根骨头（0-based），顺序与 ultralytics 的 skeleton 一致
SKELETON_17: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

KEYPOINT_NAMES_17: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_KEYPOINTS = len(KEYPOINT_NAMES_17)


@dataclass
class PoseInstance:
    """一个人的姿态：框 + 17 个关键点。"""

    conf: float
    xyxy: tuple[float, float, float, float]
    keypoints: np.ndarray  # (17, 2) 像素坐标
    kpt_conf: np.ndarray  # (17,)


def bones(
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    *,
    threshold: float,
    skeleton: tuple[tuple[int, int], ...] = SKELETON_17,
) -> list[np.ndarray]:
    """把骨架连成线段（供 rerun 的 LineStrips2D 用）。

    两端关键点都过阈值才连，否则画出"悬空"的骨头。纯函数，便于单测。
    """
    strips: list[np.ndarray] = []
    for start, end in skeleton:
        if kpt_conf[start] < threshold or kpt_conf[end] < threshold:
            continue
        strips.append(np.array([keypoints[start], keypoints[end]], dtype=np.float32))
    return strips


def visible_keypoints(
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    *,
    threshold: float,
    names: tuple[str, ...] = KEYPOINT_NAMES_17,
) -> tuple[np.ndarray, list[str]]:
    """挑出过阈值的关键点，返回 (坐标 (M,2), 名字列表)。"""
    keep = np.asarray(kpt_conf) >= threshold
    return np.asarray(keypoints, dtype=np.float32)[keep], [
        names[i] for i in range(len(keep)) if keep[i]
    ]


class PoseEstimator:
    """ultralytics YOLO pose；输入 BGR，输出 (PoseInstance 列表, ultralytics Results)。

    不再无条件 ``result.plot()``（它会在整图上画框/骨架/关键点），rerun 侧用
    ``Boxes2D``/``Points2D``/``LineStrips2D`` 结构化数据重画；只有要存盘时才调
    ``annotate()`` 按需生成标注图。
    """

    def __init__(
        self,
        weights,
        *,
        device: str | None = None,
        conf: float = 0.35,
        imgsz: int = 640,
        kpt_conf: float = 0.5,
    ):
        import torch
        from ultralytics import YOLO

        self.weights = str(weights)
        if not Path(self.weights).is_file():
            raise FileNotFoundError(f"找不到权重: {self.weights}")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conf, self.imgsz, self.kpt_conf = conf, imgsz, kpt_conf
        self.model = YOLO(self.weights)
        if self.model.task != "pose":
            raise ValueError(
                f"{self.weights} 是 {self.model.task!r} 任务，不是 pose；"
                f"检查 config.toml 的 [detector].weights"
            )
        self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}

    def __call__(self, bgr: np.ndarray) -> tuple[list[PoseInstance], object]:
        result = self.model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        instances: list[PoseInstance] = []
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is not None and keypoints is not None and len(boxes) > 0:
            xyxy_all = boxes.xyxy.tolist()
            conf_all = boxes.conf.tolist()
            xy = keypoints.xy.cpu().numpy()
            kpt_conf = (
                keypoints.conf.cpu().numpy()
                if keypoints.conf is not None
                else np.ones(xy.shape[:2], dtype=np.float32)
            )
            for xyxy, box_conf, points, points_conf in zip(
                xyxy_all, conf_all, xy, kpt_conf, strict=True
            ):
                instances.append(
                    PoseInstance(
                        conf=float(box_conf),
                        xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        keypoints=np.asarray(points, dtype=np.float32),
                        kpt_conf=np.asarray(points_conf, dtype=np.float32),
                    )
                )

        return instances, result

    def annotate(self, result) -> np.ndarray:
        """把 Results 画成 BGR 标注图（框 + 关键点 + 骨架）。按需调用，别每帧调。"""
        return np.ascontiguousarray(result.plot())
