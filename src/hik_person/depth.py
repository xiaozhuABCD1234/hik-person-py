# SPDX-License-Identifier: AGPL-3.0-or-later
"""单目深度估计（yolo26*-depth）：给 2D 关键点补一个 z 坐标。

`ultralytics` 的 depth 任务输出一张与原图同分辨率的公制深度图（米）。它不是标定过的
双目/结构光深度，只是模型从单张图里估的，绝对尺度会有出入；用来把骨架立起来看结构
够用，别拿它当测量。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DepthEstimator:
    """ultralytics YOLO depth；输入 BGR，输出 ``(H, W)`` float32 深度图（米）。"""

    def __init__(self, weights, *, device: str | None = None, imgsz: int = 320):
        import torch
        from ultralytics import YOLO

        self.weights = str(weights)
        if not Path(self.weights).is_file():
            raise FileNotFoundError(f"找不到深度权重: {self.weights}")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.model = YOLO(self.weights)
        if self.model.task != "depth":
            raise ValueError(
                f"{self.weights} 是 {self.model.task!r} 任务，不是 depth；"
                f"检查 config.toml 的 [pose3d].weights"
            )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        result = self.model.predict(bgr, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        depth_map = result.depth
        if depth_map is None:
            raise RuntimeError("depth 任务没有返回 DepthMap；检查权重是否真的是 depth 模型")
        # 形状与输入图一致（模型内部已缩回原分辨率），单位米
        return np.ascontiguousarray(depth_map.data.cpu().numpy(), dtype=np.float32)
