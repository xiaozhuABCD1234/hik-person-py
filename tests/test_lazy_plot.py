# SPDX-License-Identifier: AGPL-3.0-or-later
"""推理不再无条件画标注图（结构性浪费那次的回归测试）。

`result.plot()` 会在整图上画框/骨架，rerun 侧又用结构化实体重画一遍，等于同一份
信息画两遍、传两遍。所以：

* ``__call__`` 只返回 ultralytics 的 ``Results``，**绝不**调用 ``plot()``；
* 只有显式 ``annotate(result)`` 才画（存盘路径用）。

用鸭子类型的假 ``Results`` 构造 estimator，不加载 YOLO、不需要相机。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hik_person.detect import PersonDetector
from hik_person.pose import PoseEstimator


class Boxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = np.asarray(xyxy, np.float32)
        self.conf = np.asarray(conf, np.float32)
        self.cls = np.asarray(cls, np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class Tensor:
    """够用的 torch 张量替身：只要 ``.cpu().numpy()``。"""

    def __init__(self, array) -> None:
        self._array = np.asarray(array, np.float32)

    def cpu(self) -> Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def exploding_plot():
    raise AssertionError("__call__ 里不该调 result.plot()；标注图要按需生成")


def make_detector(result) -> PersonDetector:
    detector = object.__new__(PersonDetector)
    detector.model = SimpleNamespace(predict=lambda *a, **k: [result])
    detector.device, detector.conf, detector.iou, detector.imgsz = "cpu", 0.35, 0.5, 640
    detector.names = {0: "person"}
    detector.classes = [0]
    return detector


def make_pose_estimator(result) -> PoseEstimator:
    estimator = object.__new__(PoseEstimator)
    estimator.model = SimpleNamespace(predict=lambda *a, **k: [result])
    estimator.device, estimator.conf, estimator.imgsz, estimator.kpt_conf = "cpu", 0.35, 640, 0.5
    estimator.weights, estimator.names = "fake.pt", {0: "person"}
    return estimator


def detect_result() -> SimpleNamespace:
    return SimpleNamespace(
        boxes=Boxes([[0, 0, 10, 10]], [0.9], [0]),
        plot=exploding_plot,
    )


def pose_result() -> SimpleNamespace:
    return SimpleNamespace(
        boxes=Boxes([[0, 0, 10, 10]], [0.8], [0]),
        keypoints=SimpleNamespace(
            xy=Tensor(np.full((1, 17, 2), 5.0)),
            conf=Tensor(np.full((1, 17), 0.9)),
        ),
        plot=exploding_plot,
    )


def test_detect_call_does_not_plot() -> None:
    result = detect_result()
    detections, returned = make_detector(result)(np.zeros((8, 8, 3), np.uint8))
    assert returned is result
    assert [d.label for d in detections] == ["person"]
    assert detections[0].conf == pytest.approx(0.9)


def test_pose_call_does_not_plot() -> None:
    result = pose_result()
    poses, returned = make_pose_estimator(result)(np.zeros((8, 8, 3), np.uint8))
    assert returned is result
    assert len(poses) == 1
    assert poses[0].keypoints.shape == (17, 2)


def test_annotate_renders_on_demand() -> None:
    result = detect_result()
    result.plot = lambda: np.zeros((8, 8, 3), np.uint8)
    detector = make_detector(result)
    annotated = detector.annotate(result)
    assert annotated.shape == (8, 8, 3)
    assert annotated.flags["C_CONTIGUOUS"]

    pose = pose_result()
    pose.plot = lambda: np.ones((8, 8, 3), np.uint8)
    estimator = make_pose_estimator(pose)
    assert estimator.annotate(pose).shape == (8, 8, 3)
