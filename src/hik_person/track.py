# SPDX-License-Identifier: AGPL-3.0-or-later
"""3D 关键点的卡尔曼滤波 + 简易 IoU 跟踪。

深度是逐帧估的，抖动比 2D 关键点大得多；直接画 3D 骨架会看到它“呼吸”。这里给每个人
的每个关键点坐标（17×3 个标量）各配一个**恒定速度**卡尔曼滤波：

    状态 x = [位置, 速度]，F = [[1, dt], [0, 1]]，H = [1, 0]

跨帧得把“第 i 个人”对上号，所以前后帧用检测框 IoU 贪心配对：配对上的继续更新，配不
上的新目标新建轨迹，连续多帧配不上的轨迹就删掉。没有外部依赖（不用 scipy 的匈牙利算法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import KEYPOINT_NAMES_17, SKELETON_17

NUM_KP = len(KEYPOINT_NAMES_17)
# dt 的夹取范围（秒）：太小会让 F 退化，太大会让一帧预测飞出画面。
MIN_DT, MAX_DT = 1e-3, 0.5


def iou(a, b) -> float:
    """两个 ``xyxy`` 框的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class ScalarKalman:
    """一维恒定速度卡尔曼：状态 ``[位置, 速度]``。"""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = np.zeros(2)
        self.p = np.eye(2)

    def init(self, z: float) -> None:
        """用第一个观测初始化：位置=观测，速度=0（速度方差先给大一点）。"""
        self.x = np.array([z, 0.0])
        self.p = np.diag([self.r, self.q * 100.0])

    def predict(self, dt: float) -> None:
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def update(self, z: float) -> None:
        h = np.array([1.0, 0.0])
        s = float(h @ self.p @ h) + self.r
        k = (self.p @ h) / s
        self.x = self.x + k * (z - float(h @ self.x))
        self.p = (np.eye(2) - np.outer(k, h)) @ self.p

    @property
    def position(self) -> float:
        return float(self.x[0])


@dataclass
class _Track:
    box: tuple[float, float, float, float]
    filters: list[list[ScalarKalman]] = field(default_factory=list)
    seen: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP, bool))
    missed_kpt: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP, int))
    missed: int = 0
    last_t: float = 0.0


class Pose3DTracker:
    """把逐帧 3D 关键点平滑成稳定的轨迹；返回的形状与 ``pose3d.to_display`` 一致。"""

    def __init__(
        self,
        *,
        process_noise: float = 10.0,
        measurement_noise: float = 0.1,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        max_missed_kpt: int = 3,
    ):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.max_missed_kpt = int(max_missed_kpt)
        self.tracks: list[_Track] = []

    # ------------------------------------------------------------------ 关联
    def _match(self, boxes) -> dict[int, int]:
        """贪心 IoU 配对，返回 ``{检测下标: 轨迹下标}``（互斥）。"""
        pairs = [
            (iou(track.box, box), ti, bi)
            for ti, track in enumerate(self.tracks)
            for bi, box in enumerate(boxes)
        ]
        pairs.sort(key=lambda item: item[0], reverse=True)
        used_tracks: set[int] = set()
        matched: dict[int, int] = {}
        for score, ti, bi in pairs:
            if score < self.iou_threshold or ti in used_tracks or bi in matched:
                continue
            used_tracks.add(ti)
            matched[bi] = ti
        return matched

    # ------------------------------------------------------------- 单条轨迹
    def _new_track(self, box, points, ok, t: float) -> _Track:
        filters = [[ScalarKalman(self.q, self.r) for _ in range(3)] for _ in range(NUM_KP)]
        for i in range(NUM_KP):
            if ok[i]:
                for axis in range(3):
                    filters[i][axis].init(float(points[i, axis]))
        return _Track(
            box=tuple(box),
            filters=filters,
            seen=ok.copy(),
            missed_kpt=np.where(ok, 0, 1),
            missed=0,
            last_t=t,
        )

    def _advance(self, track: _Track, box, points, ok, t: float) -> None:
        dt = min(max(t - track.last_t, MIN_DT), MAX_DT)
        for i in range(NUM_KP):
            for axis in range(3):
                filt = track.filters[i][axis]
                filt.predict(dt)
                if ok[i]:
                    filt.update(float(points[i, axis]))
        track.seen |= ok
        track.missed_kpt = np.where(ok, 0, track.missed_kpt + 1)
        track.box = tuple(box)
        track.missed = 0
        track.last_t = t

    def _freeze(self, track: _Track, t: float) -> None:
        """没配上的轨迹：只预测，不更新，计漏检。"""
        dt = min(max(t - track.last_t, MIN_DT), MAX_DT)
        for i in range(NUM_KP):
            for axis in range(3):
                track.filters[i][axis].predict(dt)
        track.missed += 1
        track.missed_kpt += 1
        track.last_t = t

    def _display(self, track: _Track) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
        show = track.seen & (track.missed_kpt <= self.max_missed_kpt)
        points = np.full((NUM_KP, 3), np.nan, dtype=np.float32)
        for i in range(NUM_KP):
            if show[i]:
                for axis in range(3):
                    points[i, axis] = track.filters[i][axis].position
        joints = points[show]
        names = [KEYPOINT_NAMES_17[i] for i in range(NUM_KP) if show[i]]
        strips = [
            np.stack([points[start], points[end]]).astype(np.float32)
            for start, end in SKELETON_17
            if show[start] and show[end]
        ]
        return joints, names, strips

    # ------------------------------------------------------------------ 入口
    def update(
        self,
        boxes,
        points3d: np.ndarray,
        kpt_conf: np.ndarray,
        *,
        threshold: float,
        t: float,
    ) -> list[tuple[np.ndarray, list[str], list[np.ndarray]]]:
        """``boxes`` 是每人的 ``xyxy``，``points3d`` 是 ``(N,17,3)``（nan=无效）。"""
        points3d = np.asarray(points3d, dtype=np.float64)
        if points3d.size:
            ok_all = np.isfinite(points3d).all(axis=2) & (np.asarray(kpt_conf) >= threshold)
        else:
            ok_all = np.zeros((0, NUM_KP), bool)

        matched = self._match(boxes)
        for bi, box in enumerate(boxes):
            if bi in matched:
                self._advance(self.tracks[matched[bi]], box, points3d[bi], ok_all[bi], t)
            else:
                self.tracks.append(self._new_track(box, points3d[bi], ok_all[bi], t))
        for ti, track in enumerate(self.tracks):
            if ti not in matched.values():
                self._freeze(track, t)
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return [self._display(track) for track in self.tracks]
