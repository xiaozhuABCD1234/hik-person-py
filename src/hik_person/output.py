# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON Lines 输出：按 ``[output].hz`` 每行记一帧里的人。

用途是把跟踪结果接到别处去（底盘、看板、离线分析）：一行一个 JSON 对象，
直接 ``json.loads`` 就能读。文件是**追加**写，所以多次运行会接在同一个文件后面，
``t`` 是相对该次运行的秒数（``wall`` 是绝对时间，方便跨次对齐）。

尺度提醒：``cam`` / ``world`` / ``distance`` / ``speed`` 都建立在单目估计深度上，
**绝对公制值不可全信**（同一次运行内的相对比较才有意义）。``valid_keypoints`` /
``coverage`` 是尺度无关的可信度自检。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .world import CameraToWorld


def person_record(person, world: CameraToWorld | None = None) -> dict:
    """一个人 -> 一个 JSON 可序列化的 dict。世界坐标只在 ``[world]`` 开着时才有。"""
    position = np.asarray(person.position, dtype=np.float64)
    finite = bool(np.isfinite(position).all())
    record = {
        "id": int(person.id),
        "conf": round(float(person.conf), 4),
        "distance": None if not np.isfinite(person.distance) else round(person.distance, 4),
        "cam": [round(float(v), 4) for v in position] if finite else None,
        "world": [round(float(v), 4) for v in world.apply(position)] if finite and world else None,
        "speed": None if not np.isfinite(person.speed) else round(float(person.speed), 4),
        "speed_ground": (
            None if not np.isfinite(person.speed_ground) else round(float(person.speed_ground), 4)
        ),
        "state": person.state,
        "velocity": [round(float(v), 4) for v in person.velocity],
        "box": [round(float(v), 2) for v in person.box],
        "valid_keypoints": int(person.valid_count),
        "coverage": round(float(person.coverage), 4),
        "age": int(person.age),
        "missed": int(person.missed),
    }
    return record


def frame_record(
    *,
    t: float,
    wall: float,
    frame_number: int,
    device_timestamp_ms: int,
    people,
    world: CameraToWorld | None = None,
) -> dict:
    """一帧 -> 一个 JSON 对象（``people`` 只放本帧真命中的人）。"""
    live = [person for person in people if person.missed == 0]
    return {
        "t": round(float(t), 4),
        "wall": round(float(wall), 3),
        "frame": int(frame_number),
        "device_timestamp_ms": int(device_timestamp_ms),
        "count": len(live),
        "people": [person_record(person, world) for person in live],
    }


class JsonlWriter:
    """按频率限流写 JSON Lines（append 模式，写完即 flush，崩溃也不丢已写的行）。"""

    def __init__(self, path: Path | str, *, hz: float = 5.0, t0: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.period = 1.0 / hz if hz > 0 else 0.0
        self.t0 = time.monotonic() if t0 is None else float(t0)
        self.last_t = -1e18

    def due(self, now: float) -> bool:
        return self.period <= 0 or now - self.last_t >= self.period

    def maybe_write(
        self,
        now: float,
        *,
        frame_number: int,
        device_timestamp_ms: int,
        people,
        world: CameraToWorld | None = None,
        wall: float | None = None,
    ) -> bool:
        """到点了就写一行；返回是否真的写了。"""
        if not self.due(now):
            return False
        self.last_t = now
        record = frame_record(
            t=now - self.t0,
            wall=time.time() if wall is None else wall,
            frame_number=frame_number,
            device_timestamp_ms=device_timestamp_ms,
            people=people,
            world=world,
        )
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()
        return True

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
