# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON Lines 输出的单元测试 —— 不需要相机，写临时文件。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hik_person.output import JsonlWriter, frame_record, person_record
from hik_person.track import TrackedPerson
from hik_person.world import CameraToWorld


def person(**overrides) -> TrackedPerson:
    fields = {
        "id": 2,
        "conf": 0.87,
        "box": (300.0, 100.0, 340.0, 400.0),
        "keypoints2d": np.full((17, 2), 320.0, np.float32),
        "kpt_conf": np.full(17, 0.9, np.float32),
        "joints": np.zeros((14, 3), np.float32),
        "names": ["nose"] * 14,
        "strips": [],
        "position": np.array([0.1, -0.05, 2.31], np.float32),
        "velocity": np.array([0.0, 0.0, 0.41], np.float32),
        "speed": 0.41,
        "speed_ground": 0.41,
        "trail": np.zeros((2, 3), np.float32),
        "age": 96,
        "missed": 0,
        "valid_count": 14,
    }
    fields.update(overrides)
    return TrackedPerson(**fields)


# --------------------------------------------------------------------- 单个人
def test_record_has_the_documented_fields() -> None:
    record = person_record(person())
    assert record["id"] == 2
    assert record["conf"] == 0.87
    assert record["distance"] == pytest.approx(np.linalg.norm([0.1, -0.05, 2.31]), abs=1e-4)
    assert record["cam"] == [0.1, -0.05, 2.31]
    assert record["world"] is None  # 没开 [world] 就没有这个世界字段的坐标
    assert record["speed"] == 0.41
    assert record["state"] in {"静止", "行走"}
    assert record["valid_keypoints"] == 14
    assert record["coverage"] == pytest.approx(14 / 17, abs=1e-4)  # 写盘时四舍五入


def test_record_is_json_serializable() -> None:
    """必须是纯 JSON：numpy 标量漏出来 json.dumps 就炸。"""
    text = json.dumps(person_record(person()), ensure_ascii=False)
    assert json.loads(text)["id"] == 2


def test_record_writes_world_when_enabled() -> None:
    world = CameraToWorld(height=1.5, pitch_deg=15.0)
    record = person_record(person(), world)
    assert record["world"] == pytest.approx(
        [float(v) for v in world.apply(np.array([0.1, -0.05, 2.31]))], abs=1e-4
    )


def test_record_keeps_none_when_position_is_unknown() -> None:
    """位置测不到时给 null，不能报成原点 (0,0,0)。"""
    item = person(position=np.full(3, np.nan, np.float32), speed=float("nan"))
    record = person_record(item)
    assert record["cam"] is None
    assert record["world"] is None
    assert record["distance"] is None
    assert record["speed"] is None


# ------------------------------------------------------------------------- 帧
def test_frame_record_counts_only_hits() -> None:
    record = frame_record(
        t=1.5,
        wall=1700000000.0,
        frame_number=8123,
        device_timestamp_ms=45678,
        people=[person(id=1), person(id=2, missed=3)],
    )
    assert record["t"] == 1.5
    assert record["frame"] == 8123
    assert record["count"] == 1  # 漏检的不算人
    assert [item["id"] for item in record["people"]] == [1]


# ------------------------------------------------------------------------ 写盘
def test_writer_appends_one_json_line_per_call(tmp_path) -> None:
    path = tmp_path / "people.jsonl"
    writer = JsonlWriter(path, hz=0.0, t0=100.0)
    assert writer.maybe_write(100.5, frame_number=1, device_timestamp_ms=10, people=[person()])
    assert writer.maybe_write(101.0, frame_number=2, device_timestamp_ms=20, people=[])
    writer.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["t"] == pytest.approx(0.5)  # 相对 t0
    assert first["count"] == 1
    assert json.loads(lines[1])["count"] == 0


def test_writer_rate_limits_by_hz(tmp_path) -> None:
    writer = JsonlWriter(tmp_path / "p.jsonl", hz=2.0, t0=0.0)
    assert writer.maybe_write(0.0, frame_number=1, device_timestamp_ms=0, people=[])
    assert not writer.maybe_write(0.2, frame_number=2, device_timestamp_ms=0, people=[])
    assert writer.maybe_write(0.5, frame_number=3, device_timestamp_ms=0, people=[])
    writer.close()


def test_writer_appends_to_an_existing_file(tmp_path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"t": 0.0}\n', encoding="utf-8")
    writer = JsonlWriter(path, hz=0.0)
    writer.maybe_write(0.0, frame_number=1, device_timestamp_ms=0, people=[])
    writer.close()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
