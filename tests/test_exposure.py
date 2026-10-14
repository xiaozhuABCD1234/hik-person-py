# SPDX-License-Identifier: AGPL-3.0-or-later
"""AE 起点（exposure_start_us）与"等 AE 收敛"的单元测试 —— 不需要真相机。"""

from __future__ import annotations

from types import SimpleNamespace

from hik_person.mvs import Camera

NOT_FOUND = 0x80000109  # MV_E_GC_NODE_NOT_FOUND


class _Sdk:
    """configure()/settings() 只用到这几个常量和空结构体。"""

    MV_TRIGGER_MODE_OFF = 0
    MV_GrabStrategy_OneByOne = 0
    MV_GrabStrategy_LatestImagesOnly = 1
    MV_GrabStrategy_LatestImages = 2
    MV_GrabStrategy_UpcomingImage = 3

    @staticmethod
    def MVCC_ENUMVALUE():
        return SimpleNamespace(nCurValue=0, nSupportedNum=0, nSupportValue=[])

    @staticmethod
    def MVCC_INTVALUE():
        return SimpleNamespace(nCurValue=0)

    @staticmethod
    def MVCC_FLOATVALUE():
        return SimpleNamespace(fCurValue=0.0)


class _Recorder:
    """记录所有写入的假相机；节点类型按名字查表，没登记的一律当作不存在。"""

    def __init__(self, kinds: dict[str, str] | None = None):
        self.kinds = dict(kinds or {})
        self.writes: list[tuple[str, object]] = []
        self.values: dict[str, float] = {"ExposureTime": 15000.0, "Gain": 0.0}

    def _write(self, key: str, value: object, kind: str) -> int:
        self.writes.append((key, value))
        if self.kinds.get(key) != kind:
            return NOT_FOUND
        if kind == "float":
            self.values[key] = float(value)  # type: ignore[arg-type]
        return 0

    def MV_CC_SetEnumValueByString(self, key, value):
        return self._write(key, value, "enum")

    def MV_CC_SetEnumValue(self, key, value):
        return self._write(key, value, "enum_int")

    def MV_CC_SetIntValue(self, key, value):
        return self._write(key, value, "int")

    def MV_CC_SetFloatValue(self, key, value):
        return self._write(key, value, "float")

    def MV_CC_SetBoolValue(self, key, value):
        return self._write(key, value, "bool")

    def MV_CC_SetBayerCvtQuality(self, quality):
        return 0

    def MV_CC_SetGrabStrategy(self, strategy):
        return 0

    def MV_CC_SetImageNodeNum(self, count):
        return 0

    # 读：只给 settings() 用；没登记的节点返回"读不到"
    def MV_CC_GetEnumValue(self, key, out):
        return NOT_FOUND

    def MV_CC_GetIntValue(self, key, out):
        return NOT_FOUND

    def MV_CC_GetFloatValue(self, key, out):
        if key in self.values:
            out.fCurValue = self.values[key]
            return 0
        return NOT_FOUND

    def MV_CC_GetBoolValue(self, key, out):
        return NOT_FOUND


# configure() 会写的节点（类型写错会拿到 NOT_FOUND，跟真相机一样）
DEFAULT_KINDS = {
    "ExposureAuto": "enum",
    "GainAuto": "enum",
    "TriggerMode": "enum_int",
    "AcquisitionMode": "enum_int",
    "AcquisitionFrameRateEnable": "bool",
    "ExposureTime": "float",
    "Gain": "float",
    "AcquisitionFrameRate": "float",
}


def make_camera(kinds: dict[str, str] | None = None) -> Camera:
    camera = Camera(_Sdk())
    camera.cam = _Recorder(DEFAULT_KINDS | (kinds or {}))  # type: ignore[assignment]
    return camera


def write_order(camera: Camera) -> list[tuple[str, object]]:
    return camera.cam.writes  # type: ignore[attr-defined]


# ----------------------------------------------------------------- AE 起点
def test_start_point_is_written_between_auto_off_and_auto_on() -> None:
    """顺序必须是：关自动 -> 写固定曝光 -> 开自动；开着自动时相机拒绝写固定值。"""
    camera = make_camera({"ExposureAuto": "enum", "GainAuto": "enum"})
    camera.configure(
        exposure_auto="Continuous",
        gain_auto="Continuous",
        exposure_start_us=8000.0,
    )

    writes = write_order(camera)
    off = writes.index(("ExposureAuto", "Off"))
    start = writes.index(("ExposureTime", 8000.0))
    on = writes.index(("ExposureAuto", "Continuous"))
    assert off < start < on
    assert writes.index(("GainAuto", "Off")) < start
    assert ("GainAuto", "Continuous") in writes


def test_start_point_is_reported_in_notes() -> None:
    camera = make_camera({"ExposureAuto": "enum", "GainAuto": "enum"})
    camera.configure(exposure_auto="Continuous", gain_auto="Continuous", exposure_start_us=8000.0)
    assert any("AE 起点" in note for note in camera.notes)


def test_no_start_point_leaves_device_exposure_alone() -> None:
    """不设起点就一个 ExposureTime 都不写（用相机现存值）。"""
    camera = make_camera({"ExposureAuto": "enum", "GainAuto": "enum"})
    camera.configure(exposure_auto="Continuous", gain_auto="Continuous")
    assert all(name != "ExposureTime" for name, _ in write_order(camera))
    assert not any("AE 起点" in note for note in camera.notes)


def test_manual_mode_ignores_start_point() -> None:
    """manual 模式下起点没意义，写的是固定的 exposure_us / gain_db。"""
    camera = make_camera({"ExposureAuto": "enum", "GainAuto": "enum"})
    camera.configure(
        exposure_us=5000.0,
        gain_db=3.0,
        exposure_auto="Off",
        gain_auto="Off",
        exposure_start_us=8000.0,
    )
    writes = write_order(camera)
    assert ("ExposureTime", 5000.0) in writes
    assert ("Gain", 3.0) in writes
    assert ("ExposureTime", 8000.0) not in writes
    assert not any("AE 起点" in note for note in camera.notes)


# ------------------------------------------------------------- 等 AE 收敛
def scripted(camera: Camera, seq: list[tuple[float, float]]) -> dict[str, int]:
    """让 read()/settings() 按脚本走：settings() 依次吐出 (曝光, 增益)。"""
    values = list(seq)
    state = {"i": 0, "reads": 0}

    def fake_settings() -> dict:
        exposure, gain = values[min(state["i"], len(values) - 1)]
        state["i"] += 1
        return {"exposure_us": exposure, "gain_db": gain}

    def fake_read(*args, **kwargs):
        state["reads"] += 1
        return None

    camera.read = fake_read  # type: ignore[method-assign]
    camera.settings = fake_settings  # type: ignore[method-assign]
    return state


def test_wait_auto_settled_stops_once_values_stop_moving() -> None:
    camera = make_camera()
    state = scripted(camera, [(800.0, 0.0), (1600.0, 0.0), (3200.0, 0.0)] + [(6400.0, 0.0)] * 40)

    settled, converged = camera.wait_auto_settled(max_frames=40, stable_frames=3)

    assert converged is True
    assert settled["exposure_us"] == 6400.0
    assert state["reads"] < 40  # 收敛就停，不空转到上限


def test_wait_auto_settled_treats_tiny_jitter_as_settled() -> None:
    """AE 稳定后有 1% 以内的抖动，不该被当成"还在变"。"""
    camera = make_camera()
    scripted(camera, [(6400.0, 0.0), (6432.0, 0.0)] * 20)  # 0.5% 来回抖

    settled, converged = camera.wait_auto_settled(max_frames=30, stable_frames=3)

    assert converged is True
    assert abs(settled["exposure_us"] - 6400.0) <= 32.0


def test_wait_auto_settled_gives_up_after_max_frames() -> None:
    """一直在爬（没收敛）就报未收敛，而不是假装收敛。"""
    camera = make_camera()
    state = scripted(camera, [(100.0, 0.0), (20000.0, 0.0)] * 30)

    settled, converged = camera.wait_auto_settled(max_frames=10, stable_frames=3)

    assert converged is False
    assert state["reads"] == 10
    assert "exposure_us" in settled


def test_wait_auto_settled_notices_gain_still_moving() -> None:
    """曝光不动但增益还在爬，也算没收敛。"""
    camera = make_camera()
    scripted(camera, [(15000.0, 0.0), (15000.0, 4.0), (15000.0, 8.0)] + [(15000.0, 8.0)] * 20)

    settled, converged = camera.wait_auto_settled(max_frames=30, stable_frames=3)

    assert converged is True
    assert settled["gain_db"] == 8.0
