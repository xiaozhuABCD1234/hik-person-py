# SPDX-License-Identifier: AGPL-3.0-or-later
"""config.toml 读取与命令行覆盖的单元测试 —— 不需要相机。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hik_person.config import (
    CLI_FIELDS,
    PROJECT_ROOT,
    Settings,
    apply_cli,
    load_settings,
    resolve_path,
)


def write_config(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_file_missing(tmp_path) -> None:
    settings, warnings = load_settings(tmp_path / "nope.toml")
    assert settings == Settings()
    assert warnings and "没找到配置文件" in warnings[0]


def test_reads_all_sections(tmp_path) -> None:
    path = write_config(
        tmp_path,
        """
[camera]
serial = "DB1505178"
index = 2
camera_fps = 249.0
exposure_us = 3500
gain_db = 0
brightness = 120.5
brightness_tolerance = 8
exposure_fps_headroom = 0.8
read_timeout_ms = 500
feature = ["ADCBitDepth=Bits_8", "GammaEnable=true"]

[detector]
weights = "models/other.pt"
conf = 0.5
iou = 0.4
imgsz = 480
device = "cpu"

[run]
detect_hz = 20.0
rate_tolerance = 0.95
duration = 30.0
max_frames = 100
save_dir = "/tmp/shots"
save_min_interval = 2.0

[rerun]
enabled = false
save = "/tmp/out.rrd"
""",
    )
    settings, warnings = load_settings(path)

    assert warnings == []
    assert settings.serial == "DB1505178"
    assert settings.index == 2
    assert settings.camera_fps == 249.0
    assert settings.exposure_us == 3500.0  # 整数写成浮点
    assert settings.gain_db == 0.0  # 0 dB 是合法值，不能退化
    assert settings.brightness == 120.5
    assert settings.brightness_tolerance == 8.0
    assert settings.exposure_fps_headroom == 0.8
    assert settings.read_timeout_ms == 500
    assert settings.feature == ["ADCBitDepth=Bits_8", "GammaEnable=true"]
    assert settings.weights == "models/other.pt"
    assert settings.imgsz == 480
    assert settings.device == "cpu"
    assert settings.detect_hz == 20.0
    assert settings.rate_tolerance == 0.95
    assert settings.duration == 30.0
    assert settings.max_frames == 100
    assert settings.save_min_interval == 2.0
    assert settings.rerun_enabled is False
    assert settings.rerun_save == "/tmp/out.rrd"


def test_unknown_key_is_warned_not_silently_ignored(tmp_path) -> None:
    """参数文件最怕"改了没生效还不知道"。"""
    path = write_config(tmp_path, '[camera]\nserail = "typo"\nserial = "ok"\n')
    settings, warnings = load_settings(path)
    assert settings.serial == "ok"
    assert any("未知配置项 [camera].serail" in w for w in warnings)


def test_wrong_type_warns_and_keeps_default(tmp_path) -> None:
    path = write_config(tmp_path, '[camera]\ncamera_fps = "fast"\n[run]\nmax_frames = 1.5\n')
    settings, warnings = load_settings(path)
    assert settings.camera_fps == Settings().camera_fps
    assert settings.max_frames == Settings().max_frames
    assert len(warnings) == 2


def test_boolean_must_be_boolean(tmp_path) -> None:
    path = write_config(tmp_path, '[rerun]\nenabled = "yes"\n')
    settings, warnings = load_settings(path)
    assert settings.rerun_enabled is True  # 默认值保住了
    assert any("需要 true/false" in w for w in warnings)


def test_broken_toml_raises(tmp_path) -> None:
    path = write_config(tmp_path, "[camera\nserial = ")
    with pytest.raises(ValueError):
        load_settings(path)


def make_args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{name: None for name in CLI_FIELDS} | overrides)


def test_cli_none_does_not_override() -> None:
    settings = apply_cli(Settings(), make_args())
    assert settings == Settings()


def test_cli_overrides_config_values() -> None:
    settings = apply_cli(Settings(serial="from-config", imgsz=640), make_args(serial="from-cli"))
    assert settings.serial == "from-cli"
    assert settings.imgsz == 640  # 没给的就保留


def test_cli_feature_replaces_the_whole_list() -> None:
    settings = apply_cli(
        Settings(feature=["ADCBitDepth=Bits_8"]), make_args(feature=["GammaEnable=true"])
    )
    assert settings.feature == ["GammaEnable=true"]


def test_cli_can_set_gain_to_zero() -> None:
    """0 是合法增益，不能被当成"没给"。"""
    settings = apply_cli(Settings(gain_db=-1.0), make_args(gain_db=0.0))
    assert settings.gain_db == 0.0


def test_cli_can_disable_rerun() -> None:
    settings = apply_cli(Settings(rerun_enabled=True), make_args(rerun_enabled=False))
    assert settings.rerun_enabled is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("models/best.pt", PROJECT_ROOT / "models/best.pt"),
        ("/abs/path.pt", Path("/abs/path.pt")),
    ],
)
def test_resolve_path(value: str, expected: Path) -> None:
    assert resolve_path(value) == expected


def test_resolve_path_expands_user() -> None:
    assert str(resolve_path("~/x")).startswith("/")
    assert "~" not in str(resolve_path("~/x"))
