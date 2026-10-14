# SPDX-License-Identifier: AGPL-3.0-or-later
"""config.toml 的读取与命令行覆盖。

优先级：**命令行 > config.toml > 代码里的兜底默认值**。

`Settings` 里那些默认值只是"连配置文件都没有"时的兜底；正常参数都应当写在
config.toml 里。字段用 `[section].key` 的扁平映射描述，写错的键名会给出警告而不是
被静默忽略 —— 参数文件最怕的就是改了没生效还不知道。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/hik_person/config.py -> 项目根
DEFAULT_CONFIG_NAME = "config.toml"


@dataclass
class Settings:
    """全部可调参数。"""

    # ---------------------------------------------------------------- [camera]
    serial: str = ""
    index: int = 0
    camera_fps: float = 30.0
    exposure_us: float = 8000.0
    gain_db: float = -1.0
    brightness: float = 110.0
    brightness_tolerance: float = 12.0
    exposure_fps_headroom: float = 0.9
    read_timeout_ms: int = 1000
    feature: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- [detector]
    weights: str = "models/best.pt"
    conf: float = 0.35
    iou: float = 0.50
    imgsz: int = 640
    device: str = ""

    # ------------------------------------------------------------------- [run]
    detect_hz: float = 15.0
    rate_tolerance: float = 0.9
    duration: float = 0.0
    max_frames: int = 0
    save_dir: str = ""
    save_min_interval: float = 1.0

    # ----------------------------------------------------------------- [rerun]
    rerun_enabled: bool = True
    rerun_save: str = ""


# (section, key) -> Settings 字段名
CONFIG_FIELDS: dict[tuple[str, str], str] = {
    ("camera", "serial"): "serial",
    ("camera", "index"): "index",
    ("camera", "camera_fps"): "camera_fps",
    ("camera", "exposure_us"): "exposure_us",
    ("camera", "gain_db"): "gain_db",
    ("camera", "brightness"): "brightness",
    ("camera", "brightness_tolerance"): "brightness_tolerance",
    ("camera", "exposure_fps_headroom"): "exposure_fps_headroom",
    ("camera", "read_timeout_ms"): "read_timeout_ms",
    ("camera", "feature"): "feature",
    ("detector", "weights"): "weights",
    ("detector", "conf"): "conf",
    ("detector", "iou"): "iou",
    ("detector", "imgsz"): "imgsz",
    ("detector", "device"): "device",
    ("run", "detect_hz"): "detect_hz",
    ("run", "rate_tolerance"): "rate_tolerance",
    ("run", "duration"): "duration",
    ("run", "max_frames"): "max_frames",
    ("run", "save_dir"): "save_dir",
    ("run", "save_min_interval"): "save_min_interval",
    ("rerun", "enabled"): "rerun_enabled",
    ("rerun", "save"): "rerun_save",
}

# 命令行能覆盖的字段（argparse 里 default=None 表示"没给"）
CLI_FIELDS: tuple[str, ...] = tuple(dict.fromkeys(CONFIG_FIELDS.values()))


def default_config_path() -> Path:
    return PROJECT_ROOT / DEFAULT_CONFIG_NAME


def _coerce(value: Any, target: type, attr: str) -> Any:
    """把 TOML 读出来的值转成 Settings 字段期望的类型，类型不对就报错。"""
    if target is bool:
        if not isinstance(value, bool):
            raise TypeError(f"需要 true/false，收到 {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"需要数字，收到 {value!r}")
        return float(value)
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"需要整数，收到 {value!r}")
        return value
    if target is list:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise TypeError(f"需要字符串数组，收到 {value!r}")
        return list(value)
    return str(value)


def load_settings(path: Path | str | None = None) -> tuple[Settings, list[str]]:
    """读配置文件；返回 ``(设置, 警告列表)``。文件不存在就用兜底默认值。"""
    resolved = Path(path) if path else default_config_path()
    settings = Settings()
    warnings: list[str] = []

    if not resolved.is_file():
        warnings.append(f"没找到配置文件 {resolved}，用代码里的兜底默认值")
        return settings, warnings

    with resolved.open("rb") as handle:
        data = tomllib.load(handle)

    for (section, key), attr in CONFIG_FIELDS.items():
        table = data.get(section)
        if not isinstance(table, dict) or key not in table:
            continue
        target = type(getattr(settings, attr))
        try:
            setattr(settings, attr, _coerce(table[key], target, attr))
        except (TypeError, ValueError) as exc:
            warnings.append(f"[{section}].{key} 不合法（{exc}），已忽略")

    for section, table in data.items():
        if not isinstance(table, dict):
            warnings.append(f"{section!r} 不是表（table），已忽略")
            continue
        for key in table:
            if (section, key) not in CONFIG_FIELDS:
                warnings.append(f"未知配置项 [{section}].{key}，已忽略")

    return settings, warnings


def apply_cli(settings: Settings, args) -> Settings:
    """把命令行显式给出的值盖到配置上（argparse 的 ``default=None`` = 没给）。"""
    for attr in CLI_FIELDS:
        value = getattr(args, attr, None)
        if value is not None:
            setattr(settings, attr, list(value) if isinstance(value, list) else value)
    return settings


def resolve_path(value: str) -> Path:
    """相对路径按项目根解析，避免受当前工作目录影响。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
