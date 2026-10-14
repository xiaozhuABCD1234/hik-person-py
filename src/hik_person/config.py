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
# 官方权重的下载地址：x 档体积大（240 MB），不入库，用 README 里的命令下到 models/
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
# 曝光/增益模式：sdk = 相机自动；manual = 固定值
EXPOSURE_MODES: tuple[str, ...] = ("sdk", "manual")

# 已经从 config.toml 里拿掉的键。旧文件里留着它们只会拿到一个泛泛的"未知配置项"，
# 这里专门说一句该怎么办，免得把"已移除"当成"拼错了"。
REMOVED_KEYS: dict[tuple[str, str], str] = {
    ("detector", "task"): "现在只有 pose 一条线（关键点/骨架），删掉这一行",
    ("detector", "iou"): "pose 线不用 NMS 的 iou 参数（用 ultralytics 默认值），删掉这一行",
}


@dataclass
class Settings:
    """全部可调参数。"""

    # ---------------------------------------------------------------- [camera]
    serial: str = ""
    index: int = 0
    camera_fps: float = 30.0
    exposure_us: float = 8000.0
    gain_db: float = -1.0
    # sdk = 相机自己的自动曝光 + 自动增益（默认）；manual = 固定用 exposure_us / gain_db
    exposure_mode: str = "sdk"
    # AE 允许的最大增益（dB）。增益比曝光噪得多，所以先大幅放宽曝光上限，再压住增益。
    # -1 = 不动它（用相机当前值），0 = 不允许增益。
    auto_gain_max_db: float = 16.98
    brightness: float = 110.0
    # sdk 模式下开 AE 前先把曝光钉到这个值当起点（us）；0 = 不重置（用相机现存值）。
    # 相机把上次的曝光存在自己身上，换相机或别人用过之后 AE 的起跑线不一样。
    exposure_start_us: float = 8000.0
    exposure_fps_headroom: float = 0.9
    read_timeout_ms: int = 1000
    feature: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- [detector]
    # 姿态权重：x 档精度最高，但在 8G 显存 + 15 Hz 下实测 36.3 ms/帧（m 档 16.6 ms）。
    # x 档不在仓库里（体积大），按 README「模型」一节的命令下载到 models/。
    weights: str = "models/yolo26x-pose.pt"
    conf: float = 0.35
    imgsz: int = 640
    device: str = ""
    kpt_conf: float = 0.5

    # ------------------------------------------------------------------- [run]
    # 推理频率（Hz）。键名沿用 detect_hz：改名字只会让已写的 config.toml 报未知配置项。
    detect_hz: float = 15.0
    rate_tolerance: float = 0.9
    duration: float = 0.0
    max_frames: int = 0
    save_dir: str = ""
    save_min_interval: float = 1.0

    # ----------------------------------------------------------------- [rerun]
    rerun_enabled: bool = True
    # 非空才写 .rrd（默认纯显示，不往磁盘记任何东西）
    rerun_save: str = ""
    # False（默认）= 图像/框/关键点用 static 记录，viewer 只保留最新一份，没有时间轴可回拖
    rerun_keep_history: bool = False
    # viewer 内存上限，到了就丢最旧数据；rerun 自己的默认是 75%%（本机约 23 GB）
    rerun_memory_limit: str = "512MB"
    # 与 viewer 同进程的 gRPC 缓冲（给后到的 viewer 补数据用）
    rerun_server_memory_limit: str = "512MB"
    # 图像送进 viewer 前先压成 JPEG；0 = 原图（4.67 MB/帧，viewer 内存约差 10 倍）
    rerun_jpeg_quality: int = 75
    rerun_port: int = 9876
    # 右列放什么：数值行（每人一行）/ 坐标曲线 / 人数曲线。默认前两个，人数默认不上。
    rerun_show_values: bool = True
    rerun_show_coords: bool = True
    rerun_show_count: bool = False
    # 3D 视图里要不要画运动轨迹（只影响显示；轨迹记录本身由 [track] 管）。
    rerun_show_trails: bool = True

    # ---------------------------------------------------------------- [pose3d]
    # 单目 3D：yolo26x-depth 给每个关键点补深度，rerun 右下角小窗画 3D 骨架。
    pose3d_enabled: bool = True
    pose3d_weights: str = "models/yolo26x-depth.pt"
    pose3d_imgsz: int = 640
    # 深度推理频率（Hz）。深度模型不便宜，没必要跟满 detect_hz。
    pose3d_hz: float = 15.0
    # 假定的水平视场角（度）。没标定，只能按普通镜头估，用来把像素+深度还原成 3D。
    pose3d_fov_deg: float = 60.0
    # 左列下面再放一块估计深度图（Turbo 伪彩）。
    pose3d_depth_view: bool = True
    # 固定 3D 视角（相机坐标系，米）：eye 位置 / 看向的点 / 上方向。
    # 不固定的话 rerun 会按数据包围盒自动取景，深度噪声会让画面忽大忽小。
    pose3d_eye: tuple[float, float, float] = (0.0, 0.0, -1.5)
    pose3d_look_at: tuple[float, float, float] = (0.0, 0.0, 2.5)
    pose3d_eye_up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    # 卡尔曼滤波：平滑单目深度逐帧的抖动（每个人的每个关键点一个恒定速度滤波）。
    pose3d_kalman: bool = True
    # 过程噪声（加速度，m/s²）越大越跟手、越不平滑；观测噪声（m）越大越平滑。
    pose3d_kalman_process: float = 10.0
    # 单目估计深度的抖动比真深度大得多，观测噪声不能设得很小（d435i 那边是 0.02）。
    pose3d_kalman_measurement: float = 0.1

    # ----------------------------------------------------------------- [track]
    # 跨帧关联（前后帧 IoU 贪心配对）与轨迹。id 单调递增、永不复用。
    track_iou_threshold: float = 0.3
    # >0 时配对还要满足 3D 距离（米）不超过它，两人交叉时不容易换 id；0 = 只看 IoU。
    track_match_max_distance: float = 0.0
    # 连续漏检多少帧后删轨迹（期间位置由卡尔曼预测，短遮挡不会丢 id）。
    track_max_missed: int = 5
    # 单个关键点连续丢这么多帧后不再画（人的其他部位继续画）。
    track_max_missed_kpt: int = 3
    # 轨迹点数上限与采样间隔（0 = 不留轨迹）。
    track_trail_length: int = 300
    track_trail_interval: float = 0.1
    # 速度指数平滑时间常数（秒），0 = 不平滑；死区只影响显示状态（静止/行走）。
    track_speed_smoothing: float = 0.3
    track_speed_deadband: float = 0.15

    # ---------------------------------------------------------------- [world]
    # 相机坐标系 -> 世界（地面）系的静态外参。默认关：不量安装高度/俯仰角时
    # 绝对坐标整体是错的，不如不开。开了以后“上方向”也用世界 Z（速度分解更准）。
    world_enabled: bool = False
    # 光心离地高度（米）。卷尺量，或把人放到已知距离看深度读数校准。
    world_height: float = 1.0
    # 镜头下俯角度（度，正 = 向下）。手机水平仪贴在相机外壳上量，或用已知身高反推。
    world_pitch_deg: float = 0.0
    world_yaw_deg: float = 0.0
    world_roll_deg: float = 0.0
    # 右下角再开一个世界系 3D 视图（相机系那个照旧）
    world_view: bool = False

    # --------------------------------------------------------------- [output]
    # 非空则把人写成 JSON Lines（追加写，一行一帧）。
    output_jsonl: str = ""
    # 写 JSONL 的频率（Hz）。跟踪是高频的，日志没必要跟满。
    output_hz: float = 5.0


# (section, key) -> Settings 字段名
CONFIG_FIELDS: dict[tuple[str, str], str] = {
    ("camera", "serial"): "serial",
    ("camera", "index"): "index",
    ("camera", "camera_fps"): "camera_fps",
    ("camera", "exposure_mode"): "exposure_mode",
    ("camera", "auto_gain_max_db"): "auto_gain_max_db",
    ("camera", "exposure_us"): "exposure_us",
    ("camera", "gain_db"): "gain_db",
    ("camera", "brightness"): "brightness",
    ("camera", "exposure_start_us"): "exposure_start_us",
    ("camera", "exposure_fps_headroom"): "exposure_fps_headroom",
    ("camera", "read_timeout_ms"): "read_timeout_ms",
    ("camera", "feature"): "feature",
    ("detector", "weights"): "weights",
    ("detector", "conf"): "conf",
    ("detector", "imgsz"): "imgsz",
    ("detector", "device"): "device",
    ("detector", "kpt_conf"): "kpt_conf",
    ("run", "detect_hz"): "detect_hz",
    ("run", "rate_tolerance"): "rate_tolerance",
    ("run", "duration"): "duration",
    ("run", "max_frames"): "max_frames",
    ("run", "save_dir"): "save_dir",
    ("run", "save_min_interval"): "save_min_interval",
    ("rerun", "enabled"): "rerun_enabled",
    ("rerun", "save"): "rerun_save",
    ("rerun", "keep_history"): "rerun_keep_history",
    ("rerun", "memory_limit"): "rerun_memory_limit",
    ("rerun", "server_memory_limit"): "rerun_server_memory_limit",
    ("rerun", "jpeg_quality"): "rerun_jpeg_quality",
    ("rerun", "port"): "rerun_port",
    ("rerun", "show_values"): "rerun_show_values",
    ("rerun", "show_coords"): "rerun_show_coords",
    ("rerun", "show_count"): "rerun_show_count",
    ("rerun", "show_trails"): "rerun_show_trails",
    ("pose3d", "enabled"): "pose3d_enabled",
    ("pose3d", "weights"): "pose3d_weights",
    ("pose3d", "imgsz"): "pose3d_imgsz",
    ("pose3d", "hz"): "pose3d_hz",
    ("pose3d", "fov_deg"): "pose3d_fov_deg",
    ("pose3d", "depth_view"): "pose3d_depth_view",
    ("pose3d", "eye"): "pose3d_eye",
    ("pose3d", "look_at"): "pose3d_look_at",
    ("pose3d", "eye_up"): "pose3d_eye_up",
    ("pose3d", "kalman"): "pose3d_kalman",
    ("pose3d", "kalman_process"): "pose3d_kalman_process",
    ("pose3d", "kalman_measurement"): "pose3d_kalman_measurement",
    ("track", "iou_threshold"): "track_iou_threshold",
    ("track", "match_max_distance"): "track_match_max_distance",
    ("track", "max_missed"): "track_max_missed",
    ("track", "max_missed_kpt"): "track_max_missed_kpt",
    ("track", "trail_length"): "track_trail_length",
    ("track", "trail_interval"): "track_trail_interval",
    ("track", "speed_smoothing"): "track_speed_smoothing",
    ("track", "speed_deadband"): "track_speed_deadband",
    ("world", "enabled"): "world_enabled",
    ("world", "height"): "world_height",
    ("world", "pitch_deg"): "world_pitch_deg",
    ("world", "yaw_deg"): "world_yaw_deg",
    ("world", "roll_deg"): "world_roll_deg",
    ("world", "view"): "world_view",
    ("output", "jsonl"): "output_jsonl",
    ("output", "hz"): "output_hz",
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
    if target is tuple:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        ):
            raise TypeError(f"需要 3 个数字的数组，收到 {value!r}")
        return tuple(float(v) for v in value)
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
            if (section, key) in CONFIG_FIELDS:
                continue
            removed = REMOVED_KEYS.get((section, key))
            warnings.append(
                f"[{section}].{key} 已移除，{removed}（已忽略）"
                if removed
                else f"未知配置项 [{section}].{key}，已忽略"
            )

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
