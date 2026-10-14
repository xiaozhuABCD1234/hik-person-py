# SPDX-License-Identifier: AGPL-3.0-or-later
"""权重缺失时的提示：默认权重是 x 档且不入库，得让人知道去哪下、怎么退回。"""

from __future__ import annotations

from dataclasses import fields

from hik_person.__main__ import build_parser, weights_download_hint
from hik_person.config import ASSETS_URL, CLI_FIELDS, Settings

# 这些参数不是 Settings 字段：它们控制的是流程（早退/枚举/文件位置）或 argparse 自带。
NON_SETTING_DESTS = {"config", "show_config", "list", "help"}

# 只在 config.toml 里给、故意不给命令行开关的字段。加新字段时要么补开关，
# 要么把它加到这里 —— 免得又出现"命令行给了值却什么都没发生"。
CONFIG_ONLY_FIELDS = {
    "pose3d_eye",
    "pose3d_look_at",
    "pose3d_eye_up",
    "pose3d_kalman_process",
    "pose3d_kalman_measurement",
    "rate_tolerance",
    "rerun_port",
    "rerun_server_memory_limit",
    "save_min_interval",
    "track_max_missed_kpt",
    "track_trail_interval",
    "track_speed_smoothing",
    "track_speed_deadband",
}


def test_every_cli_flag_lands_on_a_settings_field() -> None:
    """argparse 的 dest 必须和 Settings 字段同名，否则 ``apply_cli`` 静默忽略它。

    这是真出过的错：``--world-pitch`` 的 dest 是 ``world_pitch``，而字段叫
    ``world_pitch_deg``，命令行给了值却什么都不发生。
    """
    names = {spec.name for spec in fields(Settings)} | NON_SETTING_DESTS
    strays = sorted(action.dest for action in build_parser()._actions if action.dest not in names)
    assert strays == [], f"这些命令行参数的 dest 不对应任何 Settings 字段：{strays}"


def test_only_known_fields_are_config_file_only() -> None:
    """新增字段要有命令行开关（或明确登记为只走配置文件）。"""
    dests = {action.dest for action in build_parser()._actions}
    missing = set(CLI_FIELDS) - dests
    assert missing <= CONFIG_ONLY_FIELDS, f"这些新字段没有命令行开关：{sorted(missing)}"
    assert set(CONFIG_ONLY_FIELDS) <= set(CLI_FIELDS), "登记表里有已经不是配置项的字段"


def test_hint_points_at_the_official_asset_url() -> None:
    hint = weights_download_hint(Settings().weights, "--weights models/yolo26m-pose.pt")
    assert ASSETS_URL in hint
    assert "yolo26x-pose.pt" in hint
    assert "--weights models/yolo26m-pose.pt" in hint


def test_hint_follows_the_given_path_and_fallback() -> None:
    hint = weights_download_hint("models/custom.pt", "自备权重")
    assert f"{ASSETS_URL}/custom.pt" in hint
    assert hint.endswith("自备权重")


def test_world_pitch_flag_reaches_the_settings() -> None:
    """--world-pitch 必须真的改到 [world].pitch_deg（不是只解析出来就丢掉）。"""
    from hik_person.config import apply_cli

    args = build_parser().parse_args(["--world", "--world-height", "1.35", "--world-pitch", "20"])
    settings = apply_cli(Settings(), args)
    assert settings.world_enabled is True
    assert settings.world_height == 1.35
    assert settings.world_pitch_deg == 20.0
