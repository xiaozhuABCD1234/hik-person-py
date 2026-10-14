# SPDX-License-Identifier: AGPL-3.0-or-later
"""摔倒检测的单元测试 —— 纯状态机（不需要相机/模型）+ 一条合成骨架的端到端。

合成骨架那部分走真链路（2D 投影 -> 深度采样 -> 3D 提升 -> 卡尔曼 -> 状态机），
所以能验证"高度/竖直速度/躯干角"这三个信号真的能从骨架里取对。

阈值口径：这些米制阈值在单目方案里是"估计尺度"下的经验值（见 fall.py 的说明），
所以这里的用例钉的是**判据逻辑**（该报的报、不该报的不报），不是绝对精度。
"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.fall import NORMAL, FallConfig, FallDetector, torso_angle
from hik_person.geometry import Intrinsics
from hik_person.pose import NUM_KEYPOINTS, PoseInstance
from hik_person.track import PersonTracker

HZ = 15.0
STAND, LYING = 0.95, 0.15  # 髀中点沿"上"方向的高度：站着 0.95 m、躺着 0.15 m


def ramp(t: float, t0: float, dur: float, start: float, end: float) -> float:
    k = min(max((t - t0) / dur, 0.0), 1.0)
    return start + (end - start) * k


def feed(script, total: float, config: FallConfig | None = None) -> list[tuple[float, int]]:
    """按脚本喂信号，返回 [(t, level)] 的逐帧等级。"""
    detector = FallDetector(config)
    out = []
    for i in range(int(total * HZ)):
        t = i / HZ
        height, v_down, angle = script(t)
        detector.update(t, height=height, v_down=v_down, angle=angle)
        out.append((t, detector.level))
    return out


def transitions(log: list[tuple[float, int]]) -> list[tuple[float, int]]:
    out, prev = [], NORMAL
    for t, level in log:
        if level != prev:
            out.append((round(t, 1), level))
            prev = level
    return out


def fall_script(t: float):
    """站 3 s -> 0.4 s 摔倒 -> 躺 3 s -> 0.6 s 起身 -> 站。"""
    if t < 3.0:
        return STAND, 0.0, 3.0
    if t < 3.4:
        return ramp(t, 3.0, 0.4, STAND, LYING), 2.0, ramp(t, 3.0, 0.4, 3.0, 85.0)
    if t < 6.5:
        return LYING, 0.0, 85.0
    if t < 7.1:
        return ramp(t, 6.5, 0.6, LYING, STAND), -1.3, ramp(t, 6.5, 0.6, 85.0, 3.0)
    return STAND, 0.0, 3.0


# ------------------------------------------------------------ 状态机：该报的
def test_fall_is_detected_and_then_cleared() -> None:
    log = transitions(feed(fall_script, 9.0))
    assert log[0] == (3.0, 1)  # 冲击 -> 疑似
    assert log[1][1] == 2  # 躺倒够久 -> 报警
    assert 4.0 <= log[1][0] <= 5.6  # 冲击后 1-2.6 s 内确认（含 lying_time=1.2 s）
    assert log[2][1] == 0  # 起身直立 -> 解除
    assert 7.0 <= log[2][0] <= 8.2


def test_lying_without_angle_still_alarms() -> None:
    """躺着时关键点常被遮：角度测不到，靠"比站立参考低得多"也要能报。"""

    def script(t: float):
        if t < 2.0:
            return STAND, 0.0, 3.0
        if t < 2.4:
            return ramp(t, 2.0, 0.4, STAND, LYING), 2.0, ramp(t, 2.0, 0.4, 3.0, 85.0)
        return LYING, 0.0, float("nan")

    levels = transitions(feed(script, 8.0))
    assert levels[-1][1] == 2


def test_lying_time_is_configurable() -> None:
    """躺倒判据窗口调小，报警更早（阈值确实起作用，而不是写死的）。"""
    slow = transitions(feed(fall_script, 6.0))[1][0]
    quick = transitions(feed(fall_script, 6.0, FallConfig(lying_time=0.4)))[1][0]
    assert quick < slow


# ------------------------------------------------------------ 状态机：不该报的
def test_slow_lying_down_is_not_a_fall() -> None:
    """慢慢躺下（4 s，无冲击）不该报警：这是"主动躺下"和"摔倒"的分界。"""

    def script(t: float):
        if t < 3.0:
            return STAND, 0.0, 3.0
        if t < 7.0:
            return ramp(t, 3.0, 4.0, STAND, LYING), 0.2, ramp(t, 3.0, 4.0, 3.0, 85.0)
        if t < 10.0:
            return LYING, 0.0, 85.0
        if t < 13.0:
            return ramp(t, 10.0, 3.0, LYING, STAND), -0.27, ramp(t, 10.0, 3.0, 85.0, 3.0)
        return STAND, 0.0, 3.0

    assert max(level for _, level in feed(script, 15.0)) == NORMAL


def test_fast_squat_only_becomes_suspicious() -> None:
    """快速蹲下（下沉 0.5 m 但躯干一直直立）最多到"疑似"，不能报警，之后回到正常。"""

    def script(t: float):
        if t < 3.0:
            return STAND, 0.0, 3.0
        if t < 3.35:
            return ramp(t, 3.0, 0.35, STAND, 0.45), 1.5, ramp(t, 3.0, 0.35, 3.0, 28.0)
        if t < 4.4:
            return 0.45, 0.0, 28.0
        if t < 4.75:
            return ramp(t, 4.4, 0.35, 0.45, STAND), -1.4, ramp(t, 4.4, 0.35, 28.0, 3.0)
        return STAND, 0.0, 3.0

    log = feed(script, 7.0)
    assert max(level for _, level in log) == 1
    assert log[-1][1] == NORMAL  # 站起来后回到正常


def test_bending_down_is_not_a_fall() -> None:
    """弯腰捡东西（躯干到 75° 但只下沉 0.3 m、很慢）不该报。"""

    def script(t: float):
        if t < 2.0:
            return STAND, 0.0, 3.0
        if t < 3.0:
            return ramp(t, 2.0, 1.0, STAND, 0.65), 0.3, ramp(t, 2.0, 1.0, 3.0, 75.0)
        if t < 4.5:
            return 0.65, 0.0, 75.0
        if t < 5.5:
            return ramp(t, 4.5, 1.0, 0.65, STAND), -0.3, ramp(t, 4.5, 1.0, 75.0, 3.0)
        return STAND, 0.0, 3.0

    assert max(level for _, level in feed(script, 7.0)) == NORMAL


def test_disabled_detector_never_alarms() -> None:
    assert max(level for _, level in feed(fall_script, 9.0, FallConfig(enabled=False))) == NORMAL


# ------------------------------------------------------------------ 小工具
def test_torso_angle() -> None:
    up = np.array([0.0, -1.0, 0.0])
    assert torso_angle(np.array([0.0, -0.5, 0.0]), up) == pytest.approx(0.0, abs=1e-6)
    assert torso_angle(np.array([0.0, 0.0, 0.5]), up) == pytest.approx(90.0, abs=1e-6)
    assert np.isnan(torso_angle(np.zeros(3), up))
    assert np.isnan(torso_angle(None, up))


# --------------------------------------------------------------- 端到端合成
K = Intrinsics(fx=605.97, fy=604.48, cx=327.20, cy=245.09, width=640, height=480)
FLOOR = 0.95  # 合成场景里相机离地 0.95 m（相机系 y 向下，所以地面在 y = +0.95）
# 各关节离地高度（米）与横向偏移：按 1.72 m 的人大致摆
BODY = {
    0: (1.60, 0.00),  # nose（鼻尖，加 0.12 m 补偿就是头顶）
    1: (1.63, -0.03),
    2: (1.63, 0.03),
    3: (1.58, -0.09),
    4: (1.58, 0.09),
    5: (1.42, -0.20),
    6: (1.42, 0.20),
    7: (1.13, -0.23),
    8: (1.13, 0.23),
    9: (0.85, -0.25),
    10: (0.85, 0.25),
    11: (0.95, -0.15),
    12: (0.95, 0.15),
    13: (0.48, -0.15),
    14: (0.48, 0.15),
    15: (0.06, -0.08),
    16: (0.06, 0.08),
}


def skeleton(theta_deg: float, x0: float = 0.0, z0: float = 3.0) -> np.ndarray:
    """站直/倒下的骨架：绕 X 轴倒向 +Z，脚（踝）留在地面。"""
    theta = np.radians(theta_deg)
    axis = np.array([0.0, -np.cos(theta), np.sin(theta)])  # 身体轴："上" -> 前方
    points = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for i, (height, lateral) in BODY.items():
        points[i] = np.array([x0 + lateral, FLOOR, z0]) + height * axis
    return points


def render(pose3d: np.ndarray, rng: np.random.Generator, noise: float = 0.006):
    """一个骨架 -> (2D 姿态, 深度图)：把每个关节的真深度贴在自己的像素上。"""
    depth = np.full((480, 640), 6.0, dtype=np.float32)  # 背景：远处（关节处会被覆盖）
    uv = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    uv[:, 0] = K.cx + K.fx * pose3d[:, 0] / pose3d[:, 2]
    uv[:, 1] = K.cy + K.fy * pose3d[:, 1] / pose3d[:, 2]
    for i in range(NUM_KEYPOINTS):
        u, v = round(uv[i, 0]), round(uv[i, 1])
        if 2 <= u < 638 and 2 <= v < 478:
            value = pose3d[i, 2] + rng.normal(0.0, noise)
            depth[v - 2 : v + 3, u - 2 : u + 3] = value
    low, high = uv.min(axis=0), uv.max(axis=0)
    box = (float(low[0] - 10), float(low[1] - 10), float(high[0] + 10), float(high[1] + 10))
    pose = PoseInstance(
        conf=0.9, xyxy=box, keypoints=uv, kpt_conf=np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)
    )
    return pose, depth


def test_fall_through_full_pipeline() -> None:
    """站 4 s -> 0.45 s 摔倒 -> 躺 4 s：报警要在摔倒开始后 3 s 内出现。"""
    rng = np.random.default_rng(7)
    tracker = PersonTracker()
    levels: list[tuple[float, int]] = []
    lying_seen = False
    for i in range(int(9.0 * HZ)):
        t = i / HZ
        theta = 0.0 if t < 4.0 else min(90.0, 90.0 * (t - 4.0) / 0.45)
        pose, depth = render(skeleton(theta), rng)
        people = tracker.update([pose], depth, K, threshold=0.5, t=t)
        assert people, f"t={t:.2f} 没跟到人"
        person = people[0]
        levels.append((t, person.fall_level))
        lying_seen |= person.lying

    assert max(level for _, level in levels) == 2
    assert lying_seen
    alarm_t = next(t for t, level in levels if level == 2)
    assert 4.5 <= alarm_t <= 7.0, f"报警时刻 {alarm_t:.2f} 不合理"
    # 摔倒前一直正常（站着不会自己报）
    assert all(level == 0 for t, level in levels if t < 4.0)
    # 指标里也要能看到摔倒曲线用的那两个量
    assert "fall_level" in people[0].metrics
    assert "torso_angle" in people[0].metrics
    assert people[0].metrics["torso_angle"] > 60.0  # 躺着时躯干基本横过来
