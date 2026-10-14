# SPDX-License-Identifier: AGPL-3.0-or-later
"""跟踪（卡尔曼 + IoU 关联 + 持久 id/轨迹）的单元测试 —— 不需要相机、不加载模型。

和 d435i-person-py 的 ``test_track.py`` 是同一套用例，去掉摔倒与人体尺寸/步态两节
（单目方案不出绝对量指标），补上"没有深度时只做 2D 关联"这一节。
"""

from __future__ import annotations

import numpy as np
import pytest

from hik_person.geometry import Intrinsics
from hik_person.pose import NUM_KEYPOINTS, PoseInstance
from hik_person.track import PersonTracker, ScalarKalman, iou

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def make_pose(x: float = 320.0, y: float = 240.0, conf: float = 0.8) -> PoseInstance:
    """一个 40×300 的框，关键点集中在框中间一条竖线上。"""
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    points[:, 0] = x - 4 + np.arange(NUM_KEYPOINTS) * 0.5
    points[:, 1] = y + np.arange(NUM_KEYPOINTS)
    return PoseInstance(
        conf=conf,
        xyxy=(x - 20.0, y - 150.0, x + 20.0, y + 150.0),
        keypoints=points,
        kpt_conf=CONF.copy(),
    )


def flat_depth(value: float = 2.0) -> np.ndarray:
    return np.full((480, 640), value, dtype=np.float32)


def tracker(**kwargs) -> PersonTracker:
    defaults = {
        "kalman": True,
        "iou_threshold": 0.3,
        "max_missed": 5,
        "max_missed_kpt": 3,
        "trail_length": 100,
        "trail_interval": 0.1,
    }
    defaults.update(kwargs)
    return PersonTracker(**defaults)


def step(track: PersonTracker, poses, t: float, depth: float | np.ndarray = 2.0):
    """一次 ``update``：``depth`` 给数字就当均匀深度图，给 None 就只做 2D 关联。"""
    plane = None if depth is None else flat_depth(depth)
    return track.update(poses, plane, None if plane is None else K, threshold=0.5, t=t)


# ------------------------------------------------------------------------ IoU
def test_iou_identical_is_one_and_disjoint_is_zero() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --------------------------------------------------------------------- Kalman
def test_kalman_holds_a_constant_signal() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.01)
    kf.init(5.0)
    for _ in range(10):
        kf.predict(0.1)
        kf.update(5.0)
    assert kf.position == pytest.approx(5.0, abs=1e-3)


def test_kalman_follows_a_ramp_with_lag() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.05)
    kf.init(0.0)
    for step_index in range(1, 30):
        kf.predict(1.0)
        kf.update(float(step_index))
    assert 25.0 < kf.position < 29.5


def test_kalman_predict_moves_along_velocity() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.01)
    kf.init(0.0)
    kf.predict(1.0)
    kf.update(1.0)  # 学到速度
    before = kf.position
    kf.predict(1.0)  # 纯预测
    assert kf.position > before


def test_kalman_smooths_a_noisy_measurement() -> None:
    rng = np.random.default_rng(0)
    kf = ScalarKalman(process_noise=0.1, measurement_noise=0.5)
    kf.init(0.0)
    filtered: list[float] = []
    raw: list[float] = []
    for _ in range(60):
        kf.predict(0.05)
        noisy = 1.0 + float(rng.normal(scale=0.3))
        kf.update(noisy)
        filtered.append(kf.position)
        raw.append(noisy)
    # 滤波后的波动必须比原始观测小（同样的样本，只比后半段，躲过初始化瞬态）
    filtered_error = np.abs(np.array(filtered[10:]) - 1.0).mean()
    raw_error = np.abs(np.array(raw[10:]) - 1.0).mean()
    assert filtered_error < raw_error


# ---------------------------------------------------------------- id 与轨迹
def test_same_person_keeps_its_id() -> None:
    track = tracker()
    for index in range(5):
        people = step(track, [make_pose(320.0 + index)], index * 0.1)
        assert [person.id for person in people] == [1]
    person = people[0]
    assert person.age == 5
    assert person.missed == 0
    assert person.valid_count == NUM_KEYPOINTS
    assert person.distance == pytest.approx(np.linalg.norm(person.position))


def test_two_people_get_different_ids() -> None:
    people = step(tracker(), [make_pose(200.0), make_pose(450.0)], 0.0)
    assert [person.id for person in people] == [1, 2]


def test_missing_person_is_predicted_then_dropped_and_id_is_not_reused() -> None:
    track = tracker(max_missed=2)
    people = step(track, [make_pose()], 0.0)
    # 第一帧就是命中帧，不该被当成漏检
    assert (people[0].missed, people[0].age) == (0, 1)

    # 漏检 1 帧：还在，标成 missed，位置靠卡尔曼预测，2D 关键点不再画
    people = step(track, [], 0.1)
    assert len(people) == 1
    assert people[0].missed == 1
    assert np.isfinite(people[0].position).all()
    assert people[0].kpt_conf.max() == 0.0

    # 漏检到 max_missed 帧还能留住，再多一帧就删
    assert len(step(track, [], 0.2)) == 1
    assert step(track, [], 0.3) == []

    # 新人拿到新 id（旧 id 永不复用）
    people = step(track, [make_pose()], 0.4)
    assert [person.id for person in people] == [2]


def test_trail_grows_with_time_and_respects_interval() -> None:
    track = tracker(trail_interval=0.1)
    step(track, [make_pose()], 0.0)
    assert len(step(track, [make_pose()], 0.05)[0].trail) == 1
    assert len(step(track, [make_pose()], 0.1)[0].trail) == 2


def test_trail_can_be_disabled_and_is_capped() -> None:
    people = step(tracker(trail_length=0), [make_pose()], 0.0)
    assert len(people[0].trail) == 0

    track = tracker(trail_length=3, trail_interval=0.0)
    for index in range(6):
        people = step(track, [make_pose()], index * 0.1)
    assert len(people[0].trail) == 3  # deque 只留最近 3 个点


def test_walking_person_trail_is_monotonic_in_x() -> None:
    track = tracker(trail_interval=0.0)
    for index in range(5):
        people = step(track, [make_pose(300.0 + index * 10)], index * 0.1)
    xs = people[0].trail[:, 0]
    assert np.all(np.diff(xs) > 0)


# --------------------------------------------------------- 3D 距离门限与开关
def test_match_max_distance_prevents_swallowing_a_far_person() -> None:
    # 框完全重合（IoU=1），但深度从 2 m 跳到 8 m：默认会被当成同一人
    loose = tracker()
    step(loose, [make_pose()], 0.0, depth=2.0)
    people = step(loose, [make_pose()], 0.1, depth=8.0)
    assert [person.id for person in people] == [1]

    strict = tracker(match_max_distance=1.0)
    step(strict, [make_pose()], 0.0, depth=2.0)
    people = step(strict, [make_pose()], 0.1, depth=8.0)
    assert sorted(person.id for person in people) == [1, 2]


def test_without_kalman_joints_come_straight_from_the_observation() -> None:
    people = step(tracker(kalman=False), [make_pose()], 0.0, depth=3.0)
    assert np.allclose(people[0].joints[:, 2], 3.0)
    assert people[0].valid_count == NUM_KEYPOINTS


def test_kalman_keeps_joints_and_position_on_the_same_depth() -> None:
    people = step(tracker(), [make_pose()], 0.0, depth=4.0)
    # 第一帧的滤波值就是观测值
    assert np.allclose(people[0].joints[:, 2], 4.0, atol=1e-3)
    assert people[0].position[2] == pytest.approx(4.0, abs=1e-3)


def test_no_depth_means_no_position_but_still_tracked() -> None:
    track = tracker()
    people = track.update(
        [make_pose()], np.full((480, 640), np.nan, np.float32), K, threshold=0.5, t=0.0
    )
    assert people[0].valid_count == 0
    # 位置未被观测初始化过 -> nan（绝不能报成原点 (0,0,0)）
    assert np.isnan(people[0].position).all()
    assert np.isnan(people[0].distance)
    assert len(people[0].trail) == 0


def test_position_filter_initializes_on_the_first_valid_frame() -> None:
    track = tracker()
    empty = np.full((480, 640), np.nan, np.float32)
    track.update([make_pose()], empty, K, threshold=0.5, t=0.0)
    people = step(track, [make_pose()], 0.1, depth=2.0)
    assert people[0].position[2] == pytest.approx(2.0, abs=1e-3)
    assert len(people[0].trail) == 1


# --------------------------------------------------- 只有 2D 的时候（--no-pose3d）
def test_depth_none_only_tracks_in_2d() -> None:
    """深度模型比 pose 贵：没到深度时刻的帧只关 2D 关联，不能拿旧深度抬新帧。"""
    track = tracker()
    people = step(track, [make_pose()], 0.0, depth=2.0)
    assert people[0].valid_count == NUM_KEYPOINTS

    people = step(track, [make_pose(322.0)], 0.1, depth=None)
    assert [person.id for person in people] == [1]  # id 不变
    assert people[0].missed == 0  # 这一帧命中了（框还在）
    assert np.isfinite(people[0].position).all()  # 位置走卡尔曼预测
    assert people[0].kpt_conf.max() > 0.0  # 2D 关键点照画


def test_depth_none_never_invents_a_3d_skeleton() -> None:
    """从没见过深度时不能凭空画出 3D：没有观测就没有关节（更不能画到原点）。"""
    people = step(tracker(), [make_pose()], 0.0, depth=None)
    assert people[0].valid_count == 0
    assert len(people[0].joints) == 0
    assert len(people[0].strips) == 0
    assert np.isnan(people[0].distance)


def test_depth_none_still_associates_two_people() -> None:
    track = tracker()
    people = step(track, [make_pose(200.0), make_pose(450.0)], 0.0, depth=None)
    assert [person.id for person in people] == [1, 2]
    people = step(track, [make_pose(205.0), make_pose(445.0)], 0.1, depth=None)
    assert [person.id for person in people] == [1, 2]


# ------------------------------------------------------------------- 速度
def test_walking_speed_is_positive_and_state_flips_to_walking() -> None:
    """速度是相对量：单目尺度会影响数值大小，但“在走 / 站着”分得开。"""
    track = tracker(speed_smoothing=0.0)
    for index in range(6):
        people = step(track, [make_pose(320.0 + index * 6)], index * 0.1)
    # 6 px / 0.1 s、深度 2 m、fx=500 -> 0.24 m/s 左右
    assert people[0].speed == pytest.approx(0.24, abs=0.05)
    assert people[0].speed_ground <= people[0].speed + 1e-6
    assert people[0].state == "行走"

    still = step(tracker(speed_deadband=0.15), [make_pose()], 0.0)
    assert still[0].state == "静止"
