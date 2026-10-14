# SPDX-License-Identifier: AGPL-3.0-or-later
"""摔倒检测：单人一条时序状态机（纯逻辑，可单测）。

判据分三步，缺一步都不报警，这样才不会把"弯腰系鞋带""快速蹲一下""主动慢慢躺下"
当成摔倒：

1. **冲击**——躯干中心（髀中点优先）沿"上"方向快速下沉：要么下沉速度超过
   ``impact_speed``，要么在 ``impact_window`` 秒内累计掉了 ``drop_height``。
2. **躺倒**——躯干与"上"方向夹角超过 ``lying_angle``（人横过来了）；角度测不到时
   （躺着时关键点常被床/腿挡住）用兜底判据：中心高度低于站立参考高度
   ``lying_height`` 以上。
3. **持续**——躺倒连续保持 ``lying_time`` 秒才报警（把"蹲一下又起来"排除掉），
   而且必须在冲击后 ``confirm_window`` 秒内完成（慢慢躺下没有冲击，不报）。
   报警后要**直立保持** ``recover_time`` 秒才解除（人爬起来的过程中不闪烁）。

**这里的米制阈值都建立在单目估计深度上**（见 geometry/pose3d 的尺度说明），
所以 ``impact_speed`` / ``drop_height`` / ``lying_height`` 是"估计尺度"下的经验值：
人的绝对远近会随模型飘，但同一场景内的相对下沉量与躯干夹角是可靠的。
真深度方案（d435i）的阈值是实测的，两边不要互相照搬。
d435i 那边的判据说明里写着"深度相机测的是真位移，所以 0.4 m 只用了 0.3 s 这种判据
是实测的"——这边没有这个底气，调阈值请对着自己的场景看曲线。

已知边界：主动快速躺到地上会报警（速度和姿态与摔倒根本分不开）；被人抱起来搬走
也可能误报。这类只能靠报警后人工确认，本模块只负责报，不做抑制。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

# 状态：0 正常、1 疑似（有冲击 / 正在倒地）、2 摔倒（报警）
NORMAL, SUSPECT, FALLEN = 0, 1, 2
STATE_NAMES: dict[int, str] = {NORMAL: "", SUSPECT: "疑似摔倒", FALLEN: "摔倒"}


@dataclass(frozen=True)
class FallConfig:
    """摔倒判据。全部来自 config.toml 的 [fall]。"""

    enabled: bool = True
    # 冲击：向下速度阈值（m/s）、窗口内累计下沉阈值（m）与窗口长度（s）。
    impact_speed: float = 1.0
    drop_height: float = 0.35
    impact_window: float = 0.7
    # 躺倒：躯干与"上"方向夹角（度），以及"比站立参考高度低这么多"（米）的兜底判据。
    lying_angle: float = 60.0
    lying_height: float = 0.35
    # 时序：躺倒要连续保持多久才确认、冲击后多久内确认有效、直立多久才解除。
    lying_time: float = 1.2
    confirm_window: float = 3.0
    recover_angle: float = 35.0
    recover_time: float = 1.0


class FallDetector:
    """单人的摔倒状态机：每帧喂 ``update()``，读 ``level`` / ``state_name``。"""

    def __init__(self, config: FallConfig | None = None, *, ref_tau: float = 3.0):
        self.config = config or FallConfig()
        self.level: int = NORMAL
        self.since: float | None = None  # 进入当前状态的时刻（报警时刻看它）
        self.ref: float = float("nan")  # 站立参考高度（EMA，只在正常+直立时更新）
        self.angle: float = float("nan")  # 最近一次测到的躯干夹角（度，给曲线用）
        self.lying: bool = False  # 本帧是不是躺倒姿态（给显示用，区分“只是震了一下”）
        self._ref_tau = max(float(ref_tau), 1e-3)
        self._heights: deque = deque()  # (t, 高度) -> "窗口内下沉"判据
        self._last_t: float | None = None
        self._impact_t: float | None = None
        self._lying_since: float | None = None
        self._upright_since: float | None = None

    # ------------------------------------------------------------------ 对外
    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.level, "")

    @property
    def alarmed(self) -> bool:
        return self.level == FALLEN

    def reset(self) -> None:
        """清空时序（轨迹被删/重新创建时用）。"""
        self.level = NORMAL
        self.since = None
        self.ref = float("nan")
        self.angle = float("nan")
        self.lying = False
        self._heights.clear()
        self._last_t = None
        self._impact_t = None
        self._lying_since = None
        self._upright_since = None

    # ------------------------------------------------------------------ 每帧
    def update(
        self,
        t: float,
        *,
        height: float = float("nan"),
        v_down: float = 0.0,
        angle: float = float("nan"),
    ) -> None:
        """喂一帧信号（缺哪个给 nan，缺的判据自动跳过）。

        ``height`` 是躯干中心沿"上"方向的高度（相对量即可，内部只比较差值）、
        ``v_down`` 是向下为正的竖直速度（m/s）、``angle`` 是躯干与"上"方向的夹角（度）。
        """
        cfg = self.config
        if not cfg.enabled:
            return
        if not math.isfinite(angle):
            pass  # 保留上一次的 angle 曲线值，避免躺着时曲线断成一片
        else:
            self.angle = float(angle)
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        upright = self._is_upright(angle, height)
        # 站立参考高度只在"正常 + 直立"时更新：躺着的高度会把参考值带歪
        if self.level == NORMAL and upright and math.isfinite(height):
            alpha = 1.0 if not math.isfinite(self.ref) else 1.0 - math.exp(-dt / self._ref_tau)
            self.ref = (
                height if not math.isfinite(self.ref) else (1 - alpha) * self.ref + alpha * height
            )

        impact = False
        if math.isfinite(v_down) and v_down >= cfg.impact_speed:
            impact = True
        if math.isfinite(height):
            self._heights.append((t, float(height)))
            while self._heights and t - self._heights[0][0] > cfg.impact_window:
                self._heights.popleft()
            if len(self._heights) >= 2:
                # 从窗口内的高点“掉下来”多少（不能用 max-min：人起身时也会被算成下沉）
                drop = max(h for _, h in self._heights) - float(height)
                if drop >= cfg.drop_height:
                    impact = True
        lying = self._is_lying(angle, height)
        self.lying = lying

        if self.level == NORMAL:
            if impact:
                self._enter(SUSPECT, t)
        elif self.level == SUSPECT:
            if impact:  # 二次冲击（撞家具）继续算有效
                self._impact_t = t
            if self._impact_t is not None and t - self._impact_t > cfg.confirm_window:
                self._enter(NORMAL, t)  # 冲击完又站起来/一直立着 -> 没摔
                return
            if lying:
                self._lying_since = t if self._lying_since is None else self._lying_since
                if t - self._lying_since >= cfg.lying_time:
                    self._enter(FALLEN, t)
            else:
                self._lying_since = None
        else:  # FALLEN：等直立恢复
            if upright:
                self._upright_since = t if self._upright_since is None else self._upright_since
                if t - self._upright_since >= cfg.recover_time:
                    self._enter(NORMAL, t)
            else:
                self._upright_since = None

    # ------------------------------------------------------------------ 内部
    def _enter(self, level: int, t: float) -> None:
        self.level = level
        self.since = t
        self._impact_t = None
        self._lying_since = None
        self._upright_since = None
        self._heights.clear()

    def _is_lying(self, angle: float, height: float) -> bool:
        cfg = self.config
        if math.isfinite(angle):
            return angle >= cfg.lying_angle
        # 角度测不到（躺着常被遮）：退回"比站立参考低得多"这一条
        return (
            math.isfinite(height)
            and math.isfinite(self.ref)
            and height <= self.ref - cfg.lying_height
        )

    def _is_upright(self, angle: float, height: float) -> bool:
        cfg = self.config
        if math.isfinite(angle):
            return angle <= cfg.recover_angle
        return (
            math.isfinite(height)
            and math.isfinite(self.ref)
            and height >= self.ref - cfg.lying_height * 0.5
        )


def torso_angle(torso, up) -> float:
    """躯干向量与"上"方向的夹角（度，0 = 站直、90 = 完全横过来）；退化输入给 nan。"""
    if torso is None or up is None:
        return float("nan")
    norm = float((torso**2).sum()) ** 0.5
    if not math.isfinite(norm) or norm < 1e-6:
        return float("nan")
    cos = float((torso * up).sum()) / norm
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))
