"""per-track 两态事件状态机。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

from .. import spec


class State(IntEnum):
    """与 `fg_event_t.state` 的取值一一对应（shm v3 契约）。"""

    SUSPECTED = 0 # V1 留位不产生
    CONFIRMED = 1
    RETRACTED = 2

    def label(self) -> str:
        return self.name.lower()


@dataclass
class Event:
    """一条 per-person 事件。字段对齐 `fg_event_t` v3 的可填充部分。"""

    person_id: int
    state: State
    window_start: int
    window_end: int
    prob_fall: float
    coverage: float
    t_mono: float
    consec_hits: int
    consec_miss: int

    def key(self) -> tuple[int, int]:
        """`(person_id, window_start)` —— 幂等键去掉 `model_ver` 后的部分。"""
        return (self.person_id, self.window_start)


@dataclass
class TrackEventState:
    """一个人的状态。`MultiPipeline` 每个 track 一个。"""

    person_id: int
    consec_hits: int = 0
    consec_miss: int = 0
    state: State = State.SUSPECTED
    last_event_t: float = -1e9 # CLOCK_MONOTONIC 秒
    n_windows: int = 0
    max_prob: float = 0.0
 # 最近一次窗口的元数据，事件要用
    last_window_start: int = -1
    last_window_end: int = -1
    last_coverage: float = 0.0
    last_prob: float = 0.0
    ever_confirmed: bool = False
 # 这次确认是不是「强确认」（确认窗原始 prob >= cfg.strong_confirm）。
 # 只在 cfg.strong_confirm > 0 时有意义；强确认的事件不撤回，只重新武装。
    strong: bool = False
 # ③ 强窗破冷却：最近连续多少个评分窗的 prob >=
 # `cfg.cooldown_break_prob`。低于阈值/被否决（喂 0）的窗会清零。
 # 只在 cfg.cooldown_break_prob > 0 时参与判定；默认关时恒为 0。
    strong_streak: int = 0
 # 确认发生的那一窗。撤回事件必须沿用这一窗（见 `_event`）。
    confirm_window_start: int = -1
    confirm_window_end: int = -1

 # ------------------------------------------------------------------ 更新

    def update(self, prob_fall: float, window_start: int, window_end: int,
               coverage: float, now: float, cfg,
               counters: "Counters") -> list[Event]:
        """吃一个窗口的概率，返回这一步产生的事件（0 或 1 条）。"""
        self.n_windows += 1
        self.max_prob = max(self.max_prob, prob_fall)
        self.last_window_start = window_start
        self.last_window_end = window_end
        self.last_coverage = float(coverage)
        self.last_prob = float(prob_fall)

 # ③ 强窗破冷却：只统计喂给状态机的概率（veto/pregate feed0 为 0，
 # 天然打断连续）。默认 0 = 关闭，一律不累计。
        if cfg.cooldown_break_prob > 0 and prob_fall >= cfg.cooldown_break_prob:
            self.strong_streak += 1
        else:
            self.strong_streak = 0

        hit = prob_fall >= spec.THRESHOLD

        if hit:
            self.consec_hits += 1
            self.consec_miss = 0
        else:
            self.consec_miss += 1
            self.consec_hits = 0

 # ---- 未 confirmed：等够 confirm_windows 才升级 ----
        if not self.ever_confirmed:
            if hit and self.consec_hits >= cfg.confirm_windows:
                return self._confirm(now, cfg, counters)
            return []

 # ---- 已 confirmed：看是否要撤回 ----
        if self.state is State.CONFIRMED:
            if not hit and self.consec_miss >= cfg.retract_windows:
 # 强确认不撤回（`strong_confirm` > 0 才启用）：
 # 真跌在模型分数掉下去后仍会被撤回 —— 172531 真跌在无否决
 # 配置下 6 秒后照样撤回（p=0.001），几何否决只是把它提前约 3 秒。
 # 所以对「确认时就很确信」的事件，撤回的语义不成立。
 # 必须把状态重置成可再次确认（SUSPECTED），否则 172939 那种
 # 5 段蒙太奇只会出 1 条告警（离线实测 10 段→7 段）。
                if cfg.strong_confirm > 0 and self.strong:
                    self.state = State.SUSPECTED
                    self.consec_hits = 0
                    self.consec_miss = 0
                    counters.rearmed += 1
 # 重新武装必须同时清冷却。
 # `strong_confirm` 已经认定「这是真跌、这条事件保留」，并把
 # 人重置成可再次确认 —— 语义上就是「允许他再跌一次」。
 # 冷却若还按上一次跌倒的时间算，这个重新武装就是空的。
 # 实测（`clip_0831_173910`，生产配置）：早先一次误报
 # （窗口 450-510，p 0.922，t=17.0 s）把 14 秒后那次真跌
 # （窗口 870-930，p 0.996，t=31.0 s）静音 ⇒ GT 段 901-922 丢告警。
 # 与 `_retract` 那条「撤回不重置 `last_event_t`」不矛盾：
 # 撤回归撤回（判定这次跌完了、事件失效、冷却照常），重新武装是
 # 另一码事 —— 它没有判定跌完了，只是拒绝撤回。两处都保留
 # ⇒ 普通事件的冷却一个字没变。
 # 实测代价（19 视频、生产配置）：段召回 7/17→11/17、
 # confirmed 事件 13→17（与新增段一比一，不是刷屏）、
 # no_fall 误报 1/7 不变。
 # 只在 `strong_confirm > 0` 时执行 ⇒ `strong_confirm=0`
 # （915 窗对账用的口径）这一行永远不跑，对账天然不受影响。
                    self.last_event_t = -1e9
                    return []
                return self._retract(now, counters)
            return []

 # ---- 已 retracted：允许再次 confirmed（二次跌倒） ----
        if hit and self.consec_hits >= cfg.confirm_windows:
            return self._confirm(now, cfg, counters)
        return []

 # ------------------------------------------------------------------ 迁移

    def _confirm(self, now: float, cfg, counters: "Counters") -> list[Event]:
        """升级到 confirmed，除非还在冷却期。"""
        if now - self.last_event_t < cfg.cooldown_s:
 # ③ 强窗破冷却：连续 N 个评分窗都很强 ⇒ 放行这次确认。
 # 靶子 174226：首跌确认（p 0.536，非强）后 4.5 s，seg1 的
 # 0.9998 被 30 s 冷却静音；它前两窗是 0.9991/0.9998 的连续平台。
 # 只放行这一次确认；确认后仍按正常路径写 last_event_t。
            if (cfg.cooldown_break_prob > 0
                    and self.strong_streak >= cfg.cooldown_break_windows):
                counters.cooldown_broken += 1
                counters.cooldown_broken_by_person[self.person_id] = (
                    counters.cooldown_broken_by_person.get(self.person_id, 0) + 1)
                self.last_event_t = -1e9
            else:
                if self.consec_hits == cfg.confirm_windows:
                    counters.cooldown_suppressed += 1
                    counters.cooldown_suppressed_by_person[self.person_id] = (
                        counters.cooldown_suppressed_by_person.get(
                            self.person_id, 0) + 1)
 # 冷却期内不推进状态：保持原状态，等冷却过去后下一次命中再判
                return []
        self.state = State.CONFIRMED
        self.ever_confirmed = True
        self.last_event_t = now
 # 强确认标记：取确认窗的原始概率（`last_prob`），不是否决后的 0。
        self.strong = (cfg.strong_confirm > 0
                       and self.last_prob >= cfg.strong_confirm)
 # 记下确认窗：撤回事件要沿用它做幂等键（见 `_event`）
        self.confirm_window_start = self.last_window_start
        self.confirm_window_end = self.last_window_end
        counters.confirmed += 1
        return [self._event(State.CONFIRMED, now)]

    def _retract(self, now: float, counters: "Counters") -> list[Event]:
        """已 confirmed 后连续回落 ⇒ retracted。"""
        self.state = State.RETRACTED
        self.consec_hits = 0
        counters.retracted += 1
        return [self._event(State.RETRACTED, now)]

    def _event(self, state: State, now: float) -> Event:
        """撤回沿用确认那一窗的 window_start/end，不用当前窗。"""
        retract = (state is State.RETRACTED and self.confirm_window_start >= 0)
        return Event(
            person_id=self.person_id,
            state=state,
            window_start=(self.confirm_window_start if retract
                          else self.last_window_start),
            window_end=(self.confirm_window_end if retract
                        else self.last_window_end),
            prob_fall=self.last_prob,
            coverage=self.last_coverage,
            t_mono=now,
            consec_hits=self.consec_hits,
            consec_miss=self.consec_miss,
        )


@dataclass
class Counters:
    """运行期计数。全部是 per-run 的累加值，不是全局状态机。"""

    confirmed: int = 0
    retracted: int = 0
 # 强确认本应撤回、但被 `strong_confirm` 改成「重新武装」的次数。
 # 它和 `retracted` 是互斥的：同一次回落只会进其中一个计数器。
    rearmed: int = 0
    cooldown_suppressed: int = 0
    cooldown_suppressed_by_person: dict[int, int] = field(default_factory=dict)
 # ③ 强窗破冷却：冷却期内因「连续强窗」被放行的确认次数。
 # 它与 `cooldown_suppressed` 互斥：同一窗只会进其中一个计数器。
    cooldown_broken: int = 0
    cooldown_broken_by_person: dict[int, int] = field(default_factory=dict)
 # 几何否决：被 `geo_veto_ar_max` 判为「从没横过来」的窗口数。
 # 只在 `status == scored` 且真被否决时加，不是「所有窗都查了一遍」。
    geo_vetoed: int = 0
 # 被否决、且该窗原始概率本来 >= threshold 的次数 —— 这才是「少报了几条告警」。
    geo_vetoed_hits: int = 0
 # ---- 几何前置闸门（`FG_MULTI_PREGATE`，默认关）----
 # `skipped` = 模型没跑、状态机当窗口不存在；`feed0` = 模型没跑、喂 0。
 # 两者之和 × `geo_vetoed` 的比值就是「省了多少次 NPU 调用」的直接读数。
 # 关闭闸门时三项恒为 0。
    pregate_skipped: int = 0
    pregate_feed0: int = 0
 # 影子采样：被闸条目里仍然跑了模型的反事实样本数（1/N 抽样）。
    pregate_sampled: int = 0
    stgcn_dropped_max_persons: int = 0
    stgcn_dropped_queue_full: int = 0
    stgcn_dropped_by_window: dict[int, int] = field(default_factory=dict)
    track_expired: int = 0
    windows_scored: int = 0
    persons_scored: int = 0

    def as_dict(self) -> dict:
        return {
            "confirmed_total": self.confirmed,
            "retracted_total": self.retracted,
            "rearmed_total": self.rearmed,
            "cooldown_suppressed_total": self.cooldown_suppressed,
            "cooldown_suppressed_by_person": dict(
                sorted(self.cooldown_suppressed_by_person.items())),
            "cooldown_broken_total": self.cooldown_broken,
            "cooldown_broken_by_person": dict(
                sorted(self.cooldown_broken_by_person.items())),
            "geo_vetoed_total": self.geo_vetoed,
            "geo_vetoed_hits_total": self.geo_vetoed_hits,
            "pregate_total": {
                "skip": self.pregate_skipped,
                "feed0": self.pregate_feed0,
                "sampled": self.pregate_sampled,
            },
            "stgcn_dropped_total": {
                "max_persons": self.stgcn_dropped_max_persons,
                "queue_full": self.stgcn_dropped_queue_full,
            },
            "stgcn_dropped_by_window": dict(
                sorted(self.stgcn_dropped_by_window.items())),
            "track_expired_total": self.track_expired,
            "windows_scored": self.windows_scored,
            "persons_scored": self.persons_scored,
        }


class FrameClock:
    """回放时钟：`t = 当前窗口结束帧 / fps`。"""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps) if fps and float(fps) > 0 else 30.0
        self.t = 0.0

    def set_window_end(self, frame: int) -> None:
        self.t = float(frame) / self.fps

    def __call__(self) -> float:
        return self.t


class EventBook:
    """所有 track 的状态表 + 时钟。`MultiPipeline` 持有一个。"""

    def __init__(self, cfg, clock=None) -> None:
        self.cfg = cfg
        self._states: dict[int, TrackEventState] = {}
        self.counters = Counters()
 # 注入时钟，回放时用帧时间；生产用 monotonic。
 # 绝不用 time.time（REALTIME）—— 见模块 docstring。
        self._clock = clock if clock is not None else time.monotonic

 # ------------------------------------------------------------------ 时钟

    def now(self) -> float:
        return float(self._clock())

 # ------------------------------------------------------------------ 状态

    def state_of(self, person_id: int) -> TrackEventState:
        st = self._states.get(person_id)
        if st is None:
            st = self._states[person_id] = TrackEventState(person_id=person_id)
        return st

    def get(self, person_id: int) -> TrackEventState | None:
        """只读查询，不创建。对账时用来区分「没这个人」和「有但没事件」。"""
        return self._states.get(person_id)

    def person_ids(self) -> list[int]:
        return sorted(self._states)

    def __len__(self) -> int:
        return len(self._states)

 # ------------------------------------------------------------------ 更新

    def observe(self, person_id: int, prob_fall: float, window_start: int,
                window_end: int, coverage: float) -> list[Event]:
        """吃一个人的一个窗口。返回事件列表（0 或 1 条）。"""
        st = self.state_of(person_id)
        return st.update(prob_fall, window_start, window_end, coverage,
                         self.now(), self.cfg, self.counters)

    def expire(self, person_ids) -> int:
        """清掉已消失 track 的状态。"""
        n = 0
        for pid in person_ids:
            if self._states.pop(int(pid), None) is not None:
                n += 1
        self.counters.track_expired += n
        return n

    def snapshot(self) -> list[dict]:
        """当前所有 track 的状态，供对账 / 调试输出。"""
        return [{
            "person_id": st.person_id,
            "state": st.state.label(),
            "consec_hits": st.consec_hits,
            "consec_miss": st.consec_miss,
            "ever_confirmed": st.ever_confirmed,
            "n_windows": st.n_windows,
            "max_prob": st.max_prob,
            "last_event_t": st.last_event_t,
        } for st in (self._states[k] for k in sorted(self._states))]
