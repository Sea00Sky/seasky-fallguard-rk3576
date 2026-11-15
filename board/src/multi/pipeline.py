"""per-person 管线：在 `src.pipeline.Pipeline` 上包一层，输出每人一条记录。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import spec
from ..gate import WindowScore
from ..pipeline import Pipeline, PipelineConfig, VideoResult
from .boxgeo import BoxHistory
from .config import MultiConfig
from .events import Counters, Event, EventBook, State

# 每人每窗在本次运行里的结局
STATUS_SCORED = "scored" # 过了覆盖率门并真的跑了模型
STATUS_GATED = "gated" # coverage < 0.70，D-043 门拒掉，没跑模型
STATUS_CAPPED = "capped" # 过了覆盖率门但被 MAX_PERSONS_STGCN 丢掉
STATUS_PREGATE = "pregate" # 几何前置闸门在 NPU 之前拦下（原因见 `PersonWindow.pregate`）


@dataclass
class PersonWindow:
    """`(window, person)` 一条记录 —— 旧管线在 `score_windows` 里丢掉的就是它。"""

    window_id: int
    window_start: int
    window_end: int
    person_id: int # D-042 链根；`stitch=False` 时就是 track id
    prob_fall: float
    coverage: float
    status: str
    members: list[int] = field(default_factory=list)
 # ---- 检测框几何（`geo_veto_ar_max` 关闭时全部是 None/0/False）----
 # `prob_fall` 保持模型原值，否决只影响喂给状态机的那一个数。
 # 这样 dump 里既能看到原分、又能看到是否被否决 —— 可审计。
    geo_ar_max: float | None = None
    geo_bottom_shift: float | None = None
    geo_r_end: float | None = None
    geo_n_box: int = 0
    vetoed: bool = False
 # ---- 几何前置闸门（`pregate=1` 且该窗被闸时才非 None）----
 # `"skip"` = 几何组命中：模型没跑，状态机当这个窗口不存在
 # `"feed0"` = r_end 组命中：模型没跑，但概率按 0.0 喂给状态机（= 今天的 veto）
 # 与 `vetoed` 的分工：`vetoed` 仍是「喂给状态机的数被换成 0 / 状态机看不见」，
 # `pregate` 说明是哪种、以及模型到底跑没跑。老 dump 没有本字段。
    pregate: str | None = None
 # 影子采样得到的反事实概率（`pregate_sample=N`，1/N 抽样）。
 # 只做审计：不喂状态机、不进 `_scored` ⇒ 不改变任何判定。
    pregate_prob: float | None = None

    @property
    def hit(self) -> bool:
        """与 `WindowScore.hit` 同口径（`spec.THRESHOLD`）。"""
        from .. import spec
        return self.prob_fall >= spec.THRESHOLD


@dataclass
class MultiVideoResult:
    """`VideoResult` + 每人记录 + 事件 + 计数。"""

    base: VideoResult
    per_person: list[PersonWindow]
    events: list[Event]
    counters: Counters
    multi_cfg: dict
    n_persons: int # 本次运行出现过的 person_id 个数

 # ---- 透传，方便调用方少写 `.base.` ----
    @property
    def video_id(self) -> str:
        return self.base.video_id

    @property
    def n_frames(self) -> int:
        return self.base.n_frames

    @property
    def windows(self) -> list[WindowScore]:
        return self.base.windows

    def confirmed(self) -> list[Event]:
        return [e for e in self.events if e.state is State.CONFIRMED]

    def retracted(self) -> list[Event]:
        return [e for e in self.events if e.state is State.RETRACTED]


class MultiPipeline(Pipeline):
    """`Pipeline` + per-track 事件状态机 + `MAX_PERSONS_STGCN` 闸门。"""

    def __init__(self, cfg: PipelineConfig, detector=None, stgcn_session=None,
                 mcfg: MultiConfig | None = None, clock=None) -> None:
 # 必须在 super.__init__ 之前赋值：父类 __init__ 会调用 self.reset，
 # 而我们的 reset 要读 self.mcfg / self._clock。
        self.mcfg = mcfg if mcfg is not None else MultiConfig()
        self._clock = clock
        super().__init__(cfg, detector=detector, stgcn_session=stgcn_session)

 # ------------------------------------------------------------------ 状态

    def reset(self) -> None:
        super().reset()
        self.events_book = EventBook(self.mcfg, self._clock)
        self._per_person: list[PersonWindow] = []
        self._events: list[Event] = []
 # 每人最近一个窗口，只为渲染（`display_persons`）。
 # 不能每帧扫 `_per_person`：那个列表只追加、从不裁剪（生产进程
 # 长跑几小时就是几十万条），扫它是 O(n)/帧。
        self._last_pw: dict[int, PersonWindow] = {}
 # (window_id, person_id) -> 被人数闸门丢掉。窗口收尾时即 pop，不长期持有。
        self._capped: dict[tuple[int, int], int] = {}
 # ---- 几何前置闸门（`FG_MULTI_PREGATE=1`）----
 # (wid,pid) -> "skip"/"feed0"：NPU 之前拦下的窗口，`_on_window` 收尾时 pop。
 # `_shadow`：(wid,pid) 的影子采样集合（模型照跑，分数只做审计）。
 # `_geo_cache`：(wid,pid) -> 6 元组几何，闸门与 `_before_score` 共用一份，
 # 避免同一窗同一个人被算两遍（`geo_vetoed` 会重复计数）。
 # 三个容器只在主线程读写；闸门关闭时恒为空，一个代码路径都不执行。
        self._pregate: dict[tuple[int, int], str] = {}
        self._shadow: set[tuple[int, int]] = set()
        self._geo_cache: dict[tuple[int, int], tuple] = {}
 # 检测框历史。两个阈值都为 0 时是 None ——
 # 关闭时 `push` 里每一处都 `is not None` 挡住，一个代码路径都不执行，
 # 所以默认配置下逐位行为不变（ 基线可复现）。
        self._boxhist: BoxHistory | None = (
            BoxHistory(self.cfg.window_frames)
            if (self.mcfg.geo_veto_ar_max > 0
                or self.mcfg.geo_veto_bottom_shift > 0
                or self.mcfg.geo_veto_r_end > 0) else None)

    @property
    def counters(self) -> Counters:
        return self.events_book.counters

 # -------------------------------------------------------------- 人数闸门

    def _gate_items(self, items):
        """在真正跑 NPU 之前施加 `MAX_PERSONS_STGCN`。"""
        if not items:
            return []
        cap = int(self.mcfg.max_persons_stgcn)
        kept = items
        if len(items) > cap:
 # (-coverage, person_id) 升序 = 覆盖率降序、同覆盖率按 id 升序
            order = sorted(range(len(items)),
                           key=lambda i: (-float(items[i][2]), int(items[i][1])))
            keep_idx = set(order[:cap])
            kept = [it for i, it in enumerate(items) if i in keep_idx]
            for i, (wid, pid, cov, _x) in enumerate(items):
                if i in keep_idx:
                    continue
                self._capped[(int(wid), int(pid))] = 1
                c = self.counters
                c.stgcn_dropped_max_persons += 1
                c.stgcn_dropped_by_window[int(wid)] = (
                    c.stgcn_dropped_by_window.get(int(wid), 0) + 1)
        return self._pregate_items(kept)

    def _pregate_items(self, items):
        """几何前置闸门：在 NPU 之前把「今天反正会被 veto 的人」拿掉。"""
        cfg = self.mcfg
        if not int(cfg.pregate):
            return items
        if self._boxhist is None: # 三个 geo_veto_* 全 0 ⇒ 无条件可判
            return items
        sample = int(cfg.pregate_sample)
        out = []
        for wid, pid, cov, x in items:
 # `_candidates[wid]` 的下标就是 wid（push 里 `wid = len(_candidates)`）
            _w, start, end = self._candidates[int(wid)]
            ar, bs, r_end, n_box, group = self._geo_cached(
                int(wid), int(pid), int(start), int(end))
            if group is None:
                out.append((wid, pid, cov, x))
                continue
            kind = "skip" if group == "geo" else "feed0"
            key = (int(wid), int(pid))
            self._pregate[key] = kind
            c = self.counters
            if kind == "skip":
                c.pregate_skipped += 1
            else:
                c.pregate_feed0 += 1
            if sample > 0 and (c.pregate_skipped + c.pregate_feed0) % sample == 0:
                c.pregate_sampled += 1
                self._shadow.add(key)
                out.append((wid, pid, cov, x))
        return out

 # ------------------------------------------------------------------ 帧

    def push(self, frame_bgr: np.ndarray) -> list[Event]:
        """喂一帧；返回这一步产生的事件（0 或 1 条，通常 0 条）。"""
        n_cand = len(self._candidates)
        super().push(frame_bgr) # ← 父类全部算术，一行没改

 # ---- track 消失 ⇒ 清状态----
 # 父类 push 里已经调过 ring.expire，但没返回被丢的 id。
 # 这里用「ring 里已经没有它」来判断，避免再调一次 expire 改变行为。
 # 条件与位置都没动：仍然只在有窗口关闭的那一帧跑。
        if len(self._candidates) != n_cand:
            live = set(self.ring.track_ids())
            stale = [p for p in self.events_book.person_ids()
                     if p not in live and not self._chain_alive(p)]
            if stale:
                self.events_book.expire(stale)

        return list(self._events_out)

 # ------------------------------------------------------- 父类的四个主线程钩子

    def _after_frame(self, frame_index: int) -> None:
        """把本帧真的被检测到的框写进 `_boxhist`（仅几何否决开启时才做）。"""
        if self._boxhist is not None:
            self._record_boxes(frame_index)

    def _before_score(self, wid, start, end, obs, items) -> dict:
        """开窗那一刻、NPU 之前：回放时钟 + 几何采样 + 拼接链成员快照。"""
 # 回放时钟：把「当前窗口结束帧」推给 `FrameClock`。
 # 生产传 None ⇒ `getattr` 返回 None，这里一行不做。
 # 必须在 `events_book.observe` 之前 —— 冷却读的就是这一刻的时间。
 # （异步模式下 observe 晚 1–2 帧，但生产用的是真墙钟、不用 FrameClock。）
        setter = getattr(self._clock, "set_window_end", None)
        if setter is not None:
            setter(int(end))

        geo: dict[int, tuple] = {}
        if self._boxhist is not None:
            for _w, pid, _cov, _x in items:
                pid = int(pid)
                geo[pid] = self._geo_veto(int(wid), pid, int(start), int(end))
 # 被前置闸门拿掉的人：几何在 `_gate_items` 里已经算过（缓存），
 # 这里补进 `extra` —— dump 里要能看到「他为什么被闸」的证据。
 # 闸门关闭时 `_pregate` 恒空 ⇒ 这一段一次都不跑。
            for _w, pid, _cov in obs:
                pid = int(pid)
                if pid in geo:
                    continue
                key = (int(wid), pid)
                if key in self._pregate:
                    geo[pid] = self._geo5(self._geo_cache[key])
        plain = (None, 0.0, None, 0, False)
        out = {int(pid): (self._members_of(int(pid)), geo.get(int(pid), plain))
               for _w, pid, _cov in obs}
 # 缓存只服务「开窗 → 落账」这一小段；`_on_window` 用 `extra`，不读缓存。
        self._geo_cache.clear()
        return out

    def _on_window(self, wid, start, end, obs, items, probs, extra) -> None:
        """结果到手：写 `_scored`（父类）+ 每人一条记录 + 跑状态机。"""
 # 影子采样的人不进 `_scored`：他们是被闸掉的窗口，模型只是为
 # 反事实审计才跑。不进 `_scored` ⇒ 每窗 max 与他们无关。
        if self._shadow:
            keep_items, keep_probs = [], []
            for it, p in zip(items, probs):
                if (int(it[0]), int(it[1])) in self._shadow:
                    continue
                keep_items.append(it)
                keep_probs.append(p)
            super()._on_window(wid, start, end, obs, keep_items, keep_probs, extra)
        else:
            super()._on_window(wid, start, end, obs, items, probs, extra)

        scored_by_pid = {int(pid): (float(cov), float(p))
                         for (_w, pid, cov, _x), p in zip(items, probs)}

        out: list[Event] = []
        for _w, pid, cov in obs:
            pid = int(pid)
            cov = float(cov)
            capped = self._capped.pop((int(wid), pid), 0) > 0
            pre = self._pregate.pop((int(wid), pid), None)
            shadow_prob = None
            if pre is not None:
                status = STATUS_PREGATE
                prob = 0.0
                if pid in scored_by_pid: # 影子采样：模型确实跑过
                    shadow_prob = scored_by_pid[pid][1]
                self._shadow.discard((int(wid), pid))
            elif pid in scored_by_pid:
                status = STATUS_SCORED
                prob = scored_by_pid[pid][1]
            elif capped:
                status = STATUS_CAPPED
                prob = 0.0
            else:
                status = STATUS_GATED
                prob = 0.0

 # 成员与几何都用开窗那一刻的快照（见 `_before_score`）。
            info = extra.get(pid)
            if info is None:
                members = self._members_of(pid)
                geo = (None, 0.0, None, 0, False)
            else:
                members, geo = info

            pw = PersonWindow(
                window_id=int(wid), window_start=int(start), window_end=int(end),
                person_id=pid, prob_fall=prob, coverage=cov, status=status,
                members=members, pregate=pre, pregate_prob=shadow_prob)
            self._per_person.append(pw)
            self._last_pw[pid] = pw # 渲染用索引，O(1) 取最近一个窗口

 # ---- 几何前置闸门：模型没跑，状态机怎么吃见 config 的混合语义 ----
            if status == STATUS_PREGATE:
                ar, bshift, r_end, n_box, vetoed = geo
                pw.geo_ar_max = ar
                pw.geo_bottom_shift = bshift
                pw.geo_r_end = r_end
                pw.geo_n_box = n_box
                pw.vetoed = vetoed
 # 「本来会报警」的温度计：只有影子采样的人还答得出来。
                if shadow_prob is not None and shadow_prob >= spec.THRESHOLD:
                    self.counters.geo_vetoed_hits += 1
                if pre == "feed0":
 # 与关闭闸门时的 veto 路径逐项相同：喂 0，照常计人。
                    self.counters.persons_scored += 1
                    out.extend(self.events_book.observe(
                        pid, 0.0, int(start), int(end), cov))
 # `skip`：状态机当这个窗口不存在（不推进 consec，也不 observe）
                continue

 # ---- 状态机只吃「真的被评分」的人 ----
 # 覆盖率门外 / 被闸门丢掉的窗口对这个人而言是「没有数据」，
 # 不是 miss（见模块 docstring）。
            if status != STATUS_SCORED:
                continue
            self.counters.persons_scored += 1

 # ---- 检测框几何否决（默认关闭）----
 # 关掉时 `_before_score` 给的是默认值，与原来 `_geo_veto`
 # 在 `_boxhist is None` 时的返回完全相同。
            ar, bshift, r_end, n_box, vetoed = geo
            pw.geo_ar_max = ar
            pw.geo_bottom_shift = bshift
            pw.geo_r_end = r_end
            pw.geo_n_box = n_box
            pw.vetoed = vetoed
            if vetoed and prob >= spec.THRESHOLD:
 # `geo_vetoed` 在 `_before_score` 里已经记过（那里还不知道
 # 概率）；这一条要概率，所以只能等结果回来再记。
                self.counters.geo_vetoed_hits += 1
            out.extend(self.events_book.observe(
                pid, 0.0 if vetoed else prob, int(start), int(end), cov))

        self.counters.windows_scored += 1
        self._events.extend(out)
        self._events_out.extend(out)

 # ------------------------------------------------------------------ 辅助

    def _record_boxes(self, frame_index: int) -> None:
        """把本帧真的被检测到的框写进 `_boxhist`。"""
        hist = self._boxhist
        if hist is None:
            return
        live = []
        for tid in self.ring.track_ids():
            tid = int(tid)
            live.append(tid)
            if self.ring.last_frame(tid) != frame_index:
                continue
            box = self.tracker.bbox_of(tid)
            if box is not None:
                hist.add(tid, frame_index, box)
        hist.expire(live)

    def _geo_eval(self, person_id: int, start: int, end: int):
        """算这一窗这个人的检测框几何，并判定命中哪一组。纯函数，不记数。"""
        hist = self._boxhist
        if hist is None:
            return None, 0.0, None, 0, None
        members = self._members_of(int(person_id))
        ar, bshift, r_end, n_box = hist.window_geo(members, start, end)
        if ar is None:
            return None, 0.0, None, 0, None
        if n_box < self.mcfg.geo_veto_min_box:
            return ar, bshift, r_end, n_box, None

        cfg = self.mcfg
 # 几何组：所有已启用的几何条件同时满足
        geo = []
        if cfg.geo_veto_ar_max > 0:
            geo.append(ar < cfg.geo_veto_ar_max)
        if cfg.geo_veto_bottom_shift > 0:
            geo.append(bshift < cfg.geo_veto_bottom_shift)
        if geo and all(geo):
            return ar, bshift, r_end, n_box, "geo"
 # 时序组：窗口结束时已经站起来
        if (cfg.geo_veto_r_end > 0 and r_end is not None
                and r_end > cfg.geo_veto_r_end):
            return ar, bshift, r_end, n_box, "r_end"
        return ar, bshift, r_end, n_box, None

    def _geo_cached(self, window_id: int, person_id: int, start: int, end: int):
        """`_geo_eval` + 每窗每人只算一次的缓存 + `geo_vetoed` 计数。"""
        key = (int(window_id), int(person_id))
        got = self._geo_cache.get(key)
        if got is None:
            got = self._geo_eval(int(person_id), int(start), int(end))
            self._geo_cache[key] = got
            if got[4] is not None:
                self.counters.geo_vetoed += 1
        return got

    @staticmethod
    def _geo5(t: tuple) -> tuple:
        """6 元组（含组别）→ 旧口径 5 元组 `(ar, bs, r_end, n_box, vetoed)`。"""
        return t[0], t[1], t[2], t[3], t[4] is not None

    def _geo_veto(self, window_id: int, person_id: int, start: int, end: int):
        """`_geo_cached` 的 5 元组包装 —— `_before_score` / 审计用的旧口径。"""
        return self._geo5(self._geo_cached(window_id, person_id, start, end))

    def _members_of(self, person_id: int) -> list[int]:
        if not self.cfg.stitch:
            return [int(person_id)]
        try:
            return sorted(int(t) for t in self.stitcher.members_of(int(person_id)))
        except Exception: # noqa: BLE001 — 只用于展示
            return [int(person_id)]

    def bbox_of_person(self, person_id: int):
        """该 person 的最近 bbox（拼接链多成员时取面积最大的一个）。"""
        best, best_area = None, -1.0
        for tid in self._members_of(int(person_id)):
            box = self.tracker.bbox_of(tid)
            if box is None:
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if area > best_area:
                best, best_area = box, area
        return best

    def display_persons(self, max_age_frames: int = 2) -> list[dict]:
        """渲染用只读快照：每个此刻还看得见的人一条。"""
        if self._frame <= 0:
            return []
        now = self._frame - 1
 # track → person 反查（拼接链多成员时一个 person 对应多条 track）
        owner: dict[int, int] = {}
        for pid in self.events_book.person_ids():
            for tid in self._members_of(pid):
                owner[int(tid)] = int(pid)

        out: list[dict] = []
        for tid in self.ring.track_ids():
            tid = int(tid)
            if now - self.ring.last_frame(tid) > max_age_frames:
                continue # 已经看不见了，不画鬼影
            got = self.ring.latest(tid)
            box = self.tracker.bbox_of(tid)
            if got is None or box is None:
                continue
            kp, sc = got
            pts = np.empty((len(kp), 3), np.float32)
            pts[:, :2] = kp
            pts[:, 2] = sc
            pid = owner.get(tid)
            pw = self._last_pw.get(pid) if pid is not None else None
            out.append({
                "track_id": tid,
                "person_id": pid,
                "bbox": [float(v) for v in box],
                "keypoints": pts,
                "prob": None if pw is None else float(pw.prob_fall),
                "coverage": None if pw is None else float(pw.coverage),
                "status": None if pw is None else str(pw.status),
            })
        return out

    def _chain_alive(self, person_id: int) -> bool:
        """`person_id` 是链根：只要链里还有活着的 track，状态就不能清。"""
        if not self.cfg.stitch:
            return False
        try:
            return any(self.ring.has(int(t))
                       for t in self.stitcher.members_of(int(person_id)))
        except Exception: # noqa: BLE001
            return False

 # ------------------------------------------------------------------ 收尾

    def finish(self, video_id: str) -> MultiVideoResult:
        """收尾。`base.windows` 由父类算 —— 与旧管线逐位相同。"""
        base = super().finish(video_id)
 # 产物里的 `multi_cfg` 要能自证闸门开没开。
 # `as_dict` 本身不许加键（云端 config_hash 契约，见 config.py），
 # 所以这里单独补 —— `MultiConfig(**multi_cfg)` 仍能原样重建。
        mcfg_dump = dict(self.mcfg.as_dict())
        mcfg_dump["pregate"] = int(self.mcfg.pregate)
        mcfg_dump["pregate_sample"] = int(self.mcfg.pregate_sample)
        return MultiVideoResult(
            base=base,
            per_person=list(self._per_person),
            events=list(self._events),
            counters=self.counters,
            multi_cfg=mcfg_dump,
            n_persons=len(self.events_book),
        )

    def run_video(self, video_id: str, frames) -> MultiVideoResult:
        for frame in frames:
            self.push(frame)
        return self.finish(video_id)
