"""D-042 ID-handoff chaining, incrementally."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import spec


@dataclass
class _TrackGeom:
    first: int
    last: int
 # frame -> (centre_xy, diagonal)
    frames: dict[int, tuple[tuple[float, float], float]] = field(default_factory=dict)


def _centre_diag(box: np.ndarray) -> tuple[tuple[float, float], float]:
    b = np.asarray(box, dtype=np.float64)
    return (((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0),
            float(np.hypot(b[2] - b[0], b[3] - b[1])))


class Stitcher:
    """Incremental D-042 chaining."""

 # 桥接候选的排序惩罚：比任何真实共享帧距离都大，保证真 handoff 优先。
    BRIDGE_PENALTY = 1e6

    def __init__(self, gap_max: int = spec.STITCH_GAP_MAX,
                 overlap_max: int = spec.STITCH_OVERLAP_MAX,
                 dist_factor: float = spec.STITCH_DIST_FACTOR,
                 track_max_age: int = spec.MAX_AGE,
                 allow_gap_bridge: bool = False) -> None:
        self.gap_max = int(gap_max)
        self.overlap_max = int(overlap_max)
        self.dist_factor = float(dist_factor)
        self.track_max_age = int(track_max_age)
        self.allow_gap_bridge = bool(allow_gap_bridge)
 # 冻结视界必须 >= tracker 的 `max_age`。
 # 用 `gap_max`（5）冻结会造出「同一个 track id 同时属于冻结链和活链」：
 # tracker 要到 `max_age`（30）才丢轨迹，所以在 `last+6 … last+30` 这
 # 25 帧里它仍能把检测配回已被冻结的轨迹 ⇒ `observe` 重建
 # `_geom[t]` ⇒ `_resolve` 把同一 id 放进两条链 ⇒ `_collect` 索引到
 # 已 pop 的成员 ⇒ `KeyError`（ 实测：`clip_0831_174016` 在
 # `max_age=30` 下第 240 帧崩）。冻结视界对齐 `max_age` 之后，
 # 「成员已被 tracker 丢弃」与「几何已被 pop」才等价。
        self.freeze_after = max(self.gap_max, self.track_max_age)
        self._geom: dict[int, _TrackGeom] = {}
        self._chains: list[list[int]] = []
 # track_id -> index into `_chains`, once the track has been placed.
        self._chain_of: dict[int, int] = {}

 # ---- 长期运行：淘汰（见模块 docstring 最后一节）----
 # 已冻结的链：(链头 first, 链头 last, 成员表)。链头那两个数只用于
 # 复现 `_resolve` 原来的链顺序；几何全部丢掉。
        self._frozen: list[tuple[int, int, list[int]]] = []
 # 见过的最大 frame_index。`observe` 是唯一的时间来源。
        self._now: int = -1
        self._last_collect: int = -(1 << 60)
 # 多久跑一次淘汰。冻结不需要及时 —— 多留几十帧的几何是几十 KB，
 # 而 `_resolve` 要遍历全部冻结链，跑太勤就是白烧 CPU。
        self._collect_interval = max(self.freeze_after, 60)
 # 只增计数，给探针/日志看；不参与任何判据。
        self.stats: dict[str, int] = {"frames_trimmed": 0,
                                      "tracks_frozen": 0,
                                      "collects": 0}

    def reset(self) -> None:
        self.__init__(self.gap_max, self.overlap_max, self.dist_factor,
                      self.track_max_age, self.allow_gap_bridge)

 # ------------------------------------------------------------------ writes

    def observe(self, track_id: int, frame_index: int, bbox: np.ndarray) -> None:
        """Record that `track_id` was seen at `frame_index` with this box."""
        self._advance(frame_index)
        g = self._geom.get(track_id)
        if g is None:
            g = _TrackGeom(first=frame_index, last=frame_index)
            self._geom[track_id] = g
        g.last = max(g.last, frame_index)
        g.frames[frame_index] = _centre_diag(bbox)
        self._trim(g)

 # ------------------------------------------------------------------ chains

    def _handoff_distance(self, a: int, b: int) -> float | None:
        """Worst centre distance over the shared frames, or None if not a handoff."""
        fa, fb = self._geom[a].frames, self._geom[b].frames
        shared = sorted(set(fa) & set(fb))
        if not shared:
            if not self.allow_gap_bridge:
                return None
            ga, gb = self._geom[a], self._geom[b]
            gap = gb.first - ga.last
            if gap < 1 or gap > self.gap_max:
                return None
 # `_trim` 只保留头/尾 overlap_max+1 帧 ⇒ 这两个边界帧一定还在。
            if ga.last not in fa or gb.first not in fb:
                return None
            (ca, da), (cb, db) = fa[ga.last], fb[gb.first]
            dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
            if dist > self.dist_factor * (da + db):
                return None
            return dist + self.BRIDGE_PENALTY
        if len(shared) > self.overlap_max:
            return None
        worst = 0.0
        for t in shared:
            (ca, da), (cb, db) = fa[t], fb[t]
            dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
            if dist > self.dist_factor * (da + db):
                return None
            worst = max(worst, dist)
        return worst

    def _resolve(self) -> None:
        """Rebuild the chain list over every live track, plus the frozen ones."""
 # 只对「还没冻结」的 track 做两两比较 —— 冻结链的链尾已经死透
 # （gap > gap_max），本来就会被跳过，所以把它们挪出内循环不改变结果，
 # 但把 `_resolve` 从 O(链总数²) 拉回 O(在场上限²)。
 # 冻结链的成员表直接引用（只读），不每次复制 —— 链多时这是主要开销之一。
 # 任何会返回给调用方的路径都自己复制（`chains_so_far` / `members_of`）。
        entries: list[tuple[int, int, list[int]]] = list(self._frozen)
        live: list[list[int]] = []

        for b in sorted(self._geom, key=lambda k: (self._geom[k].first,
                                                   self._geom[k].last)):
            gb = self._geom[b]
            best: tuple[float, int] | None = None
            for ci, chain in enumerate(live):
                a = chain[-1]
                gap = gb.first - self._geom[a].last
                if gap < -self.overlap_max or gap > self.gap_max:
                    continue
                dist = self._handoff_distance(a, b)
                if dist is None:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, ci)
            if best is None:
                live.append([b])
            else:
                live[best[1]].append(b)

 # 原实现在遍历中「谁先建链谁在前」，而遍历是按链头的 (first,last) 升序，
 # 所以按同一个键排序就能复现完全相同的顺序（含冻结链）。
        for chain in live:
            head = self._geom[chain[0]]
            entries.append((head.first, head.last, chain))
        entries.sort(key=lambda e: (e[0], e[1]))

        self._chains = [e[2] for e in entries]
        self._chain_of = {}
        for i, chain in enumerate(self._chains):
            for t in chain:
                self._chain_of[t] = i

 # -------------------------------------------------------------- 淘汰/裁剪

    def _advance(self, frame_index: int) -> None:
        """推进时钟，并按 `_collect_interval` 节流触发一次淘汰。"""
        if frame_index <= self._now:
            return
        self._now = frame_index
        if frame_index - self._last_collect < self._collect_interval:
            return
        self._last_collect = frame_index
        self._collect()

    def _collect(self) -> None:
        """冻结「再也长不出新枝」的链，并丢掉它们的几何。"""
        if not self._geom:
            return
        self.stats["collects"] += 1
        self._resolve()
        n = 0
        for chain in self._chains:
 # 查全部成员，不只链头。冻结视界对齐 `max_age` 之后这条守卫
 # 永不该命中（冻结链的成员全都不在 `_geom`）；留着是纵深防御 ——
 # 万一不变式再被破坏，跳过而不是让下一行 `self._geom[t]` 抛 KeyError。
            if any(t not in self._geom for t in chain):
                continue
            newest = max(self._geom[t].last for t in chain)
            if self._now - newest <= self.freeze_after:
                continue
            head = self._geom[chain[0]]
            self._frozen.append((head.first, head.last, list(chain)))
            for t in chain:
                self._geom.pop(t, None)
            n += 1
        if n:
            self.stats["tracks_frozen"] += n
 # `_chains` 里还留着刚被删掉几何的成员，作废等下次读时重建。
            self._chains = []
            self._chain_of = {}

    def _trim(self, g: _TrackGeom) -> None:
        """只留链判据真正会读的两段：头部 `overlap_max+1` 帧 + 尾部同样多。"""
        if len(g.frames) <= 4 * (self.overlap_max + 1):
            return
        head_hi = g.first + self.overlap_max
        tail_lo = g.last - self.overlap_max
        keep = {f: v for f, v in g.frames.items()
                if f <= head_hi or f >= tail_lo}
        self.stats["frames_trimmed"] += len(g.frames) - len(keep)
        g.frames = keep

    def chains_so_far(self) -> list[list[int]]:
        """The chains as they stand right now. Recomputed on demand."""
        self._resolve()
        return [list(c) for c in self._chains]

    def members_of(self, track_id: int) -> set[int]:
        """All tracks currently believed to be the same body as `track_id`."""
        self._resolve()
        ci = self._chain_of.get(track_id)
        return set(self._chains[ci]) if ci is not None else {track_id}

    def root_of(self, track_id: int) -> int:
        """Smallest id in the chain -- the source project's chain label."""
        m = self.members_of(track_id)
        return min(m) if m else track_id
