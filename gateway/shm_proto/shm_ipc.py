#!/usr/bin/env python3
"""FallGuard 共享内存 Python 镜像 — 与 edge_gateway/include/fallguard_shm.h 逐字段一致。"""

from ctypes import CDLL


import ctypes
import mmap
import os
import struct
import time
import zlib

FG_SHM_MAGIC = 0x4647414C
FG_SHM_VERSION = 3
FG_EVT_RING_CAP = 32          # v3: 8 → 32（多人场景下 8 条会被覆盖）
FG_JPEG_MAX = 200 * 1024
FG_LIVE_W = 320               # 快照帧默认尺寸（沿用旧直播帧尺寸）
FG_LIVE_H = 240

# 事件状态机（v3 新增；V1 只产生 confirmed / retracted 两态）
FG_EVT_ST_SUSPECTED = 0
FG_EVT_ST_CONFIRMED = 1
FG_EVT_ST_RETRACTED = 2

FG_SHM_SIZE = 414344          # sizeof(fg_shm_t)（v3: 820320 → 414344，删了 live_frame[2]）
FG_EVENT_SIZE = 144           # sizeof(fg_event_t)（v3: 120 → 144）
FG_FRAME_SIZE = 204816        # sizeof(fg_frame_t)

O_RDWR = os.O_RDWR
O_CREAT = os.O_CREAT

# ---------------- libc 封装 ----------------
_libc: CDLL = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
_libc.shm_open.restype = ctypes.c_int
_libc.shm_unlink.argtypes = [ctypes.c_char_p]
_libc.shm_unlink.restype = ctypes.c_int
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]
_libc.ftruncate.restype = ctypes.c_int
_libc.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
_libc.sem_open.restype = ctypes.c_void_p
_libc.sem_wait.argtypes = [ctypes.c_void_p]
_libc.sem_wait.restype = ctypes.c_int
_libc.sem_trywait.argtypes = [ctypes.c_void_p]
_libc.sem_trywait.restype = ctypes.c_int
_libc.sem_post.argtypes = [ctypes.c_void_p]
_libc.sem_post.restype = ctypes.c_int
_libc.sem_close.argtypes = [ctypes.c_void_p]
_libc.sem_close.restype = ctypes.c_int
_libc.sem_unlink.argtypes = [ctypes.c_char_p]
_libc.sem_unlink.restype = ctypes.c_int


def _err(prefix):
    e = ctypes.get_errno()
    return OSError(e, os.strerror(e), prefix)


def _sem_open(name, init):
    sem = _libc.sem_open(name.encode(), O_CREAT | O_RDWR, 0o600, init)
    if not sem:
        raise _err("sem_open " + name)
    return sem


# ---------------- 结构镜像（与 C 头文件逐字段一致，自然对齐） ----------------
class FgEvent(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("ts_sec", ctypes.c_int64),
        ("ts_ms", ctypes.c_int32),
        ("person_id", ctypes.c_int32),
        ("class_id", ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("bbox", ctypes.c_float * 4),
        ("hybrid_score", ctypes.c_float),
        ("reason", ctypes.c_char * 64),
        ("snapshot_seq", ctypes.c_uint32),
        # ---- v3 新增 ----
        ("state", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("window_start", ctypes.c_uint32),
        ("window_end", ctypes.c_uint32),
        ("prob_fall", ctypes.c_float),
        ("coverage", ctypes.c_float),
        ("model_ver", ctypes.c_uint32),
    ]


class FgFrame(ctypes.Structure):
    _fields_ = [
        ("frame_seq", ctypes.c_uint32),
        ("w", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("jpeg_len", ctypes.c_uint32),
        ("jpeg", ctypes.c_uint8 * FG_JPEG_MAX),
    ]


class FgShm(ctypes.Structure):
    _fields_ = [
        # 固定头部
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("shm_size", ctypes.c_uint32),   # v3 新增：读端第三道校验
        ("sensor_pid", ctypes.c_int32),
        ("gateway_pid", ctypes.c_int32),
        ("sensor_uptime", ctypes.c_uint64),
        # 配置区
        ("cfg_seq", ctypes.c_uint32),
        ("sensitivity_level", ctypes.c_int32),
        ("push_threshold", ctypes.c_float),
        ("push_cooldown_ms", ctypes.c_int32),
        ("push_enabled", ctypes.c_int32),
        ("cfg_crc32", ctypes.c_uint32),
        # 状态区
        ("status_seq", ctypes.c_uint64),
        ("is_falling", ctypes.c_int32),
        ("people_count", ctypes.c_int32),
        ("fps", ctypes.c_float),
        ("frame_count", ctypes.c_uint64),
        ("last_evt_seq", ctypes.c_uint32),
        # 事件环形队列
        ("evt_head", ctypes.c_uint32),
        ("evt_tail", ctypes.c_uint32),
        ("evt_count", ctypes.c_uint32),
        ("events", FgEvent * FG_EVT_RING_CAP),
        # 告警快照双缓冲（v3 起直播帧不再进 shm）
        ("alarm_snap", FgFrame * 2),
    ]


def _check_layout():
    assert ctypes.sizeof(FgEvent) == FG_EVENT_SIZE, \
        "FgEvent size %d != %d" % (ctypes.sizeof(FgEvent), FG_EVENT_SIZE)
    assert ctypes.sizeof(FgFrame) == FG_FRAME_SIZE, \
        "FgFrame size %d != %d" % (ctypes.sizeof(FgFrame), FG_FRAME_SIZE)
    assert ctypes.sizeof(FgShm) == FG_SHM_SIZE, \
        "FgShm size %d != %d" % (ctypes.sizeof(FgShm), FG_SHM_SIZE)
    assert FgEvent.bbox.offset == 32, FgEvent.bbox.offset
    assert FgEvent.reason.offset == 52, FgEvent.reason.offset
    assert FgEvent.snapshot_seq.offset == 116, FgEvent.snapshot_seq.offset
    # v3 新增字段
    assert FgEvent.state.offset == 120, FgEvent.state.offset
    assert FgEvent.window_start.offset == 124, FgEvent.window_start.offset
    assert FgEvent.model_ver.offset == 140, FgEvent.model_ver.offset
    assert FgShm.shm_size.offset == 8, FgShm.shm_size.offset
    assert FgShm.events.offset == 104, FgShm.events.offset
    assert FgShm.alarm_snap.offset == 4712, FgShm.alarm_snap.offset
    print("[layout] FgEvent=%d FgFrame=%d FgShm=%d 与 C 结构一致"
          % (FG_EVENT_SIZE, FG_FRAME_SIZE, FG_SHM_SIZE))


_check_layout()


# ---------------- FallGuardShm ----------------
class FallGuardShm:
    """事件源写侧 / 诊断读侧的统一封装。"""

    SHM_DEFAULT = "/fallguard_shm"
    SEM_MUTEX = "/fg_shm_mutex"
    SEM_EVT = "/fg_evt_sem"
    SEM_LIVE = "/fg_live_sem"

    def __init__(self, name=SHM_DEFAULT, create=True, mode=0o600):
        self.name = name
        self._mmap = None
        self._sem_mutex = None
        self._sem_evt = None
        self._sem_live = None
        oflag = O_RDWR | (O_CREAT if create else 0)
        fd = _libc.shm_open(name.encode(), oflag, mode)
        if fd < 0:
            raise _err("shm_open " + name)
        try:
            # 必须先查尺寸再 ftruncate：v3 比 v2 小 400 KB，
            # 直接 ftruncate 会把一个仍在被 v2 网关映射的 shm 截断。
            # 0 字节是正常情况：`shm_open(O_CREAT)` 刚建出来的对象
            # 长度就是 0，下面 ftruncate 才给它尺寸。别把它当"旧版本残留"。
            st = os.fstat(fd)
            if st.st_size not in (0, FG_SHM_SIZE):
                if not create:
                    os.close(fd)
                    raise RuntimeError(
                        "shm %s 尺寸 %d != v%d 期望 %d —— 旧版本残留或版本不匹配。"
                        "先 `rm /dev/shm/fallguard_shm` 并同时重启 sensor 与 gateway。"
                        % (name, st.st_size, FG_SHM_VERSION, FG_SHM_SIZE))
                # create=True ⇒ 我是写侧，有权重建。先解除旧的再重开。
                print("[shm] 发现旧尺寸 %d B（期望 %d B），重建 %s"
                      % (st.st_size, FG_SHM_SIZE, name), flush=True)
                os.close(fd)
                _libc.shm_unlink(name.encode())
                fd = _libc.shm_open(name.encode(), oflag, mode)
                if fd < 0:
                    raise _err("shm_open(重建) " + name)
            if _libc.ftruncate(fd, FG_SHM_SIZE) != 0:
                raise _err("ftruncate")
            self._mmap = mmap.mmap(fd, FG_SHM_SIZE, access=mmap.ACCESS_WRITE)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        self._shm = FgShm.from_buffer(self._mmap)
        self._sem_mutex = _sem_open(self.SEM_MUTEX, 1)
        self._sem_evt = _sem_open(self.SEM_EVT, 0)
        self._sem_live = _sem_open(self.SEM_LIVE, 0)
        self._snap_ix = 0
        self._live_ix = 0
        self._frame_seq = 0
        if create and self._shm.magic != FG_SHM_MAGIC:
            self._init_shm()
        elif not create and self._shm.magic != FG_SHM_MAGIC:
            raise RuntimeError(
                "%s 未初始化（magic=0x%08X），先运行 mock_fall_generator.py" % (name, self._shm.magic))
        if self._shm.magic == FG_SHM_MAGIC:
            if self._shm.version != FG_SHM_VERSION:
                raise RuntimeError(
                    "shm %s 版本 %d != 本端 %d —— 新旧版本不能共存，"
                    "请同时重启 sensor 与 gateway 并 `rm /dev/shm/fallguard_shm`"
                    % (name, self._shm.version, FG_SHM_VERSION))
            if self._shm.shm_size != FG_SHM_SIZE:
                raise RuntimeError(
                    "shm %s 记录尺寸 %d != 本端 %d —— 结构漂移，拒绝继续"
                    % (name, self._shm.shm_size, FG_SHM_SIZE))

    # ---- 初始化 / 配置 ----
    def _init_shm(self):
        shm = self._shm
        ctypes.memset(ctypes.addressof(shm), 0, ctypes.sizeof(shm))
        shm.magic = FG_SHM_MAGIC
        shm.version = FG_SHM_VERSION
        shm.shm_size = FG_SHM_SIZE      # v3：读端第三道校验的依据
        shm.sensor_pid = os.getpid()
        shm.gateway_pid = 0
        shm.sensor_uptime = 0
        # 配置区默认（medium）
        shm.cfg_seq = 1
        shm.sensitivity_level = 1
        shm.push_threshold = 0.60
        shm.push_cooldown_ms = 3000
        shm.push_enabled = 1
        shm.cfg_crc32 = self._config_crc32()

    def _config_crc32(self):
        shm = self._shm
        payload = struct.pack(">iifii", shm.cfg_seq, shm.sensitivity_level,
                              shm.push_threshold, shm.push_cooldown_ms, shm.push_enabled)
        return zlib.crc32(payload) & 0xFFFFFFFF

    def read_config(self):
        shm = self._shm
        return {
            "cfg_seq": shm.cfg_seq,
            "sensitivity_level": shm.sensitivity_level,
            "push_threshold": shm.push_threshold,
            "push_cooldown_ms": shm.push_cooldown_ms,
            "push_enabled": shm.push_enabled,
            "cfg_crc32": shm.cfg_crc32,
        }

    # ---- 状态区 ----
    def update_status(self, is_falling, people, fps):
        shm = self._shm
        shm.is_falling = int(is_falling)
        shm.people_count = int(people)
        shm.fps = float(fps)
        shm.frame_count += 1
        shm.sensor_uptime = int(time.monotonic())
        shm.status_seq += 1          # 最后发布，读者以此判新

    # ---- 事件 + 快照 ----
    def push_event(self, evt, snapshot_jpeg=None, snap_wh=(FG_LIVE_W, FG_LIVE_H)):
        """写一条事件；snapshot_jpeg 非空则先写告警快照双缓冲，并把其 frame_seq 挂到事件。"""
        shm = self._shm
        _libc.sem_wait(self._sem_mutex)
        try:
            snap_seq = 0
            if snapshot_jpeg is not None:
                snap_seq = self._write_frame(shm.alarm_snap, self._snap_ix, snapshot_jpeg, snap_wh)
                self._snap_ix ^= 1
            e = shm.events[shm.evt_tail]
            self._fill_event(e, evt, snap_seq)
            if shm.evt_count == FG_EVT_RING_CAP:
                shm.evt_head = (shm.evt_head + 1) % FG_EVT_RING_CAP   # 队列满则覆盖最旧
            else:
                shm.evt_count += 1
            shm.evt_tail = (shm.evt_tail + 1) % FG_EVT_RING_CAP
            shm.last_evt_seq = e.seq
            shm.status_seq += 1
            shm.sensor_uptime = int(time.monotonic())
        finally:
            _libc.sem_post(self._sem_mutex)
        _libc.sem_post(self._sem_evt)
        return e.seq

    def _fill_event(self, e, evt, snap_seq):
        if "seq" in evt:
            e.seq = int(evt["seq"])
        else:
            e.seq = self._shm.last_evt_seq + 1
        if "ts_sec" in evt and "ts_ms" in evt:
            e.ts_sec = int(evt["ts_sec"])
            e.ts_ms = int(evt["ts_ms"])
        else:
            t = time.time()
            e.ts_sec = int(t)
            e.ts_ms = int((t - int(t)) * 1000)
        e.person_id = int(evt["person_id"])
        e.class_id = int(evt["class_id"])
        e.confidence = float(evt.get("confidence", 1.0))
        b = evt["bbox"]
        for i in range(4):
            e.bbox[i] = float(b[i])
        e.hybrid_score = float(evt.get("hybrid_score", 1.0))
        e.reason = str(evt.get("reason", "")).encode("utf-8")[:63]   # 自动 null 填充
        e.snapshot_seq = snap_seq
        # ---- v3 新增 ----
        # 默认 confirmed：V1 只做两态，调用方不显式传就是「确认的跌倒」
        e.state = int(evt.get("state", FG_EVT_ST_CONFIRMED)) & 0xFF
        e.window_start = int(evt.get("window_start", 0)) & 0xFFFFFFFF
        e.window_end = int(evt.get("window_end", 0)) & 0xFFFFFFFF
        e.prob_fall = float(evt.get("prob_fall", 0.0))
        e.coverage = float(evt.get("coverage", 0.0))
        e.model_ver = int(evt.get("model_ver", 0)) & 0xFFFFFFFF

    def _write_frame(self, frames, slot, jpeg, wh):
        fr = frames[slot]
        jlen = len(jpeg)
        assert jlen <= FG_JPEG_MAX, "JPEG %d > %d" % (jlen, FG_JPEG_MAX)
        self._frame_seq += 1
        new_seq = self._frame_seq
        ctypes.memmove(ctypes.addressof(fr.jpeg), jpeg, jlen)
        fr.jpeg_len = jlen
        fr.w, fr.h = int(wh[0]), int(wh[1])
        fr.frame_seq = new_seq          # 写全后再发布
        return new_seq

    # ---- 读侧 ----
    def peek_events(self):
        """只读当前事件环，不消费。"""
        shm = self._shm
        out = []
        _libc.sem_wait(self._sem_mutex)
        try:
            head, count = shm.evt_head, shm.evt_count
            for i in range(count):
                e = shm.events[(head + i) % FG_EVT_RING_CAP]
                out.append(self._event_to_dict(e))
        finally:
            _libc.sem_post(self._sem_mutex)
        return out

    def take_event(self):
        """消费一条事件（网关 T1 用）。无事件返回 None。"""
        shm = self._shm
        _libc.sem_wait(self._sem_mutex)
        try:
            if shm.evt_count == 0:
                return None
            e = shm.events[shm.evt_head]
            shm.evt_head = (shm.evt_head + 1) % FG_EVT_RING_CAP
            shm.evt_count -= 1
            shm.status_seq += 1
            return self._event_to_dict(e)
        finally:
            _libc.sem_post(self._sem_mutex)

    def pick_latest_frame(self, which="snap"):
        """取最新快照。v3 起只剩告警快照（live_frame 已从 shm 删除）。"""
        if which != "snap":
            raise ValueError("v3 起 shm 只存告警快照，which=%r 无对应区域" % (which,))
        frames = self._shm.alarm_snap
        fr = frames[0] if frames[0].frame_seq >= frames[1].frame_seq else frames[1]
        if fr.frame_seq == 0 or fr.jpeg_len == 0:
            return None
        return {
            "frame_seq": fr.frame_seq, "w": fr.w, "h": fr.h,
            "jpeg_len": fr.jpeg_len,
            "jpeg": ctypes.string_at(ctypes.addressof(fr.jpeg), fr.jpeg_len),
        }

    @staticmethod
    def _event_to_dict(e):
        return {
            "seq": e.seq, "ts_sec": e.ts_sec, "ts_ms": e.ts_ms,
            "person_id": e.person_id, "class_id": e.class_id,
            "confidence": e.confidence,
            "bbox": [e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]],
            "hybrid_score": e.hybrid_score,
            "reason": FallGuardShm._reason_str(e.reason),
            "snapshot_seq": e.snapshot_seq,
            # ---- v3 新增 ----
            "state": e.state,
            "window_start": e.window_start,
            "window_end": e.window_end,
            "prob_fall": e.prob_fall,
            "coverage": e.coverage,
            "model_ver": e.model_ver,
        }

    @staticmethod
    def _reason_str(field):
        """兼容两种 ctypes 行为：c_char*N 字段 getter 可能返回数组对象或 bytes。"""
        if not isinstance(field, bytes):
            field = field.value
        return field.decode("utf-8", "replace")

    # ---- 诊断 / 清理 ----
    def dump(self):
        shm = self._shm
        if shm.magic != FG_SHM_MAGIC:
            return "shm %s: 未初始化（magic=0x%08X）" % (self.name, shm.magic)
        L = []
        L.append("== FallGuardShm %s ==" % self.name)
        L.append("[header] magic=0x%08X v=%d size=%d sensor_pid=%d gateway_pid=%d uptime=%ds"
                 % (shm.magic, shm.version, shm.shm_size, shm.sensor_pid,
                    shm.gateway_pid, shm.sensor_uptime))
        L.append("[config] cfg_seq=%d sensitivity=%d push_threshold=%.2f cooldown=%d enabled=%d crc32=0x%08X"
                 % (shm.cfg_seq, shm.sensitivity_level, shm.push_threshold,
                    shm.push_cooldown_ms, shm.push_enabled, shm.cfg_crc32))
        L.append("[status] status_seq=%d is_falling=%d people=%d fps=%.1f frame_count=%d last_evt_seq=%d"
                 % (shm.status_seq, shm.is_falling, shm.people_count, shm.fps,
                    shm.frame_count, shm.last_evt_seq))
        L.append("[ring] head=%d tail=%d count=%d" % (shm.evt_head, shm.evt_tail, shm.evt_count))
        for i, ev in enumerate(self.peek_events()):
            L.append("  evt[%d] seq=%d ts=%d.%03d pid=%d cls=%d conf=%.2f bbox=%s hybrid=%.2f "
                     "reason=%s snap=%d state=%d win=%d..%d pf=%.3f cov=%.2f mv=%d"
                     % (i, ev["seq"], ev["ts_sec"], ev["ts_ms"], ev["person_id"], ev["class_id"],
                        ev["confidence"], ev["bbox"], ev["hybrid_score"], ev["reason"],
                        ev["snapshot_seq"], ev["state"], ev["window_start"], ev["window_end"],
                        ev["prob_fall"], ev["coverage"], ev["model_ver"]))
        for slot in range(2):
            fr = shm.alarm_snap[slot]
            L.append("[alarm_snap][%d] seq=%d %dx%d jpeg_len=%d"
                     % (slot, fr.frame_seq, fr.w, fr.h, fr.jpeg_len))
        return "\n".join(L)

    def close(self):
        self._shm = None   # 先释放本对象对 mmap 的 ctypes buffer 引用
        for sem in (self._sem_mutex, self._sem_evt, self._sem_live):
            if sem:
                _libc.sem_close(sem)
        self._sem_mutex = self._sem_evt = self._sem_live = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass   # 仍有调用方别名持有 ctypes 视图，交由进程退出/GC 释放
            self._mmap = None

    def cleanup(self):
        """解除映射并删除共享内存与信号量（仅调用方确认无其他持有者时用）。"""
        for n in (self.SEM_MUTEX, self.SEM_EVT, self.SEM_LIVE):
            _libc.sem_unlink(n.encode())
        _libc.shm_unlink(self.name.encode())
        self.close()
