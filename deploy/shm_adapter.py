#!/usr/bin/env python3
"""shm_adapter.py — 感知侧共享内存薄适配器。
把真实跌倒感知（class_display_demo.py，RKNN）接入 FallGuard 共享内存契约，
替换 mock 作为事件源。C 网关 / 契约 / 感知核心四文件一行不改。"""

import os
import sys
import time

# ---- 定位 shm_ipc.py（契约唯一事实来源，不复制文件）----
_CANDIDATE_DIRS = [
    os.environ.get("FALLGUARD_SHM_DIR", ""),
    # Windows 工程树：rk3576_deployment/ 的兄弟目录
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "产品化设计_告警闭环系统", "fallguard", "shm_proto"),
    # 板端同根部署（fallguard/ 与 rk3576_deployment/ 同目录）
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "fallguard", "shm_proto"),
    "/opt/fallguard/gateway/shm_proto",       # 板上已确认路径
]

_SHM_MODULE = None


def _ensure_shm_imported(shm_dir=None):
    """把含 shm_ipc.py 的目录加入 sys.path；找不到抛 RuntimeError。"""
    global _SHM_MODULE
    if _SHM_MODULE is not None:
        return _SHM_MODULE
    dirs = []
    if shm_dir:
        dirs.append(shm_dir)
    dirs += [d for d in _CANDIDATE_DIRS if d]
    for d in dirs:
        candidate = os.path.join(d, "shm_ipc.py")
        if os.path.isfile(candidate):
            if d not in sys.path:
                sys.path.insert(0, d)
            import shm_ipc          # noqa: E402
            _SHM_MODULE = shm_ipc
            return shm_ipc
    raise RuntimeError(
        "找不到 shm_ipc.py。请用 --shm-path 指定 fallguard/shm_proto 目录，"
        "或设环境变量 FALLGUARD_SHM_DIR。已尝试: %s" % dirs)


class SensorShmWriter:
    """感知侧 → fg_shm 共享内存 薄写适配器。"""

    LIVE_JPEG_W = 320           # 告警快照尺寸（沿用旧直播帧尺寸命名，勿改）
    LIVE_JPEG_H = 240
    CONFIG_RELOAD_FRAMES = 30   # 每 N 帧轮询一次 cfg_seq（判变才重读）

    ST_SUSPECTED = 0
    ST_CONFIRMED = 1
    ST_RETRACTED = 2

    # ── 闸门：reason 白名单（ 用户拍板；原为 push_threshold 分数闸门）──
    # 依据 `push-gate-threshold-analysis`：hybrid_score 与上游 decision 口径不一致，
    # 真机 5 次复现「上游已判 FALL、闸门按分数拒」（真摔 0 条送达）⇒ 决策只做一次，
    # 闸门只看 reason，不复查分数。五个值 = `src/hybrid_classifier.py`
    # `_fusion_decision()` 全部 FALL 分支的 reason（逐字核对，勿凭印象增删）。
    GATE_REASON_WHITELIST = (
        "geo_only",
        "dual_confirm",
        "geo_strong_ml_weak", # 补：全量核对 _fusion_decision 时发现的遗漏
        "ml_rescue_c5_lag", # 补：同上（旧白名单只核了 5 个，实际共 7 个）
        "ml_rescue",
        "ml_temporal_confirm",
        "hybrid_threshold",
    )

    def __init__(self, shm_dir=None):
        shm_ipc = _ensure_shm_imported(shm_dir)
        self._shm = shm_ipc.FallGuardShm(create=True)   # 复用已有 shm，magic 异常会初始化
        self._shm._shm.sensor_pid = os.getpid()
        # 帧序号基线延续：接管时以 shm 现有最大 告警快照 序号为起点，
        # 保证「frame_seq 单调递增」契约跨感知进程重启仍成立
        # （否则重启后序号回退到 1，读者会把旧残帧当新帧）
        # v3 起 live_frame 已从 shm 删除，基线只能看 alarm_snap。
        self._shm._frame_seq = max(
            (fr.frame_seq for fr in self._shm._shm.alarm_snap), default=0)
        self._cfg = self._shm.read_config()
        self._cfg_seq = self._cfg["cfg_seq"]
        self._last_push_ms = 0.0
        self._frame_counter = 0
        self._pushed_seq = None
        print("[shm] 已接入共享内存 %s  cfg_seq=%d sensitivity=%d "
              "threshold=%.2f cooldown=%dms enabled=%d"
              % (self._shm.name, self._cfg["cfg_seq"],
                 self._cfg["sensitivity_level"], self._cfg["push_threshold"],
                 self._cfg["push_cooldown_ms"], self._cfg["push_enabled"]),
              flush=True)

    # ---- 每帧状态 ----
    def update_status(self, is_falling, people, fps):
        self._shm.update_status(is_falling, people, fps)
        self._frame_counter += 1

    # ---- 配置热更新（网关 T4 下推灵敏度等；内部按帧数节流）----
    def maybe_reload_config(self):
        if self._frame_counter % self.CONFIG_RELOAD_FRAMES != 0:
            return False
        cfg = self._shm.read_config()
        if cfg["cfg_seq"] != self._cfg_seq:
            self._cfg = cfg
            self._cfg_seq = cfg["cfg_seq"]
            print("[shm] 配置变更已生效 cfg_seq=%d sensitivity=%d "
                  "threshold=%.2f cooldown=%dms enabled=%d"
                  % (cfg["cfg_seq"], cfg["sensitivity_level"],
                     cfg["push_threshold"], cfg["push_cooldown_ms"],
                     cfg["push_enabled"]), flush=True)
            return True
        return False

    # ---- 跌倒事件（闸门 + 字段映射 + 整帧快照）----
    def push_fall_event(self, event, person, hybrid, frame,
                        skip_cooldown=False):
        """event: hybrid.update() 返回的事件 dict；person: 当前检测人 dict。"""
        cfg = self._cfg
        # 闸门拒绝必须留痕：这三个 return None 原来是静默的，导致
        # 「板上打印了 FALL!、护士站却什么都没收到」时无法区分
        # 「没检出」/「被闸门拒」/「网络问题」。宁可噪声也不要静默。
        if not cfg["push_enabled"]:
            print("[shm] DROP reason=disabled  (push_enabled=0，网关侧关掉了推送)",
                  flush=True)
            return None
        feats = event.get("features", {})
        if "hybrid_score" not in feats:
            # 纯几何模式（--no-ml）的事件没有 features.hybrid_score —— 单独点出来。
            # （闸门已不看分数，仅上报字段退回 1.0，不影响放行）
            print("[shm] WARN 事件里没有 features.hybrid_score（--no-ml 纯几何路径？）"
                  " ⇒ 上报分数退回 1.0", flush=True)
        score = float(feats.get("hybrid_score", 1.0))
        # ── 闸门 = reason 白名单（ 起，原为 push_threshold 分数闸门）──
        # 上游 `_fusion_decision` 只做一次决策；旧分数闸门会把「上游判 FALL、
        # 分数 < 阈值」的事件全部拒掉（真机 5 次复现、真摔 0 条送达）⇒ 不再复查分数。
        reason = str(event.get("reason", "fall"))
        state = int(event.get("state", self.ST_CONFIRMED))
        is_retract = (state == self.ST_RETRACTED)
        if not is_retract and reason not in self.GATE_REASON_WHITELIST:
            print("[shm] DROP reason=not_whitelisted reason_str=%s score=%.3f"
                  % (reason, score), flush=True)
            return None
        now_ms = time.time() * 1000.0
        if not is_retract and not skip_cooldown:
            if now_ms - self._last_push_ms < cfg["push_cooldown_ms"]:
                print("[shm] DROP reason=cooldown dt=%.0fms need=%dms score=%.3f"
                      % (now_ms - self._last_push_ms,
                         cfg["push_cooldown_ms"], score), flush=True)
                return None
            self._last_push_ms = now_ms   # 撤回不占用冷却窗口

        # ---- 组装契约事件（evt dict → fg_event_t 字段）----
        ml = getattr(hybrid, "_last_ml_result", None)
        conf = float(ml["confidence"]) if ml and ml.get("confidence") else 1.0
        bbox = list(person.get("bbox", [0, 0, 0, 0]))
        evt = {
            "person_id": int(person.get("person_id", 1)),
            "class_id": 5,              # FALL（event 本身就是 FALL 确认事件）
            "confidence": conf,
            "bbox": bbox,
            "hybrid_score": score,
            "reason": str(event.get("reason", "fall")),
            # ---- v3 新增字段（幂等键 + 可追溯性）----
            # 这些字段必须显式透传：漏填不会报错，只会静默写 0，
            # 然后网关侧按 (person_id, window_start, model_ver) 去重时
            # 全部撞成同一个键 ⇒ 整个人的告警被合并成一条。
            # 第一版就漏了，靠 verify_shm_v3 的字段断言抓到。
            "state": state,
            "window_start": int(feats.get("window_start",
                                          event.get("window_start", 0))),
            "window_end": int(feats.get("window_end",
                                        event.get("window_end", 0))),
            "prob_fall": float(feats.get("prob_fall",
                                         event.get("prob_fall", 0.0))),
            "coverage": float(feats.get("coverage",
                                        event.get("coverage", 0.0))),
            "model_ver": int(event.get("model_ver", 0)),
        }
        # 告警快照：整帧缩放 320×240 JPEG（<200KB 上限）
        snap = self._frame_jpeg(frame)
        try:
            self._pushed_seq = self._shm.push_event(
                evt, snap, snap_wh=(self.LIVE_JPEG_W, self.LIVE_JPEG_H))
        except Exception as e:
            print("[shm] push_event 失败: %s" % e, flush=True)
            return None
        # wall_t = 板端墙钟（epoch 秒）。跨机算端到端时延时必须先扣掉板↔PC 时钟偏移，
        # 用 remote_tools/e2e_alarm_check.py --measure 量。
        print("[shm] PUSH seq=%d person#%d fall hybrid=%.2f conf=%.2f "
              "reason=%s bbox=%s wall_t=%.3f" % (self._pushed_seq, evt["person_id"],
                                                 score, conf, evt["reason"],
                                                 [round(float(v), 1) for v in bbox],
                                                 now_ms / 1000.0),
              flush=True)
        return self._pushed_seq

    # ---- 直播帧（节流 ≥1s）----
    def push_live_frame(self, frame):
        """v3 起空操作，恒返回 None。"""
        return None

    def _frame_jpeg(self, frame):
        try:
            import cv2
            if frame is None:
                return None
            small = cv2.resize(frame, (self.LIVE_JPEG_W, self.LIVE_JPEG_H))
            ok, buf = cv2.imencode(".jpg", small,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def read_config(self):
        return dict(self._cfg)

    def close(self):
        try:
            self._shm.close()   # 只解映射/关 sem，不 unlink（网关可能还在用）
        except Exception as e:
            print("[shm] close: %s" % e, flush=True)
