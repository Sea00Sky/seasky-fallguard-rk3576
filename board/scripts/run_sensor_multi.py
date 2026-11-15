"""板端实时入口：相机 → per-person 管线 → shm 事件 / web 直播 / 日志。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(os.environ.get("FG_DEPLOY_DIR", "/opt/fallguard/deploy"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
# 必须 append：部署目录里也有一个 `src/` 包，insert(0) 会让 `import src`
# 解析到旧部署的 src（ 首次冒烟就踩了）。
sys.path.append(str(DEPLOY))

from src import PipelineConfig, spec # noqa: E402
from src.backends import StgcnBackend, YoloBackend # noqa: E402
from src.capture import CaptureConfig, open_source # noqa: E402
from src.capture.base import Frame # noqa: E402
from src.multi import MultiConfig, MultiPipeline, State # noqa: E402
from src.multi import overlay # noqa: E402
from run_video import load_rknn # noqa: E402


def probe_video_fps(path: str) -> float | None:
    """读容器 fps；读不到或明显不可信时返回 None。"""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if n <= 0 or not (5.0 <= fps <= 120.0):
        return None
    return fps


class VideoFileSource:
    """录像回放源（验收 / demo 用）。接口与 `FrameSource` 同构。"""

    def __init__(self, path: str, fps: float = 30.0, loop: bool = False) -> None:
        self.path = str(path)
        self.fps = float(fps or 30.0)
        self.loop = bool(loop)
        self.loops = 0
        self.cap = None
        self.seq = 0
        self.t0 = 0.0
        self.dropped_total = 0

    def start(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开录像: {self.path}")
        self.t0 = time.monotonic()

    def read(self, timeout_s=None):
        ok, frame = self.cap.read()
        if not ok and self.loop:
 # cv2 必须在这里 import：`start` 里那次是局部 import，
 # read 的作用域里看不到（ 演示时 NameError 实测踩到）。
            import cv2
 # 回到片头：重开一次而不是 set(CAP_PROP_POS_FRAMES, 0)，
 # 后者在部分 OpenCV 后端上对 mp4 不生效（表现为卡在第一帧）。
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            if self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    self.loops += 1
        if not ok:
            return None
 # seq 跨循环连续递增：它是 ST-GCN 窗口号 window_start 的来源，
 # 重置会让第二轮的窗口号与第一轮撞车，per-person 状态机会误判。
        seq = self.seq
        self.seq += 1
        return Frame(seq=seq, pts_ns=None, bgr=frame)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def stats(self) -> dict:
        return {"backend": "video", "read": self.seq,
                "fps": self.seq / max(1e-9, time.monotonic() - self.t0),
                "seconds": time.monotonic() - self.t0,
                "dropped_total": 0, "gst_dropped": 0,
                "queue_dropped": 0, "gaps": 0, "max_gap_ms": 0.0}


# --------------------------------------------------------------------------- 辅助
def _fps_of(ts: deque) -> float:
    if len(ts) < 2:
        return 0.0
    return (len(ts) - 1) / max(1e-9, ts[-1] - ts[0])


def _model_ver(paths) -> int:
    """模型版本号：两个模型文件的 sha256 前 8 位 → 31 位正整数。"""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return int(h.hexdigest()[:8], 16) & 0x7FFFFFFF


def _to_shm_event(ev, model_ver: int) -> dict:
    """`src.multi.Event` → `shm_adapter.push_fall_event` 认的事件 dict。"""
    confirmed = (ev.state == State.CONFIRMED)
    return {
        "state": int(ev.state.value),
 # confirmed 必须命中 GATE_REASON_WHITELIST；retracted 走 bypass 分支。
        "reason": "ml_temporal_confirm" if confirmed else "retract_low_prob",
        "features": {
            "hybrid_score": float(ev.prob_fall),
            "window_start": int(ev.window_start),
            "window_end": int(ev.window_end),
            "prob_fall": float(ev.prob_fall),
            "coverage": float(ev.coverage),
        },
        "model_ver": int(model_ver),
    }


def _fmt_event(ev, src_seq: int, proc: int, fps: float, wall: float) -> str:
    """事件行。confirmed 逐字复刻旧入口的 `[evt] FALL!` 格式。"""
    tag = "FALL!" if ev.state == State.CONFIRMED else "RETRACT"
    reason = ("ml_temporal_confirm" if ev.state == State.CONFIRMED
              else "retract_low_prob")
    return (f"[evt] {tag} person#{ev.person_id} reason={reason} "
            f"src_frame={src_seq} video_t={proc / fps:.1f}s "
            f"wall_t={wall:.3f} proc_frame={proc}")


# --------------------------------------------------------------------------- 主流程
def _config_hash(cfg) -> str:
    """稳定配置指纹：排序后的 JSON 的 sha256。"""
    blob = json.dumps(cfg.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_status(path: Path, payload: dict) -> None:
    """原子写状态文件：先写 .tmp 再 rename，读方永远读不到半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FallGuard per-person 感知服务（相机 → shm / web / 日志）",
        formatter_class=argparse.RawTextHelpFormatter)
 # 采集（与 run_cam.py 同名同义）
    ap.add_argument("--backend", choices=["gst", "opencv"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--io-mode", type=int, default=None)
    ap.add_argument("--appsink-buffers", type=int, default=None)
    ap.add_argument("--queue-buffers", type=int, default=None)
    ap.add_argument("--raw-file", default=None)
    ap.add_argument("--video", default=None,
                    help="录像回放（验收/demo）；给了就不开相机")
    ap.add_argument("--video-fps", type=float, default=0.0,
                    help="回放时间轴帧率（仅 --video 有效）。\n"
                         "0 = 自动读容器 fps（默认）；>0 = 强制覆盖。\n"
                         " 老语料容器写错过（真 20 写成 30）⇒ 不要盲目写 30。")
    ap.add_argument("--loop", action="store_true",
                    help="回放播完从头再来（仅 --video 有效）。演示用：\n"
                         "观众不会一直盯着看，循环才不会错过那一段。\n"
                         "帧号 seq 跨循环连续递增，不会让窗口号撞车。")
 # 模型
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    ap.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    ap.add_argument("--yolo-core", type=int, default=None)
    ap.add_argument("--stgcn-core", type=int, default=None)
    ap.add_argument("--model-ver", type=int, default=None,
                    help="幂等键的 model_ver；默认由两个模型 sha256 算出")
    ap.add_argument("--slow-preprocess", action="store_true",
                    help="退回 f32 NCHW 预处理（对拍/排障用；生产默认走 f16 NHWC 快路径）")
    ap.add_argument("--zerocopy", choices=("off", "input", "full"), default="off",
                    help="YOLO 走 ctypes 直调 librknnrt 的零拷贝后端（2026-11-10 板上实测："
                         "detect 33.05 → 23.9 ms，检测结果逐位相同）。"
                         "off=rknnlite（默认，行为不变）；"
                         "input=绑输入（≈24.9 ms）；full=绑输入+绑输出（≈23.9 ms）。"
                         " 打开后**不再** load_rknn 建 YOLO 会话（后端自带）。")
    ap.add_argument("--async-stgcn", action="store_true",
                    help="T3：ST-GCN 放到后台线程算，主线程不在窗口边界上停 ~110 ms\n"
                         "（队列深度取 stgcn_queue_max，满了阻塞、不丢窗口）。\n"
                         " 默认关：关掉时与改动前逐位相同，离线基线不受影响。")
 # 判定
 # 全部 default=None ⇒ 不传就走 MultiConfig 的默认值（行为与本改动之前逐位相同）。
 # 加这几个入口只是为了现场能调，不是为了改默认行为（ 对账基线不能被悄悄动）。
    ap.add_argument("--confirm-windows", type=int, default=None)
    ap.add_argument("--retract-windows", type=int, default=None,
                    help="已 confirmed 后连续多少个窗口低分才撤回（默认 4；"
                         "0.5 s/窗、窗口 4× 重叠 ⇒ 4 窗只有 1.5 s 新帧）")
    ap.add_argument("--stgcn-queue-max", type=int, default=None,
                    help="ST-GCN 队列深度，满则丢最旧窗口（默认 4）")
    ap.add_argument("--cooldown-s", type=float, default=None)
    ap.add_argument("--max-persons-stgcn", type=int, default=None)
    ap.add_argument("--pregate", type=int, default=None, choices=[0, 1],
                    help="几何前置闸门（复用 FG_MULTI_GEO_VETO_* 阈值）：\n"
                         "几何组命中的人不跑 ST-GCN（状态机看不见），r_end 组命中的人\n"
                         "不跑 ST-GCN 但喂 0（= 今天的 veto）。默认关。\n"
                         " 打开后每窗 prob_fall 不再是模型原值，离线逐位基线不适用。")
    ap.add_argument("--pregate-sample", type=int, default=None,
                    help="影子采样：每 N 个被闸条目跑一次模型做反事实审计（推荐 8）")
 # 输出
    ap.add_argument("--shm", action=argparse.BooleanOptionalAction, default=True,
                    help="是否写 FallGuard 共享内存（默认开）")
    ap.add_argument("--snap-w", type=int, default=640,
                    help="告警快照宽（默认 640，与检测帧同宽，bbox 坐标天然对齐）")
    ap.add_argument("--snap-h", type=int, default=480,
                    help="告警快照高（默认 480；回滚旧行为给 320/240）")
    ap.add_argument("--shm-dir", default=None, help="shm_ipc.py 所在目录（一般不用给）")
    ap.add_argument("--web-port", type=int, default=0,
                    help="8888 = 开网页直播；0 = 关（默认关，unit 里显式给 8888）")
    ap.add_argument("--no-meta", action="store_true",
                    help="关闭帧元数据上行（P9 P4b：网页 canvas 叠骨架的数据源）\n"
                         "默认开；地址从 /etc/fallguard/config.json 的 push_url 推导")
    ap.add_argument("--meta-hz", type=float, default=10.0,
                    help="元数据采样率（默认 10 Hz）。4 人时约 120 kbps")
    ap.add_argument("--web-fps", type=float, default=10.0)
    ap.add_argument("--web-quality", type=int, default=70)
    ap.add_argument("--web-scale", type=float, default=1.0)
    ap.add_argument("--web-bind", default="0.0.0.0")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = 一直跑")
    ap.add_argument("--log-every", type=int, default=150, help="心跳帧间隔")
    ap.add_argument("--status-file", default=str(ROOT / "runtime" / "status.json"),
                    help="运行状态 JSON（config_agent 读它判断是否生效；默认 runtime/status.json）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

 # ---- 采集配置 ----
    over = {}
    for name in ("backend", "device", "width", "height", "fps", "io_mode",
                 "appsink_buffers", "queue_buffers", "raw_file"):
        val = getattr(args, name, None)
        if val is not None:
            over[name] = val
    ccfg = CaptureConfig.from_env(**over)

 # ---- 判定配置 ----
    mover = {}
    if args.confirm_windows is not None:
        mover["confirm_windows"] = args.confirm_windows
    if args.retract_windows is not None:
        mover["retract_windows"] = args.retract_windows
    if args.stgcn_queue_max is not None:
        mover["stgcn_queue_max"] = args.stgcn_queue_max
    if args.cooldown_s is not None:
        mover["cooldown_s"] = args.cooldown_s
    if args.max_persons_stgcn is not None:
        mover["max_persons_stgcn"] = args.max_persons_stgcn
    if args.pregate is not None:
        mover["pregate"] = args.pregate
    if args.pregate_sample is not None:
        mover["pregate_sample"] = args.pregate_sample
    mcfg = MultiConfig.from_env(**mover)

    model_ver = args.model_ver
    if model_ver is None:
        model_ver = _model_ver([args.yolo, args.stgcn])

    print(f"[banner] FallGuard per-person sensor  {time.strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    print(f"[config] capture  {ccfg.describe()}", flush=True)
    print(f"[config] multi    {mcfg.describe()}", flush=True)
    print(f"[config] yolo     {args.yolo}", flush=True)
    print(f"[config] stgcn    {args.stgcn}", flush=True)
    print(f"[config] model_ver {model_ver}", flush=True)
    print(f"[config] preprocess "
          f"{'f32 NCHW (slow)' if args.slow_preprocess else 'f16 NHWC (fast)'}",
          flush=True)

 # ---- 运行状态文件（config_agent 靠它判断「配置是否已生效」）----
 # 不要依赖 8888 的 /api/status：网页服务可能被关掉或端口被占，
 # 检测进程会继续跑，但 agent 读不到就会误判失败并回滚。
    status_path = Path(args.status_file)
    policy_rev = int(os.environ.get("FG_POLICY_REV", "0") or 0)
    config_source = os.environ.get("FG_CONFIG_SOURCE", "default")
    config_hash = _config_hash(mcfg)
    started_mono = time.monotonic()

    def _status_payload(ready: bool, now_mono: float) -> dict:
        return {
            "ready": ready,
            "policy_rev": policy_rev,
            "config_source": config_source,
            "config": mcfg.as_dict(),
            "config_hash": config_hash,
            "fps": round(_fps_of(ts), 1) if ready else 0.0,
            "frame_age_s": (round(max(0.0, now_mono - ts[-1]), 2) if ts else 0.0),
            "frames": proc,
            "falls_total": falls_total,
            "events": n_events,
            "model_ver": model_ver,
            "pid": os.getpid(),
            "started_monotonic": round(started_mono, 3),
            "ts": time.time(),
        }

 # ---- shm / web ----
    adapter = None
    if args.shm:
        from shm_adapter import SensorShmWriter
 # 告警快照分辨率：旧部署默认 320×240（注释写「沿用旧直播帧尺寸」），
 # 但检测帧本来就是 640×480，且 `fg_frame_t` 自带 w/h 字段、JPEG 缓冲
 # `FG_JPEG_MAX` = 200KB —— 640×480 q70 约 30–50KB，**完全放得下，
 # 不改 C 网关、不改 shm 结构**。
 # 改成原始分辨率后，`event.bbox`（640×480 坐标）与快照坐标系一致，
 # 将来在图上叠框/骨架不用再乘 0.5。
 # 旧部署目录「一字节不改」⇒ 只改内存里的类属性，照 web_stream._PAGE 的成例。
        SensorShmWriter.LIVE_JPEG_W = args.snap_w
        SensorShmWriter.LIVE_JPEG_H = args.snap_h
        adapter = SensorShmWriter(shm_dir=args.shm_dir)
        print(f"[shm] 告警快照 {args.snap_w}×{args.snap_h}", flush=True)
        print(f"[shm] 已连接 {adapter._shm.name}", flush=True)
    else:
        print("[shm] 关闭（--no-shm）", flush=True)

 # ---- 帧元数据上行----
 # 网页把骨架画在 <canvas> 上，板端只发坐标、不渲染。失败不影响检测。
    meta = None
    if not args.no_meta:
        try:
            import json as _json

            from src.multi.meta_uploader import MetaUploader, parse_push_url

            cfg_path = os.environ.get("FG_CONFIG", "/etc/fallguard/config.json")
            with open(cfg_path, encoding="utf-8") as fh:
                _cfg = _json.load(fh)
            _url = (_cfg.get("push_url") or "").strip()
            if _cfg.get("push_enabled", True) and _url:
                _base, _code, _key = parse_push_url(_url)
                _interval = 1.0 / args.meta_hz if args.meta_hz > 0 else 0.1
 # 五改：告诉云端「我推的是哪条流」，骨架按流分发。
 # 来源优先级：config.json 的 stream → 环境变量 FG_CLOUD_STREAMID
 # （`publish:<path>:<user>:<pass>` 的第二段）→ 默认 live。
 # 这个值必须与 SRT 实际发布路径一致，否则网页按点位绑定的流名
 # 收不到骨架（症状：有画面、没骨架，且只有改绑过病区才暴露）。
                _stream = str(_cfg.get("stream") or "").strip()
                if not _stream:
                    _sid = os.environ.get("FG_CLOUD_STREAMID", "").strip()
                    _seg = _sid.split(":")
                    if len(_seg) >= 2 and _seg[0] == "publish" and _seg[1].strip():
                        _stream = _seg[1].strip()
                if not _stream:
                    _stream = "live"
                meta = MetaUploader(_base, _code, _key, interval_s=_interval,
                                    stream=_stream)
                meta.start()
                print(f"[meta] 帧元数据上行 → {_base} @ {args.meta_hz:g} Hz · stream={_stream}",
                      flush=True)
            else:
                print("[meta] push 未启用或 push_url 为空，元数据上行关闭", flush=True)
        except Exception as exc: # noqa: BLE001 —— 元数据是加分项，配错了不能拦检测
            print(f"[meta] 初始化失败，跳过（{type(exc).__name__}: {exc}）", flush=True)
            meta = None

    hub = server = None
    if args.web_port > 0:
        import web_stream
 # 旧部署的页面模板里硬编码着「骨架 + 六类面板 + 跌倒横幅」—— 那是旧
 # `run_sensor_assembly` 推的画面。新管线没有六分类（检测器 + ST-GCN
 # 二分类），照原样显示就是在骗人。
 # 旧部署目录必须「一字节不改」（回滚保证）⇒ 只改内存里的模板字符串。
        _OLD_TXT = "骨架 + 六类面板 + 跌倒横幅"
        _NEW_TXT = "骨架 + 每人跌倒概率 + 跌倒横幅"
        if _OLD_TXT in web_stream._PAGE:
            web_stream._PAGE = web_stream._PAGE.replace(_OLD_TXT, _NEW_TXT)
        else:
            print("[web]  页面文案未命中旧模板（上游改过？）"
                  "—— 8888 标题可能仍是旧文案", flush=True)
        try:
            hub, server = web_stream.create(
                port=args.web_port, web_fps=args.web_fps,
                quality=args.web_quality, scale=args.web_scale,
                bind=args.web_bind)
            print(f"[web] http://{args.web_bind}:{args.web_port}/", flush=True)
        except OSError as exc:
            print(f"[web] 启动失败（端口被占？）: {exc}", flush=True)
            hub = server = None
    else:
        print("[web] 关闭（--web-port 0）", flush=True)

 # ---- 模型 + 管线 ----
    video_fps_eff = 0.0
    if args.video:
        video_fps_eff = float(args.video_fps or 0.0)
        if video_fps_eff <= 0:
            probed = probe_video_fps(args.video)
            if probed is None:
                video_fps_eff = float(spec.BOARD_FPS)
                print(f"[config]  读不到 {args.video} 的容器 fps，"
                      f"回退 {video_fps_eff:g}（可用 --video-fps 覆盖）", flush=True)
            else:
                video_fps_eff = probed
                print(f"[config] video fps 自动读取 = {video_fps_eff:g}", flush=True)
        src = VideoFileSource(args.video, video_fps_eff, loop=args.loop)
        fps_for_time = video_fps_eff
        print(f"[config] video    {args.video} @ {fps_for_time:g} fps"
              f"{' (loop)' if args.loop else ''}", flush=True)
    else:
        src = open_source(ccfg)
        fps_for_time = float(ccfg.fps)
    if args.zerocopy != "off" and args.slow_preprocess:
        ap.error("--zerocopy 与 --slow-preprocess 互斥：零拷贝后端只走 f16 NHWC")
    t_init = time.monotonic()
 # 零拷贝后端自带 RKNN 会话（ctypes 直调 librknnrt）⇒ 这里不能再
 # load_rknn，否则白白多占一个会话（同时 3 个会挂死 NPU，见 unit 头注释）。
    yolo = None if args.zerocopy != "off" else load_rknn(Path(args.yolo), args.yolo_core)
    stgcn = load_rknn(Path(args.stgcn), args.stgcn_core)
    print(f"[rknn] 初始化 {time.monotonic() - t_init:.1f} s", flush=True)

 # T3：`--async-stgcn` 才起后台线程。队列深度直接用云端下发得到的
 # `stgcn_queue_max`（这个字段此前只在配置里躺着，没有消费者）。
    pcfg = PipelineConfig(fps=fps_for_time,
                          score_window_frames=int(mcfg.window_frames),
                          stitch_bridge=bool(mcfg.stitch_bridge),
                          async_scoring=bool(args.async_stgcn),
                          async_queue_max=int(mcfg.stgcn_queue_max))
    if pcfg.async_scoring:
        print(f"[config] T3 后台线程 ST-GCN 开启，队列深度 "
              f"{pcfg.async_queue_max}（满则阻塞，不丢窗口）", flush=True)
    if args.zerocopy == "off":
        detector = YoloBackend(yolo, conf=pcfg.conf,
                               max_persons=pcfg.max_persons,
                               fast_f16=not args.slow_preprocess)
    else:
 # 这里才 import：默认（--zerocopy off）路径不该因为新模块的
 # 导入期问题而挂掉。后备对照永远可用：去掉 --zerocopy 就是老路。
        from src.backends.rknn_zerocopy import YoloBackendCtypes
        detector = YoloBackendCtypes(
            Path(args.yolo), conf=pcfg.conf, max_persons=pcfg.max_persons,
            core_mask=args.yolo_core,
            bind_output=(args.zerocopy == "full"))
    pipe = MultiPipeline(
        pcfg,
        detector=detector,
        stgcn_session=StgcnBackend(stgcn),
        mcfg=mcfg)

 # ---- 信号 ----
    running = True

    def _stop(signum, _frame):
        nonlocal running
        running = False
        print(f"[exit] 收到信号 {signum}，开始优雅退出...", flush=True)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

 # ---- 主循环 ----
    src.start()
    t0 = time.monotonic()
    proc = 0
    falls_total = 0
    n_events = 0
    last_confirmed_t = 0.0
    last_hb = t0
    last_status = 0.0
    ts: deque = deque(maxlen=60)
    canvas = None # 渲染画布，跨帧复用（省一次 900 KB malloc）
    rc = 0
 # 首帧之前先写一次 ready=False：agent 能确认「进程起来了，只是还没出帧」
    _write_status(status_path, _status_payload(False, t0))
    try:
        while running:
            if args.seconds and time.monotonic() - t0 >= args.seconds:
                break
            fr = src.read(timeout_s=1.0)
            if fr is None:
                if args.video:
                    print("[exit] 录像已播完", flush=True)
                    break
                continue
            proc += 1
            now = time.monotonic()
            ts.append(now)

            for ev in pipe.push(fr.bgr):
                n_events += 1
                confirmed = (ev.state == State.CONFIRMED)
                if confirmed:
                    falls_total += 1
                    last_confirmed_t = now
                if adapter is not None:
                    person = {
                        "person_id": int(ev.person_id),
                        "bbox": pipe.bbox_of_person(ev.person_id) or [0, 0, 0, 0],
                    }
                    adapter.push_fall_event(
                        _to_shm_event(ev, model_ver), person, None, fr.bgr,
                        skip_cooldown=True)
                print(_fmt_event(ev, fr.seq, proc, fps_for_time, now), flush=True)

            is_falling = int((now - last_confirmed_t) < 2.0)
            people = len(pipe.ring.track_ids())
 # 帧元数据上行。`submit` 只做 put_nowait，采样率由
 # MetaUploader 内部按 interval 自己卡，这里每帧调也不会超频。
 # 与下面 `hub.viewers('ann') > 0` 那段无关：那条是板端画布，
 # 这条是给网页 canvas 用的，两者互不影响、可各自开关。
            if meta is not None:
                _h, _w = fr.bgr.shape[:2]
                meta.submit(pipe.display_persons(), _w, _h, ts=time.time())
            if adapter is not None:
                adapter.update_status(is_falling, people, _fps_of(ts))
                adapter.maybe_reload_config()
            if hub is not None:
                hub.set_meta(frames=proc, fps=round(_fps_of(ts), 1),
                             people=people, falling=is_falling,
                             falls_total=falls_total)
 # 按需渲染：
 # 没人看 ann 就一步都不画。渲染成本 1 人 −3.34 ms / 3 人 −6.81 ms，
 # 只该在看的人身上付；21 fps 门测的是无观看者场景 ⇒ 不受影响。
                if hub.viewers("ann") > 0:
                    canvas = overlay.render(
                        fr.bgr, pipe.display_persons(),
                        falling=bool(is_falling), fps=round(_fps_of(ts), 1),
                        canvas=canvas)
                    hub.publish("ann", canvas)
                else:
 # 仍然调一次：viewers<=0 时 publish 立即返回，只为让
 # `drop_no_viewer` 计数保持旧语义（运维靠它判断"没人看"）。
                    hub.publish("ann", fr.bgr)
                hub.publish("raw", fr.bgr)

 # 状态文件：1 秒一次。config_agent 靠它判断配置是否已生效。
            if now - last_status >= 1.0:
                last_status = now
                _write_status(status_path, _status_payload(True, now))

 # 回放按视频时间轴补 sleep；处理慢于 video_fps 时不补（表现为慢放）
            if args.video and video_fps_eff > 0:
                delay = (t0 + proc / video_fps_eff) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            if not args.quiet and now - last_hb >= 5.0:
                last_hb = now
                st = src.stats
 # 几何否决的温度计：`vetoed` = 被 `geo_veto_ar_max` 判为「从没横过来」
 # 的窗口数，`vetoed_hits` = 其中原本会报警的（这才是少报了几条）。
 # 关掉否决时两者恒为 0，不影响旧日志的读法。
                _c = pipe.counters
                print(f"[hb] frames={proc} fps={_fps_of(ts):.1f} "
                      f"people={people} falling={is_falling} "
                      f"falls_total={falls_total} "
                      f"vetoed={_c.geo_vetoed} vetoed_hits={_c.geo_vetoed_hits} "
 # 前置闸门（默认关 ⇒ 全 0）：`skip` = 模型没跑、状态机看不见；
 # `feed0` = 模型没跑、喂 0（= veto）；`sampled` = 影子采样跑过的。
                      f"pregate_skip={_c.pregate_skipped} "
                      f"pregate_feed0={_c.pregate_feed0} "
                      f"pregate_sampled={_c.pregate_sampled} "
                      f"rearmed={_c.rearmed} "
                      f"read={st.get('read', proc)} "
                      f"dropped={st.get('dropped_total', 0)} "
 # 检出净化：`drop_kp` = 姿态无效丢掉的（窗帘/床沿），
 # `drop_dup` = 同一个人的重复框/局部框。两者恒 0 ⇒ 净化没生效。
                      f"drop_kp={getattr(pipe, 'filter_dropped_kp', 0)} "
                      f"drop_dup={getattr(pipe, 'filter_dropped_dup', 0)}",
                      flush=True)
    except Exception as exc: # noqa: BLE001
        rc = 1
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            src.stop()
        except Exception: # noqa: BLE001
            pass
        try:
            res = pipe.finish("live")
            print(f"[exit] done. frames={proc} falls_total={falls_total} "
                  f"events={n_events} windows={len(res.windows)}", flush=True)
        except Exception as exc: # noqa: BLE001
            print(f"[exit] pipe.finish 异常: {exc}", flush=True)
        if server is not None:
            try:
                server.stop()
            except Exception: # noqa: BLE001
                pass
        if adapter is not None:
            try:
                adapter.close()
            except Exception: # noqa: BLE001
                pass
        if meta is not None:
            try:
                meta.stop()
                _st = meta.stats()
                print(f"[exit] meta sent={_st['sent']} dropped={_st['dropped']} "
                      f"failed={_st['failed']}", flush=True)
            except Exception: # noqa: BLE001
                pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
