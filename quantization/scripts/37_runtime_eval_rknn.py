"""G3 with the delivered RKNN graphs instead of ONNX."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse `scripts/30`'s evaluation helpers verbatim rather than re-deriving the
# collapse / segment rules here. A second implementation of "what counts as a
# hit" is how two reports end up disagreeing about the same video.
g3 = _load_own("runtime_eval_30", "scripts/30_runtime_eval.py")
# Same reason for the session builder (L-125): the conversion recipe must be
# the one `scripts/21` already validated, not a paraphrase.
v21 = _load_own("verify_rknn_21", "scripts/21_verify_rknn.py")

paths = g3.paths
spec_mod = g3.spec_mod
rt = g3.rt

RKNN_FILES = {
    ("stgcn", "fp16"): paths.MODELS / "stgcnpp_v2_fp16.rknn",
    ("stgcn", "int8"): paths.MODELS / "stgcnpp_v2_int8.rknn",
    ("yolo", "fp16"): paths.MODELS / "yolo26n_pose_480x640_fp16.rknn",
    ("yolo", "int8"): paths.MODELS / "yolo26n_pose_480x640_int8.rknn",
}


# ------------------------------------------------------------------ RKNN seams


class _StgcnSession:
    """ORT-session-shaped wrapper around an RKNN simulator session."""

    class _Input:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, rknn, input_name: str = "input") -> None:
        self._rknn = rknn
        self._inputs = [self._Input(input_name)]

    def get_inputs(self):
        return self._inputs

    def run(self, _output_names, feeds):
        x = next(iter(feeds.values()))
        # `data_format="nchw"` is mandatory: omitting it defaults to nhwc for
        # 4-D input and the simulator rejects the tensor (see scripts/21's
        # `_run`, which documents both quirks). The ST-GCN graph is 5-D, where
        # the default happens to work, but the layout is stated anyway so this
        # cannot drift from the G4 path.
        outs = self._rknn.inference(inputs=[x], data_format="nchw")
        if outs is None:
            raise RuntimeError(
                "RKNN ST-GCN inference returned None -- the toolkit logs the "
                "real error instead of raising")
        return [np.asarray(outs[0])]


class RknnDetector:
    """`src.runtime.Detector`, with the ONNX session replaced by RKNN."""

    def __init__(self, rknn, conf: float = spec_mod.CONF,
                 max_persons: int = spec_mod.MAX_PERSONS) -> None:
        from src.deploy import yolo_export

        self._yolo = yolo_export
        self.session = rknn
        self.conf = float(conf)
        self.max_persons = int(max_persons)
        self.n_calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list:
        chw, params = self._yolo.preprocess(frame_bgr, imgsz=spec_mod.IMGSZ)
        if chw.shape != self._yolo.INPUT_SHAPE:
            raise ValueError(
                f"preprocess produced {chw.shape}, graph wants "
                f"{self._yolo.INPUT_SHAPE}")
        outs = self.session.inference(inputs=[chw], data_format="nchw")
        if outs is None:
            raise RuntimeError("RKNN YOLO inference returned None")
        raw = np.asarray(outs[0])
        self.n_calls += 1
        boxes, kpts, scores = self._yolo.decode(
            raw, params, conf=self.conf, max_persons=self.max_persons)
        return [rt.Detection(bbox=boxes[i], keypoints=kpts[i], scores=scores[i])
                for i in range(len(boxes))]


# -------------------------------------------------------------------- replay


def replay(rows, yolo_backend: str, stgcn_backend: str, precision: str,
           tag: str) -> dict:
    """One full pass over the board clips with the chosen backends."""
    from src.runtime import Pipeline

    cfg_kwargs = dict(
        yolo_onnx=paths.MODELS / "yolo26n_pose_480x640.onnx",
        stgcn_onnx=paths.MODELS / "stgcnpp_v2.onnx",
        stitch=True,
        # §3.3 batching is an ONNX-side optimisation (`_b4.onnx`). Feeding an
        # RKNN session through the batch path would mix backends inside one
        # forward pass, so it is off here. The per-person loop is what the
        # board's `rknn_batch_size` build replaces, and §3.3 already proved
        # the two are bit-identical -- so this choice cannot move a number.
        score_batch=1,
    )
    cfg = rt.PipelineConfig(**cfg_kwargs)

    det = None
    stgcn = None
    if "rknn" in (yolo_backend, stgcn_backend):
        print(f"[{tag}] building RKNN sessions ({precision}) ...", flush=True)
        t = time.perf_counter()
        if yolo_backend == "rknn":
            det = RknnDetector(v21._rknn_session(
                RKNN_FILES[("yolo", precision)], "yolo", precision))
        if stgcn_backend == "rknn":
            stgcn = _StgcnSession(v21._rknn_session(
                RKNN_FILES[("stgcn", precision)], "stgcn", precision))
        print(f"[{tag}]   built in {time.perf_counter() - t:.1f}s", flush=True)

    all_windows: list[dict] = []
    per_video: list[dict] = []
    t0 = time.perf_counter()
    total_frames = 0

    for i, row in enumerate(rows, 1):
        vid = row["video_id"]
        pipe = Pipeline(cfg, detector=det, stgcn_session=stgcn)
        video_path = paths.SOURCE_RAW_BOARD / row["relative_path"]
        if det is not None:
            det.n_calls = 0
        result = pipe.run_video(vid, g3.read_frames(video_path))
        total_frames += result.n_frames

        offset = len(all_windows)
        for w in result.windows:
            all_windows.append({"window_id": f"w{offset + w.window_id:06d}",
                                "video_id": vid, "start": w.start, "end": w.end,
                                "fps": float(row["fps"])})

        per_video.append({
            "video_id": vid, "n_frames": result.n_frames,
            "n_windows": len(result.windows),
            "n_scored": result.n_scored, "n_observed": result.n_observed,
            "n_yolo_calls": (det.n_calls if det is not None else None),
            # `WindowScore.prob_fall` is `score_windows`'s output: the max over
            # people past the coverage gate. Using it instead of re-collapsing
            # guarantees the two backends are compared through the same code
            # path that produced the published G3/G6 tables.
            "window_prob": [round(float(w.prob_fall), 8)
                            for w in result.windows],
        })
        print(f"[{tag} {i}/{len(rows)}] {vid} frames={result.n_frames} "
              f"windows={len(result.windows)} scored={result.n_scored}",
              flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n[{tag}] replayed {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps)", flush=True)

    return {"windows": all_windows, "per_video": per_video,
            "seconds": elapsed, "n_frames": total_frames,
            "yolo_backend": yolo_backend, "stgcn_backend": stgcn_backend}


def _window_probs(run: dict) -> np.ndarray:
    out = []
    for v in run["per_video"]:
        out.extend(v["window_prob"])
    return np.asarray(out, dtype=np.float64)


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so `json.dump` accepts them."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--backend", choices=["onnx", "rknn", "both"], default="both",
                   help="onnx / rknn / both (both also runs the two "
                        "single-model attribution arms)")
    p.add_argument("--limit", type=int, default=0, help="first N videos (smoke)")
    p.add_argument("--arms", default=None,
                   help="comma-separated subset of "
                        "onnx,yolo_rknn,stgcn_rknn,rknn. The four-arm run is "
                        "~3 h on the RKNN simulator; after one full run the "
                        "per-arm cache makes re-selecting free.")
    p.add_argument("--recompute", action="store_true",
                   help="ignore the per-arm cache (use when a .rknn changed)")
    # QAT artefacts live in `experiments/qat/` (delivery is never overwritten),
    # so the ST-GCN side needs a way to point at them. The YOLO side has no
    # equivalent yet because no YOLO QAT build exists -- if one appears, add
    # `--yolo-rknn` with the same shape rather than generalising prematurely.
    p.add_argument("--stgcn-rknn", default=None,
                   help="override the ST-GCN .rknn (e.g. the QAT build in "
                        "experiments/qat/). Its sha256 enters the cache key, "
                        "so a cached arm can never be reused across models.")
    return p.parse_args()


def _arms(backend: str) -> list[tuple[str, str, str]]:
    """`(tag, yolo_backend, stgcn_backend)` for each run this invocation makes."""
    if backend == "onnx":
        return [("onnx", "onnx", "onnx")]
    if backend == "rknn":
        return [("rknn", "rknn", "rknn")]
    return [
        ("onnx", "onnx", "onnx"),
        # yolo converted, stgcn held at ONNX -> isolates the detector
        ("yolo_rknn", "rknn", "onnx"),
        # stgcn converted, detector held at ONNX -> isolates ST-GCN
        ("stgcn_rknn", "onnx", "rknn"),
        ("rknn", "rknn", "rknn"),
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    # `--backend onnx` is a plain ONNX replay and runs anywhere; only the RKNN
    # simulator is Linux-only. Blocking both would make the ONNX half of the
    # pairing untestable on the Windows side.
    if args.backend in ("rknn", "both") and sys.platform != "linux":
        print("The RKNN simulator is Linux-only. Run this inside WSL2:\n"
              "  wsl -d Ubuntu -- bash -lc 'cd \"<project>\" && "
              "~/rknn-env/bin/python scripts/37_runtime_eval_rknn.py'")
        return 2

    if args.stgcn_rknn:
        override = Path(args.stgcn_rknn).resolve()
        if not override.exists():
            raise SystemExit(f"--stgcn-rknn not found: {override}")
        RKNN_FILES[("stgcn", args.precision)] = override
        print(f"ST-GCN override -> {override}", flush=True)

    ref = g3.load_reference()
    gt = g3.load_gt()
    rows = g3.board_rows(args.limit)

    for name, prec in (("yolo", args.precision), ("stgcn", args.precision)):
        p = RKNN_FILES[(name, prec)]
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/20_convert_rknn.py first")

    # Per-arm cache. A four-arm run is ~3 h of simulator time, and the whole
    # point of the single-model arms is to be *added* to a baseline that
    # already exists. Keying on the tag + precision + row count + the sha256 of
    # every .rknn involved is the L-125 discipline: a cached arm must never be
    # mistaken for a run of the current artefacts.
    suffix = f"_{args.precision}" + (f"_limit{args.limit}" if args.limit else "")
    cache_dir = paths.PARITY / f"g3_rknn_arms{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _arm_key(yb: str, sb: str) -> str:
        parts = [f"n{len(rows)}"]
        for model, backend in (("yolo", yb), ("stgcn", sb)):
            if backend == "rknn":
                parts.append(f"{model}={paths.sha256_file(RKNN_FILES[(model, args.precision)])}")
            else:
                parts.append(f"{model}=onnx")
        return "|".join(parts)

    wanted = _arms(args.backend)
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        unknown = keep - {t for t, _, _ in wanted}
        if unknown:
            raise SystemExit(f"unknown --arms {sorted(unknown)}; "
                             f"choose from {[t for t, _, _ in wanted]}")
        wanted = [a for a in wanted if a[0] in keep]
        # ## Always keep the ONNX arm
        #
        # Every diff below is against the ONNX baseline (`_diff(runs["onnx"],
        # ...)`), and the gate section is skipped entirely when `"rknn"` is
        # absent. Selecting only `--arms stgcn_rknn` therefore used to run 19
        # videos for 17 minutes and then emit a report with no comparison --
        # measured , the run was real and the output said "本次只跑
        # ONNX 对照臂". The baseline is what makes a single-model arm mean
        # anything, so it is added back when the caller filters it out. On a
        # second invocation it is a cache hit, so this costs nothing.
        if not any(t == "onnx" for t, _, _ in wanted):
            wanted.insert(0, ("onnx", "onnx", "onnx"))

    runs: dict[str, dict] = {}
    for tag, yb, sb in wanted:
        key = _arm_key(yb, sb)
        cache = cache_dir / f"{tag}.json"
        if cache.exists() and not args.recompute:
            d = paths.read_json(cache)
            if d.get("key") == key:
                print(f"[{tag}] reusing cached arm ({cache.name})", flush=True)
                runs[tag] = d["run"]
                continue
            print(f"[{tag}] cache stale (artefacts or row set changed), "
                  f"re-running", flush=True)
        run = replay(rows, yb, sb, args.precision, tag)
        paths.write_json(cache, {"key": key, "run": _jsonable(run)})
        runs[tag] = run

    # ---- everything below mirrors scripts/30's evaluation, so the numbers are
    # directly comparable to the published G3/G6 tables.
    report: dict = {
        "precision": args.precision,
        "backend": args.backend,
        "n_videos": len(rows),
        "rknn_sha256": {k: paths.sha256_file(v) for k, v in
                        ((f"{m}_{args.precision}", RKNN_FILES[(m, args.precision)])
                         for m in ("yolo", "stgcn"))},
        # Which graph each arm used. Carried separately from `runs` because
        # `runs[tag]` below is *reduced* to the scored fields -- reading
        # `yolo_backend` out of it raised KeyError after a 12-minute run
        # (measured ), so the provenance is kept where it lives.
        "arms": {tag: {"yolo_backend": run["yolo_backend"],
                       "stgcn_backend": run["stgcn_backend"]}
                 for tag, run in runs.items()},
        "runs": {},
    }

    for name, run in runs.items():
        wp = _window_probs(run)
        cmp_windows = g3._compare_windows(run["windows"], ref["windows"])
        ref_wp = g3._ref_collapse(ref["windows"], ref["samples"], "C")
        cmp_prob = g3._compare_probs(wp, ref_wp)
        seg = g3.evaluate_segments(run["windows"], wp, gt, g3.EXCLUDE)
        nofall = g3.evaluate_no_fall(run["windows"], wp, gt, g3.EXCLUDE)
        ref_seg, ref_nofall = g3.load_source_reference("C", spec_mod.MIN_COVERAGE)
        g6_seg = g3._compare_per_segment(seg["per_seg"], ref_seg)
        g6_nofall = g3._compare_no_fall(nofall, ref_nofall)

        report["runs"][name] = {
            "seconds": round(run["seconds"], 2),
            "fps": round(run["n_frames"] / run["seconds"], 2),
            "windows": cmp_windows,
            "segments": {k: seg[k] for k in ("seg_tot", "seg_hit", "pos_v",
                                             "pos_h", "neg_v", "neg_fa")},
            "per_segment": seg["per_seg"],
            "g6_per_segment": {k: g6_seg[k] for k in
                               ("n", "n_hit_match", "max_abs_dp", "mean_abs_dp")},
            "g6_no_fall": {k: g6_nofall[k] for k in
                           ("n", "n_verdict_match", "max_abs_dp", "mean_abs_dp")},
            "vs_reference": cmp_prob,
            "per_video": run["per_video"],
        }

    # ---- paired comparisons. Each arm is diffed against the ONNX baseline,
    # never against another RKNN arm: the baseline is the only one where both
    # graphs are the same bytes `scripts/30` used.
    def _diff(a_run: dict, b_run: dict, a_key: str, b_key: str) -> dict:
        a, b = _window_probs(a_run), _window_probs(b_run)
        n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n])
        flips = int(((a[:n] >= spec_mod.THRESHOLD)
                     != (b[:n] >= spec_mod.THRESHOLD)).sum())
        seg_a = {(s["video_id"], s["start"], s["end"]): s["hit"]
                 for s in report["runs"][a_key]["per_segment"]}
        seg_b = {(s["video_id"], s["start"], s["end"]): s["hit"]
                 for s in report["runs"][b_key]["per_segment"]}
        common = set(seg_a) & set(seg_b)
        seg_agree = sum(1 for k in common if seg_a[k] == seg_b[k])
        flipped_segments = sorted(f"{v}@{s}-{e}" for v, s, e in common
                                  if seg_a[(v, s, e)] != seg_b[(v, s, e)])
        return {
            "n_windows": int(n),
            "max_abs_dp": float(d.max()) if n else 0.0,
            "mean_abs_dp": float(d.mean()) if n else 0.0,
            "threshold_flips": int(flips),
            "n_over_1e-3": int((d > 1e-3).sum()),
            "segments_compared": int(len(common)),
            "segment_verdicts_agree": int(seg_agree),
            "flipped_segments": flipped_segments,
            "recall_a": f"{report['runs'][a_key]['segments']['seg_hit']}"
                        f"/{report['runs'][a_key]['segments']['seg_tot']}",
            "recall_b": f"{report['runs'][b_key]['segments']['seg_hit']}"
                        f"/{report['runs'][b_key]['segments']['seg_tot']}",
        }

    if "onnx" in runs:
        report["vs_onnx"] = {
            tag: _diff(runs["onnx"], runs[tag], "onnx", tag)
            for tag in runs if tag != "onnx"
        }

    gate = {}
    gate_arm = ("rknn" if "rknn" in runs else
                next((t for t in runs if t != "onnx"), None))
    if gate_arm is not None:
        r = report["runs"][gate_arm]
        n_seg = r["g6_per_segment"]["n"]
        n_nf = r["g6_no_fall"]["n"]
        gate = {
            "windows_915_of_915": bool(r["windows"]["n_missing"] == 0
                                       and r["windows"]["n_extra"] == 0
                                       and r["windows"]["order_identical"]),
            "segment_recall": bool(r["segments"]["seg_hit"] == g3.EXPECT_SEGMENTS["C"]),
            "videos": bool(r["segments"]["pos_h"] == g3.EXPECT_VIDEOS["pos_h"]
                           and r["segments"]["pos_v"] == g3.EXPECT_VIDEOS["pos_v"]),
            "false_alarms": bool(r["segments"]["neg_fa"] == g3.EXPECT_VIDEOS["neg_fa"]
                                 and r["segments"]["neg_v"] == g3.EXPECT_VIDEOS["neg_v"]),
            "g6_segment_verdicts": bool(n_seg > 0
                                        and r["g6_per_segment"]["n_hit_match"] == n_seg),
            "g6_no_fall_verdicts": bool(n_nf > 0
                                        and r["g6_no_fall"]["n_verdict_match"] == n_nf),
        }
    report["gate"] = gate
    report["gate_arm"] = gate_arm
    ok = bool(gate) and all(gate.values())
    if args.limit:
        ok = False
        print(f"\nNOTE: --limit {args.limit} is a smoke test, not a gate run.")
    report["passed"] = ok

    report = _jsonable(report)
    paths.write_json(paths.PARITY / f"g3_rknn{suffix}.json", report)
    _write_report(report, suffix)
    return 0 if ok else 1


def _write_report(report: dict, suffix: str) -> None:
    def mark(good: bool) -> str:
        return "✅" if good else "❌"

    _arm = report.get("gate_arm")
    _title_arm = ("" if _arm in (None, "rknn") else f"（`{_arm}` 单模型臂）")
    title = ("G3 — 流式 runtime，**RKNN 交付图**" + _title_arm
             if report["gate"] else "G3 — 流式 runtime（ONNX 对照）")
    lines = [f"# {title}", "",
             f"- 后端：{report['backend']}；视频 {report['n_videos']} 个",
             "- 管线（tracker / ring / 插值 / 覆盖率门 / D-042 拼接 / 每窗取最大）"
             "**一个字未改**，只换执行后端",
             "- 会话由 `scripts/21` 的 `_rknn_session` 构建（L-125：验证器必须"
             "复现构建参数）",
             "", "## 判据（业务指标；概率只列值，D-114）", "",
             "| 判据 | 实测 | 门槛 | 结果 |", "|---|---:|---:|---|"]
    if not report["gate"]:
        lines += ["| — | 本次只跑 ONNX 对照臂 | — | — |", ""]
    if report["gate"]:
        r = report["runs"][report["gate_arm"]]
        g = report["gate"]
        lines += [
            f"| 候选窗口数 | {r['windows']['n_mine']} | ={r['windows']['n_ref']} "
            f"| {mark(r['windows']['n_mine'] == r['windows']['n_ref'])} |",
            f"| 缺失 / 多余窗口 | {r['windows']['n_missing']} / "
            f"{r['windows']['n_extra']} | 0 / 0 | {mark(g['windows_915_of_915'])} |",
            f"| 段召回 | {r['segments']['seg_hit']}/{r['segments']['seg_tot']} "
            f"| 13/17 | {mark(g['segment_recall'])} |",
            f"| 有跌倒视频检出 | {r['segments']['pos_h']}/{r['segments']['pos_v']} "
            f"| 8/9 | {mark(g['videos'])} |",
            f"| no_fall 视频误报 | {r['segments']['neg_fa']}/{r['segments']['neg_v']} "
            f"| 3/7 | {mark(g['false_alarms'])} |",
            f"| 逐段命中判定 vs `per_segment.csv` "
            f"| {r['g6_per_segment']['n_hit_match']}/{r['g6_per_segment']['n']} "
            f"| 全部 | {mark(g['g6_segment_verdicts'])} |",
            f"| no_fall 逐视频误报判定 "
            f"| {r['g6_no_fall']['n_verdict_match']}/{r['g6_no_fall']['n']} "
            f"| 全部 | {mark(g['g6_no_fall_verdicts'])} |",
            "",
            "## 与冻结参考的残差（**证据，不设门**）", "",
            "| 臂 | YOLO | ST-GCN | `max\\|Δp\\|` | 均值 | 超 1e-3 | 阈值翻转 |",
            "|---|---|---|---:|---:|---:|---:|"]
        for name, run in report["runs"].items():
            c = run["vs_reference"]
            arm = report["arms"][name]
            lines.append(f"| `{name}` | {arm['yolo_backend']} "
                         f"| {arm['stgcn_backend']} | {c['max_abs_dp']:.3e} "
                         f"| {c['mean_abs_dp']:.3e} | {c['n_over_tol']}"
                         f"/{c['n_compared']} | {c['threshold_flips']} |")
        lines.append("")

    if report.get("vs_onnx"):
        lines += ["## 逐模型归因（每一臂都与 ONNX 基线配对）", "",
                  "单模型臂的意义：`rknn` 一臂同时换了两个图，指标一动就无法归因。"
                  "`yolo_rknn`（只换检测器）与 `stgcn_rknn`（只换 ST-GCN）"
                  "各自把另一个图钉在 ONNX 上——这是 D-108 分离检测器误差与"
                  "管线误差的同一手法，用在「哪一次转换移动了数字」上。", "",
                  "| 臂 | YOLO | ST-GCN | 段召回 | 逐段判定一致 | 阈值翻转 | "
                  "`max\\|Δp\\|` | 翻转的段 |",
                  "|---|---|---|---:|---:|---:|---:|---|"]
        for tag, p in report["vs_onnx"].items():
            arm = report["arms"][tag]
            flips = ("—" if not p["flipped_segments"]
                     else "、".join(f"`{s}`" for s in p["flipped_segments"]))
            lines.append(
                f"| `{tag}` | {arm['yolo_backend']} | {arm['stgcn_backend']} "
                f"| {p['recall_a']} → {p['recall_b']} "
                f"| {p['segment_verdicts_agree']}/{p['segments_compared']} "
                f"| {p['threshold_flips']} | {p['max_abs_dp']:.3e} | {flips} |")
        lines.append("")
        # The attribution sentence is generated from the numbers, not asserted:
        # if the single-model arms disagree with the combined arm, saying so is
        # more useful than picking a story.
        yo = report["vs_onnx"].get("yolo_rknn")
        st = report["vs_onnx"].get("stgcn_rknn")
        if yo is not None and st is not None:
            y_flips = len(yo["flipped_segments"])
            s_flips = len(st["flipped_segments"])
            if y_flips and not s_flips:
                lines += ["**归因：差异全部来自 YOLO 检测器那一侧。** "
                          "只换 ST-GCN 时逐段判定 17/17 不变；只换检测器时"
                          "段判定翻转。这与 FP16 的误差模型一致——检测器输出"
                          "决定 track 划分，而 track 划分是离散的（L-107）。", ""]
            elif s_flips and not y_flips:
                lines += ["**归因：差异全部来自 ST-GCN 那一侧。** "
                          "只换检测器时逐段判定不变。", ""]
            elif y_flips and s_flips:
                lines += [f"**归因：两侧都有贡献**——只换检测器翻转 {y_flips} 段，"
                          f"只换 ST-GCN 翻转 {s_flips} 段。", ""]
            else:
                lines += ["**归因：两侧单独都不改变任何段判定**，"
                          "合起来才出现差异，说明是两个小误差的叠加。", ""]

    if not report["gate"]:
        lines += ["**本次只跑 ONNX 对照臂**，不产生结论。"
                  "要与 RKNN 配对比较，去掉 `--backend onnx`。", ""]
    else:
        verdict = "通过" if report["passed"] else "未通过"
        _what = (f"`{report['gate_arm']}` 臂" if report["gate_arm"] != "rknn"
                 else "RKNN 交付图")
        lines += [f"**结论：{_what}在业务指标上{verdict}。**"
                  + (" 这正是论文可以写「转换后段召回不变」的那条证据。"
                     if report["passed"] else
                     " 不能写「转换后业务指标不变」。"), ""]
    (paths.PARITY / f"g3_rknn{suffix}.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
