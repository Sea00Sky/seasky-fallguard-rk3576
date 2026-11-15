#!/usr/bin/env python3
"""p8/regress.py — 逐窗对账"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import statistics
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 硬门：窗口总数。变了说明 tracker / ring / 窗口切分被改动了。
EXPECT_WINDOWS = 915

# 跨臂参考口径（来自 reference/g3_rknn_fp16.json 的 rknn_vs_onnx）
CROSS_ARM_REF = {
    "max_abs_dp": 0.61965125,
    "mean_abs_dp": 0.005100647683060109,
    "threshold_flips": 4,
    "n_over_1e-3": 175,
}

COVERAGE_BINS = [
    (0.00, 0.70, "cov<0.70 (gate 外)"),
    (0.70, 0.80, "cov 0.70-0.80"),
    (0.80, 0.90, "cov 0.80-0.90"),
    (0.90, 1.00, "cov 0.90-1.00"),
    (1.00, 1.01, "cov = 1.00"),
]

SNAP_FIELDS = ["video_id", "window_idx", "prob_fall",
               "track_id", "coverage", "n_candidates", "start", "end"]


# ------------------------------------------------------------------ 工具

def prob_digest(probs: list[float]) -> str:
    """概率向量的 sha256。用 float64 小端字节，保证跨进程/跨机可比。"""
    h = hashlib.sha256()
    for p in probs:
        h.update(struct.pack("<d", float(p)))
    return h.hexdigest()


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[i]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ snap

def load_run(results_dir: Path) -> tuple[dict, dict]:
    """读 run_video.py 的产物 -> (per_video, summary)。"""
    files = sorted(glob.glob(str(results_dir / "*.json")))
    per_video: dict[str, dict] = {}
    for f in files:
        name = Path(f).name
        if name in ("summary.json", "digest.json", "run_meta.json"):
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if "video_id" not in d or "windows" not in d:
            continue
        per_video[d["video_id"]] = d
    if not per_video:
        raise SystemExit(f"[FATAL] {results_dir} 下没找到 per-video JSON")

    summary = {}
    sp = results_dir / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    return per_video, summary


def cmd_snap(args) -> int:
    src = Path(args.from_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_video, summary = load_run(src)

    rows: list[dict] = []
    digest: dict[str, dict] = {}
    total = 0
    for vid in sorted(per_video):
        d = per_video[vid]
        probs = [float(w["prob_fall"]) for w in d["windows"]]
        total += len(probs)
        digest[vid] = {
            "n_frames": d.get("n_frames"),
            "n_windows": len(probs),
            "prob_sha256": prob_digest(probs),
            "max_prob": max(probs) if probs else 0.0,
        }
        for i, w in enumerate(d["windows"]):
            rows.append({
                "video_id": vid,
                "window_idx": i,
                "prob_fall": repr(float(w["prob_fall"])),
                "track_id": w.get("track_id", ""),
                "coverage": repr(float(w.get("coverage", 0.0))),
                "n_candidates": w.get("n_candidates", ""),
                "start": w.get("start", ""),
                "end": w.get("end", ""),
            })

    # 上游 summary 里的 n_scored / n_observed 是定位轨迹差异的关键诊断量
    for row in (summary.get("per_video") or []):
        vid = row.get("video_id")
        if vid in digest:
            digest[vid]["n_scored"] = row.get("n_scored")
            digest[vid]["n_observed"] = row.get("n_observed")

    csv_path = out / "windows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SNAP_FIELDS)
        w.writeheader()
        w.writerows(rows)

    all_probs = [float(r["prob_fall"]) for r in rows]
    doc = {
        "source": rel(src),
        "n_videos": len(digest),
        "n_windows": total,
        "prob_sha256_all": prob_digest(all_probs),
        "per_video": digest,
    }
    (out / "digest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[snap] {rel(src)} -> {rel(out)}")
    print(f"       视频 {len(digest)}  窗口 {total}  "
          f"all_sha256 {doc['prob_sha256_all'][:16]}…")
    if total != EXPECT_WINDOWS:
        print(f"        窗口数 {total} != 期望 {EXPECT_WINDOWS}")
    return 0


# ------------------------------------------------------------------ diff

def load_csv(path: Path) -> tuple[dict, bool]:
    """读 windows.csv 或 reference/g3_windows.csv -> (per_video, 是否含 track_id)。"""
    per_video: dict[str, list[dict]] = {}
    has_track = False
    with open(path, encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        has_track = "track_id" in fields
        for r in rd:
            vid = r["video_id"]
            per_video.setdefault(vid, []).append({
                "window_idx": int(r["window_idx"]),
                "prob_fall": float(r["prob_fall"]),
                "track_id": (int(r["track_id"]) if has_track
                             and r.get("track_id") not in (None, "") else None),
                "coverage": (float(r["coverage"]) if "coverage" in fields
                             and r.get("coverage") not in (None, "") else None),
            })
    if not per_video:
        raise SystemExit(f"[FATAL] {path} 里没有窗口")
    return per_video, has_track


def resolve_new(path: Path) -> Path:
    """--new 可以给目录（自动找 windows.csv）。"""
    if path.is_dir():
        p = path / "windows.csv"
        if not p.exists():
            raise SystemExit(f"[FATAL] {path} 下没有 windows.csv，先跑 snap")
        return p
    return path


def cmd_diff(args) -> int:
    new_path = resolve_new(Path(args.new))
    base_path = Path(args.base)
    out_md = Path(args.out) if args.out else new_path.parent / "diff.md"

    new, new_has_track = load_csv(new_path)
    base, base_has_track = load_csv(base_path)

    # ---- 窗口数 / 顺序 ------------------------------------------------
    all_vids = sorted(set(new) | set(base))
    count_problems: list[str] = []
    n_new = n_base = 0
    for vid in all_vids:
        a = base.get(vid, [])
        b = new.get(vid, [])
        n_base += len(a)
        n_new += len(b)
        if len(a) != len(b):
            # 0 窗口的视频不出现在基线里是正常的
            if not (len(a) == 0 and len(b) == 0):
                count_problems.append(f"{vid}: base={len(a)} new={len(b)}")
            continue
        ia = [x["window_idx"] for x in a]
        ib = [x["window_idx"] for x in b]
        if ia != ib:
            count_problems.append(f"{vid}: window_idx 顺序不同")

    # ---- 概率偏差 ------------------------------------------------------
    diffs: list[tuple[str, int, float, float, float, float | None]] = []
    for vid in all_vids:
        a = base.get(vid, [])
        b = new.get(vid, [])
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            diffs.append((vid, x["window_idx"], x["prob_fall"], y["prob_fall"],
                          abs(x["prob_fall"] - y["prob_fall"]), y["coverage"]))

    ds = sorted(d[4] for d in diffs)
    n = len(ds)
    stats = {
        "n_windows": n,
        "max_abs_dp": ds[-1] if ds else 0.0,
        "mean_abs_dp": (sum(ds) / n) if n else 0.0,
        "median_abs_dp": statistics.median(ds) if ds else 0.0,
        "p90_abs_dp": pct(ds, 0.90),
        "p99_abs_dp": pct(ds, 0.99),
        "n_over_1e-3": sum(1 for x in ds if x > 1e-3),
        "n_over_0.01": sum(1 for x in ds if x > 0.01),
        "n_over_0.1": sum(1 for x in ds if x > 0.1),
    }
    flips = [d for d in diffs if (d[2] >= 0.5) != (d[3] >= 0.5)]

    # ---- track_id / coverage（仅同臂可比）-------------------------------
    track_diff = cov_diff = 0
    max_dcov = 0.0
    if new_has_track and base_has_track:
        for vid in all_vids:
            a, b = base.get(vid, []), new.get(vid, [])
            if len(a) != len(b):
                continue
            for x, y in zip(a, b):
                if x["track_id"] is not None and y["track_id"] is not None \
                        and x["track_id"] != y["track_id"]:
                    track_diff += 1
                if x["coverage"] is not None and y["coverage"] is not None:
                    dc = abs(x["coverage"] - y["coverage"])
                    max_dcov = max(max_dcov, dc)
                    if dc > 0:
                        cov_diff += 1

    # ---- 判定 ----------------------------------------------------------
    # 同臂 = 基线带 track_id（本脚本 snap 的产物）
    same_arm = bool(args.expect_zero) if args.expect_zero is not None else base_has_track

    # 915 只是「全量运行」的健全性检查，不能当所有比较的硬门。
    # 部分运行（--video 单视频、--max-frames）本来就不该有 915 个窗口；
    # 那种情况下真正的硬门是「逐视频窗口数与基线一致」，已经记在 count_problems 里。
    is_full = n_new == EXPECT_WINDOWS
    window_ok = not count_problems
    exact_ok = stats["max_abs_dp"] == 0.0 and track_diff == 0

    if same_arm:
        verdict = "PASS" if (window_ok and exact_ok) else "FAIL"
    else:
        # 跨臂只报不判；窗口数仍然算硬门（它反映管线结构，与数值精度无关）
        verdict = "PASS" if window_ok else "FAIL"

    # ---- 按 coverage 分组 ----------------------------------------------
    bins: list[dict] = []
    for lo, hi, label in COVERAGE_BINS:
        sel = [d[4] for d in diffs
               if d[5] is not None and lo <= d[5] < hi]
        if not sel:
            bins.append({"label": label, "n": 0})
            continue
        s = sorted(sel)
        bins.append({
            "label": label, "n": len(s),
            "max": s[-1], "mean": sum(s) / len(s),
            "p90": pct(s, 0.90),
            "n_over_1e-3": sum(1 for x in s if x > 1e-3),
        })

    # ---- 逐视频 --------------------------------------------------------
    per_video_rows = []
    for vid in all_vids:
        a, b = base.get(vid, []), new.get(vid, [])
        sel = [d for d in diffs if d[0] == vid]
        if not sel and len(a) == len(b) == 0:
            continue
        per_video_rows.append({
            "video_id": vid,
            "n_base": len(a), "n_new": len(b),
            "max": max((d[4] for d in sel), default=0.0),
            "mean": (sum(d[4] for d in sel) / len(sel)) if sel else 0.0,
            "n_over_1e-3": sum(1 for d in sel if d[4] > 1e-3),
            "flips": sum(1 for d in sel if (d[2] >= 0.5) != (d[3] >= 0.5)),
        })

    worst = sorted(diffs, key=lambda d: -d[4])[:20]

    # ---- 写 diff.md ----------------------------------------------------
    L: list[str] = []
    A = L.append
    A("# 逐窗对账\n")
    A(f"- 新跑：`{rel(new_path)}`")
    A(f"- 基线：`{rel(base_path)}`")
    A(f"- 模式：**{'同臂（必须恰好 0）' if same_arm else '跨臂（只报不判）'}**")
    A(f"- 判定：**{verdict}**\n")

    A("## 1. 窗口数 / 顺序\n")
    A("| | 值 | 期望 |")
    A("|---|---:|---:|")
    A(f"| 新跑窗口数 | {n_new} | {EXPECT_WINDOWS}（全量） |")
    A(f"| 基线窗口数 | {n_base} | — |")
    A(f"| 逐视频不一致 | {len(count_problems)} | 0 |")
    A("")
    if not is_full:
        A(f"> ℹ 本次是**部分运行**（{n_new} 窗，非 {EXPECT_WINDOWS}）。"
          f"915 这个门只对全量运行生效，本次以「逐视频窗口数一致」为硬门。\n")
    A("")
    if count_problems:
        A("**不一致清单：**\n")
        for p in count_problems:
            A(f"- {p}")
        A("")
    else:
        A(" 每个视频的窗口数与 `window_idx` 顺序都一致。\n")

    A("## 2. 概率偏差\n")
    A("| 指标 | 值 | 参考 |")
    A("|---|---:|---:|")
    A(f"| `max\\|Δp\\|` | **{stats['max_abs_dp']:.10f}** | 同臂 0 / 跨臂 ≈{CROSS_ARM_REF['max_abs_dp']:.4f} |")
    A(f"| `mean\\|Δp\\|` | {stats['mean_abs_dp']:.10f} | 跨臂 ≈{CROSS_ARM_REF['mean_abs_dp']:.6f} |")
    A(f"| median | {stats['median_abs_dp']:.3e} | — |")
    A(f"| p90 | {stats['p90_abs_dp']:.6f} | — |")
    A(f"| p99 | {stats['p99_abs_dp']:.6f} | — |")
    A(f"| `>1e-3` | {stats['n_over_1e-3']} / {n} | 跨臂 ≈{CROSS_ARM_REF['n_over_1e-3']} |")
    A(f"| `>0.01` | {stats['n_over_0.01']} / {n} | — |")
    A(f"| `>0.1` | {stats['n_over_0.1']} / {n} | — |")
    A(f"| 跨 0.5 翻转 | {len(flips)} | 跨臂 ≈{CROSS_ARM_REF['threshold_flips']} |")
    A("")

    if new_has_track and base_has_track:
        A("## 3. 轨迹 / 覆盖率（同臂才有）\n")
        A(f"- `track_id` 不同的窗口：**{track_diff}**")
        A(f"- `coverage` 不同的窗口：**{cov_diff}**（`max\\|Δcov\\|` = {max_dcov:.10f}）")
        A("")
    else:
        A("## 3. 轨迹 / 覆盖率\n")
        A("基线没有 `track_id` / `coverage` 列（跨臂），本项跳过。\n")

    A("## 4. 按 coverage 分组\n")
    A("> `mean` 对长尾不敏感，必须分组看。\n")
    A("| 分组 | n | max\\|Δp\\| | mean\\|Δp\\| | p90 | >1e-3 |")
    A("|---|---:|---:|---:|---:|---:|")
    for b in bins:
        if b["n"] == 0:
            A(f"| {b['label']} | 0 | — | — | — | — |")
        else:
            A(f"| {b['label']} | {b['n']} | {b['max']:.6f} | "
              f"{b['mean']:.6f} | {b['p90']:.6f} | {b['n_over_1e-3']} |")
    A("")

    A("## 5. 逐视频\n")
    A("| 视频 | base 窗 | new 窗 | max\\|Δp\\| | mean\\|Δp\\| | >1e-3 | 翻转 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(per_video_rows, key=lambda x: -x["max"]):
        flag = " " if r["n_base"] != r["n_new"] else ""
        A(f"| {r['video_id']} | {r['n_base']} | {r['n_new']}{flag} | "
          f"{r['max']:.6f} | {r['mean']:.6f} | {r['n_over_1e-3']} | {r['flips']} |")
    A("")

    # ---- 诊断：gate 外窗口 -------------------------------------------------
    # 板端 cov<0.70 的窗口 prob 必须恒为 0（MIN_COVERAGE 门）。
    # 若基线在这些位置非零，说明两臂的「谁被评分」本身就不同，
    # 这会把 max|Δp| 顶到 1.0 —— 是跨臂结构性差异，不是数值噪声。
    gate_out = [d for d in diffs if d[5] is not None and d[5] < 0.70]
    gate_base_nz = [d for d in gate_out if d[2] > 0]
    gate_new_nz = [d for d in gate_out if d[3] > 0]
    if gate_out:
        A("## 5b. gate 外窗口诊断（`coverage < 0.70`）\n")
        A(f"- 新跑中 gate 外窗口：**{len(gate_out)}**")
        A(f"- 其中新跑 `prob > 0`：**{len(gate_new_nz)}**"
          + ("   门正常" if not gate_new_nz else "   **门失效**"))
        A(f"- 其中基线 `prob > 0`：**{len(gate_base_nz)}**"
          + ("（两臂「谁被评分」不同，是跨臂结构性差异）" if gate_base_nz else ""))
        if gate_base_nz:
            A("")
            A("| 视频 | idx | 基线 p | 新跑 p | 新跑 cov |")
            A("|---|---:|---:|---:|---:|")
            for vid, idx, p0, p1, _dd, cov in sorted(
                    gate_base_nz, key=lambda d: -d[2])[:10]:
                A(f"| {vid} | {idx} | {p0:.6f} | {p1:.6f} | {cov:.2f} |")
        A("")

    if worst and worst[0][4] > 0:
        A("## 6. 偏差最大的 20 个窗口\n")
        A("| 视频 | idx | base p | new p | \\|Δp\\| | coverage |")
        A("|---|---:|---:|---:|---:|---:|")
        for vid, idx, p0, p1, dd, cov in worst:
            cs = f"{cov:.2f}" if cov is not None else "—"
            A(f"| {vid} | {idx} | {p0:.6f} | {p1:.6f} | {dd:.6f} | {cs} |")
        A("")

    if flips:
        A("## 7. 跨 0.5 翻转清单\n")
        A("| 视频 | idx | base p | new p |")
        A("|---|---:|---:|---:|")
        for vid, idx, p0, p1, _dd, _c in flips:
            A(f"| {vid} | {idx} | {p0:.6f} | {p1:.6f} |")
        A("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8")

    print(f"[diff] {rel(new_path)}  vs  {rel(base_path)}")
    print(f"       模式 {'同臂' if same_arm else '跨臂'}   窗口 {n_new} "
          f"(期望 {EXPECT_WINDOWS})")
    print(f"       max|dp| = {stats['max_abs_dp']:.10f}   "
          f"mean = {stats['mean_abs_dp']:.6f}   flips = {len(flips)}")
    if same_arm:
        print(f"       track_diff = {track_diff}   max|dcov| = {max_dcov:.10f}")
    print(f"       判定 **{verdict}**  -> {rel(out_md)}")
    if count_problems:
        for p in count_problems[:10]:
            print(f"         ! {p}")
    return 0 if verdict == "PASS" else 1


# ------------------------------------------------------------------ run

def cmd_run(args) -> int:
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(ROOT / "scripts" / "run_video.py"),
           "--all", "--out", str(raw)]
    if args.videos_dir:
        cmd += ["--videos-dir", args.videos_dir]
    if args.yolo:
        cmd += ["--yolo", args.yolo]
    if args.stgcn:
        cmd += ["--stgcn", args.stgcn]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]

    log = out / "run.log"
    print(f"[run] {' '.join(cmd)}")
    print(f"      日志 -> {rel(log)}   （别用 | tail，管道块缓冲会看不到进度）")
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FATAL] run_video.py 退出码 {proc.returncode}，看 {rel(log)}")
        return 2

    rc = cmd_snap(argparse.Namespace(**{**vars(args), "from_dir": str(raw),
                                        "out": str(out)}))
    if rc != 0:
        return rc

    base = args.base or str(ROOT / "reference" / "g3_windows.csv")
    return cmd_diff(argparse.Namespace(**{
        **vars(args),
        "new": str(out),
        "base": base,
        "out": str(out / "diff.md"),
    }))


# ------------------------------------------------------------------ selftest

def cmd_selftest(args) -> int:
    """不碰 NPU / 不读视频，只验比较逻辑本身。"""
    import tempfile

    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}  got={got!r} want={want!r}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rows = [("v1", 0, 0.1), ("v1", 1, 0.9), ("v2", 0, 0.5)]

        def write(p, rows, with_track):
            with open(p, "w", newline="", encoding="utf-8") as fh:
                fields = ["video_id", "window_idx", "prob_fall"]
                if with_track:
                    fields += ["track_id", "coverage"]
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                for vid, i, p_ in rows:
                    r = {"video_id": vid, "window_idx": i, "prob_fall": repr(p_)}
                    if with_track:
                        r["track_id"] = 1
                        r["coverage"] = 1.0
                    w.writerow(r)

        a = td / "a.csv"
        write(a, rows, with_track=True)

        print("[selftest] 同臂逐位相同")
        b = td / "b.csv"
        write(b, rows, with_track=True)
        pa, _ = load_csv(a)
        pb, _ = load_csv(b)
        ds = [abs(x["prob_fall"] - y["prob_fall"])
              for v in pa for x, y in zip(pa[v], pb[v])]
        check("max|dp|", max(ds), 0.0)

        print("[selftest] 检出 1 ulp 差异")
        c = td / "c.csv"
        write(c, [("v1", 0, 0.1 + 1e-12), ("v1", 1, 0.9), ("v2", 0, 0.5)],
              with_track=True)
        pc, _ = load_csv(c)
        ds = [abs(x["prob_fall"] - y["prob_fall"])
              for v in pa for x, y in zip(pa[v], pc[v])]
        check("max|dp| > 0", max(ds) > 0, True)

        print("[selftest] 检出窗口缺失")
        d = td / "d.csv"
        write(d, rows[:2], with_track=True)
        pd_, _ = load_csv(d)
        check("v2 消失", "v2" in pd_, False)

        print("[selftest] digest 对顺序敏感")
        check("交换后 digest 变",
              prob_digest([0.1, 0.9]) != prob_digest([0.9, 0.1]), True)
        check("同值 digest 稳",
              prob_digest([0.1, 0.9]), prob_digest([0.1, 0.9]))

        print("[selftest] 无 track_id 列 -> 判为跨臂")
        e = td / "e.csv"
        write(e, rows, with_track=False)
        _pe, has_track = load_csv(e)
        check("has_track", has_track, False)

    print()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="逐窗对账（同臂必为 0 / 跨臂只报不判）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="跑一遍 run_video.py + snap + diff")
    p.add_argument("--out", required=True)
    p.add_argument("--videos-dir", default=None)
    p.add_argument("--yolo", default=None)
    p.add_argument("--stgcn", default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--base", default=None,
                   help="默认 reference/g3_windows.csv（跨臂）")
    p.add_argument("--expect-zero", action="store_true", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("snap", help="从结果目录导出 windows.csv + digest.json")
    p.add_argument("--from", dest="from_dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_snap)

    p = sub.add_parser("diff", help="逐窗对比并写 diff.md")
    p.add_argument("--new", required=True, help="结果目录或 windows.csv")
    p.add_argument("--base", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--expect-zero", action="store_true", default=None,
                   help="强制按同臂判（max|dp| 必须恰好 0）")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("selftest", help="只验比较逻辑，不碰 NPU/视频")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
