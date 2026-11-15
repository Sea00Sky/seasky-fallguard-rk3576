#!/usr/bin/env python3
"""把视频容器的 fps 重定时到真实采集帧率（只改时间戳，不重编码）。"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = ("done", "整体测试视频", "蹲姿实录视频")


def container_fps(path: Path) -> float:
    """用 PyAV 读容器帧率 —— 与重封装同一个依赖，不额外引入 cv2。"""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        rate = st.average_rate or st.guessed_rate or st.base_rate
        return float(rate) if rate else 0.0


def _frame_hash(frame) -> str:
    return hashlib.sha256(frame.to_ndarray(format="gray").tobytes()).hexdigest()


def remux_scaled(src: Path, dst: Path, scale: float) -> None:
    """把 src 重封装到 dst，时间戳乘 scale。返回后 dst 可解码。"""
    import av

    with av.open(str(src)) as inp, av.open(str(dst), "w", format="mp4") as out:
        out.metadata.update(inp.metadata)
        vmap = {}
        for st in inp.streams:
            if st.type != "video":
                continue
            ost = out.add_stream_from_template(st)
            vmap[st.index] = (st, ost)
        if not vmap:
            raise RuntimeError(f"没有视频流: {src}")

        for pkt in inp.demux():
            if pkt.stream.index not in vmap:
                continue
            if pkt.pts is None and pkt.dts is None:
                continue
            ist, ost = vmap[pkt.stream.index]
            tb_in = ist.time_base
            tb_out = ost.time_base or tb_in

            def _cv(t):
                if t is None:
                    return None
                return int(round(float(t * tb_in) * scale / float(tb_out)))

            pkt.stream = ost
            pkt.pts = _cv(pkt.pts)
            pkt.dts = _cv(pkt.dts)
            if pkt.duration is not None:
                pkt.duration = _cv(pkt.duration)
            out.mux(pkt)


def verify_frames(a: Path, b: Path, step: int) -> tuple[bool, str]:
    """逐帧解码对比：帧数必须相等；step>1 时只比对每 step 帧（第 0 帧永远比）。"""
    import av

    with av.open(str(a)) as ca, av.open(str(b)) as cb:
        da, db = ca.decode(video=0), cb.decode(video=0)
        n = 0
        while True:
            fa, fb = next(da, None), next(db, None)
            if fa is None or fb is None:
                if fa is None and fb is None:
                    return True, f"{n} 帧一致（每 {step} 帧抽 1 帧比对）"
                return False, f"帧数不同：{n} 帧后一侧先结束"
            if step <= 1 or n % step == 0:
                if _frame_hash(fa) != _frame_hash(fb):
                    return False, f"帧内容不一致: #{n}"
            n += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-fps", type=float, required=True,
                    help="真实采集帧率，例如 20")
    ap.add_argument("--src-root", default=str(ROOT / "videos"))
    ap.add_argument("--dest-root", default=None,
                    help="输出目录；默认 None = 原地替换（会先备份）")
    ap.add_argument("--backup-root", default=str(ROOT / "videos_original_30fps"),
                    help="原地替换时的原件备份目录")
    ap.add_argument("--verify", choices=["none", "sample", "full"], default="sample",
                    help="sample=每 17 帧抽 1 帧；full=逐帧")
    ap.add_argument("--only", default=None, help="只处理文件名包含该子串的片段")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dest_root = Path(args.dest_root) if args.dest_root else None
    backup_root = Path(args.backup_root)
    if args.target_fps <= 0:
        ap.error("--target-fps 必须 > 0")

    files: list[Path] = []
    for d in DATASET_DIRS:
        files += sorted((src_root / d).glob("*.mp4"))
    if args.only:
        files = [p for p in files if args.only in p.name]
    if not files:
        print(f"[ERR] {src_root} 下没找到视频", file=sys.stderr)
        return 2

    print(f"[retime] {len(files)} 个文件，目标 {args.target_fps:g} fps，"
          f"verify={args.verify}")
    failed = 0
    for src in files:
        old = container_fps(src)
        if old <= 0:
            print(f"[ERR] {src.name}: 读不到容器 fps，跳过")
            failed += 1
            continue
        scale = old / args.target_fps
        rel = src.relative_to(src_root)
        dst = (dest_root / rel) if dest_root else src
        print(f"\n== {rel}  {old:g} -> {args.target_fps:g} fps  (x{scale:.4f})")

        if abs(scale - 1.0) < 1e-6:
            print("   已经是目标帧率，跳过")
            continue
        if args.dry_run:
            print("   [dry-run] 不做修改")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dest_root is None:
            bak = backup_root / rel
            bak.parent.mkdir(parents=True, exist_ok=True)
            if not bak.exists():
                shutil.copy2(src, bak)
                print(f"   备份 -> {bak}")
            else:
                print(f"   备份已存在，保留: {bak}")

        tmp = dst.with_name(dst.stem + ".retime.tmp.mp4")
        if tmp.exists():
            tmp.unlink()
        remux_scaled(src, tmp, scale)

        new = container_fps(tmp)
        if abs(new - args.target_fps) > 0.01:
            print(f"   [ERR] 重定时后容器 fps={new:g}，不是目标值；保留临时文件 {tmp}")
            failed += 1
            continue

        if args.verify != "none":
            step = 1 if args.verify == "full" else 17
            ok, msg = verify_frames(src, tmp, step)
            print(f"   校验({args.verify}): {msg}")
            if not ok:
                print(f"   [ERR] 校验失败；原件未覆盖，临时文件 {tmp}")
                failed += 1
                continue

        if dest_root is None:
            st = src.stat()
            os.replace(tmp, src)
            os.utime(src, (st.st_atime, st.st_mtime))
            print(f"   已替换原文件（容器 {new:g} fps）")
        else:
            os.replace(tmp, dst)
            print(f"   写出 -> {dst}（容器 {new:g} fps）")

    print(f"\n[retime] 完成：{len(files) - failed}/{len(files)} 成功")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
