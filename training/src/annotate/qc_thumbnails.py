"""Build contact sheets for manually annotated fall intervals."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import cv2
import numpy as np

from src.data.paths import ANNOTATIONS, MANIFESTS, RAW, REPORTS, read_csv, write_csv

MANUAL_CSV = ANNOTATIONS / "manual" / "interval_annotations.csv"
DEFAULT_OUTPUT = REPORTS / "qc_manual_intervals"
DEFAULT_REPORT = REPORTS / "manual_annotation_qc.md"

# Frames outside the interval are drawn with a gray border; frames inside are
# green. The margin before/after is 0.4 s (or 5 frames, whichever is larger).
MARGIN_SECONDS = 0.4


def _safe_name(video_id: str) -> str:
    keep = []
    for ch in video_id:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep) + ".jpg"


def sample_spec(start: int, end: int, n_frames: int, fps: float) -> list[tuple[int, str]]:
    margin = max(5, int(round(MARGIN_SECONDS * (fps if fps > 0 else 25.0))))
    candidates = [
        (max(0, start - margin), "pre"),
        (start, "start"),
        (start + (end - start) // 4, "q1"),
        ((start + end) // 2, "mid"),
        (start + 3 * (end - start) // 4, "q3"),
        (end, "end"),
        (min(n_frames - 1, end + margin), "post"),
    ]
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index, role in candidates:
        index = max(0, min(n_frames - 1, index))
        if index in seen:
            continue
        seen.add(index)
        out.append((index, role))
    return out


def read_selected(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    found: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return found
    index = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                found[index] = frame
                wanted.discard(index)
            index += 1
    finally:
        cap.release()
    return found


def make_strip(
    frames: dict[int, np.ndarray],
    spec: list[tuple[int, str]],
    start: int,
    end: int,
    header: str,
) -> np.ndarray | None:
    tiles = []
    first_size: tuple[int, int] | None = None
    for index, role in spec:
        frame = frames.get(index)
        if frame is None:
            continue
        if first_size is None:
            first_size = (frame.shape[1], frame.shape[0])
        else:
            frame = cv2.resize(frame, first_size, interpolation=cv2.INTER_AREA)
        tile = frame.copy()
        inside = start <= index <= end
        color = (0, 190, 0) if inside else (190, 190, 190)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 3)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"f{index} {role}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return None

    strip = np.hstack(tiles)
    bar = np.zeros((34, strip.shape[1], 3), dtype=np.uint8)
    bar[:] = (32, 32, 32)
    cv2.putText(
        bar,
        header,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, strip])


def build(dataset: str, output_dir: Path, report_path: Path) -> dict:
    rows = read_csv(MANUAL_CSV)
    manifest = {r["video_id"]: r for r in read_csv(MANIFESTS / "videos.csv")}

    selected = []
    for row in rows:
        if row.get("dataset") != dataset or row.get("status") != "annotated":
            continue
        info = manifest.get(row["video_id"])
        if info is None:
            print(f"[warn] not in manifest: {row['video_id']}")
            continue
        selected.append((row, info))

    selected.sort(key=lambda item: (item[1].get("event_id", ""), item[0]["video_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    cards = []
    for row, info in selected:
        video_id = row["video_id"]
        start = int(float(row["start_frame"]))
        end = int(float(row["end_frame"]))
        n_frames = int(float(row["n_frames"]))
        fps = float(row["fps"] or info.get("fps") or 25.0)
        seconds = (end - start + 1) / fps if fps > 0 else 0.0
        spec = sample_spec(start, end, n_frames, fps)
        frames = read_selected(RAW / info["relative_path"], [i for i, _ in spec])
        header = (
            f"{video_id} | frames {start}-{end} ({end - start + 1}) | "
            f"{seconds:.2f}s | {info.get('annotation_status', '')}"
        )
        strip = make_strip(frames, spec, start, end, header)
        if strip is None:
            print(f"[warn] no frames decoded: {video_id}")
            continue

        image_name = _safe_name(video_id)
        ok, buffer = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print(f"[warn] JPEG encode failed: {video_id}")
            continue
        # cv2.imwrite fails on some Windows Chinese paths; write bytes instead.
        (output_dir / image_name).write_bytes(buffer.tobytes())
        scene = info.get("event_id", "").split(":")[1] if ":" in info.get("event_id", "") else ""
        coverage = (end - start + 1) / n_frames if n_frames > 0 else 0.0
        flags = []
        if seconds >= 2.0:
            flags.append("long>=2s")
        if coverage >= 0.5:
            flags.append("coverage>=50%")
        index_rows.append(
            {
                "video_id": video_id,
                "scene": scene,
                "original_status": info.get("annotation_status", ""),
                "start_frame": start,
                "end_frame": end,
                "interval_frames": end - start + 1,
                "n_frames": n_frames,
                "seconds": f"{seconds:.3f}",
                "coverage": f"{coverage:.3f}",
                "flags": ";".join(flags),
                "image": image_name,
            }
        )
        cards.append(
            {
                "video_id": video_id,
                "scene": scene,
                "original_status": info.get("annotation_status", ""),
                "start": start,
                "end": end,
                "frames": end - start + 1,
                "n_frames": n_frames,
                "coverage": coverage,
                "flags": flags,
                "seconds": seconds,
                "image": image_name,
            }
        )

    write_csv(output_dir / "qc_index.csv", index_rows)
    _write_html(output_dir / "index.html", cards, dataset)
    _write_markdown(report_path, cards, dataset, output_dir)
    return {
        "videos": len(cards),
        "dataset": dataset,
        "output_dir": str(output_dir),
        "report": str(report_path),
    }


def _write_html(path: Path, cards: list[dict], dataset: str) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        "<title>Manual interval QC</title>",
        "<style>",
        "body{font-family:'Microsoft YaHei UI',system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}",
        "h1{font-size:20px}",
        ".legend span{display:inline-block;margin-right:16px}",
        ".sw{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:14px 0}",
        ".meta{font-size:13px;color:#444;margin-bottom:6px}",
        "img{width:100%;max-width:1900px;height:auto;display:block}",
        "</style></head><body>",
        f"<h1>Manual fall interval QC — {html.escape(dataset)} ({len(cards)} videos)</h1>",
        '<div class="legend">'
        '<span><i class="sw" style="background:#00be00"></i>帧在标注区间内</span>'
        '<span><i class="sw" style="background:#bebebe"></i>帧在区间外</span>'
        "</div>",
        "<p>每条约 7 帧：pre / start / q1 / mid / q3 / end / post。"
        "重点看 start 之前是否已经失衡、end 时是否已经稳定贴地，以及中间是否真的发生了跌倒。</p>",
    ]
    for card in cards:
        flag_html = ""
        if card["flags"]:
            flag_html = (
                ' <span style="color:#b00000;font-weight:bold">⚠ '
                + html.escape(", ".join(card["flags"]))
                + "</span>"
            )
        parts.append('<div class="card">')
        parts.append(
            '<div class="meta">'
            f"<b>{html.escape(card['video_id'])}</b> · {html.escape(card['scene'])} · "
            f"原始状态 {html.escape(card['original_status'])} · "
            f"帧 {card['start']}–{card['end']}（{card['frames']} 帧 = {card['seconds']:.2f} s）"
            f"{flag_html}</div>"
        )
        parts.append(f'<img src="{html.escape(card["image"])}" loading="lazy">')
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_markdown(path: Path, cards: list[dict], dataset: str, output_dir: Path) -> None:
    rel_dir = output_dir.relative_to(REPORTS).as_posix()
    lines = [
        f"# 人工标注 QC — {dataset}",
        "",
        f"- 视频数：{len(cards)}",
        f"- 缩略图索引：`{rel_dir}/index.html`",
        f"- 机器可读索引：`{rel_dir}/qc_index.csv`",
        "- 每条约 7 帧：pre / start / q1 / mid / q3 / end / post；绿框 = 在标注区间内。",
        "",
        "## 复核要点",
        "",
        "1. start 前一帧还不应明显失衡；start 应是失控下落的起点。",
        "2. end 应是稳定贴地，不应包含长时间躺卧或爬起。",
        "3. 中间帧应能看到躯干/髋部下落，而不是坐下、弯腰或主动躺下。",
        "",
        "## 区间长度",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    seconds = [c["seconds"] for c in cards]
    frames = [c["frames"] for c in cards]
    if cards:
        lines += [
            f"| 数量 | {len(cards)} |",
            f"| 帧数 min / median / max | {min(frames)} / {sorted(frames)[len(frames)//2]} / {max(frames)} |",
            f"| 秒数 min / median / max | {min(seconds):.2f} / {sorted(seconds)[len(seconds)//2]:.2f} / {max(seconds):.2f} |",
        ]
    else:
        lines.append("| 数量 | 0 |")

    lines += ["", "## 重点复核", ""]
    flagged = [card for card in cards if card["flags"]]
    if flagged:
        lines += ["| video_id | 区间 | 帧数 | 秒 | 占比 | 标记 |", "|---|---|---|---|---|---|"]
        for card in flagged:
            lines.append(
                f"| {card['video_id']} | {card['start']}–{card['end']} | {card['frames']} | "
                f"{card['seconds']:.2f} | {card['coverage']:.0%} | {', '.join(card['flags'])} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## 索引", "", "| video_id | 场景 | 原始状态 | 区间 | 帧数 | 秒 |", "|---|---|---|---|---|---|"]
    for card in cards:
        lines.append(
            f"| {card['video_id']} | {card['scene']} | {card['original_status']} | "
            f"{card['start']}–{card['end']} | {card['frames']} | {card['seconds']:.2f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build QC contact sheets for manual fall intervals.")
    parser.add_argument("--dataset", default="le2i", help="dataset to review (default: le2i)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _force_utf8_stdout() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    if not MANUAL_CSV.exists():
        print(f"missing manual annotations: {MANUAL_CSV}")
        return 1
    result = build(args.dataset, args.output_dir, args.report)
    print(
        f"QC strips: {result['videos']} videos -> {result['output_dir']}  "
        f"report -> {result['report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
