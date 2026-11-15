"""QC for the MCFD manual fall-interval layer."""
from __future__ import annotations

import argparse
import hashlib
import html
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from src.annotate import mcfd_segments as ms
from src.annotate.qc_thumbnails import make_strip, read_selected, sample_spec
from src.data import paths

EXPECTED_FULL = {
    "rows": 200,
    "videos": 184,
    "segments": 200,
    "annotated_rows": 25,
    "propagated_rows": 175,
    "source_cameras": 23,
}

HASH_TARGETS = [
    paths.ANNOTATIONS / "v2" / "le2i.csv",
    paths.ANNOTATIONS / "v2" / "mcfd.csv",
    paths.ANNOTATIONS / "v2" / "urfd.csv",
    paths.MANIFESTS / "v2" / "videos.csv",
    paths.MANIFESTS / "v2" / "events.csv",
    paths.SPLITS / "v2.json",
    paths.KEYPOINTS_V2 / "extraction_index.csv",
    paths.KEYPOINTS_V2 / "interval_coverage.csv",
    paths.WINDOWS_V2 / "windows.npz",
    paths.WINDOWS_V2 / "windows.csv",
    paths.WINDOWS_V2 / "windows_summary.json",
    paths.PYSKL_V2 / "annotations.pkl",
    paths.PYSKL_V2 / "export_summary.json",
    paths.ANNOTATIONS / "manual" / "interval_annotations.csv",
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buffer.tofile(str(path))


def _row_video_path(row: dict) -> Path:
    chute = ms.to_int(row["chute"])
    cam = ms.to_int(row["cam"])
    return paths.RAW / "mcfd" / "dataset" / "dataset" / f"chute{chute:02d}" / f"cam{cam}.avi"


def render_strips(rows: list[dict], out_dir: Path, index_rows: list[dict]) -> int:
    written = 0
    for row in rows:
        chute = ms.to_int(row["chute"])
        cam = ms.to_int(row["cam"])
        seg = ms.to_int(row["segment_index"])
        start = ms.to_int(row["start_frame"])
        end = ms.to_int(row["end_frame"])
        n_frames = ms.to_int(row["n_frames"])
        fps = ms.to_float(row["fps"]) or 120.0
        spec = sample_spec(start, end, n_frames, fps)
        frames = read_selected(_row_video_path(row), [i for i, _ in spec])
        header = (
            f"{row['video_id']} seg{seg} f{start}-{end} "
            f"({(end - start + 1) / fps:.2f}s, {row['status']})"
        )
        strip = make_strip(frames, spec, start, end, header)
        if strip is None:
            index_rows.append(
                {"kind": "strip", "path": "", "chute": chute, "cam": cam, "segment": seg,
                 "start_frame": start, "end_frame": end, "status": row["status"],
                 "source_video_id": row.get("source_video_id", ""), "notes": "no frames"}
            )
            continue
        rel = Path("strips") / f"chute{chute:02d}_cam{cam}_seg{seg}.jpg"
        write_image(out_dir / rel, strip)
        index_rows.append(
            {"kind": "strip", "path": str(rel).replace("\\", "/"), "chute": chute, "cam": cam,
             "segment": seg, "start_frame": start, "end_frame": end, "status": row["status"],
             "source_video_id": row.get("source_video_id", ""), "notes": ""}
        )
        written += 1
    return written


def _frame_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (60, 60, 60), 1)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(tile, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _sheet(frames_by_cam: dict[int, np.ndarray], chute: int, seg: int, mode: str) -> np.ndarray:
    blank = np.zeros((240, 360, 3), dtype=np.uint8)
    tiles = []
    for cam in range(1, 9):
        frame = frames_by_cam.get(cam)
        tiles.append(
            _frame_tile(frame, f"cam{cam}") if frame is not None
            else _frame_tile(blank, f"cam{cam} missing")
        )
    top = np.hstack(tiles[:4])
    bottom = np.hstack(tiles[4:])
    grid = np.vstack([top, bottom])
    bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (32, 32, 32)
    cv2.putText(
        bar,
        f"chute{chute:02d} seg{seg} {mode}",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, grid])


def render_sheets(rows: list[dict], out_dir: Path, index_rows: list[dict]) -> int:
    by_chute: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        by_chute.setdefault((ms.to_int(row["chute"]), ms.to_int(row["segment_index"])), []).append(row)

    written = 0
    for (chute, seg), items in sorted(by_chute.items()):
        items.sort(key=lambda r: ms.to_int(r["cam"]))
        frames = {"start": {}, "end": {}}
        for row in items:
            cam = ms.to_int(row["cam"])
            start = ms.to_int(row["start_frame"])
            end = ms.to_int(row["end_frame"])
            got = read_selected(_row_video_path(row), [start, end])
            if start in got:
                frames["start"][cam] = got[start]
            if end in got:
                frames["end"][cam] = got[end]
        for mode in ("start", "end"):
            sheet = _sheet(frames[mode], chute, seg, mode)
            rel = Path("sheets") / f"chute{chute:02d}_seg{seg}_{mode}.jpg"
            write_image(out_dir / rel, sheet)
            index_rows.append(
                {"kind": f"sheet_{mode}", "path": str(rel).replace("\\", "/"), "chute": chute,
                 "cam": "", "segment": seg,
                 "start_frame": "", "end_frame": "", "status": "", "source_video_id": "",
                 "notes": f"{len(frames[mode])}/8 cameras"}
            )
            written += 1
    return written


def compare_propagation(
    rows: list[dict], annotated: list[dict], segments, meta, chute: int | None
) -> tuple[bool, str]:
    existing = [
        r for r in rows
        if (r.get("status") or "").strip() == ms.PROPAGATED
        and (chute is None or ms.to_int(r.get("chute")) == chute)
    ]
    expected = ms.propagate_rows(annotated, segments, meta)
    if chute is not None:
        expected = [r for r in expected if ms.to_int(r["chute"]) == chute]

    def key(row: dict):
        return ms.key_of(row)

    existing.sort(key=key)
    expected.sort(key=key)
    fields = [
        "start_frame", "end_frame", "anchor_start_frame", "offset_frames",
        "n_frames", "status", "source_video_id", "notes",
    ]
    if len(existing) != len(expected):
        return False, f"propagated rows {len(existing)} != recomputed {len(expected)}"
    for got, want in zip(existing, expected):
        if key(got) != key(want):
            return False, f"key mismatch {key(got)} != {key(want)}"
        for field in fields:
            if str(got.get(field)) != str(want.get(field)):
                return False, f"{key(got)} field {field}: {got.get(field)!r} != {want.get(field)!r}"
    return True, f"{len(existing)} propagated rows reproduce exactly"


def duration_table(rows: list[dict]) -> list[str]:
    lines = ["| status | n | min s | median s | max s | min f | median f | max f |", "|---|---|---|---|---|---|---|---|"]
    for status in (ms.ANNOTATED, ms.PROPAGATED):
        subset = [r for r in rows if (r.get("status") or "").strip() == status]
        if not subset:
            continue
        frames = [ms.to_int(r["end_frame"]) - ms.to_int(r["start_frame"]) + 1 for r in subset]
        secs = [f / (ms.to_float(r["fps"]) or 120.0) for f, r in zip(frames, subset)]
        lines.append(
            f"| {status} | {len(subset)} | {min(secs):.2f} | {statistics.median(secs):.2f} | "
            f"{max(secs):.2f} | {min(frames)} | {statistics.median(frames):.0f} | {max(frames)} |"
        )
    return lines


def write_html(path: Path, rows: list[dict]) -> None:
    by_chute: dict[int, list[dict]] = {}
    for row in rows:
        by_chute.setdefault(ms.to_int(row["chute"]), []).append(row)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MCFD interval QC</title>",
        "<style>body{font-family:sans-serif;margin:20px;background:#fafafa}"
        "img{max-width:100%;border:1px solid #ccc;margin:4px 0}"
        "details{margin:10px 0;background:#fff;padding:6px}"
        "code{background:#eee;padding:1px 3px}</style></head><body>",
        "<h1>MCFD manual fall intervals — QC</h1>",
        "<p>Each interval: pre/start/q1/mid/q3/end/post. Sheets: 8 cameras at start and end.</p>",
    ]
    for chute, items in sorted(by_chute.items()):
        parts.append(f"<details><summary><b>chute{chute:02d}</b> ({len(items)} intervals)</summary>")
        for row in sorted(items, key=lambda r: (ms.to_int(r["cam"]), ms.to_int(r["segment_index"]))):
            cam = ms.to_int(row["cam"])
            seg = ms.to_int(row["segment_index"])
            strip = f"strips/chute{chute:02d}_cam{cam}_seg{seg}.jpg"
            parts.append(
                f"<p><code>{html.escape(row['video_id'])}</code> seg{seg} "
                f"f{row['start_frame']}..{row['end_frame']} ({row['status']})</p>"
                f"<img src='{strip}' loading='lazy'>"
            )
        for seg in sorted({ms.to_int(r["segment_index"]) for r in items}):
            for mode in ("start", "end"):
                sheet = f"sheets/chute{chute:02d}_seg{seg}_{mode}.jpg"
                parts.append(f"<p>8-camera {mode}: <code>{sheet}</code></p><img src='{sheet}' loading='lazy'>")
        parts.append("</details>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(
    path: Path,
    rows: list[dict],
    stats: dict,
    prop_ok: bool,
    prop_msg: str,
    strip_count: int,
    sheet_count: int,
    missing_images: int,
    chute: int | None,
    hashes: list[tuple[str, str]],
) -> None:
    lines = [
        "# MCFD 跌倒区间人工层 QC",
        "",
        f"- 范围：{'chute ' + f'{chute:02d}' if chute is not None else '全部 23 个 chute'}",
        f"- 人工行：{stats['annotated_rows']}；传播行：{stats['propagated_rows']}；"
        f"总行数：{stats['rows']}；视频数：{stats['videos']}",
        f"- 传播可重算：{'PASS' if prop_ok else 'FAIL'} — {prop_msg}",
        f"- 缩略图条：{strip_count}；8 相机对照图：{sheet_count}；"
        f"缺失帧条：{missing_images}；合计落盘：{strip_count + sheet_count}",
        "",
        "## 区间时长",
        "",
        *duration_table(rows),
        "",
        "## v2 冻结输入 SHA256",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for name, digest in hashes:
        lines.append(f"| {name} | `{digest}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QC the MCFD manual fall-interval layer.")
    parser.add_argument("--input", type=Path, default=paths.MCFD_FALL_INTERVALS)
    parser.add_argument("--output-dir", type=Path, default=paths.REPORTS_MCFD_INTERVALS)
    parser.add_argument("--chute", type=int, default=None)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    all_rows = ms.load_mcfd_rows()
    segments = ms.merged_segments(all_rows)
    meta = ms.load_video_meta()
    rows = ms.load_manual(args.input)
    if args.chute is not None:
        rows = [r for r in rows if ms.to_int(r.get("chute")) == args.chute]
    if not rows:
        print(f"没有可 QC 的行：{args.input}")
        return 1

    stats = ms.validate_manual(rows, segments, meta, require_complete=args.chute is None)
    if args.chute is None:
        for field, expected in EXPECTED_FULL.items():
            got = stats.get(field)
            if field == "source_cameras":
                got = len(got or [])
            if got != expected:
                raise SystemExit(f"full QC: {field}={got}, expected {expected}")

    annotated = [r for r in rows if (r.get("status") or "").strip() == ms.ANNOTATED]
    prop_ok, prop_msg = compare_propagation(rows, annotated, segments, meta, args.chute)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("strips/*.jpg")) + list(out_dir.glob("sheets/*.jpg")):
        stale.unlink()
    index_rows: list[dict] = []
    strip_count = sheet_count = 0
    if not args.no_images:
        strip_count = render_strips(rows, out_dir, index_rows)
        sheet_count = render_sheets(rows, out_dir, index_rows)
    missing_images = sum(1 for r in index_rows if r["kind"] == "strip" and not r["path"])

    index_path = out_dir / "qc_index.csv"
    paths.write_csv(index_path, index_rows)

    entries = sorted(
        list(out_dir.glob("strips/*.jpg")) + list(out_dir.glob("sheets/*.jpg")),
        key=lambda p: str(p),
    )
    written_files = len(entries)
    hashes = [(str(p.relative_to(paths.PROJECT_ROOT)), sha256(p)) for p in HASH_TARGETS]
    hashes.append((str(args.input.relative_to(paths.PROJECT_ROOT)) if args.input.is_relative_to(paths.PROJECT_ROOT) else str(args.input), sha256(args.input)))

    write_html(out_dir / "index.html", rows)
    write_summary(
        out_dir / "summary.md",
        rows,
        stats,
        prop_ok,
        prop_msg,
        strip_count,
        sheet_count,
        missing_images,
        args.chute,
        hashes,
    )

    print(f"rows={stats['rows']} videos={stats['videos']} "
          f"annotated={stats['annotated_rows']} propagated={stats['propagated_rows']}")
    print(f"propagation: {'PASS' if prop_ok else 'FAIL'} — {prop_msg}")
    print(f"strips={strip_count} sheets={sheet_count} files_on_disk={written_files}")
    if args.chute is None and not args.no_images:
        expected_files = 250
        if written_files != expected_files:
            raise SystemExit(f"image count {written_files} != {expected_files}")
    if not prop_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
