"""Import the RK3576 board-recorded videos into the read-only raw layer."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402

# 板端录制的原始视频不入库；用 FG_BOARD_VIDEO_DIR 指向本机的 data_board_videos。
SOURCE = Path(os.environ.get("FG_BOARD_VIDEO_DIR", "data_board_videos"))
TARGET = paths.RAW / "board"
MANIFEST = paths.MANIFESTS / "board" / "videos.csv"

GROUP_MAP = {
    "done": "done",
    "整体测试视频": "integral_test",
    "蹲姿实录视频": "squat",
    "walk": "walk",
}

FIELDS = [
    "video_id", "dataset", "group", "event_id", "relative_path",
    "fps", "n_frames", "width", "height", "readable", "use_half", "sha256", "notes",
]


def probe(path: Path) -> dict:
    """Decode every frame once to get the exact count used by extract_video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": 0.0, "n_frames": 0, "width": 0, "height": 0, "readable": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return {"fps": round(fps, 6), "n_frames": n, "width": width, "height": height,
            "readable": n > 0}


def copy_verified(src: Path, dst: Path) -> str:
    src_hash = paths.sha256_file(src)
    if dst.exists():
        if paths.sha256_file(dst) != src_hash:
            raise RuntimeError(f"existing copy hash differs: {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        if paths.sha256_file(dst) != src_hash:
            raise RuntimeError(f"copy verification failed: {dst}")
    return src_hash


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not SOURCE.is_dir():
        print(f"source missing: {SOURCE}")
        return 1

    rows: list[dict] = []
    copied = 0
    for src in sorted(SOURCE.rglob("*")):
        if not src.is_file():
            continue
        group_src = src.relative_to(SOURCE).parts[0]
        group = GROUP_MAP.get(group_src, group_src)
        rel = Path("board") / group / src.name
        dst = paths.RAW / rel

        if src.suffix.lower() == ".mp4":
            if not dst.exists():
                copied += 1
            digest = copy_verified(src, dst)
            meta = probe(dst)
            stem = src.stem
            video_id = f"board:{group}:{stem}"
            note = ""
            if meta["n_frames"] < 20:
                note = f"only {meta['n_frames']} frames; unusable"
            rows.append({
                "video_id": video_id,
                "dataset": "board",
                "group": group,
                "event_id": video_id,
                "relative_path": rel.as_posix(),
                "fps": f"{meta['fps']:.6f}",
                "n_frames": meta["n_frames"],
                "width": meta["width"],
                "height": meta["height"],
                "readable": str(meta["readable"]),
                "use_half": "",
                "sha256": digest,
                "notes": note,
            })
            print(f"[import] {rel.as_posix()} frames={meta['n_frames']} "
                  f"{meta['width']}x{meta['height']}@{meta['fps']:.0f}")
        elif src.suffix.lower() == ".json":
            digest = copy_verified(src, dst)
            print(f"[import] {rel.as_posix()} (label file, sha256={digest[:12]}…)")

    if not rows:
        print("no mp4 found")
        return 1
    paths.write_csv(MANIFEST, rows, FIELDS)
    print(f"\nvideos: {len(rows)} (new copies: {copied})")
    print(f"manifest: {MANIFEST}")
    missing = [r["video_id"] for r in rows if r["readable"] != "True"]
    if missing:
        print(f"unreadable: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
