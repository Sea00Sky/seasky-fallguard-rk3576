"""Phase 0 environment and source-data check. Read-only."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures: list[str] = []

    print("== Python / packages ==")
    print("python", sys.version.split()[0])
    for name in ("ultralytics", "torch", "cv2", "numpy", "yaml"):
        try:
            mod = __import__(name)
            print(f"{name:12s}", getattr(mod, "__version__", "ok"))
        except Exception as exc:
            print(f"{name:12s} MISSING ({exc})")
            if name in ("cv2", "numpy", "yaml"):
                failures.append(f"missing package: {name}")

    print("\n== Source datasets ==")
    sources = {
        "le2i": paths.SOURCE_LE2I,
        "mcfd": paths.SOURCE_MCFD,
        "urfd": paths.SOURCE_URFD,
    }
    for name, root in sources.items():
        if not root.exists():
            print(f"{name:6s} MISSING {root}")
            failures.append(f"source missing: {root}")
            continue
        videos = list(paths.iter_files(root, paths.VIDEO_EXTS))
        total = sum(p.stat().st_size for p in paths.iter_files(root))
        print(f"{name:6s} ok  videos={len(videos):4d}  bytes={paths.human_bytes(total)}  {root}")

    print("\n== Destination ==")
    usage = shutil.disk_usage(str(paths.PROJECT_ROOT.drive + "\\"))
    print("project root", paths.PROJECT_ROOT)
    print("free", paths.human_bytes(usage.free), "total", paths.human_bytes(usage.total))
    if usage.free < 20 * 1024**3:
        failures.append("free disk < 20 GB")

    print("\n== Raw layer status ==")
    print("datasets/raw exists:", paths.RAW.exists())
    print("yolo26n-pose.pt cached:", any(paths.WEIGHTS.glob("yolo26n-pose.pt")))

    print("\n== Video readability sample ==")
    samples = []
    for name, root in sources.items():
        vids = sorted(paths.iter_files(root, paths.VIDEO_EXTS))
        if vids:
            samples.append((name, vids[0]))
    for name, path in samples:
        info = paths.video_info(path)
        print(
            f"{name:6s} readable={info['readable']} fps={info['fps']:.2f} "
            f"frames={info['n_frames']} {info['width']}x{info['height']}"
        )
        if not info["readable"]:
            failures.append(f"unreadable sample video: {path}")

    print("\n== Result ==")
    if failures:
        for item in failures:
            print("FAIL:", item)
        print(f"{len(failures)} failure(s)")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
