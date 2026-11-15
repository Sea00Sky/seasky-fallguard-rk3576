"""Extract keypoints for manifest videos with YOLO26n-pose."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.pose import extract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "le2i", "mcfd", "urfd"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="max videos per dataset; 0 = all")
    parser.add_argument("--overwrite", action="store_true", help="re-extract existing npz")
    parser.add_argument("--device", default="", help="override config device, e.g. 0 or cpu")
    parser.add_argument("--batch", type=int, default=0, help="override config batch_frames")
    parser.add_argument("--config", default="", help="alternative pose.yaml")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = extract.load_config(Path(args.config) if args.config else None)
    if args.device:
        cfg.device = args.device
    if args.batch:
        cfg.batch_frames = args.batch

    if not cfg.model_path.exists():
        print(f"model missing: {cfg.model_path}")
        return 1
    model_sha = paths.sha256_file(cfg.model_path)
    print(f"model: {cfg.model_path} sha256={model_sha}")
    print(f"device={cfg.device} imgsz={cfg.imgsz} conf={cfg.conf} batch={cfg.batch_frames}")

    rows = extract.select_rows(args.dataset, args.limit)
    if not rows:
        print("no videos selected")
        return 1
    print(f"videos: {len(rows)} (dataset={args.dataset}, limit={args.limit})")

    # Imported lazily so QC code paths do not need torch/ultralytics.
    from ultralytics import YOLO

    model = YOLO(str(cfg.model_path))
    model_sha = paths.sha256_file(cfg.model_path)
    print(f"loaded task={model.task} kpt_shape={getattr(model.model, 'kpt_shape', None)}")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep rows from earlier subset runs; a smoke/Dataset run must not erase
    # the rest of the index.
    existing: dict[str, dict] = {}
    if cfg.index_path.exists():
        for old in paths.read_csv(cfg.index_path):
            existing[old.get("video_id", "")] = old
    all_order = [r["video_id"] for r in extract.select_rows("all", 0)]
    results: list[dict] = []
    started = datetime.now()
    for i, row in enumerate(rows, 1):
        try:
            result = extract.extract_video(
                model, cfg, row, model_sha, low_conf=args.overwrite
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = {
                "video_id": row["video_id"], "dataset": row["dataset"],
                "event_id": row["event_id"], "split": row["split"], "label": row["label"],
                "n_frames_expected": row["n_frames"], "n_frames_read": 0,
                "width": "", "height": "", "crop_offset_x": "",
                "frames_with_person": 0, "selected_track_id": -1,
                "selected_track_frames": 0, "selected_coverage": 0.0,
                "mean_kpt_score": 0.0, "median_kpt_score": 0.0, "low_score_ratio": 1.0,
                "elapsed_s": 0.0, "status": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "model_sha256": model_sha, "npz_path": "",
            }
        existing[result["video_id"]] = result
        results.append(result)
        ordered = [existing[v] for v in all_order if v in existing]
        ordered += [v for k, v in existing.items() if k not in set(all_order)]
        paths.write_csv(cfg.index_path, ordered, fieldnames=extract.INDEX_FIELDS)
        print(
            f"[{i}/{len(rows)}] {result['dataset']} {result['video_id']} "
            f"frames={result['n_frames_read']}/{result['n_frames_expected']} "
            f"person={result['frames_with_person']} "
            f"cov={result['selected_coverage']:.2f} "
            f"score={result['median_kpt_score']:.2f} "
            f"{result['status']} {result['error']}",
            flush=True,
        )

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    elapsed = (datetime.now() - started).total_seconds()
    print(f"done in {elapsed:.1f}s; status={counts}")
    print(f"index: {cfg.index_path}")
    return 0 if all(r["status"] in ("ok", "resumed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
