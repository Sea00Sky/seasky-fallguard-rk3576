"""Phase 3.0 smoke test: torch/GPU, mmcv + PYSKL import, ST-GCN++ forward,
and loading the exported v2 annotations through PYSKL's PoseDataset pipeline."""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402

PKL = paths.PYSKL_V2 / "annotations.pkl"
CLIP_LEN = 64
EXPECTED_SPLITS = {"train": 3903, "val": 625, "test": 800}


def pipeline(clip_len: int) -> list[dict]:
    return [
        dict(type="PreNormalize2D"),
        dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
        dict(type="UniformSample", clip_len=clip_len, num_clips=1),
        dict(type="PoseDecode"),
        dict(type="FormatGCNInput", num_person=1),
        dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
        dict(type="ToTensor", keys=["keypoint"]),
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures: list[str] = []

    import torch
    import mmcv

    print(f"python    : {sys.version.split()[0]}")
    print(f"torch     : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"mmcv      : {mmcv.__version__}")

    import pyskl
    from pyskl.datasets import PoseDataset
    from pyskl.models import build_model

    print(f"pyskl     : {getattr(pyskl, '__version__', 'unknown')}")

    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is False")
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"cuda dev  : {name} sm_{cap[0]}{cap[1]}")
        try:
            a = torch.randn(256, 256, device="cuda")
            b = (a @ a).sum().item()
            torch.cuda.synchronize()
            print(f"cuda matmul ok (sum={b:.3f})")
        except Exception as exc:  # pragma: no cover - environment specific
            failures.append(f"cuda matmul failed: {exc}")

    model_cfg = dict(
        type="RecognizerGCN",
        backbone=dict(
            type="STGCN",
            gcn_adaptive="init",
            gcn_with_res=True,
            tcn_type="mstcn",
            graph_cfg=dict(layout="coco", mode="spatial"),
        ),
        cls_head=dict(type="GCNHead", num_classes=2, in_channels=256),
    )
    model = build_model(model_cfg)
    model.init_weights()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model     : ST-GCN++ coco/17j params={n_params/1e6:.2f}M")

    # GenSkeFeat appends keypoint_score, so the model consumes (x, y, score).
    x = torch.randn(2, 1, CLIP_LEN, 17, 3)
    with torch.no_grad():
        feat = model.extract_feat(x)
        score = model.cls_head(feat)
    print(f"forward   : feat={tuple(feat.shape)} score={tuple(score.shape)}")
    if tuple(score.shape) != (2, 2):
        failures.append(f"unexpected score shape {tuple(score.shape)}")

    if torch.cuda.is_available():
        model = model.cuda()
        with torch.no_grad():
            score_cuda = model.cls_head(model.extract_feat(x.cuda()))
        torch.cuda.synchronize()
        print(f"cuda fwd  : score={tuple(score_cuda.shape)}")
        model = model.cpu()

    # Annotation-level checks straight from the pickle.
    with open(PKL, "rb") as fh:
        payload = pickle.load(fh)
    annotations = payload["annotations"]
    print(f"pkl       : {len(annotations)} items")
    if len(annotations) != sum(EXPECTED_SPLITS.values()):
        failures.append("pkl item count mismatch")
    for split, expected in EXPECTED_SPLITS.items():
        got = len(payload["split"].get(split, []))
        print(f"split     : {split:5s} {got}")
        if got != expected:
            failures.append(f"split {split} size {got} != {expected}")
    labels = Counter(a["label"] for a in annotations)
    print(f"labels    : {dict(labels)}")
    if set(labels) != {0, 1}:
        failures.append(f"unexpected labels {dict(labels)}")
    bad_shape = [
        a["frame_dir"] for a in annotations
        if a["keypoint"].shape != (1, CLIP_LEN, 17, 2)
        or a["keypoint_score"].shape != (1, CLIP_LEN, 17)
    ]
    if bad_shape:
        failures.append(f"{len(bad_shape)} bad keypoint shapes, e.g. {bad_shape[:3]}")
    nonfinite = [
        a["frame_dir"] for a in annotations
        if not np.isfinite(a["keypoint"]).all()
        or not np.isfinite(a["keypoint_score"]).all()
    ]
    if nonfinite:
        failures.append(f"{len(nonfinite)} non-finite items, e.g. {nonfinite[:3]}")

    # Dataset + pipeline round trip per split.
    for split, expected in EXPECTED_SPLITS.items():
        ds = PoseDataset(
            ann_file=str(PKL), pipeline=pipeline(CLIP_LEN), split=split,
            test_mode=split != "train",
        )
        if len(ds) != expected:
            failures.append(f"dataset {split} len {len(ds)} != {expected}")
        picks = sorted({0, len(ds) // 2, len(ds) - 1})
        for i in picks:
            item = ds[i]
            kp = item["keypoint"]
            shape = tuple(kp.shape)
            arr = kp.numpy() if hasattr(kp, "numpy") else np.asarray(kp)
            # FormatGCNInput emits (num_clips, M, T, V, C); the recognizer
            # takes keypoint[:, 0] to obtain (N, M, T, V, C).
            if shape != (1, 1, CLIP_LEN, 17, 3):
                failures.append(f"{split}[{i}] keypoint shape {shape}")
            if not np.isfinite(arr).all():
                failures.append(f"{split}[{i}] non-finite after pipeline")
            if not (arr.min() >= -1.01 and arr.max() <= 1.01):
                failures.append(
                    f"{split}[{i}] out of [-1,1]: min={arr.min():.3f} max={arr.max():.3f}"
                )
            label = int(np.asarray(item["label"]).reshape(-1)[0])
            if label not in (0, 1):
                failures.append(f"{split}[{i}] bad label {label}")
        print(f"dataset   : {split:5s} len={len(ds)} sampled={picks} "
              f"shape={tuple(ds[picks[0]]['keypoint'].shape)}")

    print("failures :", json.dumps(failures, ensure_ascii=False) if failures else "none")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
