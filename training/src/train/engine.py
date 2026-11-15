"""Train and evaluate ST-GCN++ on a window-annotation layer (v2, v3, ...)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from src.data import paths

SPLITS = ("train", "val", "test")
CLIP_LEN_DEFAULT = 64


# --------------------------------------------------------------------------- config


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path else paths.PROJECT_ROOT / "configs" / "train_stgcnpp.yaml"
    with open(path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def layer_of(cfg: dict) -> str:
    """Data layer a run reads, taken from ``datasets/pyskl/<layer>/annotations.pkl``."""
    parts = resolve(cfg["data"]["ann_file"]).parts
    if len(parts) < 3 or parts[-1] != "annotations.pkl" or parts[-3] != "pyskl":
        raise ValueError(
            f"ann_file must be datasets/pyskl/<layer>/annotations.pkl, got {cfg['data']['ann_file']!r}"
        )
    layer = parts[-2]
    if layer not in paths.LAYER_VERSIONS:
        raise ValueError(f"unknown data layer {layer!r}; expected one of {paths.LAYER_VERSIONS}")
    return layer


# ----------------------------------------------------------------------- data setup


def make_pipeline(cfg: dict, split: str = "train") -> list:
    clip_len = int(cfg["data"]["clip_len"])
    num_person = int(cfg["data"]["num_person"])
    pipeline: list = [dict(type="PreNormalize2D")]
    augment_cfg = cfg["data"].get("train_augment") or {}
    if split == "train" and augment_cfg:
        from src.train.augment import SkeletonAugment

        pipeline.append(SkeletonAugment(
            rotate=float(augment_cfg.get("rotate", 0.0)),
            scale=float(augment_cfg.get("scale", 0.0)),
            translate=float(augment_cfg.get("translate", 0.0)),
            jitter=float(augment_cfg.get("jitter", 0.0)),
        ))
    pipeline += [
        dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
        dict(type="UniformSample", clip_len=clip_len, num_clips=1),
        dict(type="PoseDecode"),
        dict(type="FormatGCNInput", num_person=num_person),
        dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
        dict(type="ToTensor", keys=["keypoint"]),
    ]
    return pipeline


def build_dataset(cfg: dict, split: str):
    from pyskl.datasets import PoseDataset

    return PoseDataset(
        ann_file=str(resolve(cfg["data"]["ann_file"])),
        pipeline=make_pipeline(cfg, split),
        split=split,
        test_mode=split != "train",
    )


def build_loader(cfg: dict, split: str) -> tuple[object, DataLoader]:
    dataset = build_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=split == "train",
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dataset, loader


def build_recognizer(cfg: dict):
    from pyskl.models import build_model

    return build_model(cfg["model"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------- metrics


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Metrics for the fall class (label 1)."""
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / max(1, len(labels))
    auc = roc_auc(labels, probs)
    return {
        "n": int(len(labels)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": precision, "recall": recall,
        "f1": f1, "auc": auc, "threshold": float(threshold),
    }


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(probs)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Threshold maximising fall-class F1 on the given split."""
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    candidates = np.unique(np.concatenate([probs, [0.0, 0.5, 1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = binary_metrics(labels, probs, float(t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def _load_manifest_events(split: str, layer: str) -> list[dict]:
    path = paths.PROJECT_ROOT / "datasets" / "manifests" / layer / "events.csv"
    elements = paths.read_csv(path)
    return [r for r in elements if r.get("split") == split]


def event_metrics(
    dataset, labels: np.ndarray, probs: np.ndarray, threshold: float, split: str,
    layer: str = "v2",
) -> dict:
    """Aggregate window predictions per event_id."""
    labels = np.asarray(labels).astype(np.int64)
    pred = (np.asarray(probs) >= threshold).astype(np.int64)
    infos = dataset.video_infos
    by_event: dict[str, list[int]] = defaultdict(list)
    for i, info in enumerate(infos):
        by_event[str(info["event_id"])].append(i)

    manifest = _load_manifest_events(split, layer)
    fall_events = {r["event_id"] for r in manifest if r.get("label") == "fall"}
    nonfall_events = {r["event_id"] for r in manifest if r.get("label") == "non_fall"}

    windowed = set(by_event)
    detected = {
        eid for eid, idxs in by_event.items()
        if any(pred[i] == 1 and labels[i] == 1 for i in idxs)
    }
    false_alarms = {
        eid for eid, idxs in by_event.items()
        if eid in nonfall_events and any(pred[i] == 1 for i in idxs)
    }
    fall_windowed = fall_events & windowed
    return {
        "split": split,
        "fall_events_manifest": len(fall_events),
        "fall_events_with_windows": len(fall_windowed),
        "fall_events_detected": len(detected & fall_events),
        "event_recall_all": (
            len(detected & fall_events) / len(fall_events) if fall_events else float("nan")
        ),
        "event_recall_windowed": (
            len(detected & fall_events) / len(fall_windowed) if fall_windowed else float("nan")
        ),
        "nonfall_events": len(nonfall_events),
        "nonfall_events_with_false_alarm": len(false_alarms),
        "event_false_alarm_rate": (
            len(false_alarms) / len(nonfall_events) if nonfall_events else float("nan")
        ),
    }


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        keypoint = batch["keypoint"].to(device, non_blocking=True)
        # forward_test averages softmax over clips and returns (B, 2).
        out = np.asarray(model(keypoint, return_loss=False), dtype=np.float64)
        probs.append(out[:, 1])
        labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(probs)


# ------------------------------------------------------------------------ training


def class_weights_from(labels: np.ndarray) -> torch.Tensor:
    counts = Counter(int(x) for x in labels)
    total = sum(counts.values())
    return torch.tensor(
        [total / (2.0 * counts.get(c, 1)) for c in (0, 1)], dtype=torch.float32
    )


def dataset_labels(dataset) -> np.ndarray:
    return np.asarray([int(info["label"]) for info in dataset.video_infos], dtype=np.int64)


class SampleWeightedDataset:
    """Wrap a PoseDataset and attach one loss weight per sample."""

    def __init__(self, dataset, weights) -> None:
        self.dataset = dataset
        self.weights = np.asarray(weights, dtype=np.float32)
        if len(self.weights) != len(dataset):
            raise ValueError("weights length does not match dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def video_infos(self):
        return self.dataset.video_infos

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["sample_weight"] = torch.tensor(
            float(self.weights[index]), dtype=torch.float32
        )
        return item


def onset_sample_weights(
    dataset, full_until: float = 0.5, min_weight: float = 0.25, layer: str = "v2",
) -> tuple[np.ndarray, list]:
    """Down-weight positive windows whose fall onset sits late in the window."""
    from src.pose.qc import load_interval_specs

    specs = load_interval_specs(paths.layer(layer)["annotations"])
    weights = np.ones(len(dataset), dtype=np.float32)
    offsets: list = []
    for i, info in enumerate(dataset.video_infos):
        if int(info["label"]) != 1:
            offsets.append(None)
            continue
        start = int(info["start_frame"])
        end = int(info["end_frame"])
        falls = [
            s for s, e, tag in specs.get(info["video_id"], [])
            if tag == "fall" and e > s
        ]
        inside = [s for s in falls if start <= s < end]
        if not inside:
            offsets.append(None)
            continue
        offset = (inside[0] - start) / max(1, end - start)
        offsets.append(offset)
        if offset > full_until:
            span = max(1e-6, 1.0 - full_until)
            weights[i] = 1.0 - (1.0 - min_weight) * (offset - full_until) / span
    return weights, offsets


def train_one_epoch(
    model, loader, optimizer, device, criterion, weighted_criterion=None
) -> float:
    """One epoch with an explicit criterion so class/sample weights can be applied."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        keypoint = batch["keypoint"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        # (N, num_clips, M, T, V, C) -> recognizer takes [:, 0].
        logits = model.cls_head(model.extract_feat(keypoint[:, 0]))
        if weighted_criterion is not None and "sample_weight" in batch:
            sample_weight = batch["sample_weight"].to(device, non_blocking=True).float()
            loss_vec = weighted_criterion(logits, label)
            loss = (loss_vec * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
        else:
            loss = criterion(logits, label)
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss: {float(loss)}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss) * len(label)
        count += len(label)
    return total / max(1, count)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, cfg: dict) -> None:
    # torch's file writer opens paths with the ANSI code page on Windows, which
    # fails for non-ASCII project paths (same class as L-014). Pass a file object.
    with open(path, "wb") as fh:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
        }, fh)


def load_checkpoint(path: Path, device) -> dict:
    with open(path, "rb") as fh:
        return torch.load(fh, map_location=device)


def load_pretrained_weights(model, path: Path, device) -> dict:
    """Initialise the backbone from a PYSKL checkpoint, keeping our 2-class head."""
    # Read through a file object: torch's path handling breaks on non-ASCII
    # project paths on Windows (L-016).
    with open(path, "rb") as fh:
        raw = fh.read()
    sha256 = hashlib.sha256(raw).hexdigest()
    ckpt = torch.load(io.BytesIO(raw), map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    allowed_mismatch = {"cls_head.fc_cls.weight", "cls_head.fc_cls.bias"}
    target = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
    skipped = sorted(set(state) - set(loaded))
    missing, unexpected = model.load_state_dict(loaded, strict=False)

    bad_skipped = set(skipped) - allowed_mismatch
    bad_missing = set(missing) - allowed_mismatch
    if bad_skipped or bad_missing or unexpected:
        raise RuntimeError(
            "pretrained checkpoint mismatch: "
            f"skipped={sorted(bad_skipped)} missing={sorted(bad_missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    print(
        f"pretrained: {path.name} sha256={sha256} "
        f"loaded {len(loaded)}/{len(target)} tensors, reinit={skipped}",
        flush=True,
    )
    return {
        "path": str(path),
        "sha256": sha256,
        "loaded_tensors": len(loaded),
        "reinitialised": skipped,
    }


def run(cfg: dict, args=None) -> dict:
    import torch.nn as nn

    if args is not None:
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.batch_size is not None:
            cfg["data"]["batch_size"] = args.batch_size
        if args.device is not None:
            cfg["train"]["device"] = args.device
        if args.run_dir is not None:
            cfg["output"]["run_dir"] = args.run_dir
        if args.class_weights is not None:
            cfg["train"]["class_weights"] = args.class_weights
        if args.num_workers is not None:
            cfg["data"]["num_workers"] = args.num_workers

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = torch.device(cfg["train"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    run_dir = resolve(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    layer = layer_of(cfg)
    datasets = {s: build_dataset(cfg, s) for s in SPLITS}
    train_labels = dataset_labels(datasets["train"])
    print(f"data layer: {layer} ({cfg['data']['ann_file']})", flush=True)

    onset_cfg = cfg["train"].get("onset_weighting") or {}
    onset_enabled = bool(onset_cfg.get("enabled"))
    if onset_enabled:
        sample_weights, _offsets = onset_sample_weights(
            datasets["train"],
            full_until=float(onset_cfg.get("full_until", 0.5)),
            min_weight=float(onset_cfg.get("min_weight", 0.25)),
            layer=layer,
        )
        pos_w = sample_weights[train_labels == 1]
        print(
            "onset weights:",
            f"pos mean={pos_w.mean():.3f} min={pos_w.min():.3f} "
            f"<1: {int((pos_w < 1.0).sum())}/{len(pos_w)}",
            flush=True,
        )
        datasets["train"] = SampleWeightedDataset(datasets["train"], sample_weights)

    loaders = {}
    for s in SPLITS:
        loader = DataLoader(
            datasets[s],
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=s == "train",
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        loaders[s] = loader
    print(
        "data:",
        {s: len(datasets[s]) for s in SPLITS},
        "fall:",
        {s: int(dataset_labels(datasets[s]).sum()) for s in SPLITS},
        flush=True,
    )

    model = build_recognizer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params on {device}", flush=True)

    pretrained_info = None
    if cfg["train"].get("pretrained"):
        pretrained_info = load_pretrained_weights(model, resolve(cfg["train"]["pretrained"]), device)

    weights = None
    if cfg["train"].get("class_weights") == "balanced":
        weights = class_weights_from(train_labels).to(device)
        print(f"class weights: {weights.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    weighted_criterion = (
        nn.CrossEntropyLoss(weight=weights, reduction="none") if onset_enabled else None
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"]["momentum"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        nesterov=bool(cfg["train"]["nesterov"]),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg["train"]["lr"]) * float(cfg["train"]["eta_min_ratio"]),
    )

    history: list[dict] = []
    metrics_csv = run_dir / "metrics.csv"
    fields = [
        "epoch", "lr", "train_loss", "val_loss", "val_accuracy",
        "val_precision", "val_recall", "val_f1", "val_auc",
    ]
    best_f1, best_epoch, best_val_loss = -1.0, -1, float("inf")
    # Metrics stamped into last.pth must describe the weights in that file, i.e.
    # the epoch that just finished -- not the historical best (L-044). None
    # until the first evaluation; eval_every can skip epochs.
    last_eval: dict | None = None
    threshold = float(cfg["eval"]["threshold"])
    eval_every = max(1, int(cfg["train"]["eval_every"]))
    # Opt-in per-epoch checkpoints, used by the model-selection experiment
    # (scripts/27): "average the last N epochs" and "pick by event-level CV"
    # both need the intermediate weights, which the default run never keeps.
    # Off by default so every existing run's artefact set is unchanged.
    ckpt_cfg = cfg["train"].get("checkpoint_every") or {}
    ckpt_every = int(ckpt_cfg.get("every", 0)) if ckpt_cfg.get("enabled") else 0
    ckpt_dir = run_dir / "epochs"
    if ckpt_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, device, criterion, weighted_criterion
            )
            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step()

            row = {"epoch": epoch, "lr": f"{lr_now:.6g}", "train_loss": f"{train_loss:.6f}"}
            if epoch % eval_every == 0 or epoch == epochs:
                labels, probs = predict(model, loaders["val"], device)
                val = binary_metrics(labels, probs, threshold)
                p_true = np.where(labels == 1, probs, 1.0 - probs)
                val_loss = float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))
                row.update({
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val['accuracy']:.6f}",
                    "val_precision": f"{val['precision']:.6f}",
                    "val_recall": f"{val['recall']:.6f}",
                    "val_f1": f"{val['f1']:.6f}",
                    "val_auc": f"{val['auc']:.6f}",
                })
                last_eval = {**val, "epoch": epoch}
                is_better = (val["f1"], -val_loss) > (best_f1, -best_val_loss)
                if is_better:
                    best_f1, best_epoch, best_val_loss = val["f1"], epoch, val_loss
                    save_checkpoint(run_dir / "best.pth", model, optimizer, epoch, val, cfg)
                print(
                    f"epoch {epoch:02d}/{epochs} lr={lr_now:.4g} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"P={val['precision']:.3f} R={val['recall']:.3f} "
                    f"F1={val['f1']:.3f} AUC={val['auc']:.3f}"
                    f"{' *' if is_better else ''}",
                    flush=True,
                )
            else:
                print(f"epoch {epoch:02d}/{epochs} lr={lr_now:.4g} "
                      f"train_loss={train_loss:.4f}", flush=True)
            writer.writerow(row)
            history.append(row)
            # last.pth holds this epoch's weights, so it carries this epoch's
            # metrics (the most recent evaluation, since eval_every may skip).
            save_checkpoint(run_dir / "last.pth", model, optimizer, epoch,
                            last_eval if last_eval is not None else {}, cfg)
            if ckpt_every > 0 and epoch % ckpt_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch{epoch:03d}.pth", model, optimizer, epoch,
                    last_eval if last_eval is not None else {}, cfg,
                )

    elapsed = time.time() - t0
    checkpoint = load_checkpoint(run_dir / "best.pth", device)
    model.load_state_dict(checkpoint["model"])
    print(f"best epoch {checkpoint['epoch']} val F1={checkpoint['metrics']['f1']:.4f}",
          flush=True)

    results = {
        "history": history,
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_s": elapsed,
        "pretrained": pretrained_info,
    }
    for split in ("val", "test"):
        labels, probs = predict(model, loaders[split], device)
        window = binary_metrics(labels, probs, threshold)
        event = event_metrics(datasets[split], labels, probs, threshold, split, layer)
        t_opt, f1_opt = best_threshold(labels, probs)
        results[split] = {
            "window": window,
            "event": event,
            "best_threshold": {"threshold": t_opt, "f1": f1_opt},
        }
        print(
            f"{split}: acc={window['accuracy']:.3f} P={window['precision']:.3f} "
            f"R={window['recall']:.3f} F1={window['f1']:.3f} AUC={window['auc']:.3f} | "
            f"event recall(all)={event['event_recall_all']:.3f} "
            f"FAR={event['event_false_alarm_rate']:.3f}",
            flush=True,
        )

    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    if args is None or not getattr(args, "no_report", False):
        report_path = resolve(cfg["output"]["report"])
        write_report(report_path, cfg, datasets, results, n_params, elapsed, layer)
    return results


# --------------------------------------------------------------------------- report


def write_report(path, cfg, datasets, results, n_params, elapsed, layer: str = "v2") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = results["val"]["window"]
    t = results["test"]["window"]
    ve = results["val"]["event"]
    te = results["test"]["event"]
    changes = []
    if layer != "v2":
        changes.append(f"数据层 {layer}（v2 以外，窗口与标注不同，不可与 v2 数字混比）")
    pretrained = results.get("pretrained")
    init_desc = "从零训练"
    if pretrained:
        init_desc = (
            f"预训练 {Path(pretrained['path']).name}"
            f"（sha256 {pretrained['sha256'][:12]}…，重初始化 "
            f"{', '.join(pretrained['reinitialised'])}）"
        )
        changes.append(f"初始化 {init_desc}")
    if cfg["train"].get("class_weights"):
        changes.append(f"类权重 {cfg['train']['class_weights']}")
    base_lr = 0.02
    if float(cfg["train"]["lr"]) != base_lr:
        changes.append(f"lr {base_lr} → {cfg['train']['lr']}")
    onset = cfg["train"].get("onset_weighting") or {}
    if onset.get("enabled"):
        changes.append(
            f"onset 位置降权（≤{onset.get('full_until')} 全权重，降至 "
            f"{onset.get('min_weight')}）"
        )
    aug = cfg["data"].get("train_augment") or {}
    aug_desc = "无"
    if aug:
        aug_desc = (
            f"rotate±{aug.get('rotate', 0.0)} / scale±{aug.get('scale', 0.0)} / "
            f"translate±{aug.get('translate', 0.0)} / jitter{aug.get('jitter', 0.0)}"
        )
        changes.append(f"数据增强 {aug_desc}")
    lines = [
        f"# ST-GCN++ 训练与评估（{layer} 窗口）",
        "",
        "## 配置",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 数据层 | `{layer}`（`{cfg['data']['ann_file']}`） |",
        f"| 输入 | {cfg['data']['clip_len']} 帧 × 17 关节 × (x, y, score) |",
        f"| 模型 | ST-GCN++，{n_params / 1e6:.2f}M 参数 |",
        f"| 初始化 | {init_desc} |",
        f"| batch / lr | {cfg['data']['batch_size']} / {cfg['train']['lr']} ({cfg['train']['optimizer']}) |",
        f"| epochs | {cfg['train']['epochs']}（cosine，最优 epoch {results['best_epoch']}） |",
        f"| 类权重 | {cfg['train'].get('class_weights') or '无'} |",
        f"| 数据增强 | {aug_desc} |",
        f"| 划分 | train {len(datasets['train'])} / val {len(datasets['val'])} / test {len(datasets['test'])} 窗口 |",
        f"| 训练耗时 | {elapsed / 60:.1f} min |",
        "",
        "## 窗口级（阈值 0.5，fall 为正类）",
        "",
        "| split | n | 正样本 | acc | precision | recall | F1 | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in (("val", w), ("test", t)):
        labels = dataset_labels(datasets[name])
        lines.append(
            f"| {name} | {m['n']} | {int(labels.sum())} | {m['accuracy']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} |"
        )
    lines += [
        "",
        "## 事件级（同一 event_id 任一窗口为 fall 即判阳性）",
        "",
        "| split | fall 事件 | 有窗口 | 检出 | recall(全部) | recall(有窗口) | non_fall 事件 | 误报事件 | FAR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, e in (("val", ve), ("test", te)):
        lines.append(
            f"| {name} | {e['fall_events_manifest']} | {e['fall_events_with_windows']} | "
            f"{e['fall_events_detected']} | {e['event_recall_all']:.3f} | "
            f"{e['event_recall_windowed']:.3f} | {e['nonfall_events']} | "
            f"{e['nonfall_events_with_false_alarm']} | {e['event_false_alarm_rate']:.3f} |"
        )
    lines += [
        "",
        "## 备注",
        "",
        "- 正样本按 D-017 onset 规则；每个跌倒事件的可用窗口只有起始附近数个，窗口级 F1 不等于事件召回。",
        "- 事件召回分母按本层 manifest 的 fall 事件计，含被 D-018 覆盖率过滤到没有任何窗口的事件"
        f"（test {te['fall_events_manifest'] - te['fall_events_with_windows']} 个、"
        f"val {ve['fall_events_manifest'] - ve['fall_events_with_windows']} 个），见 `event_recall_all`。",
        f"- 相对 D-023 基线的变更：{'；'.join(changes) or '无（基线复跑）'}。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
