"""QAT upper-bound experiment: ST-GCN++ with fake-quantisation in the graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "experiments" / "qat"
# torch's zip writer rejects non-ASCII paths on Windows
# "Parent directory E:\毕业设计(...)\experiments\qat does not exist"
# even when it does. Checkpoints go to an ASCII path; everything else can stay.
CKPT_DIR = Path("qat_runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap the number of training windows")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig)

    sys.path.insert(0, str(SOURCE_ROOT))
    from src.train import engine # noqa: E402 (source project, read-only)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    cfg = engine.load_config(None)
    print(f"ann_file: {cfg['data']['ann_file']}", flush=True)

    model = engine.build_recognizer(cfg)
    ckpt = SOURCE_ROOT / "runs" / "stgcnpp_v2" / "best.pth"
    state = torch.load(str(ckpt), map_location="cpu")
    sd = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt.name}: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    if missing:
        print(f"  first missing: {missing[:3]}", flush=True)

    # --- fake-quant insertion -------------------------------------------------
    # Weight observers per-channel (matches rknn's default `quantized_method=
    # channel`); activations per-tensor. `reduce_range=False` keeps the full
    # int8 range, which is what the NPU implements.
    #
    # The qconfig entries MUST be wrapped in `FakeQuantize.with_args`, not
    # passed as bare observers. A bare observer is a valid qconfig value and
    # `prepare_qat` accepts it silently -- it inserts a module named
    # `weight_fake_quant`, but that module is the *observer itself*, so calling
    # it returns its input unchanged. The 23:14 run did exactly
    # this: three epochs converged to acc 0.9867 with zero quantisation noise
    # in the forward pass, i.e. it was a plain fine-tune. Measured afterwards:
    # bare observer -> fake_quant(w) == w, maxdiff 0.0
    # FakeQuantize-wrapped -> maxdiff 1.2e-2
    # `scripts/35` caught it because the QAT model and the fp32 model produced
    # byte-identical logits, which is impossible if fake-quant were live.
    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_tensor_affine,
        reduce_range=False)
    wt = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
        reduce_range=False)
    model.eval()
    model.qconfig = QConfig(activation=act, weight=wt)

    n_wrapped = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.qconfig = model.qconfig
            n_wrapped += 1
    print(f"qconfig set on {n_wrapped} conv/linear modules", flush=True)

    from torch.ao.quantization import prepare_qat
    model = prepare_qat(model.train(), inplace=False)
    print("prepare_qat done", flush=True)

    # Prove the fake-quant is live before spending epochs on it. A bare observer
    # in the qconfig produces a module that looks right and does nothing (see
    # the note above), so the check is empirical: round-tripping a weight
    # through its fake-quant must perturb it.
    probe = model.backbone.gcn[0].gcn.conv.weight_fake_quant
    w = model.backbone.gcn[0].gcn.conv.weight.detach()
    moved = float((probe(w) - w).abs().max())
    print(f"fake-quant liveness: max|fq(w)-w| = {moved:.3e}", flush=True)
    if moved == 0.0:
        raise SystemExit(
            "fake-quant is a no-op -- the qconfig holds a bare observer instead "
            "of FakeQuantize, so this run would be a plain fine-tune")

    # --- brief fine-tune ------------------------------------------------------
    train_set, train_loader = engine.build_loader(cfg, "train")
    print(f"train windows: {len(train_set)}", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    model.to(device)

    def logits_of(x: "torch.Tensor") -> "torch.Tensor":
        """(N, 1, T, V, C) -> (N, num_classes) without leaving torch."""
        if x.dim() == 6:            # (N, nc, M, T, V, C) -> (N, M, T, V, C)
            assert x.shape[1] == 1, f"num_clips must be 1, got {x.shape[1]}"
            x = x[:, 0]
        return model.cls_head(model.extract_feat(x))

    step = 0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in train_loader:
            kp = batch["keypoint"].to(device).float()
            y = batch["label"].to(device).long()
            opt.zero_grad()
            logits = logits_of(kp)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            step += 1
            if args.limit and step >= args.limit:
                break
        print(f"epoch {epoch+1}/{args.epochs}  loss {loss_sum/max(1,total):.4f}  "
              f"acc {correct/max(1,total):.4f}", flush=True)
        if args.limit and step >= args.limit:
            break

    model.eval()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_out = CKPT_DIR / "stgcnpp_qat.pth"
    torch.save({"model": model.state_dict()}, ckpt_out)
    print(f"saved {ckpt_out}", flush=True)
    return 0
    print(f"saved {OUT / 'stgcnpp_qat.pth'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
