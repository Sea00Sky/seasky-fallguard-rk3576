"""QAT for YOLO26n-pose by distillation, because there are no pose labels."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import paths  # noqa: E402

# ASCII checkpoint dir: torch's zip writer rejects the non-ASCII project path
# on Windows (same workaround as scripts/33).
CKPT_DIR = Path("qat_runs")
OUT = PROJECT_ROOT / "experiments" / "qat"


def build_raw_model(frozen: bool):
    """The model in the *export* form used for deployment."""
    import torch

    from src.deploy.yolo_export import YOLOPoseRaw

    ck = torch.load(str(paths.YOLO_PT), map_location="cpu", weights_only=False)
    model = ck["model"].float().eval()
    raw = YOLOPoseRaw(model, end2end_branch=True).eval()
    if frozen:
        for p in raw.parameters():
            p.requires_grad_(False)
    return raw


def insert_fake_quant(model):
    """Attach a QConfig to every Conv2d/Linear and run `prepare_qat`."""
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=__import__("torch").qint8, qscheme=__import__("torch").per_tensor_affine,
        reduce_range=False)
    wt = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=__import__("torch").qint8,
        qscheme=__import__("torch").per_channel_symmetric,
        reduce_range=False)
    qconfig = QConfig(activation=act, weight=wt)

    n = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.qconfig = qconfig
            n += 1
    print(f"qconfig set on {n} conv/linear modules", flush=True)
    prepared = prepare_qat(model.train(), inplace=False)

    # ## torch 2.11 freezes every parameter in `prepare_qat`
    #
    # Measured on this build: after `prepare_qat`, the output has
    # `requires_grad=False` and `grad_fn=None`, and `sum(p.requires_grad)` over
    # all parameters is 0. The first `loss.backward()` then fails with
    # "element 0 of tensors does not require grad and does not have a grad_fn",
    # which reads like a model/graph problem but is this.
    #
    # Whether that is a regression or an intentional default (the API is
    # deprecated in favour of torchao pt2e) does not matter here -- QAT needs
    # the weights trainable, so they are re-enabled explicitly. The observers
    # are left alone: they are buffers, not parameters, and re-enabling them
    # would be wrong.
    n_trainable = 0
    for p in prepared.parameters():
        p.requires_grad_(True)
        n_trainable += 1
    print(f"re-enabled gradients on {n_trainable} parameter tensors "
          f"(prepare_qat froze them)", flush=True)
    return prepared


def channel_balanced_loss(y_student, y_teacher):
    """MSE per channel group, normalised by that group's own scale."""
    import torch.nn.functional as F

    groups = [(0, 4), (4, 5), (5, y_teacher.shape[1])]
    total = 0.0
    for lo, hi in groups:
        a = y_student[:, lo:hi]
        b = y_teacher[:, lo:hi]
        # Per-group normaliser from the teacher itself, so the weighting
        # follows the data rather than a constant someone tuned.
        scale = b.detach().abs().mean().clamp_min(1e-6)
        total = total + F.mse_loss(a / scale, b / scale)
    return total / len(groups)


def check_liveness(model) -> float:
    """Fake-quant must actually perturb a weight, or this is a plain fine-tune."""
    import torch.nn as nn

    for _name, mod in model.named_modules():
        fq = getattr(mod, "weight_fake_quant", None)
        if fq is not None and hasattr(mod, "weight") and mod.weight is not None:
            w = mod.weight.detach()
            moved = float((fq(w) - w).abs().max())
            print(f"fake-quant liveness: max|fq(w)-w| = {moved:.3e}", flush=True)
            if moved == 0.0:
                raise SystemExit(
                    "fake-quant is a no-op -- the qconfig holds a bare observer "
                    "instead of FakeQuantize, so this run would be a plain "
                    "fine-tune (see scripts/33)")
            return moved
    raise SystemExit("no weight_fake_quant module found; prepare_qat did nothing")


def load_frames(n: int, seed: int = 0) -> list[np.ndarray]:
    """`n` preprocessed training-split frames, deterministic order."""
    import cv2

    from src.deploy import spec
    from src.deploy.yolo_export import preprocess

    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_V2)
            if r.get("readable") == "True" and r.get("split") == "train"]
    if not rows:
        raise SystemExit(f"no train-split rows in {paths.SOURCE_MANIFEST_V2}")
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    # One frame per video, cycling through the list until `n` is reached, so the
    # sample spans datasets rather than exhausting one folder.
    order = list(rng.permutation(len(rows)))
    i = 0
    while len(out) < n:
        row = rows[order[i % len(order)]]
        i += 1
        path = paths.SOURCE_RAW_V2 / row["relative_path"]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        total = int(row["n_frames"])
        pos = int(rng.integers(0, max(1, total)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        chw, _ = preprocess(frame, imgsz=spec.IMGSZ)
        # Training videos come in several resolutions (320x240 le2i, larger
        # MCFD/URFD), and `preprocess(auto=True)` is only the identity for
        # 640x480. The graph is static at (1,3,480,640), so anything else is
        # letterboxed to the model's input -- the same thing `scripts/20`'s
        # calibration builder does for the same reason. Skipping this fails
        # later with "all input arrays must have the same shape".
        if chw.shape[1:] != (480, 640):
            chw = _fit_chw(chw, 480, 640)
        out.append(chw[0])
    return out


def _fit_chw(chw: np.ndarray, h: int, w: int) -> np.ndarray:
    """`(1,3,H,W)` -> `(1,3,h,w)`; area-averaging when shrinking."""
    import cv2

    x = chw[0].transpose(1, 2, 0)                      # HWC
    interp = cv2.INTER_AREA if (x.shape[0] > h or x.shape[1] > w) else cv2.INTER_LINEAR
    y = cv2.resize(x, (w, h), interpolation=interp)
    return np.ascontiguousarray(y.transpose(2, 0, 1)[None])


def freeze_bn_stats(model):
    """Keep BatchNorm in eval mode while the Conv/Linear weights train."""
    import torch.nn as nn

    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            # `eval()` is not enough on its own: the training loop calls
            # `student.train()` at the top of every epoch, which flips them
            # back. Marking them frozen makes the intent explicit and lets the
            # loop re-apply it.
            for p in m.parameters():
                p.requires_grad_(False)
            n += 1
    print(f"pinned {n} BatchNorm layers to eval (running stats frozen)", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="small: this adapts to quantisation, it does not train")
    ap.add_argument("--n-frames", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-frames", type=int, default=120,
                    help="frames used to calibrate the fake-quant observers "
                         "before the first step (train split, A12)")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap the number of optimiser steps per epoch")
    ap.add_argument("--out", default=str(CKPT_DIR / "yolo_qat_distill.pth"))
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    torch.manual_seed(42)

    teacher = build_raw_model(frozen=True).to(device)
    student = insert_fake_quant(build_raw_model(frozen=False)).to(device)
    check_liveness(student)
    freeze_bn_stats(student)

    print(f"loading {args.n_frames} training frames ...", flush=True)
    frames = load_frames(args.n_frames)
    print(f"  got {len(frames)} frames", flush=True)

    # ## Calibrate the observers BEFORE training, not during it
    #
    # `prepare_qat` leaves every observer at its default range, so the first
    # forward pass clips activations to a range nobody chose. Measured on this
    # graph: an uncalibrated student has `mean|d| = 1.49` against the teacher
    # before any training at all, and a 32-frame calibration moves the loss
    # from 31.5 to 10.7.
    #
    # The frames are a disjoint slice of the same training split used for the
    # steps below, so no board frame enters the training layer (A12).
    bs = args.batch_size
    before: tuple[float, float] | None = None
    if args.calib_frames > 0:
        print(f"calibrating observers on {args.calib_frames} train frames ...",
              flush=True)
        calib = load_frames(args.calib_frames, seed=1)
        with torch.no_grad():
            for i in range(0, len(calib), bs):
                student(torch.from_numpy(np.stack(calib[i:i + bs])).to(device))
        for m in student.modules():
            from torch.ao.quantization import FakeQuantize as _FQ
            if isinstance(m, _FQ):
                m.disable_observer()
        print("  observers calibrated and frozen", flush=True)

    # Baseline for the after-training comparison below: the same calibrated
    # student, same frames, same eval mode -- just before the first step.
    student.eval()
    with torch.no_grad():
        _x = torch.from_numpy(np.stack(frames[:8])).to(device)
        _d = (student(_x) - teacher(_x)).abs()
    before = (float(_d.max()), float(_d.mean()))
    print(f"pre-training: max|student-teacher| = {before[0]:.4f}, "
          f"mean = {before[1]:.6f}", flush=True)

    # Only the weights move: BN parameters were frozen above, and `AdamW` is
    # handed `student.parameters()` so the optimiser never sees them either.
    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"optimising {len(trainable)} parameter tensors", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    n_steps = (len(frames) + bs - 1) // bs

    for epoch in range(args.epochs):
        student.train()
        # `student.train()` above re-enables BN train mode on every epoch --
        # re-pin after it, or the first epoch's setting silently reverts.
        freeze_bn_stats(student)
        t0 = time.perf_counter()
        loss_sum, n_seen = 0.0, 0
        for step in range(n_steps):
            batch = np.stack(frames[step * bs:(step + 1) * bs])
            x = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                y_teacher = teacher(x)
            y_student = student(x)
            loss = channel_balanced_loss(y_student, y_teacher)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(x)
            n_seen += len(x)
            if args.limit and step + 1 >= args.limit:
                break
            if (step + 1) % 10 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{n_steps} "
                      f"mse {loss_sum/n_seen:.6f}", flush=True)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch+1}/{args.epochs}  mse {loss_sum/max(1,n_seen):.6f}  "
              f"({dt:.0f}s)", flush=True)

    student.eval()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    # Save the full module so the fake-quant observers' min/max travel with the
    # weights -- those ranges ARE the QAT result (scripts/34 makes the same
    # point for ST-GCN).
    torch.save({"model": student.state_dict()}, out)
    print(f"saved {out}", flush=True)

    # ## Report the *change*, not the absolute error
    #
    # An absolute `max|student-teacher|` is uninterpretable on its own: the
    # box/keypoint channels span -37..665, so 300 px on one outlier anchor says
    # nothing about whether the run helped. What decides whether the 1-2 h was
    # worth it is whether the calibrated-but-untrained student got worse or
    # better. So measure both, on the same frames, in the same mode.
    import copy

    def _err(m) -> tuple[float, float]:
        m.eval()
        with torch.no_grad():
            d = (m(torch.from_numpy(np.stack(frames[:8])).to(device))
                 - teacher(torch.from_numpy(np.stack(frames[:8])).to(device))).abs()
        return float(d.max()), float(d.mean())

    student.eval()
    after_max, after_mean = _err(student)
    print(f"post-QAT:   max|student-teacher| = {after_max:.4f}, "
          f"mean = {after_mean:.6f}", flush=True)
    if before is not None:
        print(f"pre-training: max|student-teacher| = {before[0]:.4f}, "
              f"mean = {before[1]:.6f}", flush=True)
        verdict = "improved" if after_mean < before[1] else "REGRESSED"
        print(f"distillation {verdict}: mean {before[1]:.6f} -> {after_mean:.6f}",
              flush=True)
        if after_mean >= before[1]:
            print("  the weights moved away from the teacher -- the run did not "
                  "help and this checkpoint should not be exported", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
