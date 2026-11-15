"""Score a QAT checkpoint in the t9 layout against G5."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=228)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    v21 = _load("v21", "scripts/21_verify_rknn.py")
    q55 = _load("q55", "scripts/55_qat_ceiling_t9.py")
    paths = v21.paths
    from src.deploy import spec as spec_mod
    from src.deploy.yolo_export import decode, preprocess

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = v21.load_yolo_frames(args.n)
    if args.limit:
        frames = frames[: args.limit]
    print(f"eval {frames.shape}  dev {dev}", flush=True)

    _, params = preprocess(np.zeros((480, 640, 3), np.uint8),
                           imgsz=spec_mod.IMGSZ)

    def build_t9():
        """Same nine-output wrapper as scripts/55_qat_ceiling_t9.py."""
        ck = torch.load(str(paths.YOLO_PT), map_location="cpu",
                        weights_only=False)
        net = ck["model"].float().eval()
        head = net.model[-1]

        class T9(nn.Module):
            """Nine raw head convs, keyed as `net.model.23.*`."""

            def __init__(self, net, head):
                super().__init__()
                self.net = net
                self._nl = head.nl
                self._f = list(head.f)

            def forward(self, x):
                y = []
                for m in self.net.model[:-1]:
                    if m.f != -1:
                        x = (y[m.f] if isinstance(m.f, int)
                             else [x if j == -1 else y[j] for j in m.f])
                    x = m(x)
                    y.append(x if m.i in self.net.save else None)
                head = self.net.model[-1]
                feats = [y[i] for i in self._f]
                outs = []
                for i, f in enumerate(feats):
                    b = head.one2one_cv2[i](f)
                    sc = head.one2one_cv3[i](f)
                    # forward_head does kpts_head[i](pose_head[i](x[i])):
                    # one2one_cv4 is the intermediate 85ch conv.
                    k = head.one2one_cv4_kpts[i](head.one2one_cv4[i](f))
                    outs.extend([b, sc, k])
                return outs

        return T9(net, head)

    def qcfg():
        # Must match scripts/39: MovingAverage for activations, per-channel
        # symmetric for weights. A different observer changes the module tree
        # and `load_state_dict(strict=True)` would fail on shape, not silently
        # -- but keeping them identical also keeps the ranges interpretable.
        act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                     dtype=torch.qint8,
                                     qscheme=torch.per_tensor_affine,
                                     reduce_range=False)
        wt = FakeQuantize.with_args(observer=PerChannelMinMaxObserver,
                                    dtype=torch.qint8,
                                    qscheme=torch.per_channel_symmetric,
                                    reduce_range=False)
        return QConfig(activation=act, weight=wt)

    def forward9(model):
        outs = []
        with torch.no_grad():
            for i in range(0, len(frames), 8):
                o = model(torch.from_numpy(frames[i:i + 8]).to(dev))
                outs.append([t.cpu().numpy() for t in o])
        out9 = [np.concatenate([r[j] for r in outs], axis=0) for j in range(9)]
        return np.concatenate(
            [q55.t9_rebuild([a[i] for a in out9]) for i in range(len(frames))],
            axis=0)

    # ---- fp32 reference ---------------------------------------------------
    ref_model = build_t9().to(dev).eval()
    raw_ref = forward9(ref_model)
    print(f"reference rebuilt {raw_ref.shape}", flush=True)

    # ---- QAT student ------------------------------------------------------
    student = build_t9()
    for _n, mod in student.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.qconfig = qcfg()
    student = prepare_qat(student.train(), inplace=False)
    for mod in student.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = student.load_state_dict(sd, strict=False)
    # `strict=False` because the checkpoint was built on `YOLOPoseRaw` (t4),
    # whose module tree differs from this t9 wrapper. What must match is every
    # conv/bn weight and every fake-quant scale; anything else is a real error.
    print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}",
          flush=True)
    for k in list(missing)[:5]:
        print(f"    MISSING  {k}", flush=True)
    for k in list(unexpected)[:5]:
        print(f"    UNEXPECTED  {k}", flush=True)
    if len(missing) > 20:
        raise SystemExit(
            f"{len(missing)} missing keys -- the checkpoint does not fit this "
            f"graph; refusing to score a half-loaded model")

    student = student.to(dev).eval()
    raw_q = forward9(student)
    print(f"student rebuilt {raw_q.shape}", flush=True)

    # ---- identical scoring to scripts/25 ----------------------------------
    ious, l2 = [], []
    cm = sw = 0
    for i in range(len(raw_ref)):
        ba, ka, _ = decode(raw_ref[i][None], params, conf=spec_mod.CONF,
                           max_persons=spec_mod.MAX_PERSONS)
        bb, kb, _ = decode(raw_q[i][None], params, conf=spec_mod.CONF,
                           max_persons=spec_mod.MAX_PERSONS)
        if len(ba) != len(bb):
            cm += 1
            continue
        fi, pairs, s = v21._match_boxes(ba, bb)
        sw += s
        ious.extend(fi)
        for ja, jb in pairs:
            l2.extend(np.linalg.norm(ka[ja] - kb[jb], axis=1).tolist())

    a = np.asarray(ious, dtype=np.float64)
    b = np.asarray(l2, dtype=np.float64)
    diag = float(np.hypot(spec_mod.IMGSZ, spec_mod.IMGSZ * 0.75))
    dconf = np.abs(raw_ref[:, 4, :] - raw_q[:, 4, :])
    crit = {
        "anchor_ge_0.5": float((a >= 0.5).mean()) if a.size else 0.0,
        "anchor_min": v21.G5_ANCHOR_MIN,
        "iou_median": float(np.median(a)) if a.size else 0.0,
        "iou_median_min": v21.G5_IOU_MEDIAN_INT8,
        "iou_p5": float(np.percentile(a, 5)) if a.size else 0.0,
        "iou_p5_min": v21.G5_IOU_P5_INT8,
        "kpt_l2_max_px": float(b.max()) if b.size else float("inf"),
        "kpt_l2_max_allowed": v21.G5_KPT_L2_DIAG_FRAC * diag,
        "kpt_frac_within_1px": float((b <= 1.0).mean()) if b.size else 0.0,
        "conf_abs_err_max": float(dconf.max()),
        "conf_abs_err_p99": float(np.percentile(dconf, 99)),
        "count_mismatch_frames": int(cm),
        "count_mismatch_frac": cm / len(raw_ref),
        "count_mismatch_frac_max": v21.TOL_COUNT_MISMATCH_FRAC,
        "order_swaps": int(sw),
    }
    passed = (crit["anchor_ge_0.5"] >= v21.G5_ANCHOR_MIN
              and crit["iou_median"] >= v21.G5_IOU_MEDIAN_INT8
              and crit["iou_p5"] >= v21.G5_IOU_P5_INT8
              and crit["kpt_l2_max_px"] <= v21.G5_KPT_L2_DIAG_FRAC * diag
              and crit["conf_abs_err_max"] <= v21.TOL_CONF_ABS_FP16
              and crit["count_mismatch_frac"] <= v21.TOL_COUNT_MISMATCH_FRAC)

    tag = Path(args.ckpt).stem
    print(f"\nG5 yolo qat_t9 {tag}: {'PASS' if passed else 'FAIL'}", flush=True)
    for k, v in crit.items():
        print(f"    {k:32s} {v}", flush=True)

    # Cache the raw tensors so `scripts/_probe_mismatch.py` can characterise
    # the failures (swap vs drift) without re-running the whole model. Shape
    # matches scripts/25's cache: (N, 1, 56, 6300).
    npz = paths.PARITY / f"yolo_raw_qat_t9_{tag}.npz"
    np.savez_compressed(npz, raw_onnx=raw_ref[:, None], raw_rknn=raw_q[:, None])
    print(f"wrote {npz}", flush=True)

    out = Path(args.out) if args.out else (
        paths.PARITY / f"g5_yolo_qat_t9_{tag}.json")
    out.write_text(json.dumps({
        "gate": "G5", "model": "yolo", "layout": "t9",
        "variant": f"qat_t9_{tag}", "ckpt": str(args.ckpt),
        "n_frames": int(len(frames)),
        "criteria": crit, "passed": bool(passed),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
