"""QAT 值不值得跑：在 t9 布局上重测量化天花板。"""
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


# Same constants as scripts/25_verify_truncated.py::_T9_SCALES.
T9_SCALES = ((8, 60, 80), (16, 30, 40), (32, 15, 20))


def t9_grid(h: int, w: int):
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    return xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5


def t9_rebuild(outs: list[np.ndarray]) -> np.ndarray:
    """Nine pre-decode tensors -> (1, 56, 6300), same as scripts/25."""
    boxes, confs, kpts = [], [], []
    for i, (stride, h, w) in enumerate(T9_SCALES):
        b = np.asarray(outs[i * 3]).reshape(1, 4, -1)[0]
        c = np.asarray(outs[i * 3 + 1]).reshape(1, 1, -1)[0]
        k = np.asarray(outs[i * 3 + 2]).reshape(1, 17, 3, -1)[0]
        cx, cy = t9_grid(h, w)
        boxes.append(np.stack([(cx - b[0]) * stride, (cy - b[1]) * stride,
                               (cx + b[2]) * stride, (cy + b[3]) * stride]))
        confs.append(1.0 / (1.0 + np.exp(-c[0])))
        kpts.append(np.stack([(cx + k[:, 0]) * stride, (cy + k[:, 1]) * stride,
                              1.0 / (1.0 + np.exp(-k[:, 2]))], axis=1))
    box = np.concatenate(boxes, axis=-1)[None].astype(np.float32)
    conf = np.concatenate(confs, axis=-1)[None, None].astype(np.float32)
    kpt = np.concatenate(kpts, axis=-1).reshape(1, 51, -1).astype(np.float32)
    return np.concatenate([box, conf, kpt], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=228, help="scoring frames")
    ap.add_argument("--calib-frames", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap frames")
    ap.add_argument("--out", default="reports/parity/g5_yolo_qat_ceiling_t9.json")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    v21 = _load("v21", "scripts/21_verify_rknn.py")
    q39 = _load("qat39", "scripts/39_yolo_qat_distill.py")
    paths = v21.paths
    from src.deploy import spec as spec_mod
    from src.deploy.yolo_export import decode, preprocess

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = v21.load_yolo_frames(args.n)
    if args.limit:
        frames = frames[: args.limit]
    calib = q39.load_frames(args.calib_frames)
    print(f"eval {frames.shape}  calib {len(calib)}  dev {dev}", flush=True)

    _, params = preprocess(np.zeros((480, 640, 3), np.uint8),
                           imgsz=spec_mod.IMGSZ)

    # ---- the t9 graph: neck outputs -> nine raw head convs ----------------
    def build_t9():
        ck = torch.load(str(paths.YOLO_PT), map_location="cpu",
                        weights_only=False)
        net = ck["model"].float().eval()
        head = net.model[-1]

        class T9(nn.Module):
            """Replays the neck, then applies the nine one2one head convs."""

            def __init__(self, net, head):
                super().__init__()
                self.net = net
                self.head = head

            def forward(self, x):
                y = []
                for m in self.net.model[:-1]:
                    if m.f != -1:
                        x = (y[m.f] if isinstance(m.f, int)
                             else [x if j == -1 else y[j] for j in m.f])
                    x = m(x)
                    y.append(x if m.i in self.net.save else None)
                feats = [y[i] for i in self.head.f]
                outs = []
                for i, f in enumerate(feats):
                    b = self.head.one2one_cv2[i](f)
                    s = self.head.one2one_cv3[i](f)
                    # `forward_head` does `kpts_head[i](pose_head[i](x[i]))`:
                    # `one2one_cv4` is the intermediate 85ch conv, `_kpts` is
                    # the final 51ch one. Calling `_kpts` on the neck output
                    # directly fails with "expected 85 channels, got 64".
                    k = self.head.one2one_cv4_kpts[i](self.head.one2one_cv4[i](f))
                    outs.extend([b, s, k])
                return outs

        return T9(net, head).eval()

    ref = build_t9().to(dev)
    with torch.no_grad():
        raw_ref = []
        for i in range(0, len(frames), 8):
            outs = ref(torch.from_numpy(frames[i:i + 8]).to(dev))
            raw_ref.append([o.cpu().numpy() for o in outs])
    # list of 9 arrays, each (N, ...)
    ref9 = [np.concatenate([r[j] for r in raw_ref], axis=0) for j in range(9)]
    print("reference t9 shapes:", [a.shape for a in ref9], flush=True)
    rebuilt_ref = np.concatenate(
        [t9_rebuild([a[i] for a in ref9]) for i in range(len(frames))], axis=0)
    print("rebuilt reference:", rebuilt_ref.shape, flush=True)

    def qcfg():
        act = FakeQuantize.with_args(observer=MinMaxObserver, dtype=torch.qint8,
                                     qscheme=torch.per_tensor_affine,
                                     reduce_range=False)
        wt = FakeQuantize.with_args(observer=PerChannelMinMaxObserver,
                                    dtype=torch.qint8,
                                    qscheme=torch.per_channel_symmetric,
                                    reduce_range=False)
        return QConfig(activation=act, weight=wt)

    def score(raw_a, raw_b, tag):
        ious, l2 = [], []
        cm = sw = 0
        for i in range(len(raw_a)):
            ba, ka, _ = decode(raw_a[i][None], params, conf=spec_mod.CONF,
                               max_persons=spec_mod.MAX_PERSONS)
            bb, kb, _ = decode(raw_b[i][None], params, conf=spec_mod.CONF,
                               max_persons=spec_mod.MAX_PERSONS)
            if len(ba) != len(bb):
                cm += 1
                continue
            fi, pairs, s = v21._match_boxes(ba, bb)
            sw += s
            ious.extend(fi)
            for ja, jb in pairs:
                l2.extend(np.linalg.norm(ka[ja] - kb[jb], axis=1).tolist())
        a = np.asarray(ious)
        b = np.asarray(l2)
        # `rebuilt` is (N, 56, 6300) -- no leading batch dim. scripts/21 uses
        # `raw[:, 0, 4, :]` because its cache keeps (N, 1, 56, N); this script
        # rebuilds per frame, so the conf row is index 4 of axis 1.
        dconf = np.abs(raw_a[:, 4, :] - raw_b[:, 4, :])
        r = {
            "variant": tag,
            "iou_median": float(np.median(a)) if a.size else 0.0,
            "iou_p5": float(np.percentile(a, 5)) if a.size else 0.0,
            "anchor_ge_0.5": float((a >= 0.5).mean()) if a.size else 0.0,
            "kpt_l2_max_px": float(b.max()) if b.size else float("inf"),
            "conf_abs_err_max": float(dconf.max()),
            "conf_abs_err_p99": float(np.percentile(dconf, 99)),
            "count_mismatch_frames": int(cm),
            "count_mismatch_frac": cm / len(raw_a),
            "order_swaps": int(sw),
        }
        r["passes"] = bool(
            r["anchor_ge_0.5"] >= v21.G5_ANCHOR_MIN
            and r["iou_median"] >= v21.G5_IOU_MEDIAN_INT8
            and r["iou_p5"] >= v21.G5_IOU_P5_INT8
            and r["kpt_l2_max_px"] <= v21.G5_KPT_L2_DIAG_FRAC * float(
                np.hypot(spec_mod.IMGSZ, spec_mod.IMGSZ * 0.75))
            and r["conf_abs_err_max"] <= v21.TOL_CONF_ABS_FP16
            and r["count_mismatch_frac"] <= v21.TOL_COUNT_MISMATCH_FRAC)
        print(f"\n{tag}: {'PASS' if r['passes'] else 'FAIL'}", flush=True)
        for k, v in r.items():
            print(f"    {k:24s} {v}", flush=True)
        return r

    def run_variant(weight_int8: bool, act_int8: bool, tag: str):
        m = build_t9()
        for _n, mod in m.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.qconfig = qcfg()
        q = prepare_qat(m.train(), inplace=False).to(dev)
        for p in q.parameters():
            p.requires_grad_(True)
        for mod in q.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        q.eval()
        with torch.no_grad():
            for i in range(0, len(calib), 8):
                q(torch.from_numpy(np.stack(calib[i:i + 8])).to(dev))
        for name, mod in q.named_modules():
            if not isinstance(mod, FakeQuantize):
                continue
            mod.disable_observer()
            is_act = name.endswith("activation_post_process")
            if is_act and not act_int8:
                mod.disable_fake_quant()
            if (not is_act) and not weight_int8:
                mod.disable_fake_quant()
        q.eval()
        with torch.no_grad():
            out9 = []
            for i in range(0, len(frames), 8):
                outs = q(torch.from_numpy(frames[i:i + 8]).to(dev))
                out9.append([o.cpu().numpy() for o in outs])
        out9 = [np.concatenate([r[j] for r in out9], axis=0) for j in range(9)]
        rebuilt = np.concatenate(
            [t9_rebuild([a[i] for a in out9]) for i in range(len(frames))],
            axis=0)
        return score(rebuilt_ref, rebuilt, tag)

    results = [
        run_variant(True, True, "W8A8 (the board's scheme)"),
        run_variant(True, False, "W8A32 (weights only)"),
        run_variant(False, True, "W32A8 (activations only)"),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gate": "G5", "model": "yolo", "layout": "t9",
        "note": "fake-quant ceiling, NOT a rknn build; see scripts/44",
        "n_frames": int(len(frames)),
        "n_calib_frames": len(calib),
        "calib_domain": "train split (A12)",
        "variants": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
