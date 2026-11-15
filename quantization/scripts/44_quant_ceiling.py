"""Which half of 8-bit hurts: the weights, the activations, or both?"""
from __future__ import annotations

import importlib.util
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
    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    v21 = _load("v21", "scripts/21_verify_rknn.py")
    paths = v21.paths
    from src.deploy import spec as spec_mod
    from src.deploy.yolo_export import YOLOPoseRaw, decode, preprocess

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = v21.load_yolo_frames(v21.N_YOLO_FRAMES)
    q39 = _load("qat39", "scripts/39_yolo_qat_distill.py")
    calib = q39.load_frames(120)
    print(f"eval {frames.shape}  calib(train split) {len(calib)}", flush=True)

    _, params = preprocess(np.zeros((480, 640, 3), np.uint8), imgsz=spec_mod.IMGSZ)

    def build():
        ck = torch.load(str(paths.YOLO_PT), map_location="cpu", weights_only=False)
        return YOLOPoseRaw(ck["model"].float().eval(), end2end_branch=True).eval()

    ref_model = build().to(dev)
    with torch.no_grad():
        raw_ref = np.concatenate(
            [ref_model(torch.from_numpy(frames[i:i + 8]).to(dev)).cpu().numpy()
             for i in range(0, len(frames), 8)], axis=0)
    print(f"reference {raw_ref.shape}", flush=True)

    # Both observers are always inserted -- the module graph must stay the same
    # across variants, or a difference could come from graph structure rather
    # than from 8-bit. A variant is switched off afterwards with
    # `disable_fake_quant()`, which makes that node the identity.
    def qcfg() -> QConfig:
        act = FakeQuantize.with_args(observer=MinMaxObserver, dtype=torch.qint8,
                                     qscheme=torch.per_tensor_affine,
                                     reduce_range=False)
        wt = FakeQuantize.with_args(observer=PerChannelMinMaxObserver,
                                    dtype=torch.qint8,
                                    qscheme=torch.per_channel_symmetric,
                                    reduce_range=False)
        return QConfig(activation=act, weight=wt)

    def run_variant(weight_int8: bool, act_int8: bool, tag: str):
        m = build()
        for _n, mod in m.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.qconfig = qcfg()
        q = prepare_qat(m.train(), inplace=False).to(dev)
        for p in q.parameters():
            p.requires_grad_(True)
        # BN stays on its checkpoint statistics -- the graph being compared
        # against is eval mode, so train-mode BN would measure a different
        # function (scripts/39 documents the 6.20 vs 1.77 measurement).
        for mod in q.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

        q.eval()
        with torch.no_grad():
            for i in range(0, len(calib), 8):
                q(torch.from_numpy(np.stack(calib[i:i + 8])).to(dev))
        # Freeze the learned ranges, then silence whichever half this variant
        # does not quantise. `disable_observer` only stops range updates;
        # `disable_fake_quant` removes the rounding itself.
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
            out = np.concatenate(
                [q(torch.from_numpy(frames[i:i + 8]).to(dev)).cpu().numpy()
                 for i in range(0, len(frames), 8)], axis=0)
        return score(raw_ref, out, params, v21, spec_mod, decode, tag)

    def score(raw_ref, raw_fq, params, v21, spec_mod, decode, tag):
        ious, l2 = [], []
        cm = sw = 0
        for i in range(len(raw_ref)):
            ba, ka, _ = decode(raw_ref[i][None], params, conf=spec_mod.CONF,
                               max_persons=spec_mod.MAX_PERSONS)
            bb, kb, _ = decode(raw_fq[i][None], params, conf=spec_mod.CONF,
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
        dconf = np.abs(raw_ref[:, 4, :] - raw_fq[:, 4, :])
        r = {
            "variant": tag,
            "iou_median": float(np.median(a)), "iou_p5": float(np.percentile(a, 5)),
            "anchor_ge_0.5": float((a >= 0.5).mean()),
            "kpt_l2_max_px": float(b.max()),
            "conf_abs_err_max": float(dconf.max()),
            "conf_abs_err_p99": float(np.percentile(dconf, 99)),
            "count_mismatch_frames": int(cm),
            "count_mismatch_frac": cm / len(raw_ref),
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

    results = []
    results.append(run_variant(True, True, "W8A8 (the board's scheme)"))
    results.append(run_variant(True, False, "W8A32 (weights only)"))
    results.append(run_variant(False, True, "W32A8 (activations only)"))

    out = paths.PARITY / "g5_yolo_quant_ceiling.json"
    paths.write_json(out, {
        "gate": "G5", "model": "yolo", "layout": "t4",
        "n_frames": int(len(frames)),
        "n_calib_frames": len(calib),
        "calib_domain": "train split (A12)",
        "variants": results,
        "note": "fake-quant simulation in torch, no RKNN. Separates weight "
                "error from activation error to decide whether QAT training "
                "has anything to fix.",
    })
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
