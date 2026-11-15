"""Gate 0 for the official yolo26_pose recipe: does `subgraph_dtype_target` exist here?"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

TARGET = "rk3576"
LAYOUT = "t4"

# `convert.py`'s subgraphs, transcribed. Each entry is (label, inputs, outputs).
# The endpoints are tensor names *before* the ONNX exporter renames them, so
# they are resolved against our graph by `--verify-names` rather than trusted.
OFFICIAL_SUBGRAPHS = [
    ("backbone_cv1_10_13",
     ["/model.10/cv1/conv/Conv_output_0"],
     ["/model.13/cv1/conv/Conv_output_0"]),
    ("backbone_cv1_22",
     ["/model.22/cv1/conv/Conv_output_0"],
     ["/model.22/cv2/conv/Conv_output_0"]),
    ("head_cv3_2",
     ["/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv_output_0"],
     ["/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0"]),
    ("head_cv3_1",
     ["/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_output_0"],
     ["/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_output_0"]),
    ("head_cv3_0",
     ["/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_output_0"],
     ["/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_output_0"]),
    ("attn_22",
     ["/model.22/m.0/m.0.1/attn/qkv/conv/Conv_output_0"],
     ["/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0"]),
    ("backbone_cv1_19",
     ["/model.19/cv1/conv/Conv_output_0"],
     ["/model.19/cv2/conv/Conv_output_0"]),
]

# Our graph has no `/model.23/` prefix on the detection head (the exporter drops
# the Detect module wrapper), while the official names carry it. So the mapping
# is "try the name as-is, then with `/model.23` stripped", not a suffix match.
_DROP_PREFIX = "/model.23"


def resolve(name: str, available: set[str]) -> str | None:
    """Map an official tensor name onto ours, tolerating the dropped prefix."""
    if name in available:
        return name
    if name.startswith(_DROP_PREFIX):
        stripped = name[len(_DROP_PREFIX):]
        if stripped in available:
            return stripped
    return None


def tensor_names(onnx_path: Path) -> set[str]:
    """Every tensor name the graph can produce, including implicit ones."""
    import onnx

    m = onnx.load(str(onnx_path))
    names: set[str] = set()
    for n in m.graph.node:
        names.update(n.output)
        for o in n.output:
            names.add(f"{n.name}_output_0")
    names.update(o.name for o in m.graph.output)
    names.update(i.name for i in m.graph.input)
    return names


def verify_names(onnx_path: Path) -> int:
    names = tensor_names(onnx_path)
    print(f"{onnx_path.name}: {len(names)} tensor names\n")
    bad = 0
    for label, ins, outs in OFFICIAL_SUBGRAPHS:
        ri = [resolve(x, names) for x in ins]
        ro = [resolve(x, names) for x in outs]
        ok = all(ri) and all(ro)
        bad += 0 if ok else 1
        print(f"  {'OK ' if ok else 'MISS'} {label}")
        for src, got in zip(ins + outs, ri + ro):
            if got is None:
                print(f"        missing: {src}")
    print(f"\n{len(OFFICIAL_SUBGRAPHS) - bad}/{len(OFFICIAL_SUBGRAPHS)} subgraphs resolvable")
    return 0 if bad == 0 else 1


def calib_list() -> Path:
    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:200]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}")
    txt = paths.CALIB / "yolo_calib.txt"
    txt.write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    return txt


def mode_config(onnx_path: Path, out: Path) -> dict:
    """Does `config()` accept the parameter at all? Honoured is decided later."""
    from rknn.api import RKNN

    names = tensor_names(onnx_path)
    sub = []
    for label, ins, outs in OFFICIAL_SUBGRAPHS:
        ri = [resolve(x, names) for x in ins]
        ro = [resolve(x, names) for x in outs]
        if all(ri) and all(ro):
            sub.append([ri, ro, "w16a16"])
    print(f"resolved {len(sub)}/{len(OFFICIAL_SUBGRAPHS)} subgraphs to w16a16")

    rknn = RKNN(verbose=False)
    base = dict(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET, quantized_method="channel",
                optimization_level=3)
    accepted = True
    try:
        ret = rknn.config(subgraph_dtype_target=sub, **base)
        print(f"config(subgraph_dtype_target=...) -> ret={ret}")
        accepted = ret == 0
    except TypeError as e:
        print(f"config() REJECTED the parameter: {e}")
        accepted = False
    except Exception as e:  # noqa: BLE001 - report whatever the toolkit raises
        print(f"config() raised {type(e).__name__}: {e}")
        accepted = False

    if not accepted:
        rknn.release()
        return {"mode": "config", "accepted": False}

    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")
    calib = calib_list()
    if rknn.build(do_quantization=True, dataset=str(calib)) != 0:
        raise SystemExit("build failed")
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()
    return {"mode": "config", "accepted": True, "rknn": out.name,
            "sha256": paths.sha256_file(out), "size": out.stat().st_size}


def mode_cfg(onnx_path: Path, out: Path) -> dict:
    """Fallback: step1, rewrite dtype of matching tensors, step2."""
    from rknn.api import RKNN

    names = tensor_names(onnx_path)
    want: set[str] = set()
    for label, ins, outs in OFFICIAL_SUBGRAPHS:
        for x in ins + outs:
            got = resolve(x, names)
            if got:
                want.add(got)

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET, quantized_method="channel",
                optimization_level=3)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")
    calib = calib_list()
    print(f"  hybrid_quantization_step1 over "
          f"{len(calib.read_text(encoding='utf-8').split())} frames", flush=True)
    if rknn.hybrid_quantization_step1(dataset=str(calib), proposal=False) != 0:
        raise SystemExit("hybrid_quantization_step1 failed")

    stem = onnx_path.stem
    cfg = Path(f"{stem}.quantization.cfg")
    if not cfg.exists():
        raise SystemExit(f"step1 did not write {cfg}")

    text = cfg.read_text(encoding="utf-8")
    # Only rewrite tensors that step1 actually quantised; anything else is a
    # name that did not survive into `quantize_parameters`.
    changed = 0
    for name in sorted(want):
        pat = re.compile(
            rf"(^    {re.escape(name)}:\n(?:        .*\n)*?        dtype: )int8$",
            flags=re.M)
        text, n = pat.subn(r"\g<1>int16", text)
        changed += n
    cfg.write_text(text, encoding="utf-8")
    print(f"  rewrote dtype to int16 on {changed} of {len(want)} target tensors")
    if changed == 0:
        rknn.release()
        raise SystemExit(
            "no target tensor was found in quantize_parameters -- the subgraph "
            "names do not survive into the cfg; use --mode config instead")

    rknn2 = RKNN(verbose=False)
    if rknn2.hybrid_quantization_step2(
            model_input=f"{stem}.model", data_input=f"{stem}.data",
            model_quantization_cfg=str(cfg)) != 0:
        raise SystemExit("hybrid_quantization_step2 failed")
    if rknn2.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn2.release()
    rknn.release()
    return {"mode": "cfg", "accepted": True, "n_rewritten": changed,
            "rknn": out.name, "sha256": paths.sha256_file(out),
            "size": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["verify", "config", "cfg"], default="verify")
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    onnx_path = Path(args.onnx) if args.onnx else (
        paths.MODELS / f"yolo26n_pose_480x640_{LAYOUT}.onnx")
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    if args.mode == "verify":
        return verify_names(onnx_path)

    tag = f"int8_{LAYOUT}_subw16_{args.mode}"
    out = paths.MODELS / f"yolo26n_pose_480x640_{tag}.rknn"
    print(f"building {out.name}", flush=True)
    rec = mode_config(onnx_path, out) if args.mode == "config" else \
        mode_cfg(onnx_path, out)
    rec["gate"] = "G5"
    rec["probe"] = "official yolo26_pose subgraph w16a16"
    rep = paths.REPORTS / "parity" / f"g5_probe_subw16_{args.mode}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {rep}")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
