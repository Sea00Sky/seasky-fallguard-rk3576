"""Prove `rknn_batch_size=N` is bit-identical to N batch-1 calls (PLAN §3.3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ONNX = PROJECT_ROOT / "models" / "stgcnpp_v2.onnx"
CALIB = PROJECT_ROOT / "calib" / "stgcn_calib.npy"
CALIB_TXT = PROJECT_ROOT / "calib" / "stgcn_calib.txt"
OUT_JSON = PROJECT_ROOT / "reports" / "parity" / "g1b_rknn_batch.json"


def build(batch: int, precision: str):
    from rknn.api import RKNN

    r = RKNN(verbose=False)
    # Same config as scripts/20: mean 0 / std 1 because `spec.to_model_input`
    # already applied the model's convention. Image-style [0,0,0] raises for a
    # 5-D input (the toolkit expects exactly one channel value).
    r.config(mean_values=[[0]], std_values=[[1]], target_platform="rk3576")
    assert r.load_onnx(model=str(ONNX)) == 0, f"load_onnx batch={batch}"
    kw = {"rknn_batch_size": batch} if batch > 1 else {}
    if precision == "int8":
        assert r.build(do_quantization=True, dataset=str(CALIB_TXT), **kw) == 0
    else:
        assert r.build(do_quantization=False, **kw) == 0
    assert r.init_runtime(target=None) == 0, "init_runtime"
    return r


def infer(r, data: np.ndarray) -> np.ndarray:
    return np.asarray(r.inference(inputs=[data], data_format="nchw")[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n", type=int, default=8, help="windows to compare")
    p.add_argument("--check-broadcast", action="store_true",
                   help="also demonstrate the single-sample broadcast trap")
    args = p.parse_args()

    for f in (ONNX, CALIB, CALIB_TXT):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    data = np.load(CALIB)[:args.n].astype(np.float32)
    print(f"probe {data.shape}, batch={args.batch}", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {"batch": args.batch, "n_probe": int(len(data)), "runs": {}}

    for precision in ("fp16", "int8"):
        print(f"\n=== {precision} ===", flush=True)
        rb = build(args.batch, precision)
        bN = infer(rb, data[:args.batch])
        rb.release()
        print(f"  batch-{args.batch} done {bN.shape}", flush=True)

        r1 = build(1, precision)
        b1 = np.concatenate([infer(r1, data[i:i + 1]) for i in range(args.batch)])
        r1.release()
        print(f"  batch-1 loop done {b1.shape}", flush=True)

        d = np.abs(bN - b1)
        ident = bool(np.array_equal(bN, b1))
        print(f"  batch{args.batch} vs batch1-loop: MAX_ABS={d.max():.9f} "
              f"bitwise={ident}", flush=True)
        print(f"  batch-{args.batch} rows:\n"
              f"{np.array2string(bN, precision=6)}", flush=True)
        result["runs"][precision] = {
            "max_abs": float(d.max()),
            "bitwise_identical": ident,
            "batch_out": bN.tolist(),
        }

    if args.check_broadcast:
        print("\n=== broadcast trap ===", flush=True)
        rb = build(args.batch, "fp16")
        one = infer(rb, data[0:1])
        print(f"  fed 1 sample to a batch-{args.batch} model -> shape {one.shape}")
        print(f"  rows identical: "
              f"{bool(np.allclose(one, np.broadcast_to(one[0], one.shape)))}")
        result["broadcast"] = {
            "fed": 1, "model_batch": args.batch,
            "out_shape": list(one.shape),
            "all_rows_equal": bool(np.allclose(
                one, np.broadcast_to(one[0], one.shape))),
        }
        rb.release()

    ok = all(v["bitwise_identical"] for v in result["runs"].values())
    result["passed"] = bool(ok)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'}: rknn_batch_size is a pure speed change")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
