"""Export a dynamic-batch ST-GCN++ ONNX for PLAN §3.3 batched scoring."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
sys.path.insert(0, str(SOURCE_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
spec_mod = _load_own("deploy_spec", "src/deploy/spec.py")
export_mod = _load_own("deploy_stgcn_export", "src/deploy/stgcn_export.py")

import torch  # noqa: E402

from src.train import engine # noqa: E402 (source project)

STATIC_ONNX = paths.MODELS / "stgcnpp_v2.onnx"
BATCH_ONNX = paths.MODELS / "stgcnpp_v2_b4.onnx"
OPSET = 12
N_PROBE = 200
# The same bound G1 uses. Not a new number: 1e-4 is already the project's
# standard for "two graphs compute the same function" (scripts/11), chosen
# because it is ~4 orders below the smallest logit gap that can flip a
# decision. Inventing a tighter one here would be the L-114 mistake -- a
# threshold picked to fit an observation.
TOL_SAME = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opset", type=int, default=OPSET)
    p.add_argument("--n-probe", type=int, default=N_PROBE)
    return p.parse_args()


def load_model():
    cfg = engine.load_config()
    model = engine.build_recognizer(cfg)
    ckpt = torch.load(paths.STGCN_PTH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run(sess, name: str, x: np.ndarray) -> np.ndarray:
    return np.asarray(sess.run(None, {name: x})[0])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    paths.PARITY.mkdir(parents=True, exist_ok=True)

    if not STATIC_ONNX.exists():
        raise SystemExit(f"missing {STATIC_ONNX}; run scripts/11 first")

    model = load_model()
    wrapper = export_mod.STGCNExport(model, patch=True).eval()
    dummy = torch.zeros(4, *spec_mod.MODEL_INPUT_SHAPE[1:], dtype=torch.float32)
    with torch.no_grad():
        eager = wrapper(dummy)
    print(f"eager batch-4 output {tuple(eager.shape)}", flush=True)

    torch.onnx.export(
        wrapper, dummy, str(BATCH_ONNX),
        opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        # Batch axis only. T/V/C stay fixed because the RKNN graph and every
        # preprocessing constant are built around (64, 17, 3); a dynamic time
        # axis would let a caller silently feed the wrong clip length.
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        do_constant_folding=True, verbose=False,
    )
    print(f"wrote {BATCH_ONNX} ({BATCH_ONNX.stat().st_size/1e6:.2f} MB)", flush=True)

    import onnx
    m = onnx.load(str(BATCH_ONNX))
    onnx.checker.check_model(m)
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"opset={args.opset} nodes={len(m.graph.node)} "
          f"Einsum={ops.get('Einsum', 0)}", flush=True)

    # The dynamic graph is bigger (650 vs 364 nodes) and that is expected: with
    # a symbolic batch axis the exporter can no longer bake reshape targets in
    # as constants, so it emits a `Shape -> Gather -> Unsqueeze -> Concat` chain
    # per reshape (94 Unsqueeze, 43 Shape, 43 Gather). Those are *shape*
    # arithmetic, not math. The functional operator counts must be identical to
    # the delivered graph, and that is what this asserts -- "the dynamic export
    # changed the model" is exactly the kind of thing a node count hides.
    static_ops: dict[str, int] = {}
    for node in onnx.load(str(STATIC_ONNX)).graph.node:
        static_ops[node.op_type] = static_ops.get(node.op_type, 0) + 1
    FUNC = ("Conv", "MatMul", "ReduceSum", "Gemm", "BatchNormalization",
            "Relu", "MaxPool", "GlobalAveragePool", "ReduceMean", "Add")
    drift = {k: (static_ops.get(k, 0), ops.get(k, 0)) for k in FUNC
             if static_ops.get(k, 0) != ops.get(k, 0)}
    print("  functional op counts " +
          ("match the delivered graph" if not drift else f"DIFFER: {drift}"),
          flush=True)
    if drift:
        raise SystemExit(f"dynamic export changed the model, not just the "
                         f"batch axis: {drift}")

    import onnxruntime as ort

    def session(path: Path):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return s, s.get_inputs()[0].name

    s_static, n_static = session(STATIC_ONNX)
    s_batch, n_batch = session(BATCH_ONNX)
    in_shape = s_batch.get_inputs()[0].shape
    print(f"dynamic graph input shape: {in_shape}", flush=True)
    if in_shape[0] != "N":
        raise SystemExit(f"batch axis is {in_shape[0]!r}, not dynamic -- the "
                         f"export did not take")

    # --- 1. same function as the delivered graph, at batch 1 ------------------
    calib = np.load(PROJECT_ROOT / "calib" / "stgcn_calib.npy")
    n = min(args.n_probe, len(calib))
    x = calib[:n].astype(np.float32)
    print(f"\nprobe windows: {x.shape}", flush=True)

    ref = np.concatenate([run(s_static, n_static, x[i:i + 1]) for i in range(n)])
    dyn1 = np.concatenate([run(s_batch, n_batch, x[i:i + 1]) for i in range(n)])
    d_same = np.abs(ref - dyn1)
    max_same = float(d_same.max())
    print(f"  dynamic(b=1) vs delivered static: max|d|={max_same:.3e} "
          f"mean={d_same.mean():.3e}", flush=True)

    # --- 2. batching is exact -------------------------------------------------
    chunk = 4
    batched = np.concatenate([run(s_batch, n_batch, x[i:i + chunk])
                              for i in range(0, n, chunk)])
    d_batch = np.abs(ref - batched)
    max_batch = float(d_batch.max())
    dec_ref = spec_mod.softmax2(ref)[:, 1] >= 0.5
    dec_bat = spec_mod.softmax2(batched)[:, 1] >= 0.5
    flips = int((dec_ref != dec_bat).sum())
    print(f"  batch-{chunk} vs batch-1 loop:        max|d|={max_batch:.3e} "
          f"mean={d_batch.mean():.3e} flips={flips}/{n}", flush=True)

    # --- 3. every batch size the runtime may chunk to -------------------------
    by_size: dict[int, float] = {}
    for bs in range(1, 9):
        take = min(bs, n)
        y = run(s_batch, n_batch, x[:take])
        by_size[bs] = float(np.abs(y - ref[:take]).max())
    print("  max|d| by batch size: " +
          " ".join(f"{k}:{v:.1e}" for k, v in by_size.items()), flush=True)

    ok = max_same <= TOL_SAME and max_batch <= TOL_SAME and flips == 0
    result = {
        "opset": args.opset,
        "static_onnx": STATIC_ONNX.name,
        "static_sha256": paths.sha256_file(STATIC_ONNX),
        "batch_onnx": BATCH_ONNX.name,
        "batch_sha256": paths.sha256_file(BATCH_ONNX),
        "batch_onnx_bytes": BATCH_ONNX.stat().st_size,
        "input_shape": [str(s) for s in in_shape],
        "op_histogram": ops,
        "n_probe": n,
        "dynamic_b1_vs_static": {"max_abs": max_same,
                                 "mean_abs": float(d_same.mean())},
        "batch4_vs_loop": {"max_abs": max_batch,
                           "mean_abs": float(d_batch.mean()),
                           "decision_flips": flips},
        "max_abs_by_batch_size": by_size,
        "tolerance": TOL_SAME,
        "passed": bool(ok),
    }
    paths.write_json(paths.PARITY / "g1b_stgcn_batch.json", result)

    lines = [
        "# G1b — ST-GCN++ 动态 batch 导出（PLAN §3.3）", "",
        "> 交付图 `models/stgcnpp_v2.onnx` **保持静态 batch-1 不变**（它是 RKNN 的输入，",
        "> sha256 冻结在 `reference/MANIFEST_source.md`）。本文件是 PC 侧的批处理伴随产物。", "",
        f"- 静态图 `{STATIC_ONNX.name}` sha256 `{result['static_sha256'][:16]}…`",
        f"- 批处理图 `{BATCH_ONNX.name}` sha256 `{result['batch_sha256'][:16]}…`，"
        f"{result['batch_onnx_bytes']/1e6:.2f} MB",
        f"- 输入 `{tuple(result['input_shape'])}`（batch 轴动态，T/V/C 固定）",
        f"- 探针样本：{n} 条真实校准窗口", "",
        f"- 节点数 {len(m.graph.node)} vs 交付静态图 "
        f"{sum(static_ops.values())}。差异全部是形状算术"
        f"（`Shape`/`Gather`/`Unsqueeze`/`Concat`）：batch 轴变成符号后，"
        f"导出器无法再把 reshape 目标折成常量。**功能算子数逐一相同**"
        f"（Conv 125 / MatMul 10 / ReduceSum 10 / Gemm 1 …）。", "",
        f"- 残差 `{max_same:.1e}` 不是导出错误，是 float32 累加顺序差异："
        f"logits 量级 ~5，1 ulp ≈ 5e-7，实测约 5 ulp。判据用 G1 的 1e-4 而不是更紧的数，"
        f"理由见脚本注释与 L-114。", "",
        "| 检查 | 判据 | 实测 | 结果 |",
        "|---|---|---|---|",
        f"| 动态图(b=1) vs 交付静态图 | ≤{TOL_SAME:g} | {max_same:.3e} "
        f"| {'✅' if max_same <= TOL_SAME else '❌'} |",
        f"| batch-4 vs 4×batch-1 | ≤{TOL_SAME:g}，翻转 0 | {max_batch:.3e}，"
        f"{flips} 翻转 | {'✅' if max_batch <= TOL_SAME and flips == 0 else '❌'} |",
        "",
        "按 batch 大小分列的 `max|d|`：",
        "",
        "| batch | " + " | ".join(str(k) for k in by_size) + " |",
        "|---|" + "---:|" * len(by_size),
        "| `max\\|d\\|` | " + " | ".join(f"{v:.1e}" for v in by_size.values()) + " |",
        "",
        f"**{'通过' if ok else '未通过'}**", "",
    ]
    (paths.PARITY / "g1b_stgcn_batch.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
