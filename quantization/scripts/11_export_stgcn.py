"""Phase 1a -- export ST-GCN++ to ONNX and prove it matches PyTorch (G1)."""
from __future__ import annotations

import argparse
import importlib.util
import json
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

ONNX_PATH = paths.MODELS / "stgcnpp_v2.onnx"
OPSET = 12
TOL = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opset", type=int, default=OPSET)
    p.add_argument("--no-patch", action="store_true",
                   help="keep einsum (diagnostic only; RKNN needs the rewrite)")
    p.add_argument("--skip-v2", action="store_true", help="board samples only")
    return p.parse_args()


def load_model():
    cfg = engine.load_config()
    model = engine.build_recognizer(cfg)
    ckpt = torch.load(paths.STGCN_PTH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ------------------------------------------------------------------ input sets


def board_inputs() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Rebuild the exact tensors the reference pipeline scored."""
    from src.data import paths as spaths
    from src.window import build as wbuild

    cfg = wbuild.load_config()
    from src.pose import extract
    pose_cfg = extract.load_config()
    pose_cfg.output_root = spaths.KEYPOINTS
    pose_cfg.index_path = spaths.KEYPOINTS / "board" / "extraction_index.csv"

    rows = [r for r in spaths.read_csv(spaths.MANIFESTS / "board" / "videos.csv")
            if r.get("readable") == "True"]

    xs: list[np.ndarray] = []
    meta: list[dict] = []
    wid = 0          # global window index, matching board_candidates.csv order
    for row in rows:
        vid = row["video_id"]
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            kp, sc, tid = d["keypoints"], d["scores"], d["track_ids"]
            np_ = d["n_persons"].astype(np.int32)
            bbox = d["bboxes"]
            width, height = int(d["meta_width"]), int(d["meta_height"])
        fps = float(row["fps"])
        n = len(np_)
        win = wbuild.frames_for(fps, cfg.duration_s)
        st = wbuild.frames_for(fps, cfg.stride_s)
        if n < win:
            continue
        chains = wbuild.stitch_id_handoffs(tid, np_, bbox)

        def add_raw(k_arr, s_arr, cov, kind, track, wid):
            """For B/C: `_track_series` gives raw pixels, so normalise here."""
            k = k_arr.copy()
            k[:, :, 0] /= max(1, width)
            k[:, :, 1] /= max(1, height)
            idx = wbuild.resample_indices(len(k), cfg.resample_frames)
            xs.append(spec_mod.to_model_input(k[idx].astype(np.float32),
                                              s_arr[idx].astype(np.float32)))
            meta.append({"video_id": vid, "strategy": kind, "track": int(track),
                         "coverage": float(cov), "window": int(wid)})

        def add_ready(k_arr, s_arr, cov, kind, track, wid):
            """For A: `extract_window` already normalised, resampled and cast."""
            xs.append(spec_mod.to_model_input(np.asarray(k_arr, dtype=np.float32),
                                              np.asarray(s_arr, dtype=np.float32)))
            meta.append({"video_id": vid, "strategy": kind, "track": int(track),
                         "coverage": float(cov), "window": int(wid)})

        for w0 in range(0, n - win + 1, st):
            w1 = w0 + win

            # A: what extract_window picks. It does not gate.
            base = wbuild.extract_window(kp, sc, tid, np_, w0, w1, cfg, width, height)
            if base is not None:
                add_ready(base["keypoints"], base["scores"], base["coverage"],
                          "A", base["track_id"], wid)

            # B: every single track in the window.
            ids = sorted({int(tid[t, p]) for t in range(w0, w1)
                          for p in range(int(np_[t])) if int(tid[t, p]) >= 0})
            for track in ids:
                series = wbuild._track_series(kp[w0:w1], sc[w0:w1], tid[w0:w1],
                                              np_[w0:w1], track,
                                              cfg.max_interp_gap_frames)
                k_arr, s_arr, present, _ = series
                add_raw(k_arr, s_arr, float(present.mean()), "B", track, wid)

            # C: B + ID-handoff chains.
            for chain in chains:
                members = set(chain)
                series = wbuild._group_series(kp[w0:w1], sc[w0:w1], tid[w0:w1],
                                              np_[w0:w1], members,
                                              cfg.max_interp_gap_frames)
                k_arr, s_arr, present, _ = series
                add_raw(k_arr, s_arr, float(present.mean()), "C", min(members), wid)
            wid += 1
    arr = spec_mod.stack_batch(xs)
    return arr, np.array([m["coverage"] for m in meta], dtype=np.float64), meta


def v2_test_inputs() -> tuple[np.ndarray, np.ndarray]:
    """The 800 v2 test windows, straight from the frozen windows layer."""
    import yaml

    rows = paths.read_csv(SOURCE_ROOT / "datasets" / "windows" / "v2" / "windows.csv")
    test = [r for r in rows if r["split"] == "test"]
    with np.load(SOURCE_ROOT / "datasets" / "windows" / "v2" / "windows.npz",
                 allow_pickle=False) as d:
        kp = d["keypoints"]
        sc = d["scores"]
        labels = d["labels"]
    # windows.npz is in csv order; index by the row position.
    pos = {r["window_id"]: i for i, r in enumerate(rows)}
    idx = [pos[r["window_id"]] for r in test]
    kp = kp[idx].astype(np.float32)
    sc = sc[idx].astype(np.float32)
    xs = [spec_mod.to_model_input(kp[i], sc[i]) for i in range(len(idx))]
    return spec_mod.stack_batch(xs), labels[idx]


# ------------------------------------------------------------------- the gates


def torch_logits(model, x: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch])).numpy())
    return np.concatenate(out)


def onnx_logits(path: Path, x: np.ndarray) -> np.ndarray:
    """Run the static batch-1 graph one sample at a time."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(len(x)):
        out.append(sess.run(None, {name: x[i:i + 1]})[0])
    return np.concatenate(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    paths.PARITY.mkdir(parents=True, exist_ok=True)

    print("weights:")
    for name, want in paths.EXPECTED_SHA256.items():
        got = paths.sha256_file(paths.WEIGHTS / name)
        print(f"  {name:24s} {got[:16]}... {'OK' if got == want else 'MISMATCH'}")
        if got != want:
            raise SystemExit("wrong weight")

    model, ckpt = load_model()
    print(f"checkpoint: epoch={ckpt.get('epoch')} metrics_keys={list((ckpt.get('metrics') or {}).keys())[:6]}")

    # ---- export
    wrapper = export_mod.STGCNExport(model, patch=not args.no_patch).eval()
    print(f"unit_gcn blocks patched: {wrapper.n_patched}")
    dummy = torch.zeros(*spec_mod.MODEL_INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        eager = wrapper(dummy)
    print(f"eager output {tuple(eager.shape)} on input {tuple(dummy.shape)}")

    torch.onnx.export(
        wrapper, dummy, str(ONNX_PATH),
        opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=None, do_constant_folding=True, verbose=False,
    )
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.2f} MB)")

    import onnx
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"opset={args.opset} nodes={len(m.graph.node)} Einsum={ops.get('Einsum', 0)}")
    print("op histogram:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]))

    # ---- G1 on board samples
    xb, cov_b, meta_b = board_inputs()
    print(f"board inputs: {xb.shape}")
    # Guard against silently rebuilding a subset: an earlier version of this
    # function skipped strategy A and quietly produced 12,310 of 13,223 samples,
    # which would have let G1 pass on 93% of the board.
    ref_rows = paths.read_csv(paths.REFERENCE / "board_samples.csv")
    if len(xb) != len(ref_rows):
        raise SystemExit(
            f"rebuilt {len(xb)} board samples but the frozen reference has "
            f"{len(ref_rows)}; strategies A/B/C must all be present")
    n_by = {}
    for m in meta_b:
        n_by[m["strategy"]] = n_by.get(m["strategy"], 0) + 1
    print(f"  by strategy: {n_by}")
    # compare against the *unpatched* model: that is the definition of correct
    unpatched = export_mod.STGCNLogits(model).eval()
    pt_b = torch_logits(unpatched, xb)
    on_b = onnx_logits(ONNX_PATH, xb)
    d_b = np.abs(pt_b - on_b)
    max_b = float(d_b.max())
    flip_b = int(((pt_b[:, 1] - pt_b[:, 0] > 0) != (on_b[:, 1] - on_b[:, 0] > 0)).sum())
    print(f"G1 board: max|dlogit|={max_b:.3e} decision flips={flip_b}/{len(pt_b)}")

    # ---- G1 on v2 test
    result = {"opset": args.opset, "patched": not args.no_patch,
              "onnx_bytes": ONNX_PATH.stat().st_size,
              "op_histogram": ops, "board": {}, "v2_test": {}}
    result["board"] = {"n": int(len(pt_b)), "max_abs_dlogit": max_b,
                       "decision_flips": flip_b}
    if not args.skip_v2:
        xv, yv = v2_test_inputs()
        print(f"v2 test inputs: {xv.shape}")
        pt_v = torch_logits(unpatched, xv)
        on_v = onnx_logits(ONNX_PATH, xv)
        d_v = np.abs(pt_v - on_v)
        max_v = float(d_v.max())
        flip_v = int(((pt_v[:, 1] - pt_v[:, 0] > 0) != (on_v[:, 1] - on_v[:, 0] > 0)).sum())
        # decision at the project threshold, on softmax(prob_fall) >= 0.5
        pf_pt = spec_mod.softmax2(pt_v)[:, 1]
        pf_on = spec_mod.softmax2(on_v)[:, 1]
        dec_flip = int(((pf_pt >= 0.5) != (pf_on >= 0.5)).sum())
        print(f"G1 v2:    max|dlogit|={max_v:.3e} argmax flips={flip_v}/{len(pt_v)} "
              f"threshold flips={dec_flip}")
        result["v2_test"] = {"n": int(len(pt_v)), "max_abs_dlogit": max_v,
                             "argmax_flips": flip_v, "threshold_flips": dec_flip,
                             "n_positive": int((yv == 1).sum())}

    ok = (max_b <= TOL and flip_b == 0
          and (args.skip_v2 or (result["v2_test"]["max_abs_dlogit"] <= TOL
                                and result["v2_test"]["threshold_flips"] == 0)))
    result["tolerance"] = TOL
    result["passed"] = bool(ok)

    # manifest line
    result["onnx_sha256"] = paths.sha256_file(ONNX_PATH)
    result["source_weight_sha256"] = paths.EXPECTED_SHA256["stgcnpp_v2_best.pth"]
    result["input_shape"] = list(spec_mod.MODEL_INPUT_SHAPE)
    result["output_shape"] = [1, spec_mod.NUM_CLASSES]
    paths.write_json(paths.PARITY / "g1_stgcn.json", result)

    lines = ["# G1 — ST-GCN++ ONNX 数值对账", "",
             f"- 源权重 `weights/stgcnpp_v2_best.pth` sha256 `{result['source_weight_sha256'][:16]}…`",
             f"- 产出 `models/stgcnpp_v2.onnx` sha256 `{result['onnx_sha256'][:16]}…`，"
             f"{result['onnx_bytes']/1e6:.2f} MB",
             f"- opset {args.opset}，静态 shape，输入 `{tuple(spec_mod.MODEL_INPUT_SHAPE)}`，"
             f"输出 `(1, {spec_mod.NUM_CLASSES})` logits（softmax 留给 CPU）",
             f"- einsum→matmul 改写：{'已应用（' + str(wrapper.n_patched) + ' 个 unit_gcn 块）' if not args.no_patch else '未应用（诊断模式）'}",
             f"- 图中 `Einsum` 节点数：**{ops.get('Einsum', 0)}**",
             "", "| 输入集 | 样本数 | `max\\|Δlogit\\|` | 判定翻转 | 判据 | 结果 |",
             "|---|---:|---:|---:|---|---|",
             f"| 板端 915 窗（A/B/C 全样本） | {result['board']['n']} | {max_b:.3e} | {flip_b} "
             f"| ≤{TOL:g}，翻转 0 | {'✅' if max_b <= TOL and flip_b == 0 else '❌'} |"]
    if not args.skip_v2:
        v = result["v2_test"]
        lines.append(f"| v2 test 800 窗 | {v['n']} | {v['max_abs_dlogit']:.3e} "
                     f"| {v['threshold_flips']} | ≤{TOL:g}，翻转 0 "
                     f"| {'✅' if v['max_abs_dlogit'] <= TOL and v['threshold_flips'] == 0 else '❌'} |")
    lines += ["", f"**G1：{'通过' if ok else '未通过'}**", ""]
    (paths.PARITY / "g1_stgcn.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
