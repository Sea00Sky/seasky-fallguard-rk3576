"""Probe: can the toolkit's two-step hybrid quantisation keep the output layer
in fp16, and does that fix G5?"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deploy import paths  # noqa: E402

MODELS = {
    "stgcn": {
        "onnx": "stgcnpp_v2.onnx",
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 1, 64, 17, 3],
        "mean": [[0]],
        "std": [[1]],
        "calib": "stgcn_calib.txt",
        "stem": "stgcnpp_v2",
    },
    "yolo": {
        "onnx": "yolo26n_pose_480x640.onnx",
        "input_name": "images",
        "output_name": "output0",
        "input_shape": [1, 3, 480, 640],
        "mean": [[0, 0, 0]],
        "std": [[1, 1, 1]],
        "calib": "yolo_calib.txt",
        "stem": "yolo26n_pose_480x640",
    },
}


def _layer_blocks(cfg_text: str) -> dict[str, tuple[int, int]]:
    """Return {layer_name: (start_line, end_line)} for `quantize_parameters`."""
    lines = cfg_text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.rstrip() == "quantize_parameters:")
    except StopIteration:
        raise SystemExit("no `quantize_parameters:` in the cfg; schema changed")
    blocks: dict[str, tuple[int, int]] = {}
    cur_name, cur_start = None, None
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if not l.strip():
            continue
        indent = len(l) - len(l.lstrip())
        if indent == 0:                       # left quantize_parameters
            break
        if indent == 4 and l.rstrip().endswith(":"):
            if cur_name is not None:
                blocks[cur_name] = (cur_start, i)
            cur_name, cur_start = l.strip()[:-1], i
    if cur_name is not None:
        blocks[cur_name] = (cur_start, len(lines))
    return blocks


def _set_dtype(lines: list[str], span: tuple[int, int], dtype: str) -> bool:
    """Rewrite the `dtype:` line inside one layer block. Returns True if found."""
    for i in range(*span):
        if re.match(r"^\s+dtype:\s*\S+\s*$", lines[i]):
            lines[i] = re.sub(r"(dtype:\s*)\S+", r"\g<1>" + dtype, lines[i])
            return True
    return False


def resolve_targets(blocks: dict[str, tuple[int, int]], want: str) -> list[str]:
    """Map the user's `--protect` argument onto real cfg layer names."""
    if want == "output":
        # Whatever the graph declares as its output. For ST-GCN that is exactly
        # `logits`; for YOLO it is `output0`, which is NOT a key in the cfg --
        # the concat's inputs are. Report that honestly instead of pretending.
        return [n for n in blocks if n in ("logits", "output0")]
    if want.startswith("re:"):
        pat = re.compile(want[3:])
        return [n for n in blocks if pat.search(n)]
    return [want] if want in blocks else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--protect", default="output",
                    help="cfg layer name, `re:<regex>`, or `output` for the "
                         "graph output itself")
    ap.add_argument("--dtype", default="float16",
                    help="quant dtype to force on the protected layers")
    ap.add_argument("--via", choices=["params", "custom"], default="custom",
                    help="`params` edits quantize_parameters[name].dtype (the "
                         "toolkit rejects this for the graph output and its "
                         "upstream); `custom` fills the cfg's "
                         "custom_quantize_layers map, which is the separate "
                         "top-level key the toolkit provides for this")
    ap.add_argument("--work", required=True,
                    help="scratch directory; step1 writes its three files here")
    ap.add_argument("--list", action="store_true",
                    help="only list candidate layer names, then exit")
    args = ap.parse_args()

    spec = MODELS[args.model]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=spec["mean"], std_values=spec["std"],
                target_platform="rk3576", quantized_dtype="w8a8",
                quantized_algorithm="normal", optimization_level=3)
    rknn.load_onnx(model=str(paths.MODELS / spec["onnx"]),
                   inputs=[spec["input_name"]],
                   input_size_list=[spec["input_shape"]],
                   outputs=[spec["output_name"]])

    cwd = Path.cwd()
    try:
        import os
        os.chdir(work)                       # step1 writes into the cwd
        if rknn.hybrid_quantization_step1(
                dataset=str(paths.CALIB / spec["calib"])) != 0:
            raise SystemExit("hybrid_quantization_step1 failed")
    finally:
        os.chdir(cwd)

    cfg_path = work / f"{spec['stem']}.quantization.cfg"
    model_path = work / f"{spec['stem']}.model"
    data_path = work / f"{spec['stem']}.data"
    for p in (cfg_path, model_path, data_path):
        if not p.exists():
            raise SystemExit(
                f"step1 did not produce {p.name}; it wrote "
                f"{sorted(x.name for x in work.iterdir())}")

    text = cfg_path.read_text(encoding="utf-8")
    blocks = _layer_blocks(text)
    print(f"cfg has {len(blocks)} layer entries")

    if args.list:
        for n in blocks:
            print("   ", n)
        return 0

    targets = resolve_targets(blocks, args.protect)
    if not targets:
        near = [n for n in blocks if args.protect.split(":")[-1] in n]
        raise SystemExit(
            f"`--protect {args.protect}` matched no cfg entry. "
            f"Closest names containing that substring: {near[:10]}. "
            f"Run with --list to see all {len(blocks)} entries.")
    print(f"protecting {len(targets)} layer(s) as {args.dtype}: {targets}")

    # Two routes, and which one works is not documented anywhere -- the toolkit
    # core is compiled. Learned the hard way:
    # * editing `quantize_parameters[name]['dtype']` is REJECTED for the graph
    # output and for its immediate upstream:
    # `ValueError: The quantize_parameters['logits']['dtype'] is not
    # allowed to be modified!`
    # * the cfg's first line is `custom_quantize_layers: {}`, a separate
    # top-level key. That is presumably the sanctioned route.
    lines = text.splitlines()
    if args.via == "custom":
        if not lines or not lines[0].startswith("custom_quantize_layers:"):
            raise SystemExit("cfg does not start with custom_quantize_layers; "
                             "schema changed, do not guess")
        payload = ", ".join(f'"{n}": "{args.dtype}"' for n in targets)
        lines[0] = "custom_quantize_layers: {" + payload + "}"
        print(f"custom_quantize_layers -> {lines[0]}")
    else:
        for name in targets:
            if not _set_dtype(lines, blocks[name], args.dtype):
                raise SystemExit(f"{name}: no dtype line found in its cfg block")
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if rknn.hybrid_quantization_step2(
            model_input=str(model_path), data_input=str(data_path),
            model_quantization_cfg=str(cfg_path)) != 0:
        raise SystemExit("hybrid_quantization_step2 failed")

    out = work / f"{spec['stem']}_int8_hybrid.rknn"
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()
    # L-116: exit code 0 is not evidence. Assert the artefact.
    if not out.exists() or out.stat().st_size < 1_000_000:
        raise SystemExit(f"export claimed success but {out} is missing or tiny")
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    print(f"copy it to models/ under the name scripts/21 expects, then verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
