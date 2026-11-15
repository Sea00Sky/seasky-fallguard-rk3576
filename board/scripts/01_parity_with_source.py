"""验证 `src/yolo_decode.py` 与源项目的 `yolo_export.py` 逐位一致。"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import spec as spec_mod                       # noqa: E402
from src import yolo_decode as mine                    # noqa: E402

SOURCE_ROOT = ROOT.parent / "YOLOV26n-pose+ST-GCN++-量化部署"


def load_source():
    """把源项目的 yolo_export 当成独立模块加载（它 `import torch`，PC 上有）。"""
    p = SOURCE_ROOT / "src" / "deploy" / "yolo_export.py"
    if not p.exists():
        raise SystemExit(f"找不到源文件: {p}")
    # 源文件用相对导入 `from ..deploy import spec` 吗？检查一下。
    text = p.read_text(encoding="utf-8")
    if "from .." in text or "from ." in text:
        # 用包路径导入，让相对导入能解析
        sys.path.insert(0, str(SOURCE_ROOT))
        import src.deploy.yolo_export as mod   # noqa: F401
        return mod
    spec = importlib.util.spec_from_file_location("source_yolo_export", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["source_yolo_export"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_frames(path: Path, n: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开 {path}")
    out = []
    try:
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
    finally:
        cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=120)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    src = load_source()
    print(f"源实现: {src.__file__}")
    print(f"本实现: {mine.__file__}")

    # 找一段有人的视频
    if args.video:
        vp = Path(args.video)
    else:
        cands = sorted((ROOT / "videos").rglob("*.mp4"))
        if not cands:
            raise SystemExit("videos/ 下没有 mp4")
        vp = cands[0]
    print(f"视频: {vp}")

    frames = read_frames(vp, args.n_frames)
    print(f"帧数: {len(frames)}")

    # ---- preprocess -------------------------------------------------------
    max_pre = 0.0
    params_list = []
    for f in frames:
        a, pa = src.preprocess(f, imgsz=spec_mod.IMGSZ)
        b, pb = mine.preprocess(f, imgsz=spec_mod.IMGSZ)
        max_pre = max(max_pre, float(np.abs(a - b).max()))
        params_list.append((pa, pb))
    same_params = all(pa == pb for pa, pb in params_list)
    print(f"\npreprocess  max|diff| = {max_pre:.3e}   params 一致 = {same_params}")

    # ---- decode -----------------------------------------------------------
    # 用一个合成输出，保证有框跨过阈值（真实输出需要跑模型）。
    rng = np.random.default_rng(0)
    max_box = max_kpt = max_sc = 0.0
    n_kept = 0
    for pa, _ in params_list[:20]:
        raw = np.zeros((1, 56, mine.N_ANCHORS), np.float32)
        # 前 6 个 anchor 给高置信度，位置随机
        raw[0, 4, :6] = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.3], np.float32)
        raw[0, 0:4, :6] = rng.uniform(0, 600, (4, 6)).astype(np.float32)
        raw[0, 5:, :6] = rng.uniform(0, 1, (51, 6)).astype(np.float32)

        ba, ka, sa = src.decode(raw, pa, conf=spec_mod.CONF,
                                max_persons=spec_mod.MAX_PERSONS)
        bb, kb, sb = mine.decode(raw, pa, conf=spec_mod.CONF,
                                 max_persons=spec_mod.MAX_PERSONS)
        if ba.shape != bb.shape or ka.shape != kb.shape or sa.shape != sb.shape:
            print(f"  FAIL 形状不一致: {ba.shape} vs {bb.shape}")
            return 1
        n_kept += len(ba)
        if len(ba):
            max_box = max(max_box, float(np.abs(ba - bb).max()))
            max_kpt = max(max_kpt, float(np.abs(ka - kb).max()))
            max_sc = max(max_sc, float(np.abs(sa - sb).max()))

    print(f"decode      max|dbox| = {max_box:.3e}  "
          f"max|dkpt| = {max_kpt:.3e}  max|dscore| = {max_sc:.3e}")
    print(f"            共解码 {n_kept} 个框")

    ok = (max_pre == 0.0 and same_params and max_box == 0.0
          and max_kpt == 0.0 and max_sc == 0.0 and n_kept > 0)
    print()
    if ok:
        print("逐位一致 ")
        return 0
    print("不一致 ")
    if n_kept == 0:
        print("  注意：一个框都没解码出来，说明合成数据没跨过阈值，这次比较没有信息量")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
