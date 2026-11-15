"""从 `reference/` 复现 PC 基线，验证本项目的评测口径与上游一致。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import spec                             # noqa: E402
from scripts.run_video import EXCLUDE, load_fall_intervals   # noqa: E402

REF = ROOT / "reference"

# 取自量化侧 g3 对拍结果（g3_rknn_fp16.json 的 `runs.rknn` 节），该产物不入库
EXPECT = {
    "n_windows": 915,
    "segment_recall": (14, 17),
    "videos_with_fall": (9, 9),
    "no_fall_false_alarms": (3, 7),
}

FAILS: list[str] = []


def check(name, got, want) -> None:
    ok = got == want
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:34s} got={got}  want={want}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("=" * 70)
    print("PC 基线口径核对（只读 reference/）")
    print("=" * 70)

    # ---- 1. 段召回 --------------------------------------------------------
    # `g3_per_segment.csv` 是 ONNX 臂的逐段结果（13/17），不是 RKNN 的。
    # RKNN 基线在 `g3_rknn_fp16.json` 的 `runs.rknn.segments` 里（14/17）。
    # 两个数字都对，只是问的不是同一件事——把这一条当"基线"会直接搞错预期。
    print("\n-- 段召回（g3_per_segment.csv = ONNX 臂）--")
    seg = list(csv.DictReader(open(REF / "g3_per_segment.csv",
                                   encoding="utf-8-sig")))
    check("per_segment 行数", len(seg), 17)
    check("ONNX 臂段召回", (sum(int(r["hit"]) for r in seg), len(seg)), (13, 17))

    # ---- 2. 排除集 --------------------------------------------------------
    print("\n-- 排除集 --")
    gt = load_fall_intervals(REF / "board_fall_intervals.csv")
    n_seg_all = sum(len(v) for v in gt.values())
    n_excluded = sum(len(gt.get(v, [])) for v in EXCLUDE)
    check("GT 总段数", n_seg_all, 19)
    check("被排除的段数", n_excluded, 2)
    check("19 - 2 = 17", n_seg_all - n_excluded, 17)

    # ---- 3. 有跌倒视频 ----------------------------------------------------
    print("\n-- 有跌倒视频 --")
    with_fall = {v for v in gt if gt[v]} - EXCLUDE
    check("有跌倒视频数", len(with_fall), 9)

    # ---- 4. no_fall 误报 --------------------------------------------------
    print("\n-- no_fall 误报（扫描表，按 C/0.70 过滤）--")
    all_rows = list(csv.DictReader(open(REF / "per_video_nofall.csv",
                                        encoding="utf-8-sig")))
    check("扫描表总行数", len(all_rows), 84)
    c70 = [r for r in all_rows
           if r["strategy"] == "C" and r["gate"] == "0.70"]
    check("C/0.70 行数", len(c70), 7)
    fa = sum(int(r["false_alarm"]) for r in c70)
    check("误报", (fa, len(c70)), EXPECT["no_fall_false_alarms"])

    # 阈值口径：false_alarm 应该等于 max_prob >= 0.5
    recomputed = sum(1 for r in c70
                     if float(r["max_prob"]) >= spec.THRESHOLD)
    check("误报 = max_prob >= 0.5", recomputed, fa)

    # ---- 5. 全链路基线 ----------------------------------------------------
    print("\n-- 全链路基线（g3_rknn_fp16.json）--")
    import json
    g3 = json.loads((REF / "g3_rknn_fp16.json").read_text(encoding="utf-8"))
    rk = g3["runs"]["rknn"]
    check("窗口数", rk["windows"]["n_mine"], EXPECT["n_windows"])
    check("段召回", (rk["segments"]["seg_hit"], rk["segments"]["seg_tot"]),
          EXPECT["segment_recall"])
    check("有跌倒视频", (rk["segments"]["pos_h"], rk["segments"]["pos_v"]),
          EXPECT["videos_with_fall"])
    check("no_fall 误报", (rk["segments"]["neg_fa"], rk["segments"]["neg_v"]),
          EXPECT["no_fall_false_alarms"])

    # 与 ONNX 的差异（说明板端预期不是 13/17）
    print("\n-- 与 ONNX 的差异（决定板端预期值）--")
    onnx_seg = g3["runs"]["onnx"]["segments"]
    print(f"     ONNX  段召回 {onnx_seg['seg_hit']}/{onnx_seg['seg_tot']}")
    print(f"     RKNN  段召回 {rk['segments']['seg_hit']}/{rk['segments']['seg_tot']}")
    check("RKNN 比 ONNX 多检出一段（预期方向）",
          rk["segments"]["seg_hit"] - onnx_seg["seg_hit"], 1)
    flips = g3["rknn_vs_onnx"]["threshold_flips"]
    print(f"     阈值翻转 {flips} 个窗口（证据，不设门）")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项 —— 口径与上游不一致，板端数字将不可比")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("口径一致   run_video.py 的数字可以直接与基线对照")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
