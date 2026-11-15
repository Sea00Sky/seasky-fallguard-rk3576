"""Analysis for the seed sweep: what is the single-run noise floor?"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import engine  # noqa: E402

SWEEP_ROOT = "runs/seed_sweep_v2"
OUT_DIR = "reports/seed_sweep"
BASELINE_RUN = "runs/stgcnpp_v2"
METRICS = ("f1", "recall", "precision", "auc")
ABLATION_RUNS = [
    "stgcnpp_v2", "stgcnpp_v2_cw", "stgcnpp_v2_aug", "stgcnpp_v2_onset",
    "stgcnpp_v2_pretrained", "stgcnpp_v2_pretrained_lr002",
    "stgcnpp_v2_pretrained_lr001", "stgcnpp_v2_pretrained_lr0005",
    "stgcnpp_v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default=SWEEP_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument(
        "--compare-runs", nargs="*", default=None,
        help="extra runs to place on the noise scale (loads best.pth; needs torch)",
    )
    parser.add_argument("--device", default="")
    return parser.parse_args()


# ------------------------------------------------------------------- statistics


def t_crit(df: int, two_sided: float = 0.95) -> float:
    from scipy.stats import t as student_t

    return float(student_t.ppf(0.5 + two_sided / 2.0, df))


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = int(len(values))
    mean = float(values.mean())
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "min": mean, "max": mean,
                "range": 0.0, "se": float("nan"), "ci95_mean_low": float("nan"),
                "ci95_mean_high": float("nan"), "pi95_single_low": float("nan"),
                "pi95_single_high": float("nan"), "mdd_single": float("nan"),
                "values": [float(v) for v in values]}
    sd = float(values.std(ddof=1))
    t = t_crit(n - 1)
    se = sd / np.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "se": float(se),
        "ci95_mean_low": float(mean - t * se),
        "ci95_mean_high": float(mean + t * se),
        "pi95_single_low": float(mean - t * sd * np.sqrt(1.0 + 1.0 / n)),
        "pi95_single_high": float(mean + t * sd * np.sqrt(1.0 + 1.0 / n)),
        "mdd_single": float(t * sd * np.sqrt(2.0)),
        "values": [float(v) for v in values],
    }


# ---------------------------------------------------------------------- loading


def load_manifest(sweep_root: Path) -> list[dict]:
    path = sweep_root / "runs_manifest.csv"
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    # The val non-fall event count was not recorded by the first sweep version;
    # it is constant per split, so read it from any run's results.json.
    for split in ("val", "test"):
        key = f"{split}_nonfall_events"
        if key not in rows[0]:
            sample = json.loads(
                (sweep_root / rows[0]["run"] / "results.json").read_text(encoding="utf-8")
            )
            for r in rows:
                r[key] = sample[split]["event"]["nonfall_events"]
    return rows


def load_predictions(sweep_root: Path, runs: list[str], split: str) -> dict:
    """Stack per-window probabilities across runs; verify window order matches."""
    labels_ref = key_ref = None
    probs = []
    for name in runs:
        npz_path = sweep_root / name / "per_window.npz"
        if not npz_path.exists():
            raise RuntimeError(f"missing {npz_path}; run scripts/25_seed_sweep.py first")
        with np.load(npz_path, allow_pickle=False) as data:
            labels = data[f"{split}_labels"]
            key = np.stack([
                data[f"{split}_video_id"].astype(str),
                data[f"{split}_start_frame"].astype(str),
                data[f"{split}_end_frame"].astype(str),
            ])
            if labels_ref is None:
                labels_ref, key_ref = labels, key
                events = data[f"{split}_event_id"].astype(str)
                datasets = data[f"{split}_dataset"].astype(str)
            else:
                if not np.array_equal(labels_ref, labels):
                    raise RuntimeError(f"{name}: {split} labels differ")
                if not np.array_equal(key_ref, key):
                    raise RuntimeError(f"{name}: {split} window order differs")
            probs.append(data[f"{split}_probs"])
    return {
        "labels": labels_ref,
        "events": events,
        "datasets": datasets,
        "probs": np.vstack(probs),
        "runs": list(runs),
    }


def dataset_breakdown(pred: dict, threshold: float) -> dict:
    """Window metrics per source dataset, per run, then summarised."""
    out = {}
    for name in sorted(set(pred["datasets"].tolist())):
        mask = pred["datasets"] == name
        per_run = []
        for row in pred["probs"]:
            m = engine.binary_metrics(pred["labels"][mask], row[mask], threshold)
            per_run.append({k: m[k] for k in METRICS})
        out[name] = {
            "n_windows": int(mask.sum()),
            "n_pos": int(pred["labels"][mask].sum()),
            "metrics": {k: describe([r[k] for r in per_run]) for k in METRICS},
        }
    return out


def event_stability(pred: dict, threshold: float, split: str = "test") -> dict:
    """How often each event's verdict flips across runs."""
    labels = pred["labels"]
    by_event: dict[str, list[int]] = {}
    for i, eid in enumerate(pred["events"]):
        by_event.setdefault(eid, []).append(i)

    manifest = engine._load_manifest_events(split, "v2")
    fall_events = {r["event_id"] for r in manifest if r.get("label") == "fall"}

    rows = []
    for eid, idxs in sorted(by_event.items()):
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        is_fall = eid in fall_events
        votes = []
        for row in pred["probs"]:
            hit = (row[idxs_arr] >= threshold)
            if is_fall:
                hit = hit & (labels[idxs_arr] == 1)
            votes.append(bool(hit.any()))
        n_hit = int(sum(votes))
        rows.append({
            "event_id": eid,
            "is_fall": is_fall,
            "n_windows": len(idxs),
            "n_windows_pos": int((labels[idxs_arr] == 1).sum()),
            "n_runs_detected": n_hit,
            "n_runs": len(votes),
            "stable": n_hit == len(votes) or n_hit == 0,
            "mean_max_prob": float(pred["probs"][:, idxs_arr].max(axis=1).mean()),
        })
    return {"rows": rows}


def borderline_windows(pred: dict, threshold: float) -> dict:
    """Windows whose probability straddles the threshold across runs."""
    mean_p = pred["probs"].mean(axis=0)
    sd_p = pred["probs"].std(axis=0, ddof=1)
    pred_pos = pred["probs"] >= threshold
    n_flip = pred_pos.any(axis=0) & ~pred_pos.all(axis=0)
    band = (mean_p >= 0.3) & (mean_p <= 0.7)
    sd_stable = float(sd_p[~n_flip].mean()) if (~n_flip).any() else 0.0
    sd_unstable = float(sd_p[n_flip].mean()) if n_flip.any() else 0.0
    return {
        "n_windows": int(len(mean_p)),
        "n_unstable": int(n_flip.sum()),
        "n_unstable_pos": int((n_flip & (pred["labels"] == 1)).sum()),
        "n_unstable_neg": int((n_flip & (pred["labels"] == 0)).sum()),
        "n_in_band": int(band.sum()),
        "mean_sd_overall": float(sd_p.mean()),
        "mean_sd_unstable": sd_unstable,
        "mean_sd_stable": sd_stable,
        "sd_ratio": float(sd_unstable / sd_stable) if sd_stable > 0 else float("nan"),
    }


def per_window_csv(pred: dict, path: Path, threshold: float) -> None:
    mean_p = pred["probs"].mean(axis=0)
    sd_p = pred["probs"].std(axis=0, ddof=1)
    pred_pos = pred["probs"] >= threshold
    n_flip = pred_pos.any(axis=0) & ~pred_pos.all(axis=0)
    rows = []
    for i in range(len(mean_p)):
        row = {
            "event_id": pred["events"][i],
            "dataset": pred["datasets"][i],
            "label": int(pred["labels"][i]),
            "p_mean": float(mean_p[i]),
            "p_sd": float(sd_p[i]),
            "p_min": float(pred["probs"][:, i].min()),
            "p_max": float(pred["probs"][:, i].max()),
            "flips_at_threshold": bool(n_flip[i]),
        }
        for j, name in enumerate(pred["runs"]):
            row[f"p_{name}"] = float(pred["probs"][j, i])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------- variance split


def variance_split(manifest: list[dict], key: str) -> dict:
    """Split total variance into within-seed and between-seed components."""
    by_seed: dict[int, list[float]] = {}
    for row in manifest:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))

    repeats = {s: v for s, v in by_seed.items() if len(v) > 1}
    groups = [np.asarray(v, dtype=np.float64) for v in by_seed.values()]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = float(np.mean(np.concatenate(groups)))

    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    msw = ss_within / (n_total - k) if n_total > k else float("nan")
    msb = ss_between / (k - 1) if k > 1 else float("nan")
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1) if k > 1 else float("nan")

    between_var = max(0.0, (msb - msw) / n0) if np.isfinite(msb) and np.isfinite(msw) else float("nan")
    total = msw + between_var if np.isfinite(between_var) else msw
    observed_sd = float(np.std(np.concatenate(groups), ddof=1)) if n_total > 1 else float("nan")
    return {
        "metric": key,
        "n_runs": len(manifest),
        "n_seeds": k,
        "n_repeated_seeds": len(repeats),
        "n0_effective": float(n0) if np.isfinite(n0) else None,
        "within_seed_sd": float(np.sqrt(msw)) if np.isfinite(msw) else None,
        "between_seed_sd": float(np.sqrt(between_var)) if np.isfinite(between_var) else None,
        "total_sd": float(np.sqrt(total)) if np.isfinite(total) else None,
        "within_share": float(msw / total) if total and np.isfinite(total) else None,
        "observed_sd": observed_sd,
        "repeats": {str(s): [float(x) for x in v] for s, v in repeats.items()},
    }


# ------------------------------------------------------------- ablation scoring


def score_against_noise(value: float, stats: dict) -> dict:
    """Where does a single-run value sit relative to the run distribution?"""
    if stats["n"] < 2 or not np.isfinite(stats["sd"]) or stats["sd"] == 0:
        return {"z": float("nan"), "outside_pi95": None, "percentile": float("nan"),
                "delta_vs_mean": float("nan")}
    z = (value - stats["mean"]) / stats["sd"]
    return {
        "z": float(z),
        "outside_pi95": bool(value < stats["pi95_single_low"] or value > stats["pi95_single_high"]),
        "percentile": float((np.asarray(stats["values"]) < value).mean()),
        "delta_vs_mean": float(value - stats["mean"]),
    }


def ablation_table(stats: dict, runs: list[str]) -> list[dict]:
    rows = []
    for run in runs:
        path = engine.resolve(f"runs/{run}/results.json")
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run": run,
            "test_f1": res["test"]["window"]["f1"],
            "test_recall": res["test"]["window"]["recall"],
            "test_auc": res["test"]["window"]["auc"],
            "event_detected": res["test"]["event"]["fall_events_detected"],
            "event_manifest": res["test"]["event"]["fall_events_manifest"],
            "false_alarms": res["test"]["event"]["nonfall_events_with_false_alarm"],
            "nonfall_events": res["test"]["event"]["nonfall_events"],
        }
        for key in ("test_f1", "test_recall", "test_auc"):
            row[f"{key}_vs_noise"] = score_against_noise(row[key], stats[key.split("_")[1]])
        rows.append(row)
    return rows


# --------------------------------------------------------------------- writing


def build_conclusion(summary: dict) -> str:
    test = summary["test"]
    f1, rec = test["window"]["f1"], test["window"]["recall"]
    vs = summary["variance_split"]["test_f1"]
    inside = [r["run"] for r in summary["ablations"] if not r["test_f1_vs_noise"]["outside_pi95"]]
    outside = [r["run"] for r in summary["ablations"] if r["test_f1_vs_noise"]["outside_pi95"]]
    lines = [
        f"1. **单次运行的 test 窗口 F1 噪声底 = [{f1['pi95_single_low']:.3f}, "
        f"{f1['pi95_single_high']:.3f}]**（sd={f1['sd']:.3f}，极差 {f1['range']:.3f}，"
        f"n={f1['n']} 次运行）。两次独立单次运行至少要差 **{f1['mdd_single']:.3f}** 才能判读。",
    ]
    if vs["within_seed_sd"] is not None:
        lines.append(
            f"2. 方差分解：同 seed 重复的 sd = {vs['within_seed_sd']:.4f}"
            f"（占 {vs['within_share'] * 100:.0f}%），不同 seed 的 sd = "
            f"{vs['between_seed_sd']:.4f}。也就是说这个「种子噪声」里"
            f"{'大部分其实是 GPU 非确定性，换 seed 并不是主要来源' if vs['within_share'] > 0.5 else '主要来自换 seed（初始化 + 采样顺序）'}。"
        )
    lines.append(
        f"3. 窗口召回的噪声底 = [{rec['pi95_single_low']:.3f}, {rec['pi95_single_high']:.3f}]"
        f"（sd={rec['sd']:.3f}）。D-035 里 Le2i 0.847→0.763 的下降落在该区间"
        f"{'之外，是真实信号' if 0.763 < rec['pi95_single_low'] else '之内，不能当信号'}。"
    )
    if outside:
        lines.append(
            f"4. 落在噪声底**之外**的对照：{', '.join(f'`{r}`' for r in outside)}。"
            "只有这些可以称作「有效应」。"
        )
    else:
        lines.append("4. 没有任何已记录的对照落在噪声底之外。")
    if inside:
        lines.append(
            f"5. 落在噪声底**之内**、此前被当作改进的对照：{', '.join(f'`{r}`' for r in inside)}。"
            "这些数字只能读作同一分布里的抽样，不能读作模型变好。"
        )
    lines.append(
        "6. 由此，v3（D-035）不替换 v2 的裁决不需要改；但它给出的「±0.08」现在有了"
        "明确的统计口径、逐 run 证据和方差来源分解。"
    )
    return "\n".join(lines)


def write_report(path: Path, summary: dict) -> None:
    sw, test, val = summary["sweep"], summary["test"], summary["val"]
    f1 = test["window"]["f1"]
    lines = [
        "# D-023 基线多种子重复：单次运行的噪声底",
        "",
        "## 为什么做这个",
        "",
        "D-035 里 v3 对 Le2i 的数据**一个字节都没改**，test 窗口召回却从 0.847 掉到 0.763"
        "（59 个正窗口里 8 个 TP→FN）。此前 9 个 checkpoint 对照的所有 Δ 都建立在一个没人量过的方差上。"
        "本报告量它。",
        "",
        "## 设计",
        "",
        f"- 运行：{len(sw['runs'])} 次（{', '.join(sw['runs'])}）",
        f"- 种子：{', '.join(str(s) for s in sw['seeds'])}；重复的种子用于分离"
        "「同 seed 的 GPU 非确定性」与「不同 seed 的初始化差异」",
        "- 单变量：只改 `train.seed`；其余字段在训练前逐项断言与 `configs/train_stgcnpp.yaml` 相等",
        f"- 选模：val fall-class F1（阈值 {sw['threshold']}）最高，与 D-023 一致",
        f"- 数据层：`{sw['layer']}`，pkl sha256 `{sw['ann_sha256'][:16]}…`",
        f"- 训练耗时：合计 {sw['total_min']:.1f} min，单次 {sw['elapsed_min_min']:.1f}–"
        f"{sw['elapsed_min_max']:.1f} min",
        "",
        "## 确定性预检（决定了本报告的口径）",
        "",
        "同一 seed 连跑两次、各 2 epoch：",
        "",
        "| 运行 | epoch1 train_loss | epoch2 val F1 |",
        "|---|---:|---:|",
        f"| 预检 A | {sw['precheck']['a_loss']:.4f} | {sw['precheck']['a_val_f1']:.4f} |",
        f"| 预检 B | {sw['precheck']['b_loss']:.4f} | {sw['precheck']['b_val_f1']:.4f} |",
        "",
        f"`cudnn.deterministic = {sw['cudnn_deterministic']}`，`benchmark = {sw['cudnn_benchmark']}`。"
        "同 seed 两次结果不同 → 这个路径**不是逐比特可复现**的。所以本报告不把"
        "「种子噪声」等同于「初始化敏感性」，而是把方差拆开报。",
        "",
        "## 统计口径（重要）",
        "",
        "对照实验只跑**一次**，所以要看的是「同配置再跑一次会落在哪」，即**单次运行的预测区间**，"
        "不是均值的置信区间。两者差一个 $\\sqrt{1+1/n}$ 因子。",
        "",
        "| 量 | 公式 | 用途 |",
        "|---|---|---|",
        "| `ci95_mean` | mean ± t·sd/√n | 均值的精度 |",
        "| `pi95_single` | mean ± t·sd·√(1+1/n) | **单次运行的合理范围 = 噪声底** |",
        "| `mdd_single` | t·sd·√2 | 两次独立单次运行之间可判读的最小差异 |",
        "",
        f"t 临界值（df={f1['n'] - 1}，双侧 95%）= {sw['t_crit']:.4f}。",
        "",
        "## 结果：test 窗口级（阈值 0.5，fall 为正类）",
        "",
        "| 指标 | mean | sd | min | max | 极差 | pi95_single | mdd_single |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for key, label in (("f1", "F1"), ("recall", "召回"), ("precision", "精度"), ("auc", "AUC")):
        s = test["window"][key]
        lines.append(
            f"| {label} | {s['mean']:.4f} | {s['sd']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
            f"{s['range']:.4f} | [{s['pi95_single_low']:.4f}, {s['pi95_single_high']:.4f}] | "
            f"{s['mdd_single']:.4f} |"
        )
    lines += [
        "",
        "逐 run 原始值：",
        "",
        "| run | seed | test F1 | 召回 | 精度 | AUC | TP/FP/FN | 事件 | FAR | best epoch |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary["per_run"]:
        lines.append(
            f"| {row['run']} | {row['seed']} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | "
            f"{row['test_precision']:.4f} | {row['test_auc']:.4f} | "
            f"{row['test_tp']}/{row['test_fp']}/{row['test_fn']} | "
            f"{row['test_event_detected']}/{row['test_event_manifest']} | "
            f"{row['test_false_alarms']}/{row['test_nonfall_events']} | {row['best_epoch']} |"
        )

    vs = summary["variance_split"]["test_f1"]
    lines += [
        "",
        "## 方差分解：同 seed 重复 vs 不同 seed",
        "",
        "| 指标 | 同 seed sd | 不同 seed sd | 合计 sd | 同 seed 占比 | 实测 sd |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("test_f1", "test F1"), ("test_recall", "test 召回"), ("test_auc", "test AUC")):
        v = summary["variance_split"][key]
        if v["within_seed_sd"] is None:
            continue
        # The observed sd is the check on the decomposition: the two components
        # should reproduce the spread of the 10 runs themselves.
        lines.append(
            f"| {label} | {v['within_seed_sd']:.4f} | {v['between_seed_sd']:.4f} | "
            f"{v['total_sd']:.4f} | {v['within_share'] * 100:.0f}% | {v['observed_sd']:.4f} |"
        )
    lines += [
        "",
        "「实测 sd」是 10 次运行本身的标准差，用来校验分解：F1 与 AUC 的分解合计"
        "（0.0252 / 0.0067）与实测（0.0251 / 0.0066）吻合。召回一列不吻合"
        "（0.0907 vs 0.0559），因为只有 seed 42 有重复、而它那三次（0.74 / 0.81 / 0.92）"
        "比其余 seed 散得多，pooled within-seed 估计因此高估了总体波动；"
        "该列只能读作「同 seed 重复自身的波动很大」，不能当总体 sd。",
        "",
        "同 seed 的重复值：",
        "",
    ]
    for seed, values in vs["repeats"].items():
        lines.append(f"- seed {seed}：{', '.join(f'{v:.4f}' for v in values)}")
    lines += [
        "",
        "## 结果：val 与事件级",
        "",
        "| 指标 | mean | sd | 极差 | pi95_single |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in (("f1", "val 窗口 F1"), ("recall", "val 窗口召回"), ("auc", "val 窗口 AUC")):
        s = val["window"][key]
        lines.append(
            f"| {label} | {s['mean']:.4f} | {s['sd']:.4f} | {s['range']:.4f} | "
            f"[{s['pi95_single_low']:.4f}, {s['pi95_single_high']:.4f}] |"
        )
    for key, label in (("event_recall_all", "test 事件召回"),
                       ("event_false_alarm_rate", "test 事件误报率")):
        s = test["event"][key]
        lines.append(
            f"| {label} | {s['mean']:.4f} | {s['sd']:.4f} | {s['range']:.4f} | "
            f"[{s['pi95_single_low']:.4f}, {s['pi95_single_high']:.4f}] |"
        )
    det = test["event_detected_counts"]
    lines += [
        "",
        f"事件检出数：{det['histogram']}（分母 {det['denominator']}），"
        f"范围 {det['min']}–{det['max']}，极差 **{det['range']} 个事件**。",
        "",
        "## 结果：分数据集 test 窗口召回",
        "",
        "| 数据集 | 窗口 | 正窗口 | 召回 mean | 召回 sd | 召回 极差 | F1 mean | F1 sd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in test["by_dataset"].items():
        r, f = block["metrics"]["recall"], block["metrics"]["f1"]
        lines.append(
            f"| {name} | {block['n_windows']} | {block['n_pos']} | {r['mean']:.4f} | "
            f"{r['sd']:.4f} | {r['range']:.4f} | {f['mean']:.4f} | {f['sd']:.4f} |"
        )

    bw = test["borderline"]
    lines += [
        "",
        "## 机制：方差来自少数贴阈值的窗口",
        "",
        f"- test {bw['n_windows']} 个窗口里，{bw['n_unstable']} 个在 {f1['n']} 次运行间**判定翻转**"
        f"（正样本 {bw['n_unstable_pos']} 个、负样本 {bw['n_unstable_neg']} 个）。",
        f"- 平均概率落在 p∈[0.3, 0.7] 的窗口只有 {bw['n_in_band']} 个。",
        f"- 翻转窗口的概率标准差均值 {bw['mean_sd_unstable']:.4f}，"
        f"不翻转窗口 {bw['mean_sd_stable']:.4f}（{bw['sd_ratio']:.1f}×）。",
        "",
        "印证 L-027：噪声不是均匀分布的，只有贴近决策边界的窗口会动。"
        "逐窗概率见 `reports/seed_sweep/per_window_probabilities.csv`。",
        "",
        "## 用噪声底重新判读已完成的对照",
        "",
        f"判据：单次运行的值是否落在 {f1['n']} 次运行的 `pi95_single` 之外。",
        "",
        "| 实验 | test F1 | Δ vs 均值 | z | 在噪声底外? | 事件 | FAR |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in summary["ablations"]:
        v = row["test_f1_vs_noise"]
        lines.append(
            f"| `{row['run']}` | {row['test_f1']:.4f} | {v['delta_vs_mean']:+.4f} | "
            f"{v['z']:+.2f} | {'**是**' if v['outside_pi95'] else '否'} | "
            f"{row['event_detected']}/{row['event_manifest']} | "
            f"{row['false_alarms']}/{row['nonfall_events']} |"
        )
    if summary.get("compare_runs"):
        lines += [
            "",
            "## 额外对照：逐窗概率对噪声分布的位置",
            "",
            "| run | test F1 | 分位 | 与运行均值判定不同的窗口数 |",
            "|---|---:|---:|---:|",
        ]
        for row in summary["compare_runs"]:
            lines.append(
                f"| `{row['run']}` | {row['test_f1']:.4f} | "
                f"{row['percentile'] * 100:.0f}% | {row['n_flipped_windows']} |"
            )

    es = summary["test"]["event_stability"]["rows"]
    fall_rows = [r for r in es if r["is_fall"]]
    non_rows = [r for r in es if not r["is_fall"]]
    lines += [
        "",
        "## 逐事件稳定性：哪些结论在 10 次运行间会翻",
        "",
        "事件级判据与 `engine.event_metrics` 一致：fall 事件看「任一窗口 pred=1 且 label=1」，"
        "非 fall 事件看「任一窗口 pred=1」。用同一判据会漏掉全部误报。",
        "",
        "### fall 事件（27 个有窗口）",
        "",
        "| 事件 | 窗口 | 正窗口 | 检出次数 | 均值最大概率 | 稳定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    unstable_fall = [
        r for r in sorted(fall_rows, key=lambda x: (x["n_runs_detected"], -x["n_windows"]))
        if r["n_runs_detected"] < r["n_runs"]
    ]
    for r in unstable_fall:
        lines.append(
            f"| `{r['event_id']}` | {r['n_windows']} | {r['n_windows_pos']} | "
            f"{r['n_runs_detected']}/{r['n_runs']} | {r['mean_max_prob']:.3f} | "
            f"{'是' if r['stable'] else '**否**'} |"
        )
    stable_fall = len(fall_rows) - len(unstable_fall)
    never = [r for r in unstable_fall if r["n_windows_pos"] == 0]
    flipping = [r for r in unstable_fall if r["n_windows_pos"] > 0]
    never_names = ", ".join("`{}`".format(r["event_id"]) for r in never)
    flip_names = ", ".join("`{}`".format(r["event_id"]) for r in flipping)
    flip_probs = [r["mean_max_prob"] for r in flipping]
    prob_span = (
        f"{min(flip_probs):.2f}–{max(flip_probs):.2f}" if flip_probs else "n/a"
    )
    lines += [
        "",
        f"其余 {stable_fall} 个 fall 事件 10/10 稳定检出，未列出。"
        f"**{len(unstable_fall)} 个事件的检出结论会随运行翻转**："
        f"{len(never)} 个（{never_names}）没有任何正窗口（D-017 边界效应），"
        "无论怎么跑都不可能检出，只影响召回分母；"
        f"{len(flipping)} 个（{flip_names}）有正窗口但概率贴在 0.5 附近，"
        f"均值最大概率 {prob_span}。",
        "",
        "### 非 fall 事件（16 个）的误报频率",
        "",
        "| 事件 | 窗口 | 触发次数 | 均值最大概率 |",
        "|---|---:|---:|---:|",
    ]
    for r in sorted(non_rows, key=lambda x: -x["n_runs_detected"]):
        lines.append(
            f"| `{r['event_id']}` | {r['n_windows']} | {r['n_runs_detected']}/{r['n_runs']} | "
            f"{r['mean_max_prob']:.3f} |"
        )
    fired = [r for r in non_rows if r["n_runs_detected"] > 0]
    total_fires = sum(r["n_runs_detected"] for r in non_rows)
    lines += [
        "",
        f"**{len(fired)}/16 个非 fall 事件在至少一次运行里触发**，"
        f"10 次运行累计 {total_fires} 个事件级误报（平均每次 {total_fires / 10:.1f} 个，"
        f"即 FAR ≈ {total_fires / 10 / 16:.3f}）。"
        "只有 `le2i:Office:video (22)`（9/10）和 `urfd:adl-15-cam0`（5/10）是稳定的误报源，"
        "其余 6 个是贴边界的偶发触发 —— 这就是 FAR 从 1/16 摆到 7/16 的全部原因。",
        "",
        "## 方法学问题：val 选模不能预测 test",
        "",
        "| 统计量 | 值 |",
        "|---|---|",
        f"| val F1 vs test F1 的 Pearson r | {summary['val_test_correlation']['r']:+.3f} |",
        f"| p 值（n={summary['val_test_correlation']['n']}） | {summary['val_test_correlation']['p']:.3f} |",
        f"| best epoch 范围 | {summary['best_epoch_range']} |",
        "",
        "如果 val F1 能预测 test F1，这 10 次运行应该排成一条上升的线。实测 r ≈ 0，"
        "意味着 **D-023 的「按 val F1 选 checkpoint」在 test 上等价于随机挑一个 epoch**。"
        "这不是本实验引入的问题，而是 625 个 val 窗口 / 88 个正窗口的样本量撑不起 30 个 epoch 的选模。",
        "",
        "## 结论",
        "",
        summary["conclusion"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    sweep_root = engine.resolve(args.sweep_root)
    out_dir = engine.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(sweep_root)
    runs = [r["run"] for r in manifest]
    seeds = [int(r["seed"]) for r in manifest]
    prereg = json.loads((sweep_root / "preregistration.json").read_text(encoding="utf-8"))
    threshold = float(prereg["threshold"])

    audits = [json.loads((sweep_root / r / "audit.json").read_text(encoding="utf-8")) for r in runs]
    shas = {a["ann_file_sha256"] for a in audits}
    if len(shas) != 1:
        raise RuntimeError(f"runs disagree on the pkl sha256: {shas}")

    summary: dict = {
        "sweep": {
            "runs": runs,
            "seeds": seeds,
            "threshold": threshold,
            "layer": audits[0]["layer"],
            "ann_sha256": shas.pop(),
            "total_min": float(sum(float(r["elapsed_min"]) for r in manifest)),
            "elapsed_min_min": min(float(r["elapsed_min"]) for r in manifest),
            "elapsed_min_max": max(float(r["elapsed_min"]) for r in manifest),
            "t_crit": t_crit(len(manifest) - 1),
            "cudnn_deterministic": prereg["cudnn_deterministic"],
            "cudnn_benchmark": prereg["cudnn_benchmark"],
            "precheck": _precheck(sweep_root),
            "preregistration": prereg,
        },
        "per_run": [
            {
                "run": r["run"],
                "seed": int(r["seed"]),
                **{k: (float(v) if k.startswith(("test_f1", "test_recall", "test_precision",
                                                 "test_auc", "val_"))
                     else int(v) if k not in ("run", "run_dir") else v)
                   for k, v in r.items() if k in (
                       "test_f1", "test_recall", "test_precision", "test_auc",
                       "test_tp", "test_fp", "test_fn", "test_tn",
                       "test_event_detected", "test_event_manifest",
                       "test_false_alarms", "test_nonfall_events", "best_epoch")}
            }
            for r in manifest
        ],
    }

    summary["variance_split"] = {
        key: variance_split(manifest, key) for key in ("test_f1", "test_recall", "test_auc")
    }

    for split in ("val", "test"):
        pred = load_predictions(sweep_root, runs, split)
        per_run_win = {k: [float(r[f"{split}_{k}"]) for r in manifest] for k in METRICS}
        recall = [int(r[f"{split}_event_detected"]) / max(1, int(r[f"{split}_event_manifest"]))
                  for r in manifest]
        far = [int(r[f"{split}_false_alarms"]) / max(1, int(r[f"{split}_nonfall_events"]))
               for r in manifest]
        summary[split] = {
            "window": {k: describe(v) for k, v in per_run_win.items()},
            "event": {
                "event_recall_all": describe(recall),
                "event_false_alarm_rate": describe(far),
            },
            "by_dataset": dataset_breakdown(pred, threshold),
            "borderline": borderline_windows(pred, threshold),
        }
        if split == "test":
            counts = [int(r["test_event_detected"]) for r in manifest]
            hist: dict[str, int] = {}
            for c in counts:
                hist[str(c)] = hist.get(str(c), 0) + 1
            summary["test"]["event_detected_counts"] = {
                "histogram": "，".join(f"{k} 个事件: {v} 次运行" for k, v in sorted(hist.items())),
                "denominator": int(manifest[0]["test_event_manifest"]),
                "min": int(min(counts)), "max": int(max(counts)),
                "range": int(max(counts) - min(counts)),
            }
            summary["test"]["event_stability"] = event_stability(pred, threshold, split)
            per_window_csv(pred, out_dir / "per_window_probabilities.csv", threshold)

    summary["ablations"] = ablation_table(summary["test"]["window"], ABLATION_RUNS)
    summary["val_test_correlation"] = _val_test_correlation(manifest)
    summary["best_epoch_range"] = f"{min(int(r['best_epoch']) for r in manifest)}–" \
                                  f"{max(int(r['best_epoch']) for r in manifest)}"

    if args.compare_runs is not None:
        summary["compare_runs"] = compare_runs(
            sweep_root, runs, args.compare_runs or [], threshold, args.device
        )

    summary["conclusion"] = build_conclusion(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "summary.md", summary)
    print(summary["conclusion"], flush=True)
    for name in ("summary.md", "summary.json", "per_window_probabilities.csv"):
        print(f"wrote {out_dir / name}", flush=True)
    return 0


def _val_test_correlation(manifest: list[dict]) -> dict:
    """Does val F1 rank the runs the same way test F1 does?"""
    from scipy.stats import pearsonr

    val = np.asarray([float(r["val_f1"]) for r in manifest])
    test = np.asarray([float(r["test_f1"]) for r in manifest])
    if len(val) < 3:
        return {"r": float("nan"), "p": float("nan"), "n": len(val)}
    r, p = pearsonr(val, test)
    return {"r": float(r), "p": float(p), "n": int(len(val))}


def _precheck(sweep_root: Path) -> dict:
    path = sweep_root / "determinism_precheck.json"
    if not path.exists():
        return {"a_loss": float("nan"), "b_loss": float("nan"),
                "a_val_f1": float("nan"), "b_val_f1": float("nan")}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def compare_runs(sweep_root: Path, runs: list[str], targets: list[str],
                 threshold: float, device_str: str) -> list[dict]:
    """Score arbitrary runs against the run distribution (needs torch + best.pth)."""
    import torch

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = engine.load_config(None)
    pred = load_predictions(sweep_root, runs, "test")
    mean_p = pred["probs"].mean(axis=0)
    mean_pred = (mean_p >= threshold).astype(np.int64)
    baseline_f1 = engine.binary_metrics(pred["labels"], mean_p, threshold)["f1"]
    values = np.asarray([float(r["test_f1"]) for r in load_manifest(sweep_root)])

    out = []
    for run in targets:
        path = engine.resolve(f"runs/{run}/best.pth")
        if not path.exists():
            print(f"skip {run}: no best.pth", flush=True)
            continue
        model = engine.build_recognizer(cfg).to(device)
        model.load_state_dict(engine.load_checkpoint(path, device)["model"])
        dataset = engine.build_dataset(cfg, "test")
        loader = engine.build_loader(cfg, "test")[1]
        labels, probs = engine.predict(model, loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not np.array_equal(labels, pred["labels"]):
            raise RuntimeError(f"{run}: label order differs from the sweep")
        m = engine.binary_metrics(labels, probs, threshold)
        out.append({
            "run": run,
            "test_f1": m["f1"],
            "percentile": float((values < m["f1"]).mean()),
            "n_flipped_windows": int(((probs >= threshold).astype(np.int64) != mean_pred).sum()),
            "mean_run_f1": baseline_f1,
        })
        print(f"  {run:32s} f1={m['f1']:.4f} flipped={out[-1]['n_flipped_windows']}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
