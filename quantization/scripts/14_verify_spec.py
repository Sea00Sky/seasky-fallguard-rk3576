"""Phase 0 audit -- does `src/deploy/spec.py` still describe the source project?"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
spec_mod = _load_own("deploy_spec", "src/deploy/spec.py")
# `offline` needs the source project's `src`, so it must come before any import
# of this project's `src.runtime` (see its module docstring).
ref = _load_own("deploy_reference", "src/reference/offline.py")

SOURCE = paths.SOURCE_ROOT
SPEC_DOC = paths.REFERENCE / "preprocess_spec.md"

TOL_COVERAGE = 1e-12          # float64 mean of a boolean vector: exact or wrong


# ------------------------------------------------------------------ check 1


def _yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


def _fn_default(path: Path, func: str, arg: str) -> float:
    """Read a literal default like `gap_max: int = 5` out of a source file."""
    text = path.read_text(encoding="utf-8")
    start = text.index(f"def {func}(")
    body = text[start:start + 1200]
    m = re.search(rf"{re.escape(arg)}\s*:\s*int\s*=\s*(\d+)", body)
    if not m:
        raise RuntimeError(f"cannot find default `{arg}` in {func}() of {path}")
    return float(m.group(1))


def check_constants() -> list[dict]:
    """Every spec constant against the source project's own configuration."""
    pose = _yaml(SOURCE / "configs" / "pose.yaml")
    labels = _yaml(SOURCE / "configs" / "labels.yaml")
    inf, trk = pose["inference"], pose["tracking"]
    win = labels["window"]
    build_py = SOURCE / "src" / "window" / "build.py"
    extract_py = SOURCE / "src" / "pose" / "extract.py"

    # (name, spec value, source value, where the source value lives)
    rows = [
        ("IMGSZ", spec_mod.IMGSZ, inf["imgsz"], "configs/pose.yaml:30 inference.imgsz"),
        ("CONF", spec_mod.CONF, inf["conf"], "configs/pose.yaml:34 inference.conf"),
        ("IOU", spec_mod.IOU, inf["iou"], "configs/pose.yaml:35 inference.iou"),
        ("MAX_PERSONS", spec_mod.MAX_PERSONS, inf["max_persons"],
         "configs/pose.yaml:39 inference.max_persons"),
        ("PERSON_CLASS", spec_mod.PERSON_CLASS, inf["person_class"],
         "configs/pose.yaml:40 inference.person_class"),
        ("TRACK_IOU", spec_mod.TRACK_IOU, trk["iou_threshold"],
         "configs/pose.yaml:44 tracking.iou_threshold"),
        ("MAX_AGE", spec_mod.MAX_AGE, trk["max_age"],
         "configs/pose.yaml:45 tracking.max_age"),
        ("DURATION_S", spec_mod.DURATION_S, win["duration_s"],
         "configs/labels.yaml:20 window.duration_s"),
        ("STRIDE_S", spec_mod.STRIDE_S, win["stride_s"],
         "configs/labels.yaml:21 window.stride_s"),
        ("GUARD_S", spec_mod.GUARD_S, win["guard_s"],
         "configs/labels.yaml:22 window.guard_s"),
        ("NEGATIVE_MIN_DISTANCE_S", spec_mod.NEGATIVE_MIN_DISTANCE_S,
         win["negative_min_distance_from_fall_s"], "configs/labels.yaml:23"),
        ("RESAMPLE_FRAMES", spec_mod.RESAMPLE_FRAMES, win["resample_frames"],
         "configs/labels.yaml:24 window.resample_frames"),
        ("MIN_COVERAGE", spec_mod.MIN_COVERAGE, win["min_coverage"],
         "configs/labels.yaml:25 window.min_coverage"),
        ("MAX_INTERP_GAP_FRAMES", spec_mod.MAX_INTERP_GAP_FRAMES,
         win["max_interp_gap_frames"], "configs/labels.yaml:26"),
        ("STITCH_GAP_MAX", spec_mod.STITCH_GAP_MAX,
         _fn_default(build_py, "stitch_id_handoffs", "gap_max"),
         "src/window/build.py:254 stitch_id_handoffs(gap_max=5)"),
        ("STITCH_OVERLAP_MAX", spec_mod.STITCH_OVERLAP_MAX,
         _fn_default(build_py, "stitch_id_handoffs", "overlap_max"),
         "src/window/build.py:255 stitch_id_handoffs(overlap_max=1)"),
    ]

    out = []
    for name, mine, theirs, where in rows:
        ok = float(mine) == float(theirs)
        out.append({"constant": name, "spec": mine, "source": theirs,
                    "where": where, "ok": ok})

    # The distance factor is a literal inside `_handoff_distance`, not a
    # parameter: `0.25 * (da + db)`. `spec.STITCH_DIST_FACTOR` must equal it.
    m = re.search(r"dist\s*>\s*([0-9.]+)\s*\*\s*\(da\s*\+\s*db\)",
                  build_py.read_text(encoding="utf-8"))
    theirs = float(m.group(1)) if m else None
    out.append({"constant": "STITCH_DIST_FACTOR", "spec": spec_mod.STITCH_DIST_FACTOR,
                "source": theirs, "where": "src/window/build.py:244 literal 0.25",
                "ok": theirs is not None and float(spec_mod.STITCH_DIST_FACTOR) == theirs})

    # PYSKL `PreNormalize2D(threshold=0.01, mode='fix')`: default args, so the
    # source only *names* the transform (`engine.py:72`). Verified by import.
    try:
        from pyskl.datasets.pipelines import PreNormalize2D

        import inspect

        sig = inspect.signature(PreNormalize2D.__init__)
        thr = float(sig.parameters["threshold"].default)
        mode = str(sig.parameters["mode"].default)
        got = (mode == "fix" and thr == float(spec_mod.SCORE_THRESHOLD))
        where = ("pyskl.datasets.pipelines.PreNormalize2D defaults "
                 f"(threshold={thr}, mode={mode!r}); source names it at "
                 "src/train/engine.py:72")
    except Exception as exc:                      # pyskl not installed here
        got, where = None, f"pyskl not importable in this env: {exc}"
    out.append({"constant": "SCORE_THRESHOLD", "spec": spec_mod.SCORE_THRESHOLD,
                "source": thr if got is not None else None, "where": where,
                "ok": got})

    # Model contract: clip length / person count / classes come from the
    # training config, which is frozen in weights/. Read the snapshot.
    cfg = _yaml(paths.STGCN_CFG)
    data = cfg.get("data", {})
    out.append({"constant": "CLIP_LEN", "spec": spec_mod.CLIP_LEN,
                "source": data.get("clip_len"),
                "where": "weights/stgcnpp_v2_config_used.yaml data.clip_len",
                "ok": int(spec_mod.CLIP_LEN) == int(data.get("clip_len", -1))})
    out.append({"constant": "NUM_PERSON", "spec": spec_mod.NUM_PERSON,
                "source": data.get("num_person"),
                "where": "weights/stgcnpp_v2_config_used.yaml data.num_person",
                "ok": int(spec_mod.NUM_PERSON) == int(data.get("num_person", -1))})
    return out


# ------------------------------------------------------------- checks 2, 3


def check_windows() -> dict:
    """The 915-row window table against the frozen npz."""
    ref_npz = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)
    frozen = [(str(ref_npz["video_id"][i]), int(ref_npz["start"][i]),
               int(ref_npz["end"][i])) for i in range(len(ref_npz["window_id"]))]
    mine = [(w["video_id"], w["start"], w["end"]) for w in ref.candidate_windows()]
    first_diff = next((i for i, (a, b) in enumerate(zip(mine, frozen)) if a != b),
                      None)
    out = {"n_mine": len(mine), "n_ref": len(frozen),
           "identical": mine == frozen, "first_diff": None}
    if first_diff is not None:
        out["first_diff"] = {"index": first_diff,
                             "mine": list(mine[first_diff]),
                             "ref": list(frozen[first_diff])}
    return out


def check_samples() -> dict:
    """Per-sample coverage, A/B/C, in frozen order."""
    ref_npz = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)
    kind = np.array([str(x) for x in ref_npz["sample_kind"]])
    track = ref_npz["sample_track"].astype(np.int64)
    cov = ref_npz["sample_coverage"].astype(np.float64)
    members = np.array([str(x) for x in ref_npz["sample_members"]])

    mine = ref.reference_samples()
    n = min(len(mine), len(ref_npz["sample_kind"]))
    if len(mine) != len(kind):
        return {"n_mine": len(mine), "n_ref": len(kind), "identical": False,
                "reason": "sample count differs"}
    my_kind = np.array([s["kind"] for s in mine])
    my_track = np.array([s["track"] for s in mine], dtype=np.int64)
    my_cov = np.array([s["coverage"] for s in mine], dtype=np.float64)
    my_members = np.array(["|".join(str(x) for x in s["members"]) for s in mine])

    d = np.abs(my_cov - cov)
    bad = np.nonzero((my_kind != kind) | (my_track != track))[0]
    out = {
        "n_mine": len(mine), "n_ref": len(kind),
        "kind_track_identical": bool(len(bad) == 0),
        "first_kind_track_diff": (None if not len(bad) else {
            "index": int(bad[0]), "mine": [str(my_kind[bad[0]]), int(my_track[bad[0]])],
            "ref": [str(kind[bad[0]]), int(track[bad[0]])]}),
        "max_abs_dcoverage": float(d.max()) if len(d) else 0.0,
        "members_identical": bool(np.array_equal(my_members, members)),
        "first_members_diff": None,
    }
    diff = np.nonzero(my_members != members)[0]
    if len(diff):
        out["first_members_diff"] = {
            "index": int(diff[0]), "mine": str(my_members[diff[0]]),
            "ref": str(members[diff[0]])}
    out["consistent"] = (out["kind_track_identical"]
                         and out["members_identical"]
                         and out["max_abs_dcoverage"] <= TOL_COVERAGE)
    return out


# ------------------------------------------------------------------ check 4


def dump_runtime_chains(out_path: Path) -> int:
    """Subprocess mode: chains from this project's `Stitcher`, as JSON."""
    import src.runtime.stitch as rt_stitch

    out: dict[str, list[list[int]]] = {}
    for row in ref.manifest_rows():
        vid = row["video_id"]
        try:
            arr = ref.video_arrays(vid)
        except FileNotFoundError:
            continue
        st = rt_stitch.Stitcher()
        for t in range(len(arr["n_persons"])):
            for p in range(int(arr["n_persons"][t])):
                tid = int(arr["track_ids"][t, p])
                if tid >= 0:
                    st.observe(tid, t, arr["bboxes"][t, p])
        out[vid] = st.chains_so_far()
    paths.write_json(out_path, out)
    return 0


def check_stitch() -> dict:
    """`src/runtime/stitch.py` fed the whole clip must equal the source function."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "runtime_chains.json"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dump-runtime-chains",
             str(dump)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            return {"videos_same": 0, "videos_diff": 0, "videos_missing": 0,
                    "passed": False, "error": proc.stderr[-2000:], "details": []}
        import json as _json

        got_all = _json.loads(dump.read_text(encoding="utf-8"))

    same = diff = missing = 0
    details = []
    for row in ref.manifest_rows():
        vid = row["video_id"]
        if vid not in got_all:
            missing += 1
            continue
        want = [list(c) for c in ref.stitch_chains(vid)]
        got = got_all[vid]
        if got == want:
            same += 1
        else:
            diff += 1
            details.append({"video_id": vid, "offline": want, "online": got})
    return {"videos_same": same, "videos_diff": diff, "videos_missing": missing,
            "passed": diff == 0, "details": details[:5]}


# ------------------------------------------------------------------ check 5


def check_spec_doc() -> dict:
    """Every `NAME = value` line in the doc must name a real spec constant."""
    if not SPEC_DOC.exists():
        return {"exists": False, "passed": False, "missing": [], "wrong": []}
    text = SPEC_DOC.read_text(encoding="utf-8")
    missing, wrong, seen = [], [], 0
    # Rows look like: `| `MIN_COVERAGE` | 0.70 | ... |`
    for name, value in re.findall(r"^\|\s*`([A-Z][A-Z0-9_]+)`\s*\|\s*([^|]+?)\s*\|",
                                  text, flags=re.M):
        if not hasattr(spec_mod, name):
            missing.append(name)
            continue
        seen += 1
        try:
            want = float(value.strip().strip("`"))
        except ValueError:
            continue
        if float(getattr(spec_mod, name)) != want:
            wrong.append({"constant": name, "doc": value.strip(),
                          "spec": getattr(spec_mod, name)})
    return {"exists": True, "constants_checked": seen, "missing": missing,
            "wrong": wrong, "passed": not missing and not wrong and seen >= 10}


# --------------------------------------------------------------------- main


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-samples", action="store_true",
                    help="skip check 3 (rebuilds all 13,223 samples, ~1 min)")
    ap.add_argument("--limit-videos", type=int, default=0,
                    help="only the first N videos (smoke test; the report is "
                         "written with `partial: true` and is not a gate)")
    ap.add_argument("--dump-runtime-chains", default=None, metavar="PATH",
                    help="internal: write this project's Stitcher chains as JSON "
                         "and exit (used by check 4 to keep the two `src` "
                         "packages in separate processes)")
    args = ap.parse_args()

    if args.dump_runtime_chains:
        return dump_runtime_chains(Path(args.dump_runtime_chains))

    print("check 1: spec constants vs source project")
    consts = check_constants()
    for c in consts:
        print(f"  {'OK ' if c['ok'] else 'BAD'} {c['constant']:<24} "
              f"spec={c['spec']!r:<10} source={c['source']!r}")
    # `ok is None` means "could not be checked in this interpreter" (e.g. pyskl
    # is not importable in the base env), which is NOT the same as "differs".
    # Conflating them made a base-env run report a constant mismatch that did
    # not exist -- and the message at the end then suggested re-running with
    # the very interpreter that cannot check it.
    bad_consts = [c for c in consts if c["ok"] is False]
    unverifiable = [c for c in consts if c["ok"] is None]

    print("\ncheck 4: D-042 chains, runtime stitcher vs source (whole clip)")
    lt = check_stitch()
    print(f"  {lt['videos_same']} same, {lt['videos_diff']} differ, "
          f"{lt['videos_missing']} missing")

    print("\ncheck 2: 915-window table")
    win = check_windows()
    print(f"  mine {win['n_mine']} / ref {win['n_ref']}  identical={win['identical']}")
    if not win["identical"]:
        print(f"  first difference: {win['first_diff']}")

    smp = None
    if not args.skip_samples:
        print("\ncheck 3: 13,223 sample coverages (A/B/C)")
        smp = check_samples()
        print(f"  mine {smp['n_mine']} / ref {smp['n_ref']}  "
              f"kind+track={smp['kind_track_identical']}  "
              f"members={smp['members_identical']}  "
              f"max|dcov|={smp['max_abs_dcoverage']:.3e}")
        if smp.get("first_kind_track_diff"):
            print(f"  first kind/track difference: {smp['first_kind_track_diff']}")

    print("\ncheck 5: preprocess_spec.md vs spec.py")
    doc = check_spec_doc()
    print(f"  exists={doc['exists']} constants={doc.get('constants_checked', 0)} "
          f"missing={doc['missing']} wrong={doc['wrong']}")

    # An unverifiable constant fails the audit: "we could not check it" must
    # never read as "it is fine". It is reported separately so the reader knows
    # whether to fix a number or just run in the right environment.
    passed = (not bad_consts and not unverifiable and lt["passed"]
              and win["identical"] and (smp is None or smp["consistent"])
              and doc["passed"])
    ref.PROJECT_ROOT.joinpath("reports", "reference").mkdir(parents=True,
                                                           exist_ok=True)
    payload = {
        "constants": consts, "windows": win, "samples": smp,
        "stitch": lt, "spec_doc": doc,
        "partial": bool(args.limit_videos), "passed": bool(passed),
    }
    paths.write_json(paths.REFERENCE / "spec_audit.json", payload)

    lines = ["# Phase 0 规格审计（scripts/14）", "",
             "`src/deploy/spec.py` 的每个常量、窗口几何、逐样本覆盖率与 D-042 拼接，",
             "都对回源项目的**实际配置与函数**，而不是对回本项目自己的副本。", "",
             "| 检查 | 对象 | 结果 |", "|---|---|---|",
             f"| 1 | {len(consts)} 个常量 vs `configs/*.yaml` / 函数默认值 | "
             + ("✅ 全部一致" if not bad_consts and not unverifiable else
                ("❌ 不一致：" + ", ".join(c["constant"] for c in bad_consts)
                 if bad_consts else "")
                + ("；⚠️ 本环境无法验证：" + ", ".join(
                    f"{c['constant']}（{c['where']}）" for c in unverifiable)
                   if unverifiable else ""))
             + " |",
             f"| 2 | 候选窗口表 vs 冻结 npz | "
             f"{'✅ ' + str(win['n_mine']) + ' 行逐行一致' if win['identical'] else '❌'} |"]
    if smp is not None:
        lines.append(
            f"| 3 | {smp['n_mine']} 条样本（A/B/C）的覆盖率与成员 | "
            f"{'✅ `max|Δcov| = ' + format(smp['max_abs_dcoverage'], '.0e') + '`' if smp['consistent'] else '❌'} |")
    lines.append(
        f"| 4 | `src/runtime/stitch.py` vs 源 `stitch_id_handoffs`（整段） | "
        f"{'✅ ' + str(lt['videos_same']) + '/' + str(lt['videos_same'] + lt['videos_diff']) + ' 视频一致' if lt['passed'] else '❌ ' + str(lt['videos_diff']) + ' 个视频不同'} |")
    lines.append(
        f"| 5 | `preprocess_spec.md` vs `spec.py` | "
        f"{'✅ ' + str(doc.get('constants_checked', 0)) + ' 个常量' if doc['passed'] else '❌'} |")
    if unverifiable:
        lines += ["", "**本环境缺依赖，以下常量无法验证（不是不一致）：** "
                  + "、".join(c["constant"] for c in unverifiable)
                  + "。用 pyskl 环境重跑才能给结论。"]
    lines += ["", f"**结论：{'通过' if passed else '未通过'}**", "",
              "产物：`reports/reference/spec_audit.json`。",
              # The base interpreter cannot import pyskl, so telling the reader
              # to re-run with it is how the "false mismatch" above happened.
              "重跑（**必须用 pyskl 环境**，base 环境缺 pyskl，会把 SCORE_THRESHOLD "
              "报成无法验证）：",
              "`\"<python 解释器路径>\" "
              "scripts/14_verify_spec.py`", ""]
    (paths.REFERENCE / "spec_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
