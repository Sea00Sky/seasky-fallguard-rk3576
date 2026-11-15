"""Event-level splits with leakage checks."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from . import paths

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _allocate(n: int) -> tuple[int, int, int]:
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])
    n_test = n - n_train - n_val
    if n >= 3:
        if n_val < 1:
            n_val, n_train = 1, n_train - 1
        if n_test < 1:
            n_test, n_train = 1, n_train - 1
    return n_train, n_val, n_test


def _split_group(event_ids: list[str], dataset: str, label: str) -> dict[str, str]:
    ids = sorted(event_ids)
    rng = random.Random(f"{SEED}:{dataset}:{label}")
    rng.shuffle(ids)
    n_train, n_val, _ = _allocate(len(ids))
    out = {}
    for i, eid in enumerate(ids):
        out[eid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def make(
    version: str = "v1",
    manifests_dir: Path | None = None,
    split_path: Path | None = None,
    created: str = "2026-10-19",
) -> dict:
    manifests_dir = Path(manifests_dir) if manifests_dir else paths.MANIFESTS
    videos = paths.read_csv(manifests_dir / "videos.csv")
    events = paths.read_csv(manifests_dir / "events.csv")

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ev in events:
        if ev["eligible_for_split"] == "True":
            groups[(ev["dataset"], ev["label"])].append(ev["event_id"])

    event_to_split: dict[str, str] = {}
    for (dataset, label), ids in sorted(groups.items()):
        event_to_split.update(_split_group(ids, dataset, label))

    # Fill event-level split.
    for ev in events:
        ev["split"] = event_to_split.get(ev["event_id"], "")

    # Fill video-level split from its event.
    for vid in videos:
        vid["split"] = event_to_split.get(vid["event_id"], "")

    checks = {
        "duplicate_event_in_multiple_splits": [],
        "video_event_split_mismatch": [],
        "mcfd_camera_split_mismatch": [],
        "urfd_duplicate_event_base": [],
        "duplicate_video_id": [],
        "duplicate_content_split_mismatch": [],
    }

    seen: dict[str, str] = {}
    for eid, split in event_to_split.items():
        if eid in seen and seen[eid] != split:
            checks["duplicate_event_in_multiple_splits"].append(eid)
        seen[eid] = split

    split_of_event = {ev["event_id"]: ev["split"] for ev in events}
    for vid in videos:
        if vid["split"] and split_of_event.get(vid["event_id"], "") != vid["split"]:
            checks["video_event_split_mismatch"].append(vid["video_id"])

    mcfd_splits: dict[str, set[str]] = defaultdict(set)
    for vid in videos:
        if vid["dataset"] == "mcfd" and vid["split"]:
            mcfd_splits[vid["event_id"]].add(vid["split"])
    for eid, splits in mcfd_splits.items():
        if len(splits) > 1:
            checks["mcfd_camera_split_mismatch"].append(eid)

    urfd_bases: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev["dataset"] == "urfd":
            base = ev["event_id"].replace("urfd:", "").rsplit("-cam", 1)[0]
            urfd_bases[base].append(ev["event_id"])
    checks["urfd_duplicate_event_base"] = [
        {"base": b, "events": ids} for b, ids in urfd_bases.items() if len(ids) > 1
    ]

    vids = [v["video_id"] for v in videos]
    checks["duplicate_video_id"] = [v for v, c in Counter(vids).items() if c > 1]

    # Identical files must never land in different splits.
    hash_path = paths.MANIFESTS / "raw_sha256.csv"
    if hash_path.exists():
        hashes = {r["relative_path"]: r["sha256"] for r in paths.read_csv(hash_path)}
        by_hash: dict[str, set[str]] = defaultdict(set)
        for vid in videos:
            digest = hashes.get(vid["relative_path"])
            if digest and vid["split"]:
                by_hash[digest].add(vid["split"])
        checks["duplicate_content_split_mismatch"] = [
            digest for digest, splits in by_hash.items() if len(splits) > 1
        ]

    counts = {
        "events": dict(Counter(ev["split"] for ev in events if ev["split"])),
        "fall_events": dict(
            Counter(ev["split"] for ev in events if ev["split"] and ev["label"] == "fall")
        ),
        "by_dataset": {
            ds: dict(Counter(ev["split"] for ev in events if ev["split"] and ev["dataset"] == ds))
            for ds in sorted({ev["dataset"] for ev in events})
        },
        "by_dataset_label": {
            f"{ds}:{lb}": dict(
                Counter(
                    ev["split"]
                    for ev in events
                    if ev["split"] and ev["dataset"] == ds and ev["label"] == lb
                )
            )
            for ds in sorted({ev["dataset"] for ev in events})
            for lb in sorted({ev["label"] for ev in events if ev["dataset"] == ds})
        },
    }

    payload = {
        "version": version,
        "seed": SEED,
        "ratios": RATIOS,
        "created": created,
        "event_to_split": dict(sorted(event_to_split.items())),
        "counts": counts,
        "checks": checks,
    }
    split_path = Path(split_path) if split_path else paths.SPLITS / f"{version}.json"
    paths.write_json(split_path, payload)
    paths.write_csv(manifests_dir / "events.csv", events)
    paths.write_csv(manifests_dir / "videos.csv", videos)
    return payload


if __name__ == "__main__":
    import json

    result = make()
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    bad = {k: v for k, v in result["checks"].items() if v}
    print("checks with findings:", bad if bad else "none")
