"""Copy the three raw datasets into datasets/raw/ and verify counts/sizes."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402

JOBS = [
    ("le2i", paths.SOURCE_LE2I, paths.RAW_LE2I, "dir"),
    ("mcfd_dataset", paths.SOURCE_MCFD / "dataset", paths.RAW_MCFD, "dir"),
    ("mcfd_csv", paths.SOURCE_MCFD / "data_tuple3.csv", paths.RAW_MCFD_CSV, "file"),
    ("mcfd_pdf", paths.SOURCE_MCFD / "technicalReport.pdf", paths.RAW_MCFD_PDF, "file"),
    ("urfd", paths.SOURCE_URFD, paths.RAW_URFD, "dir"),
]

ROBOCOPY = shutil.which("robocopy") or r"C:\Windows\System32\robocopy.exe"


def dir_sizes(root: Path) -> dict[str, int]:
    return {
        p.relative_to(root).as_posix(): p.stat().st_size
        for p in paths.iter_files(root)
    }


def compare_dirs(src: Path, dst: Path) -> tuple[bool, dict]:
    if not dst.exists():
        return False, {"reason": "destination missing"}
    a = dir_sizes(src)
    b = dir_sizes(dst)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    size_mismatch = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    ok = not missing and not extra and not size_mismatch
    return ok, {
        "src_files": len(a),
        "dst_files": len(b),
        "src_bytes": sum(a.values()),
        "dst_bytes": sum(b.values()),
        "missing": missing[:20],
        "extra": extra[:20],
        "size_mismatch": size_mismatch[:20],
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_size_mismatch": len(size_mismatch),
    }


def robocopy_dir(src: Path, dst: Path, log: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ROBOCOPY,
        str(src),
        str(dst),
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:2",
        "/W:2",
        "/NFL",
        "/NDL",
        "/NP",
        f"/LOG:{log}",
    ]
    print("  cmd:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Robocopy: 0-7 are success codes; >=8 is a real error.
    return proc.returncode


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths.ensure_dirs()
    print("== Preflight ==")
    total = 0
    for name, src, _dst, kind in JOBS:
        if not src.exists():
            print(f"FAIL source missing: {name} {src}")
            return 2
        if kind == "dir":
            size = sum(p.stat().st_size for p in paths.iter_files(src))
        else:
            size = src.stat().st_size
        total += size
        print(f"  {name:12s} {paths.human_bytes(size):>10s}  {src}")
    free = shutil.disk_usage(str(paths.PROJECT_ROOT)).free
    print(f"  total        {paths.human_bytes(total):>10s}")
    print(f"  free on disk {paths.human_bytes(free):>10s}")
    if free < int(total * 1.05):
        print("FAIL not enough free disk space (need 5% headroom)")
        return 2

    print("\n== Copy ==")
    all_ok = True
    summary: list[str] = []
    for name, src, dst, kind in JOBS:
        t0 = time.time()
        print(f"[{name}] {src} -> {dst}", flush=True)
        if kind == "dir":
            code = robocopy_dir(src, dst, paths.REPORTS / f"copy_{name}.log")
            ok_code = code < 8
            ok, detail = compare_dirs(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ok = dst.exists() and dst.stat().st_size == src.stat().st_size
            detail = {"src_bytes": src.stat().st_size, "dst_bytes": dst.stat().st_size}
            ok_code = True
            code = 0
        elapsed = time.time() - t0
        status = "OK" if (ok and ok_code) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {status} robocopy_code={code} elapsed={elapsed:.1f}s detail={detail}", flush=True)
        summary.append(
            f"| {name} | {paths.human_bytes(detail.get('src_bytes', 0))} | "
            f"{'OK' if ok and ok_code else 'FAIL'} | {elapsed:.1f}s | {detail} |"
        )

    print("\n== Summary ==")
    for line in summary:
        print(line)
    print("0 failures" if all_ok else "FAILURES PRESENT")

    log_lines = ["# Raw copy summary", ""]
    log_lines += [f"- {line}" for line in summary]
    log_lines.append("")
    log_lines.append("Result: " + ("PASS" if all_ok else "FAIL"))
    (paths.REPORTS / "copy_raw.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not all_ok:
        return 1

    # Freeze the raw layer with SHA256. Duplicate detection needs this file
    # before manifests/splits are built, so it is produced here.
    print("\n== SHA256 ==", flush=True)
    files = sorted(paths.iter_files(paths.RAW))
    hash_rows = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        stat = path.stat()
        hash_rows.append(
            {
                "relative_path": path.relative_to(paths.RAW).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": paths.sha256_file(path),
            }
        )
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {time.time() - t0:.1f}s", flush=True)
    paths.write_csv(paths.MANIFESTS / "raw_sha256.csv", hash_rows)

    from collections import defaultdict

    by_hash = defaultdict(list)
    for row in hash_rows:
        by_hash[row["sha256"]].append(row["relative_path"])
    duplicates = {h: v for h, v in by_hash.items() if len(v) > 1}
    print(f"  sha256 rows: {len(hash_rows)}; duplicate groups: {len(duplicates)}")
    for h, items in duplicates.items():
        print(f"    {h[:16]}... -> {items}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
