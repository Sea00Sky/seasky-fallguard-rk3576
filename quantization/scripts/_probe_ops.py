"""Throwaway: dump the per-layer dtype map of the w16a16i artefact."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rknn.api import RKNN

r = RKNN(verbose=False)
if r.load_rknn(str(ROOT / "models" / "_probe_t9_w16a16i.rknn")) != 0:
    raise SystemExit("load_rknn failed")

# The toolkit exposes the parsed graph only through the build path, so the
# usable surface here is the documented one: `get_sdk_version` / the string
# table. Dump every line that names a dtype next to a tensor so the 12 int8
# layers can be identified by grep.
import re
raw = (ROOT / "models" / "_probe_t9_w16a16i.rknn").read_bytes()
txt = raw.decode("latin-1")
names = re.findall(r"[A-Za-z0-9_./]+(?:Conv|MatMul|Add|Mul|Sub|Concat|Sigmoid)[A-Za-z0-9_./]*", txt)
from collections import Counter
c = Counter(names)
print("distinct op-ish names:", len(c))
for n, k in c.most_common(40):
    print(f"  {k:4d}  {n}")
