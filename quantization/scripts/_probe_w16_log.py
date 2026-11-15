"""Throwaway: how many Conv layers fell back in the w16a16i build."""
import re
import sys
from collections import Counter

pat = re.compile(r"Conv Not Support u8->i16 For Now")
for path in sys.argv[1:]:
    txt = open(path, "r", encoding="utf-8", errors="replace").read()
    print(f"{path}: {len(pat.findall(txt))} fallback line(s)")
    for line in txt.splitlines():
        if "Not Support" in line or "unsupported tensor dtype" in line:
            print("   ", line.strip()[:160])
