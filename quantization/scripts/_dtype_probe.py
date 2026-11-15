"""Throwaway: count dtype markers inside two .rknn artefacts."""
import os
import sys

toks = [b"int16", b"int8", b"float16", b"w16a16", b"w8a8"]
for f in sys.argv[1:]:
    b = open(f, "rb").read()
    hits = " ".join("%s=%d" % (t.decode(), b.count(t)) for t in toks)
    print("%-46s %9d B  %s" % (os.path.basename(f), len(b), hits))
