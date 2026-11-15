#!/bin/sh
# 用法:
# sh p8/regress.sh selftest
# sh p8/regress.sh run [OUT]
# sh p8/regress.sh det [OUT]
# sh p8/regress.sh diff NEW BASE
set -e
cd "$(dirname "$0")/.."

PY=${PY:-python}
export PYTHONIOENCODING=utf-8
RUN_ID=$(date +%Y%m%d_%H%M)
OUT=${2:-../results/regress/$RUN_ID}
BASE=${BASE:-../reference/g3_windows.csv}

case "${1:-help}" in
  selftest) exec $PY p8/regress.py selftest ;;
  run)      exec $PY p8/regress.py run --out "$OUT" --base "$BASE" ;;
  det)
    echo "== run 1 -> $OUT/a =="
    $PY p8/regress.py run --out "$OUT/a" --base "$OUT/a/windows.csv" || true
    echo "== run 2 -> $OUT/b（必须恰好 0）=="
    $PY p8/regress.py run --out "$OUT/b" --base "$OUT/a/windows.csv" --expect-zero
    ;;
  diff)
    [ -n "$3" ] || { echo "用法: sh p8/regress.sh diff <NEW> <BASE>"; exit 2; }
    exec $PY p8/regress.py diff --new "$2" --base "$3" ;;
  *)
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
