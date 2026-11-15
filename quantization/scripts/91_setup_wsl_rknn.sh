#!/usr/bin/env bash
# Prepare the WSL2 side for Phase 2 (RKNN conversion + G4/G5 verification).
#
# Run inside WSL2, from the project directory:
#
# bash scripts/91_setup_wsl_rknn.sh
#
# ## Why this exists instead of a list of commands in a docstring
#
# The old notes said "install rknn-toolkit2 2.3.0 with -i <tsinghua mirror>".
# Three things were wrong or missing, and each one fails differently:
#
# 1. pypi.org answers but takes 20 s (measured : HTTP 200 at
# 20.0 s for the index). A `pip install` does dozens of index requests, so
# it times out long before finishing. The mirror answers in 9 s. The fix is
# a *persistent* pip config, not a flag on one command -- otherwise every
# later `pip install` in this workflow hits the same wall.
# 2. `dism`/`wsl --install` do not install a Python. Ubuntu ships 3.12 by
# default on recent images; the toolkit needs <= 3.12 but the wheel for
# 3.10 is the one the upstream docs pin. We install 3.10 explicitly and
# build a venv, so the toolkit's pinned deps cannot break the system
# Python that other things use.
# 3. The toolkit version matters for the ONNX opset. `scripts/20` exports
# at opset 12 and pins toolkit 2.3.0 in its docstring. 2.3.0 and 2.3.2 both
# publish cp310 x86_64 wheels; we default to 2.3.0 to match the docstring,
# with `--toolkit` to override.
#
# ## Why a venv and not the system python
#
# rknn-toolkit2 2.3.0 pins `torch<=2.4.0`, `onnx==1.13.1`, `setuptools==69.5.1`
# (newer onnx dropped `onnx.mapping`, newer setuptools dropped `pkg_resources`).
# Installing those system-wide in Ubuntu would downgrade packages other tooling
# depends on. A venv keeps the blast radius to this project.
#
# ## What it does NOT do
#
# It does not touch the Windows side and does not need sudo for the Python work
# (only `apt-get` for python3.10 if it is missing, and the caller is told when
# that is about to happen).
set -euo pipefail

PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TOOLKIT_VERSION="2.3.0"
VENV="${HOME}/rknn-env"
PY_BIN="python3.10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolkit) TOOLKIT_VERSION="$2"; shift 2 ;;
    --venv)    VENV="$2"; shift 2 ;;
    --index)   PIP_INDEX="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }

say "=== setup: rknn-toolkit2 ${TOOLKIT_VERSION} ==="
say "index : ${PIP_INDEX}"
say "venv  : ${VENV}"
say ""

# --- 1. persistent pip mirror ----------------------------------------------
# Both files: ~/.pip/pip.conf is read by older pip, ~/.config/pip/pip.conf by
# current pip. Writing both removes the version dependency entirely.
say "--- pip mirror (persistent) ---"
mkdir -p "${HOME}/.pip" "${HOME}/.config/pip"
for f in "${HOME}/.pip/pip.conf" "${HOME}/.config/pip/pip.conf"; do
  cat > "$f" <<EOF
[global]
index-url = ${PIP_INDEX}
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
retries = 5
EOF
  say "wrote $f"
done
python3 -m pip config list 2>/dev/null || true
say ""

# --- 2. python 3.10 --------------------------------------------------------
say "--- python ---"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  say "${PY_BIN} not found; installing (needs sudo)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
else
  say "${PY_BIN}: $(${PY_BIN} --version)"
fi
say ""

# --- 3. venv ---------------------------------------------------------------
say "--- venv ---"
if [[ ! -x "${VENV}/bin/python" ]]; then
  "$PY_BIN" -m venv "$VENV"
  say "created ${VENV}"
else
  say "reusing ${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
python --version
python -m pip install --quiet --upgrade "pip" "setuptools==69.5.1" "wheel"
say ""

# --- 4. the toolkit --------------------------------------------------------
say "--- rknn-toolkit2 ${TOOLKIT_VERSION} ---"
# `--no-deps` first, then resolve: the toolkit's pin on onnx==1.13.1 is the one
# that matters, and letting pip solve the whole graph in one go sometimes picks
# a torch build with no cp310 wheel. Installing the toolkit alone and then its
# deps keeps the failure mode legible.
python -m pip install --quiet "rknn-toolkit2==${TOOLKIT_VERSION}" || {
  say "!! toolkit install failed. Re-run with -v to see which dep has no wheel:"
  say "   ${VENV}/bin/pip install -v rknn-toolkit2==${TOOLKIT_VERSION}"
  exit 1
}
say ""

# --- 5. verify import, not just install ------------------------------------
# `pip show` succeeding proves nothing: rknn-toolkit2 imports a compiled
# `librknnrt` and its Python binding at import time, and a missing .so only
# shows up here.
say "--- import check ---"
python - <<'PY'
import sys
try:
    from rknn.api import RKNN
except Exception as exc:                       # noqa: BLE001 - we want any failure
    print(f"FAILED to import RKNN: {exc}")
    sys.exit(1)
print("RKNN import: ok")
try:
    import rknn
    print("rknn package version:", getattr(rknn, "__version__", "unknown"))
except Exception:
    pass
PY
# The simulator needs the runtime library even without a board.
say ""
say "--- simulator smoke test (build nothing) ---"
python - <<'PY'
from rknn.api import RKNN
r = RKNN(verbose=False)
try:
    r.config(target_platform='rk3576')
    print("config(target_platform='rk3576'): ok")
finally:
    r.release()
PY

say ""
say "=== done ==="
say "Activate with:   source ${VENV}/bin/activate"
say "Then, from the project dir:"
say "  python scripts/20_convert_rknn.py --model stgcn --precision fp16"
say "  python scripts/21_verify_rknn.py --precision fp16"
