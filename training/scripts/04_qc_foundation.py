"""Run Phase 0-1 QC gates and write reports."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import qc  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(qc.main())
