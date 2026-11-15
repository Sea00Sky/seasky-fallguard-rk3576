"""Build contact sheets for manual fall intervals (QC aid)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotate.qc_thumbnails import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
