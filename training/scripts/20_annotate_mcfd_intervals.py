"""Launch the MCFD multi-segment fall interval annotator."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotate.mcfd_interval_tool import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
