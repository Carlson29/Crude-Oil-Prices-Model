"""Run the reproducible companion benchmark from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from wti_forecasting.benchmark import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
