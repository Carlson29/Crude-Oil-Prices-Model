from __future__ import annotations

import re
from pathlib import Path


def test_notebook_does_not_contain_hardcoded_credentials():
    repository = Path(__file__).resolve().parents[1]
    notebook = (repository / "code Files" / "oil_prediction.ipynb").read_text(encoding="utf-8")

    credential_patterns = (
        r"Fred\s*\(\s*api_key\s*=\s*['\"][^'\"]+",
        r"api[_-]?key\s*=\s*['\"][a-z0-9]{32}['\"]",
    )

    for pattern in credential_patterns:
        assert not re.search(pattern, notebook, re.IGNORECASE)
