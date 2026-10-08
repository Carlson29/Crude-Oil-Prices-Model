from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_references_existing_readme():
    repository = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    readme = pyproject["project"]["readme"]

    assert readme == "README.md"
    assert (repository / readme).is_file()
