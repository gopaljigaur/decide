import pathlib

import tomllib

import decide


def test_dunder_version_matches_pyproject():
    """0.2.3 shipped with `decide.__version__` still reading 0.2.2."""
    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert decide.__version__ == pyproject["project"]["version"]
