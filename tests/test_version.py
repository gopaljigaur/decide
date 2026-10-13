import pathlib
import re

import decide


def test_dunder_version_matches_pyproject():
    """0.2.3 shipped with `decide.__version__` still reading 0.2.2."""
    root = pathlib.Path(__file__).resolve().parent.parent
    # Read it with a regex rather than tomllib, which is 3.11+ while the package supports 3.10.
    pyproject = (root / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert declared is not None, "no version in pyproject.toml"
    assert decide.__version__ == declared.group(1)
