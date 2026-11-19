"""Version metadata has exactly one source of truth: pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import imgsim


def test_version_resolves_from_installed_metadata():
    """Guards against the fallback path being taken in a normal install."""
    assert imgsim.__version__ != "0.0.0+unknown", (
        "importlib.metadata could not find the imgsim-py distribution; "
        "the version fell back to a placeholder"
    )


def test_version_matches_pyproject():
    """Regression: 0.1.1 shipped with new code but __version__ still at 0.1.0.

    The literal in __init__.py was a second source of truth and drifted the moment
    pyproject.toml was bumped. It now reads from the distribution metadata, so this
    test fails loudly if anyone reintroduces a hardcoded copy.
    """
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return  # installed from a wheel, no source tree to compare against

    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "no version field in pyproject.toml"
    assert imgsim.__version__ == match.group(1)


def test_cli_version_flag_reports_the_package_version(capsys):
    import pytest

    from imgsim.cli import main

    with pytest.raises(SystemExit):
        main([".", "--version"])
    assert imgsim.__version__ in capsys.readouterr().out
