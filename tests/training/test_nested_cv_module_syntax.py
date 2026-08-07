"""Regression tests for the nested-CV module's importability."""

from __future__ import annotations

from pathlib import Path


def test_nested_cv_module_compiles() -> None:
    """The nested-CV implementation must be syntactically importable."""
    module_path = Path(__file__).parents[2] / "src" / "classiflow" / "training" / "nested_cv.py"
    compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec")
