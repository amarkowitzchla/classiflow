"""Data I/O and validation utilities."""

from classiflow.io.compatibility import (
    CompatibilityResult,
    assess_data_compatibility,
    print_compatibility_report,
)
from classiflow.io.loaders import load_data, load_data_with_groups, validate_data
from classiflow.io.schema import DataSchema

__all__ = [
    "load_data",
    "load_data_with_groups",
    "validate_data",
    "DataSchema",
    "assess_data_compatibility",
    "print_compatibility_report",
    "CompatibilityResult",
]
