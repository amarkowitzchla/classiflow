"""Data I/O and validation utilities."""

from classiflow.io.loaders import load_data, load_data_with_groups, validate_data
from classiflow.io.schema import DataSchema
from classiflow.io.compatibility import (
    assess_data_compatibility,
    print_compatibility_report,
    CompatibilityResult,
)

__all__ = [
    "load_data",
    "load_data_with_groups",
    "validate_data",
    "DataSchema",
    "assess_data_compatibility",
    "print_compatibility_report",
    "CompatibilityResult",
]
