"""Data compatibility assessment for classiflow training modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from classiflow.config import MetaConfig, HierarchicalConfig
from classiflow.io.schema import DataSchema

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """Results from data compatibility assessment."""

    is_compatible: bool
    mode: str  # "meta" or "hierarchical"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    data_summary: Optional[Dict[str, Any]] = None
    schema: Optional[Union[DataSchema, Dict[str, Any]]] = None

    def __str__(self) -> str:
        """Format results for display."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  DATA COMPATIBILITY ASSESSMENT - {self.mode.upper()} MODE")
        lines.append(f"{'='*60}")

        # Status
        status = "✓ COMPATIBLE" if self.is_compatible else "✗ INCOMPATIBLE"
        lines.append(f"\nStatus: {status}\n")

        # Data summary
        if self.data_summary:
            lines.append("Data Summary:")
            lines.append(f"  • Samples: {self.data_summary.get('n_samples', 'N/A')}")
            lines.append(f"  • Features: {self.data_summary.get('n_features', 'N/A')}")
            lines.append(f"  • Classes: {self.data_summary.get('n_classes', 'N/A')}")

            if self.mode == "hierarchical":
                # Show patient info only if using patient stratification
                if self.data_summary.get('use_patient_stratification') and "n_patients" in self.data_summary:
                    lines.append(f"  • Patients: {self.data_summary.get('n_patients', 'N/A')}")
                    lines.append(f"  • Stratification: Patient-level")
                else:
                    lines.append(f"  • Stratification: Sample-level")
                if self.data_summary.get('hierarchical'):
                    lines.append(f"  • Hierarchical: Yes")
                    lines.append(f"  • L1 Classes: {len(self.data_summary.get('l1_classes', []))}")
                    l2_branches = self.data_summary.get('l2_classes_per_branch', {})
                    if l2_branches:
                        lines.append(f"  • L2 Branches: {len(l2_branches)}")

            # Class distribution
            if "class_distribution" in self.data_summary:
                lines.append("\n  Class Distribution:")
                for cls, count in sorted(self.data_summary["class_distribution"].items()):
                    lines.append(f"    - {cls}: {count} samples")

        # Errors
        if self.errors:
            lines.append(f"\n{'─'*60}")
            lines.append("ERRORS:")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"  {i}. {err}")

        # Warnings
        if self.warnings:
            lines.append(f"\n{'─'*60}")
            lines.append("WARNINGS:")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")

        # Suggestions
        if self.suggestions:
            lines.append(f"\n{'─'*60}")
            lines.append("SUGGESTIONS:")
            for i, sug in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {sug}")

        lines.append(f"\n{'='*60}\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_compatible": self.is_compatible,
            "mode": self.mode,
            "warnings": self.warnings,
            "errors": self.errors,
            "suggestions": self.suggestions,
            "data_summary": self.data_summary,
            "schema": self.schema.model_dump() if isinstance(self.schema, DataSchema) else self.schema,
        }


def assess_data_compatibility(
    config: Union[MetaConfig, HierarchicalConfig],
    return_details: bool = True,
) -> CompatibilityResult:
    """
    Assess whether input data is compatible with the specified training mode.

    This function performs comprehensive validation of input data and configuration,
    checking for common issues that would cause training to fail. It provides
    actionable suggestions for fixing incompatibilities.

    Parameters
    ----------
    config : MetaConfig | HierarchicalConfig
        Training configuration containing data path and parameters
    return_details : bool
        Whether to include detailed data summary in results

    Returns
    -------
    CompatibilityResult
        Assessment results with compatibility status, warnings, errors, and suggestions

    Examples
    --------
    >>> from classiflow.config import MetaConfig
    >>> config = MetaConfig(
    ...     data_csv="data.csv",
    ...     label_col="diagnosis",
    ... )
    >>> result = assess_data_compatibility(config)
    >>> print(result)
    >>> if not result.is_compatible:
    ...     print("Fix these issues:", result.suggestions)
    """
    mode = "hierarchical" if isinstance(config, HierarchicalConfig) else "meta"
    result = CompatibilityResult(is_compatible=True, mode=mode)

    # Step 1: File existence check
    if not _check_file_exists(config.data_csv, result):
        return result

    # Step 2: Load and validate data
    try:
        df = pd.read_csv(config.data_csv)
    except Exception as e:
        result.is_compatible = False
        result.errors.append(f"Failed to read CSV file: {e}")
        result.suggestions.append("Ensure the file is a valid CSV format")
        return result

    if df.empty:
        result.is_compatible = False
        result.errors.append("CSV file is empty")
        return result

    # Step 3: Mode-specific validation
    if mode == "meta":
        _assess_meta_compatibility(config, df, result, return_details)
    else:
        _assess_hierarchical_compatibility(config, df, result, return_details)

    return result


def _check_file_exists(csv_path: Path, result: CompatibilityResult) -> bool:
    """Check if data file exists and is readable."""
    if not csv_path.exists():
        result.is_compatible = False
        result.errors.append(f"Data file not found: {csv_path}")
        result.suggestions.append(f"Verify the path is correct: {csv_path.absolute()}")
        return False

    if not csv_path.is_file():
        result.is_compatible = False
        result.errors.append(f"Path is not a file: {csv_path}")
        return False

    return True


def _assess_meta_compatibility(
    config: MetaConfig,
    df: pd.DataFrame,
    result: CompatibilityResult,
    return_details: bool,
) -> None:
    """Assess compatibility for meta-classifier training mode."""

    # Check label column exists
    if config.label_col not in df.columns:
        result.is_compatible = False
        result.errors.append(f"Label column '{config.label_col}' not found in CSV")
        result.suggestions.append(f"Available columns: {', '.join(df.columns)}")
        result.suggestions.append(f"Update config.label_col to one of the available columns")
        return

    # Extract labels
    y = df[config.label_col].astype(str)

    # Handle missing labels
    n_missing_labels = y.isna().sum()
    if n_missing_labels > 0:
        result.warnings.append(f"{n_missing_labels} rows have missing labels (will be dropped)")
        y = y.dropna()
        df = df[y.index]

    if len(y) == 0:
        result.is_compatible = False
        result.errors.append("No valid labels after removing missing values")
        return

    # Filter to specified classes if provided
    if config.classes is not None:
        y_before = len(y)
        mask = y.isin(config.classes)
        y = y[mask]
        df = df[mask]
        n_filtered = y_before - len(y)
        if n_filtered > 0:
            result.warnings.append(f"Filtered to {len(config.classes)} classes, removed {n_filtered} samples")

    # Extract features
    if config.feature_cols is not None:
        missing = set(config.feature_cols) - set(df.columns)
        if missing:
            result.is_compatible = False
            result.errors.append(f"Feature columns not found: {missing}")
            result.suggestions.append("Remove missing columns from config.feature_cols or update CSV")
            return
        X = df[config.feature_cols].copy()
    else:
        # Auto-select numeric columns
        X = df.drop(columns=[config.label_col]).select_dtypes(include=[np.number])

    # Check for features
    if X.shape[1] == 0:
        result.is_compatible = False
        result.errors.append("No numeric feature columns found")
        result.suggestions.append("Ensure CSV contains numeric feature columns")
        result.suggestions.append("Or specify feature_cols explicitly if columns need type conversion")
        return

    # Basic validation
    n_samples, n_features = X.shape
    n_classes = y.nunique()
    class_counts = y.value_counts().to_dict()

    # Check minimum samples
    if n_samples < 10:
        result.is_compatible = False
        result.errors.append(f"Too few samples: {n_samples} (minimum: 10)")
        result.suggestions.append("Collect more data or reduce filtering constraints")

    # Check minimum classes for meta-classifier
    if n_classes < 3:
        result.is_compatible = False
        result.errors.append(f"Too few classes: {n_classes} (meta-classifier requires at least 3)")
        result.suggestions.append("Meta-classifier is for multiclass problems (3+ classes)")
        result.suggestions.append("For binary classification, use 'classiflow train' instead")

    # Check class balance
    min_samples_per_class = min(class_counts.values())
    if min_samples_per_class < 2:
        result.is_compatible = False
        classes_too_small = [cls for cls, cnt in class_counts.items() if cnt < 2]
        result.errors.append(f"Classes with < 2 samples: {classes_too_small}")
        result.suggestions.append("Each class needs at least 2 samples for cross-validation")
        result.suggestions.append(f"Remove classes: {classes_too_small} or collect more samples")

    # Check for imbalanced classes
    max_samples_per_class = max(class_counts.values())
    imbalance_ratio = max_samples_per_class / min_samples_per_class
    if imbalance_ratio > 10:
        result.warnings.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        result.suggestions.append(f"Consider using SMOTE: set smote_mode='on' or 'both'")

    # Check CV feasibility
    if n_samples < config.outer_folds * n_classes:
        result.warnings.append(
            f"Small dataset ({n_samples} samples) for {config.outer_folds}-fold CV with {n_classes} classes"
        )
        result.suggestions.append(f"Consider reducing outer_folds (currently {config.outer_folds})")

    # Check feature quality
    _check_feature_quality(X, result)

    # Build data summary
    if return_details:
        try:
            schema = DataSchema.from_data(X, y)
            result.schema = schema
        except Exception as e:
            result.warnings.append(f"Could not create data schema: {e}")
            result.schema = None

        result.data_summary = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "feature_names": list(X.columns),
            "class_distribution": class_counts,
            "class_names": sorted(y.unique().tolist()),
        }


def _assess_hierarchical_compatibility(
    config: HierarchicalConfig,
    df: pd.DataFrame,
    result: CompatibilityResult,
    return_details: bool,
) -> None:
    """Assess compatibility for hierarchical training mode."""

    # Check patient column exists (only if provided)
    use_patient_stratification = config.patient_col is not None
    if use_patient_stratification and config.patient_col not in df.columns:
        result.is_compatible = False
        result.errors.append(f"Patient column '{config.patient_col}' not found in CSV")
        result.suggestions.append(f"Available columns: {', '.join(df.columns)}")
        result.suggestions.append(f"Update config.patient_col or set to None for sample-level stratification")
        return

    # Check L1 label column exists
    if config.label_l1 not in df.columns:
        result.is_compatible = False
        result.errors.append(f"L1 label column '{config.label_l1}' not found in CSV")
        result.suggestions.append(f"Available columns: {', '.join(df.columns)}")
        result.suggestions.append(f"Update config.label_l1 or add the column to CSV")
        return

    # Check L2 label column exists if hierarchical mode
    hierarchical = config.label_l2 is not None
    if hierarchical and config.label_l2 not in df.columns:
        result.is_compatible = False
        result.errors.append(f"L2 label column '{config.label_l2}' not found in CSV")
        result.suggestions.append(f"Available columns: {', '.join(df.columns)}")
        result.suggestions.append(f"Update config.label_l2 or set to None for single-level mode")
        return

    # Drop rows with missing L1 labels (and patient ID if using patient stratification)
    required_cols = [config.label_l1]
    if use_patient_stratification:
        required_cols.append(config.patient_col)

    df_clean = df.dropna(subset=required_cols)
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        if use_patient_stratification:
            result.warnings.append(f"Dropped {n_dropped} rows with missing patient ID or L1 label")
        else:
            result.warnings.append(f"Dropped {n_dropped} rows with missing L1 label")

    if len(df_clean) == 0:
        result.is_compatible = False
        if use_patient_stratification:
            result.errors.append("No valid rows after removing missing patient IDs and L1 labels")
        else:
            result.errors.append("No valid rows after removing missing L1 labels")
        return

    df = df_clean

    # Extract features
    exclude_cols = [config.label_l1]
    if use_patient_stratification:
        exclude_cols.append(config.patient_col)
    if hierarchical:
        exclude_cols.append(config.label_l2)

    if config.feature_cols is not None:
        missing = set(config.feature_cols) - set(df.columns)
        if missing:
            result.is_compatible = False
            result.errors.append(f"Feature columns not found: {missing}")
            result.suggestions.append("Remove missing columns from config.feature_cols or update CSV")
            return
        X = df[config.feature_cols].copy()
    else:
        # Auto-select numeric columns
        X = df.drop(columns=exclude_cols).select_dtypes(include=[np.number])

    # Check for features
    if X.shape[1] == 0:
        result.is_compatible = False
        result.errors.append("No numeric feature columns found")
        result.suggestions.append("Ensure CSV contains numeric feature columns")
        result.suggestions.append("Or specify feature_cols explicitly")
        return

    # Analyze patient and label structure
    y_l1 = df[config.label_l1].astype(str)

    n_samples = len(df)
    n_features = X.shape[1]
    n_l1_classes = y_l1.nunique()
    l1_class_counts = y_l1.value_counts().to_dict()

    # Patient-level analysis (only if patient_col is provided)
    n_patients = None
    if use_patient_stratification:
        patients = df[config.patient_col].astype(str)
        n_patients = patients.nunique()

    # Check minimum samples
    if n_samples < 10:
        result.is_compatible = False
        result.errors.append(f"Too few samples: {n_samples} (minimum: 10)")
        result.suggestions.append("Collect more data")

    # Patient-level checks (only if using patient stratification)
    if use_patient_stratification:
        # Check minimum patients
        if n_patients < 10:
            result.warnings.append(f"Few patients: {n_patients} (recommended: 30+)")
            result.suggestions.append("More patients improve generalization")

        # Check patient count vs outer folds
        if n_patients < config.outer_folds:
            result.is_compatible = False
            result.errors.append(
                f"Too few patients ({n_patients}) for {config.outer_folds}-fold CV"
            )
            result.suggestions.append(f"Reduce outer_folds to {n_patients - 1} or fewer")
            result.suggestions.append("Or collect more patient data")
    else:
        # Sample-level checks (when not using patient stratification)
        if n_samples < config.outer_folds * 2:
            result.is_compatible = False
            result.errors.append(
                f"Too few samples ({n_samples}) for {config.outer_folds}-fold CV"
            )
            result.suggestions.append(f"Reduce outer_folds or collect more samples")

    # Check L1 classes
    if n_l1_classes < 2:
        result.is_compatible = False
        result.errors.append(f"L1 has only {n_l1_classes} class (minimum: 2)")
        result.suggestions.append("Need at least 2 L1 classes for classification")

    # Check L1 class balance
    min_l1_samples = min(l1_class_counts.values())
    if min_l1_samples < 2:
        result.is_compatible = False
        small_classes = [cls for cls, cnt in l1_class_counts.items() if cnt < 2]
        result.errors.append(f"L1 classes with < 2 samples: {small_classes}")
        result.suggestions.append("Each L1 class needs at least 2 samples")

    # Hierarchical-specific checks
    l2_summary = None
    if hierarchical:
        y_l2 = df[config.label_l2].astype(str)

        # Allow missing L2 for some samples
        n_missing_l2 = y_l2.isna().sum()
        if n_missing_l2 > 0:
            result.warnings.append(
                f"{n_missing_l2} samples have missing L2 labels (will use L1 only for these)"
            )

        # Analyze L2 classes per L1 branch
        l2_classes_per_branch = {}
        for l1_cls in y_l1.unique():
            mask = (y_l1 == l1_cls) & y_l2.notna()
            if mask.sum() > 0:
                l2_classes = y_l2[mask].unique().tolist()
                l2_classes_per_branch[l1_cls] = l2_classes

        # Check if any branches have sufficient L2 classes
        valid_branches = {
            l1: l2_list
            for l1, l2_list in l2_classes_per_branch.items()
            if len(l2_list) >= config.min_l2_classes_per_branch
        }

        invalid_branches = {
            l1: l2_list
            for l1, l2_list in l2_classes_per_branch.items()
            if len(l2_list) < config.min_l2_classes_per_branch
        }

        if not valid_branches:
            result.warnings.append(
                f"No L1 branches have >= {config.min_l2_classes_per_branch} L2 classes"
            )
            result.suggestions.append(
                f"Reduce min_l2_classes_per_branch (currently {config.min_l2_classes_per_branch})"
            )
            result.suggestions.append("Or set label_l2=None for single-level classification")
        elif invalid_branches:
            # Some branches have insufficient L2 classes
            invalid_branch_names = list(invalid_branches.keys())
            result.warnings.append(
                f"{len(invalid_branches)} L1 branches have < {config.min_l2_classes_per_branch} L2 classes: {invalid_branch_names}"
            )
            result.suggestions.append(
                f"These branches will be skipped for L2 classification: {invalid_branch_names}"
            )

        l2_summary = {
            "l2_classes_per_branch": l2_classes_per_branch,
            "valid_branches": list(valid_branches.keys()),
            "n_valid_branches": len(valid_branches),
        }

    # Check feature quality
    _check_feature_quality(X, result)

    # Build data summary
    if return_details:
        result.data_summary = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_l1_classes,
            "feature_names": list(X.columns),
            "class_distribution": l1_class_counts,
            "class_names": sorted(y_l1.unique().tolist()),
            "hierarchical": hierarchical,
            "l1_classes": sorted(y_l1.unique().tolist()),
            "use_patient_stratification": use_patient_stratification,
        }

        # Add patient info only if using patient stratification
        if use_patient_stratification:
            result.data_summary["n_patients"] = n_patients

        if l2_summary:
            result.data_summary.update(l2_summary)


def _check_feature_quality(X: pd.DataFrame, result: CompatibilityResult) -> None:
    """Check feature quality issues."""

    # Check for missing values
    na_cols = X.columns[X.isna().any()].tolist()
    if na_cols:
        n_na = len(na_cols)
        result.warnings.append(f"{n_na} features have missing values")
        result.suggestions.append("Consider imputing missing values before training")
        if n_na <= 5:
            result.suggestions.append(f"Features with missing values: {', '.join(na_cols)}")

    # Check for constant features
    constant_cols = X.columns[X.std() == 0].tolist()
    if constant_cols:
        n_const = len(constant_cols)
        result.warnings.append(f"{n_const} constant features (zero variance)")
        result.suggestions.append("Constant features will be removed during training")
        if n_const <= 5:
            result.suggestions.append(f"Constant features: {', '.join(constant_cols)}")

    # Check for infinite values
    inf_cols = X.columns[np.isinf(X).any()].tolist()
    if inf_cols:
        result.is_compatible = False
        result.errors.append(f"{len(inf_cols)} features contain infinite values")
        result.suggestions.append(f"Features with inf: {', '.join(inf_cols[:5])}")
        result.suggestions.append("Replace infinite values with NaN or finite values")

    # Check for very high variance features
    high_var_threshold = X.std().quantile(0.95) * 100  # 100x the 95th percentile
    high_var_cols = X.columns[X.std() > high_var_threshold].tolist()
    if high_var_cols:
        result.warnings.append(f"{len(high_var_cols)} features have very high variance")
        result.suggestions.append("Consider scaling/normalizing features before training")


def print_compatibility_report(config: Union[MetaConfig, HierarchicalConfig]) -> CompatibilityResult:
    """
    Assess and print data compatibility report.

    Convenience function that assesses compatibility and prints formatted results.

    Parameters
    ----------
    config : MetaConfig | HierarchicalConfig
        Training configuration

    Returns
    -------
    CompatibilityResult
        Assessment results

    Examples
    --------
    >>> from classiflow.config import MetaConfig
    >>> config = MetaConfig(data_csv="data.csv", label_col="diagnosis")
    >>> result = print_compatibility_report(config)
    """
    result = assess_data_compatibility(config, return_details=True)
    print(result)
    return result
