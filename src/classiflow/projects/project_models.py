"""Pydantic models for classiflow project configuration and registries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from classiflow import __version__
from classiflow.projects.yaml_utils import load_yaml, dump_yaml


class ProjectInfo(BaseModel):
    """Core project metadata."""

    id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None


class ManifestRef(BaseModel):
    """Reference to a dataset manifest."""

    manifest: str


class DataConfig(BaseModel):
    """Training/test manifest references."""

    train: ManifestRef
    test: Optional[ManifestRef] = None


class KeyColumns(BaseModel):
    """Column naming for manifests."""

    sample_id: Optional[str] = None
    patient_id: Optional[str] = None
    label: str = "label"
    slide_id: Optional[str] = None
    specimen_id: Optional[str] = None


class TaskConfig(BaseModel):
    """Modeling task settings."""

    mode: Literal["binary", "meta", "hierarchical", "multiclass"] = "meta"
    patient_stratified: bool = True
    hierarchy_path: Optional[str] = None


class NestedCVConfig(BaseModel):
    """Nested cross-validation settings."""

    outer_folds: int = 3
    inner_folds: int = 5
    repeats: int = 2
    seed: int = 42

    @field_validator("outer_folds", "inner_folds", "repeats")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class ValidationConfig(BaseModel):
    """Validation pipeline configuration."""

    nested_cv: NestedCVConfig = Field(default_factory=NestedCVConfig)


class ModelsConfig(BaseModel):
    """Model candidates and selection criteria."""

    candidates: List[str] = Field(
        default_factory=lambda: [
            "logistic_regression",
            "svm",
            "random_forest",
            "gradient_boost",
        ]
    )
    selection_metric: str = "f1"
    selection_direction: Literal["max", "min"] = "max"


class ImbalanceConfig(BaseModel):
    """Imbalance handling and SMOTE comparison."""

    smote: Dict[str, bool] = Field(default_factory=lambda: {"enabled": False, "compare": False})


class LogisticRegressionConfig(BaseModel):
    """Logistic regression defaults for multiclass training."""

    solver: str = "saga"
    multi_class: str = "auto"
    penalty: str = "l2"  # deprecated in sklearn 1.8; retained for compatibility
    max_iter: int = 5000
    tol: float = 1e-3
    C: float = 1.0
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1


class MulticlassTrainingConfig(BaseModel):
    """Multiclass-specific training defaults."""

    group_stratify: bool = True
    estimator_mode: Literal["all", "torch_only", "cpu_only"] = "all"
    logreg: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)


class MetricsConfig(BaseModel):
    """Evaluation metrics configuration."""

    primary: List[str] = Field(default_factory=lambda: ["f1", "balanced_accuracy"])
    averaging: str = "macro"
    include_confidence_intervals: bool = False


class CalibrationConfig(BaseModel):
    """Meta classifier calibration settings."""

    calibrate_meta: bool = True
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: int = 3
    bins: int = 10
    isotonic_min_samples: int = 100


class FinalModelConfig(BaseModel):
    """Final model training configuration."""

    # Sampler option for final training
    sampler: Optional[str] = None  # "none", "smote", or None for auto-select

    # Sanity check thresholds
    sanity_min_std: float = 0.02
    sanity_max_mean_deviation: float = 0.15

    # Training options
    train_from_scratch: bool = True  # Always true in new workflow
    verify_dataset_hash: bool = True


class BundleConfig(BaseModel):
    """Bundle output configuration."""

    name: str = "model_bundle"
    include_preprocessing: bool = True
    format: str = "zip"


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    model_config = {"protected_namespaces": ()}

    project: ProjectInfo
    data: DataConfig
    key_columns: KeyColumns = Field(default_factory=KeyColumns)
    task: TaskConfig = Field(default_factory=TaskConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    multiclass: MulticlassTrainingConfig = Field(default_factory=MulticlassTrainingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    final_model: FinalModelConfig = Field(default_factory=FinalModelConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    backend: Literal["sklearn", "torch"] = "sklearn"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    model_set: Optional[str] = None
    torch_dtype: Literal["float32", "float16"] = "float32"
    torch_num_workers: int = 0
    require_torch_device: bool = False

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        """Load project configuration from YAML."""
        data = load_yaml(path)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Write project configuration to YAML."""
        dump_yaml(self.model_dump(mode="python"), path)


class DatasetSchema(BaseModel):
    """Schema summary for a dataset manifest."""

    columns: List[str]
    dtypes: Dict[str, str]
    feature_representation: Literal["wide", "feature_path"] = "wide"
    feature_columns: List[str] = Field(default_factory=list)
    feature_path_column: Optional[str] = None


class DatasetStats(BaseModel):
    """Statistics derived from a dataset manifest."""

    rows: int
    patients: Optional[int] = None
    labels: Dict[str, int] = Field(default_factory=dict)


class DatasetEntry(BaseModel):
    """Registry entry for a dataset manifest."""

    dataset_type: Literal["train", "test"]
    manifest_path: str
    sha256: str
    size_bytes: int
    registered_at: str
    classiflow_version: str = Field(default_factory=lambda: __version__)
    git_hash: Optional[str] = None
    data_schema: DatasetSchema = Field(alias="schema")
    stats: DatasetStats
    dirty: bool = False
    previous_hashes: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DatasetRegistry(BaseModel):
    """Registry for all registered datasets in a project."""

    datasets: Dict[str, DatasetEntry] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def load(cls, path: Path) -> "DatasetRegistry":
        """Load registry from YAML if present."""
        if not path.exists():
            return cls()
        data = load_yaml(path)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Write registry to YAML."""
        self.updated_at = datetime.utcnow().isoformat()
        dump_yaml(self.model_dump(mode="python", by_alias=True), path)


class StabilityGate(BaseModel):
    """Stability thresholds for CV metrics."""

    std_max: Dict[str, float] = Field(default_factory=dict)
    pass_rate_min: float = 0.8


class PhaseThresholds(BaseModel):
    """Thresholds for a project phase."""

    required: Dict[str, float] = Field(default_factory=dict)
    stability: Optional[StabilityGate] = None
    safety: Dict[str, float] = Field(default_factory=dict)


class CalibrationThresholds(BaseModel):
    """Calibration-specific promotion thresholds."""

    brier_max: float = 0.20
    ece_max: float = 0.25


class PromotionConfig(BaseModel):
    """Promotion-wide settings."""

    calibration: CalibrationThresholds = Field(default_factory=CalibrationThresholds)


class OverridePolicy(BaseModel):
    """Override policy for promotion decisions."""

    allow_override: bool = True
    require_comment: bool = True
    require_approver: bool = True


class ThresholdsConfig(BaseModel):
    """Promotion gate configuration."""

    technical_validation: PhaseThresholds = Field(default_factory=PhaseThresholds)
    independent_test: PhaseThresholds = Field(default_factory=PhaseThresholds)
    promotion_logic: Literal["ALL_REQUIRED_AND_STABILITY"] = "ALL_REQUIRED_AND_STABILITY"
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    override: OverridePolicy = Field(default_factory=OverridePolicy)

    @classmethod
    def load(cls, path: Path) -> "ThresholdsConfig":
        if not path.exists():
            return cls()
        data = load_yaml(path)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        dump_yaml(self.model_dump(mode="python"), path)
