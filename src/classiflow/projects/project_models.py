"""Pydantic models for classiflow project configuration and registries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import warnings

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from classiflow import __version__
from classiflow.projects.yaml_utils import dump_yaml, load_yaml

TaskMode = Literal["binary", "meta", "hierarchical", "multiclass"]
ExecutionEngine = Literal["sklearn", "torch", "hybrid"]
ExecutionDevice = Literal["auto", "cpu", "cuda", "mps"]

SKLEARN_CANDIDATES = {
    "logistic_regression",
    "svm",
    "random_forest",
    "gradient_boost",
}
TORCH_CANDIDATES = {
    "torch_logistic_regression",
    "torch_mlp",
    "torch_linear",
}


def _default_torch_num_workers() -> int:
    """Compute a conservative default DataLoader worker count."""
    cpu_total = os.cpu_count() or 1
    return max(cpu_total - 2, 1)


def available_project_options() -> Dict[str, List[str]]:
    """Return discoverable option enums for project config UX."""
    return {
        "task.mode": ["binary", "meta", "multiclass", "hierarchical"],
        "execution.engine": ["sklearn", "torch", "hybrid"],
        "execution.device": ["auto", "cpu", "cuda", "mps"],
        "multiclass.backend[sklearn]": ["sklearn_cpu"],
        "multiclass.backend[torch]": ["torch_auto", "torch_cpu", "torch_cuda", "torch_mps"],
        "multiclass.backend[hybrid]": ["hybrid_sklearn_meta_torch_base"],
        "models.selection_direction": ["max", "min"],
        "calibration.enabled": ["auto", "true", "false"],
        "calibration.method": ["sigmoid", "isotonic"],
        "calibration.binning": ["uniform", "quantile"],
        "execution.torch.dtype": ["float32", "float16"],
    }


def _map_legacy_multiclass_backend(
    estimator_mode: str,
    engine: str,
    device: str,
) -> str:
    mode = estimator_mode.lower()
    eng = engine.lower()
    dev = device.lower()

    if mode == "cpu_only":
        return "sklearn_cpu"
    if mode == "torch_only":
        return f"torch_{dev}"
    if mode == "all":
        return "hybrid_sklearn_meta_torch_base" if eng != "torch" else f"torch_{dev}"
    return "sklearn_cpu"


def normalize_project_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize legacy project config keys into current schema."""
    data = dict(payload)
    warnings_out: List[str] = []

    execution = data.get("execution") if isinstance(data.get("execution"), dict) else None

    legacy_keys = {
        "backend",
        "device",
        "model_set",
        "torch_dtype",
        "torch_num_workers",
        "require_torch_device",
    }
    has_legacy_execution = any(k in data for k in legacy_keys)

    if execution is None and has_legacy_execution:
        backend = str(data.pop("backend", "sklearn")).lower()
        device = str(data.pop("device", "auto")).lower()
        model_set = data.pop("model_set", None)
        torch_dtype = str(data.pop("torch_dtype", "float32")).lower()
        legacy_torch_num_workers = data.pop("torch_num_workers", None)
        if legacy_torch_num_workers is None:
            torch_num_workers = _default_torch_num_workers()
        else:
            torch_num_workers = int(legacy_torch_num_workers)
        require_torch_device = bool(data.pop("require_torch_device", False))

        normalized_execution: Dict[str, Any] = {"engine": backend}
        if backend in {"torch", "hybrid"}:
            normalized_execution["device"] = device
            normalized_execution["torch"] = {
                "dtype": torch_dtype,
                "num_workers": torch_num_workers,
                "require_device": require_torch_device,
            }
        elif model_set is not None:
            normalized_execution["model_set"] = model_set

        if model_set is not None and backend in {"torch", "hybrid"}:
            normalized_execution["model_set"] = model_set

        data["execution"] = normalized_execution
        warnings_out.append(
            "Legacy execution keys (backend/device/torch_*) were normalized into execution.*. "
            "Please run `classiflow config normalize` to persist the new format."
        )
    elif execution is not None:
        for key in legacy_keys:
            if key in data:
                data.pop(key, None)
                warnings_out.append(
                    f"Dropped legacy key '{key}' because execution.* is present."
                )

    if isinstance(data.get("multiclass"), dict):
        mc = dict(data["multiclass"])

        if "logreg" in mc and "sklearn" not in mc:
            mc["sklearn"] = {"logreg": mc.pop("logreg")}
            warnings_out.append("Legacy multiclass.logreg moved to multiclass.sklearn.logreg.")

        if "estimator_mode" in mc and "backend" not in mc:
            est_mode = str(mc.pop("estimator_mode"))
            execution_block = data.get("execution", {}) if isinstance(data.get("execution"), dict) else {}
            engine = str(execution_block.get("engine", "sklearn"))
            device = str(execution_block.get("device", "auto"))
            mc["backend"] = _map_legacy_multiclass_backend(est_mode, engine, device)
            warnings_out.append(
                "Legacy multiclass.estimator_mode was normalized to multiclass.backend."
            )

        data["multiclass"] = mc

    if isinstance(data.get("calibration"), dict):
        calibration = dict(data["calibration"])
        legacy_toggle = calibration.pop("calibrate_meta", None)
        if legacy_toggle is not None and "enabled" not in calibration:
            calibration["enabled"] = "true" if bool(legacy_toggle) else "false"
            warnings_out.append(
                "Legacy calibration.calibrate_meta was normalized to calibration.enabled."
            )
        data["calibration"] = calibration

    return data, warnings_out


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

    mode: TaskMode = "meta"
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


class MulticlassSklearnConfig(BaseModel):
    """Sklearn-specific multiclass settings."""

    logreg: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)


class MulticlassTorchConfig(BaseModel):
    """Torch-specific multiclass settings."""

    model_set: Literal["torch_basic", "torch_fast"] = "torch_basic"

    model_config = {"protected_namespaces": ()}


class MulticlassTrainingConfig(BaseModel):
    """Multiclass-specific training defaults and runtime selection."""

    group_stratify: bool = True
    backend: Optional[str] = Field(
        default=None,
        description=(
            "Explicit multiclass runtime backend. "
            "sklearn_cpu for engine=sklearn, torch_<device> for engine=torch, "
            "hybrid_sklearn_meta_torch_base for engine=hybrid."
        ),
    )
    sklearn: MulticlassSklearnConfig = Field(default_factory=MulticlassSklearnConfig)
    torch: Optional[MulticlassTorchConfig] = None

    @property
    def estimator_mode(self) -> Literal["all", "torch_only", "cpu_only"]:
        """Legacy adapter for training.multiclass config contract."""
        backend = (self.backend or "sklearn_cpu").lower()
        if backend.startswith("torch_"):
            return "torch_only"
        if backend.startswith("hybrid_"):
            return "all"
        return "cpu_only"

    @property
    def logreg(self) -> LogisticRegressionConfig:
        """Legacy adapter for old multiclass.logreg access."""
        return self.sklearn.logreg


class MetricsConfig(BaseModel):
    """Evaluation metrics configuration."""

    primary: List[str] = Field(default_factory=lambda: ["f1", "balanced_accuracy"])
    averaging: str = "macro"
    include_confidence_intervals: bool = False


class CalibrationConfig(BaseModel):
    """Meta classifier calibration settings."""

    enabled: Literal["false", "true", "auto"] = "auto"
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: int = 3
    bins: int = 10
    binning: Literal["uniform", "quantile"] = "quantile"
    isotonic_min_samples: int = 100
    policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "apply_to_modes": ["binary", "multiclass", "hierarchical", "meta"],
            "force_keep": False,
            "probability_quality_checks": {
                "enabled": True,
                "apply_to_modes": ["binary", "multiclass", "hierarchical", "meta"],
            },
            "thresholds": {
                "underconfidence_gap": -0.10,
                "high_accuracy": 0.90,
                "near_perfect_accuracy": 0.97,
                "min_calibration_n": 200,
                "min_class_n": 25,
                "min_brier_improvement": 0.002,
                "max_log_loss_regression": 0.01,
                "max_ece_ovr_regression": 0.01,
            },
        }
    )


class FinalModelConfig(BaseModel):
    """Final model training configuration."""

    sampler: Optional[str] = None
    sanity_min_std: float = 0.02
    sanity_max_mean_deviation: float = 0.15
    train_from_scratch: bool = True
    verify_dataset_hash: bool = True


class BundleConfig(BaseModel):
    """Bundle output configuration."""

    name: str = "model_bundle"
    include_preprocessing: bool = True
    format: str = "zip"


class ExecutionTorchSettings(BaseModel):
    """Torch execution options used when engine is torch or hybrid."""

    dtype: Literal["float32", "float16"] = Field(default="float32", description="Torch tensor dtype")
    num_workers: int = Field(
        default_factory=_default_torch_num_workers, description="DataLoader worker count"
    )
    require_device: bool = Field(
        default=False,
        description="Fail instead of falling back if requested torch device is unavailable",
    )


class ExecutionConfig(BaseModel):
    """Execution engine/runtime settings."""

    engine: ExecutionEngine = Field(
        default="sklearn",
        description="Execution engine for estimators/orchestration",
    )
    device: Optional[ExecutionDevice] = Field(
        default=None,
        description="Compute device; required for torch/hybrid engines",
    )
    model_set: Optional[str] = Field(
        default=None,
        description="Optional backend model set key (for torch/hybrid)",
    )
    torch: Optional[ExecutionTorchSettings] = Field(
        default=None,
        description="Torch runtime tuning (required for torch/hybrid engines)",
    )

    model_config = {"protected_namespaces": ()}


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "populate_by_name": True,
    }

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
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Experiment tracking (optional)
    tracker: Optional[Literal["mlflow", "wandb"]] = None
    experiment_name: Optional[str] = None

    # Internal migration notes to surface in CLI normalization flows.
    migration_notes: List[str] = Field(default_factory=list, alias="_migration_warnings", exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized, migration_warnings = normalize_project_payload(value)
        if migration_warnings:
            normalized["_migration_warnings"] = migration_warnings
        return normalized

    @model_validator(mode="after")
    def _validate_execution_and_mode(self) -> "ProjectConfig":
        engine = self.execution.engine
        mode = self.task.mode

        if engine == "sklearn":
            if self.execution.device is not None:
                raise ValueError(
                    "execution.device is not allowed when execution.engine=sklearn. "
                    "Suggestion: remove execution.device or use --engine torch/hybrid."
                )
            if self.execution.torch is not None:
                raise ValueError(
                    "execution.torch is not allowed when execution.engine=sklearn. "
                    "Suggestion: use --engine torch to enable torch settings."
                )
        else:
            if self.execution.device is None:
                raise ValueError(
                    "execution.device is required when execution.engine is torch or hybrid. "
                    "Allowed: auto|cpu|cuda|mps."
                )
            if self.execution.torch is None:
                raise ValueError(
                    "execution.torch is required when execution.engine is torch or hybrid."
                )

        if mode in {"binary", "meta", "hierarchical"} and engine == "hybrid":
            raise ValueError(
                f"execution.engine=hybrid is not supported for task.mode={mode}. "
                "Suggestion: use --engine sklearn or --engine torch."
            )

        if mode == "hierarchical" and engine == "torch":
            raise ValueError(
                "task.mode=hierarchical currently uses dedicated hierarchical training and does not support "
                "execution.engine=torch. Suggestion: use --engine sklearn."
            )

        if mode == "multiclass":
            device = self.execution.device or "auto"
            if self.multiclass.backend is None:
                if engine == "sklearn":
                    self.multiclass.backend = "sklearn_cpu"
                elif engine == "torch":
                    self.multiclass.backend = f"torch_{device}"
                else:
                    self.multiclass.backend = "hybrid_sklearn_meta_torch_base"

            backend = self.multiclass.backend
            assert backend is not None

            if engine == "sklearn":
                if backend != "sklearn_cpu":
                    raise ValueError(
                        "multiclass.backend must be 'sklearn_cpu' when execution.engine=sklearn. "
                        "Suggestion: set execution.engine=torch/hybrid for torch multiclass backends."
                    )
                if self.multiclass.torch is not None:
                    raise ValueError(
                        "multiclass.torch is not allowed when execution.engine=sklearn."
                    )
            elif engine == "torch":
                allowed = {"torch_auto", "torch_cpu", "torch_cuda", "torch_mps"}
                if backend not in allowed:
                    raise ValueError(
                        "multiclass.backend must be one of torch_auto|torch_cpu|torch_cuda|torch_mps "
                        "when execution.engine=torch."
                    )
                if self.multiclass.torch is None:
                    self.multiclass.torch = MulticlassTorchConfig()
            else:  # hybrid
                if backend != "hybrid_sklearn_meta_torch_base":
                    raise ValueError(
                        "multiclass.backend must be 'hybrid_sklearn_meta_torch_base' when execution.engine=hybrid."
                    )
                if self.multiclass.torch is None:
                    self.multiclass.torch = MulticlassTorchConfig()

        candidates = [c.lower() for c in self.models.candidates]
        if engine == "torch":
            unsupported = sorted(set(candidates) & SKLEARN_CANDIDATES)
            if unsupported and set(candidates) != SKLEARN_CANDIDATES:
                raise ValueError(
                    "models.candidates contains sklearn-only entries for execution.engine=torch: "
                    f"{', '.join(unsupported)}. Suggestion: use torch_* candidates or switch --engine sklearn."
                )
        if engine == "sklearn":
            unsupported = sorted(set(candidates) & TORCH_CANDIDATES)
            if unsupported:
                raise ValueError(
                    "models.candidates contains torch-only entries for execution.engine=sklearn: "
                    f"{', '.join(unsupported)}. Suggestion: use --engine torch/hybrid for torch candidates."
                )

        return self

    @property
    def backend(self) -> str:
        """Backward-compatible backend view for existing orchestration code."""
        if self.execution.engine == "torch":
            return "torch"
        return "sklearn"

    @property
    def device(self) -> str:
        """Backward-compatible device view for existing orchestration code."""
        return self.execution.device or "auto"

    @property
    def model_set(self) -> Optional[str]:
        """Backward-compatible model_set view for existing orchestration code."""
        if self.task.mode == "multiclass" and self.multiclass.torch is not None:
            return self.multiclass.torch.model_set
        return self.execution.model_set

    @property
    def torch_dtype(self) -> str:
        """Backward-compatible torch dtype view for existing orchestration code."""
        if self.execution.torch is None:
            return "float32"
        return self.execution.torch.dtype

    @property
    def torch_num_workers(self) -> int:
        """Backward-compatible torch worker view for existing orchestration code."""
        if self.execution.torch is None:
            return _default_torch_num_workers()
        return self.execution.torch.num_workers

    @property
    def require_torch_device(self) -> bool:
        """Backward-compatible torch device strictness view."""
        if self.execution.torch is None:
            return False
        return self.execution.torch.require_device

    @property
    def migration_warnings(self) -> List[str]:
        """Legacy-to-current normalization notes."""
        return list(self.__dict__.get("migration_notes", []))

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        """Load project configuration from YAML."""
        config, migration_warnings = cls.load_with_warnings(path)
        for warning_msg in migration_warnings:
            warnings.warn(warning_msg, stacklevel=2)
        return config

    @classmethod
    def load_with_warnings(cls, path: Path) -> Tuple["ProjectConfig", List[str]]:
        """Load config and return migration warnings, if any."""
        data = load_yaml(path)
        normalized, migration_warnings = normalize_project_payload(data)
        config = cls.model_validate(normalized)
        return config, migration_warnings

    def to_yaml_dict(self, *, minimal: bool = False) -> Dict[str, Any]:
        """Serialize config for YAML output."""
        dump = self.model_dump(mode="python", exclude_none=True, exclude={"migration_notes"})
        if not minimal:
            return dump

        result: Dict[str, Any] = {
            "project": dump["project"],
            "data": dump["data"],
            "key_columns": dump["key_columns"],
            "task": dump["task"],
            "validation": dump["validation"],
            "models": dump["models"],
            "imbalance": dump["imbalance"],
            "metrics": dump["metrics"],
            "final_model": dump["final_model"],
            "bundle": dump["bundle"],
            "execution": dump["execution"],
        }

        if self.task.mode == "meta":
            result["calibration"] = dump["calibration"]
        if self.task.mode == "multiclass":
            result["multiclass"] = dump["multiclass"]

        if self.tracker is not None:
            result["tracker"] = self.tracker
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name

        return result

    def save(self, path: Path, *, minimal: bool = False) -> None:
        """Write project configuration to YAML."""
        dump_yaml(self.to_yaml_dict(minimal=minimal), path)

    @classmethod
    def scaffold(
        cls,
        *,
        project_id: str,
        name: str,
        mode: TaskMode,
        engine: ExecutionEngine,
        train_manifest: str,
        test_manifest: Optional[str] = None,
        label_col: str = "label",
        patient_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        hierarchy_path: Optional[str] = None,
        device: Optional[ExecutionDevice] = None,
    ) -> "ProjectConfig":
        """Create a mode/engine-aware scaffold configuration."""
        execution: Dict[str, Any] = {"engine": engine}
        if engine in {"torch", "hybrid"}:
            execution["device"] = device or "auto"
            execution["torch"] = {
                "dtype": "float32",
                "num_workers": _default_torch_num_workers(),
                "require_device": False,
            }
            if engine == "torch" and mode in {"binary", "meta"}:
                execution["model_set"] = "torch_basic"

        multiclass: Dict[str, Any] = {"group_stratify": True}
        if mode == "multiclass":
            if engine == "sklearn":
                multiclass["backend"] = "sklearn_cpu"
            elif engine == "torch":
                resolved_device = device or "auto"
                multiclass["backend"] = f"torch_{resolved_device}"
                multiclass["torch"] = {"model_set": "torch_basic"}
            else:
                multiclass["backend"] = "hybrid_sklearn_meta_torch_base"
                multiclass["torch"] = {"model_set": "torch_basic"}

        payload: Dict[str, Any] = {
            "project": {
                "id": project_id,
                "name": name,
                "description": "",
                "owner": "",
            },
            "data": {
                "train": {"manifest": train_manifest},
                "test": {"manifest": test_manifest} if test_manifest else None,
            },
            "key_columns": {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "label": label_col,
                "slide_id": None,
                "specimen_id": None,
            },
            "task": {
                "mode": mode,
                "patient_stratified": True,
                "hierarchy_path": hierarchy_path,
            },
            "execution": execution,
            "multiclass": multiclass,
        }
        if engine == "torch":
            payload["models"] = {
                "candidates": ["torch_logistic_regression", "torch_mlp"],
                "selection_metric": "f1",
                "selection_direction": "max",
            }
        elif engine == "hybrid":
            payload["models"] = {
                "candidates": ["logistic_regression", "random_forest", "torch_mlp"],
                "selection_metric": "f1",
                "selection_direction": "max",
            }
        return cls.model_validate(payload)


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

    brier_max: Optional[float] = None
    ece_max: Optional[float] = None


_ALLOWED_GATE_METRICS = {
    "accuracy",
    "precision",
    "f1_score",
    "f1",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "sensitivity",
    "recall",
    "tpr",
    "specificity",
    "roc_auc",
    "roc_auc_ovr_macro",
    "roc_auc_macro",
    "balanced_accuracy",
    "balanced_acc",
}


class PromotionGateSpec(BaseModel):
    """Single promotion gate rule."""

    metric: str
    op: Literal[">=", ">", "<=", "<"] = ">="
    threshold: float
    scope: Literal["outer", "independent", "both"] = "both"
    aggregation: str = "mean"
    notes: Optional[str] = None

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized not in _ALLOWED_GATE_METRICS:
            allowed = ", ".join(sorted(_ALLOWED_GATE_METRICS))
            raise ValueError(f"Unsupported promotion gate metric '{value}'. Allowed: {allowed}")
        return value

    @field_validator("aggregation")
    @classmethod
    def _validate_aggregation(cls, value: str) -> str:
        aggregation = value.strip().lower()
        if aggregation in {"mean", "median", "min"}:
            return aggregation
        if re.fullmatch(r"p\d{1,3}", aggregation):
            percentile = int(aggregation[1:])
            if 0 <= percentile <= 100:
                return aggregation
        raise ValueError("aggregation must be one of: mean, median, min, or pXX (0-100)")


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
    promotion_gate_template: Optional[str] = None
    promotion_gates: List[PromotionGateSpec] = Field(default_factory=list)
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


def validate_project_payload(payload: Dict[str, Any]) -> Tuple[Optional[ProjectConfig], List[str], List[str]]:
    """Validate a project payload and return config, migration warnings, and errors."""
    normalized, migration_warnings = normalize_project_payload(payload)
    try:
        config = ProjectConfig.model_validate(normalized)
    except ValidationError as exc:
        errors: List[str] = []
        for item in exc.errors():
            path = ".".join(str(part) for part in item.get("loc", ()))
            msg = item.get("msg", "validation error")
            errors.append(f"{path}: {msg}")
        return None, migration_warnings, errors
    return config, migration_warnings, []
