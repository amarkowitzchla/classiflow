"""Artifact loading utilities for inference."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib
import pandas as pd
import numpy as np

from classiflow.inference.config import RunManifest

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """
    Load trained model artifacts for inference.

    Supports loading artifacts from:
    - Binary task pipelines (OvR/pairwise/composite)
    - Meta-classifiers
    - Hierarchical models (PyTorch-based)
    - Legacy formats with automatic migration
    """

    def __init__(self, run_dir: Path, fold: int = 1, verbose: int = 1):
        """
        Initialize artifact loader.

        Parameters
        ----------
        run_dir : Path
            Directory containing trained model artifacts
        fold : int
            Fold number to load (1-indexed)
        verbose : int
            Verbosity level
        """
        self.run_dir = Path(run_dir)
        self.fold = fold
        self.verbose = verbose

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # Resolve base run directory vs fold directory
        if self.run_dir.name.startswith("fold") and self.run_dir.parent.exists():
            self.base_dir = self.run_dir.parent
            self.fold_dir = self.run_dir
        else:
            self.base_dir = self.run_dir
            self.fold_dir = self.run_dir / f"fold{self.fold}"

        # Try to load run manifest (may not exist for legacy runs)
        self.manifest = self._load_manifest()

        # Determine run type
        self.run_type = self._detect_run_type()
        logger.info(f"Detected run type: {self.run_type}")

    def _load_manifest(self) -> Optional[RunManifest]:
        """Load run manifest if it exists."""
        # Try new format first, then legacy
        manifest_path = self.base_dir / "run.json"
        if not manifest_path.exists():
            manifest_path = self.base_dir / "run_manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)

                # TrainingRunManifest format (run.json)
                if "run_id" in data and "training_data_hash" in data:
                    return RunManifest(
                        training_config=data.get("config", {}),
                        timestamp=data.get("timestamp", ""),
                        package_version=data.get("package_version"),
                        git_hash=data.get("git_hash"),
                        label_col=(data.get("config", {}) or {}).get("label_col"),
                        feature_list=data.get("feature_list", []) or [],
                        preprocessing_steps=data.get("feature_summaries", {}).get("preprocessing", []),
                        task_type=data.get("task_type", "binary"),
                        task_definitions=data.get("task_definitions", {}),
                        best_models=data.get("best_models", {}),
                        fold_count=data.get("config", {}).get("outer_folds", 1),
                        hierarchical=bool(data.get("hierarchical")),
                        l1_classes=data.get("l1_classes"),
                        l2_classes_per_branch=data.get("l2_classes_per_branch"),
                    )

                # Inference RunManifest format (run_manifest.json)
                return RunManifest.load(manifest_path)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return None

    def _detect_run_type(self) -> str:
        """
        Detect the type of training run from available artifacts.

        Returns
        -------
        run_type : str
            One of: "binary", "meta", "hierarchical", "legacy"
        """
        fold_dir = self.fold_dir
        base_dir = self.base_dir

        if self.manifest and self.manifest.task_type:
            return self.manifest.task_type

        # Check for hierarchical models (PyTorch-based)
        config_path = base_dir / "training_config.json"
        if not config_path.exists():
            config_path = fold_dir / "training_config.json"

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if config.get("label_l2") is not None or config.get("hierarchical"):
                return "hierarchical"

        # Check for meta-classifier (check this BEFORE binary)
        if (fold_dir / "binary_none" / "meta_model.joblib").exists() or \
           (fold_dir / "binary_smote" / "meta_model.joblib").exists():
            return "meta"

        # Check for multiclass models
        if (fold_dir / "multiclass_none" / "multiclass_model.joblib").exists() or \
           (fold_dir / "multiclass_smote" / "multiclass_model.joblib").exists():
            return "multiclass"

        # Check for binary pipelines
        if (fold_dir / "binary_none" / "binary_pipes.joblib").exists() or \
           (fold_dir / "binary_smote" / "binary_pipes.joblib").exists():
            return "binary"

        # Legacy format
        return "legacy"

    def load_binary_artifacts(
        self, variant: str = "smote"
    ) -> Tuple[Dict[str, Any], Dict[str, str], Optional[List[str]]]:
        """
        Load binary task artifacts.

        Parameters
        ----------
        variant : str
            SMOTE variant: "smote" or "none"

        Returns
        -------
        pipes : Dict[str, Pipeline]
            Dictionary of trained pipelines {task__model: pipeline}
        best_models : Dict[str, str]
            Best model per task {task: model_name}
        feature_list : Optional[List[str]]
            List of features (if available)
        """
        var_dir = self.fold_dir / f"binary_{variant}"

        if not var_dir.exists():
            raise FileNotFoundError(f"Variant directory not found: {var_dir}")

        # Load binary pipes
        pipes_path = var_dir / "binary_pipes.joblib"
        if not pipes_path.exists():
            raise FileNotFoundError(f"Binary pipes not found: {pipes_path}")

        bundle = joblib.load(pipes_path)
        if isinstance(bundle, dict):
            pipes = bundle.get("pipes", {})
            best_models = bundle.get("best_models", {})
        else:
            # Legacy format
            pipes = bundle
            best_models = self._infer_best_models_from_pipes(pipes)

        logger.info(f"Loaded {len(pipes)} binary pipelines")
        logger.info(f"Best models for {len(best_models)} tasks")

        # Try to load feature list
        feature_list = self._load_feature_list()

        return pipes, best_models, feature_list

    def load_meta_artifacts(
        self, variant: str = "smote"
    ) -> Tuple[Any, List[str], List[str]]:
        """
        Load meta-classifier artifacts.

        Parameters
        ----------
        variant : str
            SMOTE variant: "smote" or "none"

        Returns
        -------
        meta_model : Any
            Trained meta-classifier
        meta_features : List[str]
            Ordered list of meta-feature names (e.g., ['TaskA_score', 'TaskB_score'])
        meta_classes : List[str]
            Ordered list of class names
        calibration_metadata : Dict[str, Any]
            Metadata about the probability calibration step
        """
        var_dir = self.fold_dir / f"binary_{variant}"

        # Load meta model
        meta_path = var_dir / "meta_model.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta model not found: {meta_path}")

        meta_model = joblib.load(meta_path)

        # Load meta features
        meta_features_path = var_dir / "meta_features.csv"
        if meta_features_path.exists():
            meta_features = pd.read_csv(meta_features_path, header=None).iloc[:, 0].astype(str).tolist()
        else:
            # Fallback: infer from model if possible
            meta_features = []
            logger.warning("meta_features.csv not found; meta-feature order may be incorrect")

        # Load meta classes
        meta_classes_path = var_dir / "meta_classes.csv"
        if meta_classes_path.exists():
            meta_classes = pd.read_csv(meta_classes_path, header=None).iloc[:, 0].astype(str).tolist()
        else:
            # Fallback to model classes if available
            if hasattr(meta_model, "classes_"):
                meta_classes = [str(c) for c in meta_model.classes_]
            else:
                meta_classes = []
                logger.warning("meta_classes.csv not found and model has no classes_ attribute")

        logger.info(f"Loaded meta-classifier with {len(meta_features)} features and {len(meta_classes)} classes")
        calibration_path = var_dir / "calibration_metadata.json"
        calibration_metadata = {}
        if calibration_path.exists():
            try:
                with open(calibration_path, "r") as handle:
                    calibration_metadata = json.load(handle)
            except Exception as exc:
                logger.warning(f"Failed to load calibration metadata: {exc}")

        return meta_model, meta_features, meta_classes, calibration_metadata

    def load_multiclass_artifacts(
        self, variant: str = "smote"
    ) -> Tuple[Any, List[str], Optional[List[str]]]:
        """
        Load multiclass model artifacts.

        Parameters
        ----------
        variant : str
            SMOTE variant: "smote" or "none"

        Returns
        -------
        model : Any
            Trained multiclass estimator
        classes : List[str]
            Ordered list of class names
        feature_list : Optional[List[str]]
            Feature list if available
        """
        var_dir = self.fold_dir / f"multiclass_{variant}"
        model_path = var_dir / "multiclass_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Multiclass model not found: {model_path}")

        model = joblib.load(model_path)

        classes_path = var_dir / "classes.csv"
        if classes_path.exists():
            classes = pd.read_csv(classes_path, header=None).iloc[:, 0].astype(str).tolist()
        else:
            classes = [str(c) for c in getattr(model, "classes_", [])]

        feature_list = None
        feature_path = var_dir / "feature_list.csv"
        if feature_path.exists():
            feature_list = pd.read_csv(feature_path, header=None).iloc[:, 0].astype(str).tolist()

        return model, classes, feature_list

    def load_hierarchical_artifacts(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load hierarchical model artifacts (PyTorch-based).

        Returns
        -------
        models : Dict[str, Any]
            Dictionary containing L1 and L2 models
        config : Dict[str, Any]
            Training configuration
        """
        from classiflow.inference.hierarchical import HierarchicalInference

        # Load using the existing hierarchical inference class
        hier_inf = HierarchicalInference(
            model_dir=self.base_dir,
            fold=self.fold,
            device="cpu",  # Will be overridden by actual inference
        )

        models = {
            "l1_model": hier_inf.model_l1,
            "l2_models": hier_inf.branch_models,
            "scaler": hier_inf.scaler,
            "l1_encoder": hier_inf.le_l1,
            "l2_encoders": hier_inf.branch_encoders,
        }

        config = {
            "hierarchical": hier_inf.hierarchical,
            "l1_classes": hier_inf.l1_classes,
            "l2_classes_per_branch": hier_inf.l2_classes_per_branch,
            "feature_cols": hier_inf.config.get("feature_cols", []),
        }

        return models, config

    def _load_feature_list(self) -> Optional[List[str]]:
        """Attempt to load feature list from various sources."""
        # Try manifest first
        if self.manifest and self.manifest.feature_list:
            return self.manifest.feature_list

        # Try training config
        config_path = self.base_dir / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if "feature_cols" in config and config["feature_cols"]:
                return config["feature_cols"]

        # Try inference config (hierarchical runs)
        inference_config_path = self.base_dir / "inference_config.json"
        if inference_config_path.exists():
            with open(inference_config_path) as f:
                config = json.load(f)
            if "feature_cols" in config and config["feature_cols"]:
                return config["feature_cols"]

        # Try to infer from a pipeline
        try:
            fold_dir = self.fold_dir
            for variant in ["none", "smote"]:
                feature_path = fold_dir / f"multiclass_{variant}" / "feature_list.csv"
                if feature_path.exists():
                    return pd.read_csv(feature_path, header=None).iloc[:, 0].astype(str).tolist()
            for variant in ["none", "smote"]:
                pipes_path = fold_dir / f"binary_{variant}" / "binary_pipes.joblib"
                if pipes_path.exists():
                    bundle = joblib.load(pipes_path)
                    if isinstance(bundle, dict) and "pipes" in bundle:
                        # Get first pipeline and extract feature names
                        first_pipe = next(iter(bundle["pipes"].values()))
                        if hasattr(first_pipe, "feature_names_in_"):
                            return first_pipe.feature_names_in_.tolist()
                        # Try to get from StandardScaler step
                        if hasattr(first_pipe, "named_steps") and "scaler" in first_pipe.named_steps:
                            scaler = first_pipe.named_steps["scaler"]
                            if hasattr(scaler, "feature_names_in_"):
                                return scaler.feature_names_in_.tolist()
        except Exception as e:
            logger.warning(f"Failed to infer feature list from pipeline: {e}")

        logger.warning("Feature list not found; will use all numeric columns from input data")
        return None

    def _infer_best_models_from_pipes(self, pipes: Dict[str, Any]) -> Dict[str, str]:
        """Infer best models from pipeline keys (legacy support)."""
        best_models = {}
        for key in pipes.keys():
            if "__" in key:
                task, model = key.split("__", 1)
                # Simple heuristic: last model per task wins
                best_models[task] = model
        return best_models

    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Get feature schema for validation.

        Returns
        -------
        schema : Dict[str, Any]
            Dictionary with:
            - feature_list: List[str]
            - feature_types: Dict[str, str] (if available)
            - preprocessing: List[str]
        """
        feature_list = self._load_feature_list()

        schema = {
            "feature_list": feature_list,
            "feature_types": {},  # Could be extended to validate dtypes
            "preprocessing": [],
        }

        if self.manifest:
            schema["preprocessing"] = self.manifest.preprocessing_steps

        return schema

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task information from manifest or artifacts.

        Returns
        -------
        task_info : Dict[str, Any]
            Dictionary with task definitions and best models
        """
        if self.manifest:
            return {
                "task_type": self.manifest.task_type,
                "task_definitions": self.manifest.task_definitions,
                "best_models": self.manifest.best_models,
            }

        # Fallback: load from artifacts
        try:
            _, best_models, _ = self.load_binary_artifacts(variant="none")
            return {
                "task_type": self.run_type,
                "task_definitions": {},
                "best_models": best_models,
            }
        except Exception:
            return {
                "task_type": "unknown",
                "task_definitions": {},
                "best_models": {},
            }
