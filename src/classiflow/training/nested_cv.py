"""Nested cross-validation orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.exceptions import FitFailedWarning

from classiflow.models import get_estimators, get_param_grids, AdaptiveSMOTE
from classiflow.metrics.scorers import get_scorers, SCORER_ORDER
from classiflow.metrics.binary import compute_binary_metrics
from classiflow.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FitFailedWarning)


class NestedCVOrchestrator:
    """
    Orchestrate nested cross-validation for binary classification.

    Outer CV: Validation folds for unbiased performance estimates
    Inner CV: Hyperparameter tuning via GridSearchCV

    Parameters
    ----------
    outer_folds : int
        Number of outer CV folds
    inner_splits : int
        Number of inner CV splits (per repeat)
    inner_repeats : int
        Number of inner CV repeats
    random_state : int
        Random seed for reproducibility
    smote_mode : Literal["off", "on", "both"]
        SMOTE control
    max_iter : int
        Max iterations for linear models
    """

    def __init__(
        self,
        outer_folds: int = 3,
        inner_splits: int = 5,
        inner_repeats: int = 2,
        random_state: int = 42,
        smote_mode: Literal["off", "on", "both"] = "off",
        max_iter: int = 10000,
    ):
        self.outer_folds = outer_folds
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.random_state = random_state
        self.smote_mode = smote_mode
        self.max_iter = max_iter

        self.estimators = get_estimators(random_state, max_iter)
        self.param_grids = get_param_grids()
        self.scorers = get_scorers()

    def run_single_task(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_name: str = "binary_task",
        outdir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run nested CV for a single binary task.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Binary labels (0/1)
        task_name : str
            Task identifier
        outdir : Optional[Path]
            Output directory for artifacts

        Returns
        -------
        results : Dict[str, Any]
            Nested CV results
        """
        logger.info(f"Running nested CV for task: {task_name}")

        # Determine SMOTE variants
        variants = self._get_smote_variants()

        # Outer CV
        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        results = {
            "task_name": task_name,
            "folds": [],
            "inner_cv_rows": [],
            "inner_cv_split_rows": [],
            "outer_rows": [],
        }

        # Collectors for averaged plots across folds
        all_roc_data = {"fpr": [], "tpr": [], "auc": []}
        all_pr_data = {"recall": [], "precision": [], "ap": []}

        for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y), 1):
            logger.info(f"  Fold {fold_idx}/{self.outer_folds}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            fold_results = self._run_fold(
                X_tr, y_tr, X_va, y_va, fold_idx, task_name, variants, outdir
            )

            results["folds"].append(fold_results)
            results["inner_cv_rows"].extend(fold_results["inner_cv_rows"])
            results["inner_cv_split_rows"].extend(fold_results["inner_cv_split_rows"])
            results["outer_rows"].extend(fold_results["outer_rows"])

            # Collect ROC/PR data for averaged plots
            if fold_results.get("roc_data"):
                all_roc_data["fpr"].append(fold_results["roc_data"]["fpr"])
                all_roc_data["tpr"].append(fold_results["roc_data"]["tpr"])
                all_roc_data["auc"].append(fold_results["roc_data"]["auc"])

            if fold_results.get("pr_data"):
                all_pr_data["recall"].append(fold_results["pr_data"]["recall"])
                all_pr_data["precision"].append(fold_results["pr_data"]["precision"])
                all_pr_data["ap"].append(fold_results["pr_data"]["ap"])

        # Generate averaged plots if we have output directory
        if outdir is not None and len(all_roc_data["fpr"]) > 0:
            logger.info("Generating averaged ROC and PR curves across folds")

            plot_averaged_roc_curves(
                all_roc_data["fpr"],
                all_roc_data["tpr"],
                all_roc_data["auc"],
                f"{task_name} – ROC Curve (Averaged Across Folds)",
                outdir / f"roc_{task_name}_averaged.png",
                show_individual=(self.outer_folds <= 5),
            )

            plot_averaged_pr_curves(
                all_pr_data["recall"],
                all_pr_data["precision"],
                all_pr_data["ap"],
                f"{task_name} – PR Curve (Averaged Across Folds)",
                outdir / f"pr_{task_name}_averaged.png",
                show_individual=(self.outer_folds <= 5),
            )

        logger.info(f"Nested CV complete for {task_name}")
        return results

    def _get_smote_variants(self):
        """Determine which SMOTE variants to run."""
        if self.smote_mode in ("on", "both"):
            return ["smote", "none"]
        return ["none"]

    def _run_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        fold_idx: int,
        task_name: str,
        variants: list,
        outdir: Optional[Path],
    ) -> Dict[str, Any]:
        """Run a single outer fold."""
        fold_results = {
            "fold_idx": fold_idx,
            "variants": {},
            "inner_cv_rows": [],
            "inner_cv_split_rows": [],
            "outer_rows": [],
        }

        # Track best model across variants for plotting
        best_estimator = None
        best_variant = None

        for variant in variants:
            logger.info(f"    Variant: {variant}")
            var_results = self._run_variant(
                X_tr, y_tr, X_va, y_va, fold_idx, task_name, variant, outdir
            )
            fold_results["variants"][variant] = var_results
            fold_results["inner_cv_rows"].extend(var_results["inner_cv_rows"])
            fold_results["inner_cv_split_rows"].extend(var_results["inner_cv_split_rows"])
            fold_results["outer_rows"].extend(var_results["outer_rows"])

            # Track best model for plotting (use first variant's best if multiple)
            if best_estimator is None and var_results["best_estimator"] is not None:
                best_estimator = var_results["best_estimator"]
                best_variant = variant

        # Generate per-fold plots if we have output directory and best model
        if outdir is not None and best_estimator is not None:
            fold_dir = outdir / f"fold{fold_idx}"
            fold_dir.mkdir(exist_ok=True)

            # Get predictions and probabilities for validation set
            y_va_pred = best_estimator.predict(X_va)
            y_va_scores = self._get_scores(best_estimator, X_va)

            # Reshape scores for plotting (binary classification needs shape (n, 2))
            y_va_proba = np.column_stack([1 - y_va_scores, y_va_scores])

            classes = ["0", "1"]

            # ROC curve
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

            fpr, tpr, _ = roc_curve(y_va.values, y_va_scores)
            roc_auc = auc(fpr, tpr)

            fold_results["roc_data"] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc,
            }

            plot_roc_curve(
                y_va.values,
                y_va_proba,
                classes,
                f"{task_name} – ROC Curve (Fold {fold_idx})",
                fold_dir / f"roc_{task_name}_fold{fold_idx}.png",
            )

            # PR curve
            precision, recall, _ = precision_recall_curve(y_va.values, y_va_scores)
            ap = average_precision_score(y_va.values, y_va_scores)

            fold_results["pr_data"] = {
                "recall": recall,
                "precision": precision,
                "ap": ap,
            }

            plot_pr_curve(
                y_va.values,
                y_va_proba,
                classes,
                f"{task_name} – PR Curve (Fold {fold_idx})",
                fold_dir / f"pr_{task_name}_fold{fold_idx}.png",
            )

            # Confusion matrix
            plot_confusion_matrix(
                y_va.values,
                y_va_pred,
                classes,
                f"{task_name} – Confusion Matrix (Fold {fold_idx})",
                fold_dir / f"cm_{task_name}_fold{fold_idx}.png",
                normalize="true",
            )

            logger.debug(f"Generated plots for fold {fold_idx} in {fold_dir}")

        return fold_results

    def _run_variant(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        fold_idx: int,
        task_name: str,
        variant: str,
        outdir: Optional[Path],
    ) -> Dict[str, Any]:
        """Run inner CV and evaluation for a single SMOTE variant."""
        # Adaptive inner CV splits based on minority class size
        min_class = int(y_tr.value_counts().min())
        n_splits_eff = max(2, min(self.inner_splits, min_class))
        if n_splits_eff < self.inner_splits:
            logger.debug(f"Reducing inner_splits {self.inner_splits} → {n_splits_eff} (minority={min_class})")

        cv_inner = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=self.inner_repeats,
            random_state=self.random_state,
        )
        n_inner_total = n_splits_eff * self.inner_repeats

        # Create sampler
        sampler = self._make_sampler(variant)

        var_results = {
            "variant": variant,
            "best_model": None,
            "best_estimator": None,
            "inner_cv_rows": [],
            "inner_cv_split_rows": [],
            "outer_rows": [],
        }

        best_score = -np.inf
        best_model_name = None
        best_estimator = None

        # Train each model type
        for model_name, est in self.estimators.items():
            # Build pipeline without VarianceThreshold to avoid removing all features
            # in small CV splits (especially problematic with scaled data)
            pipe = ImbPipeline([
                ("sampler", sampler),
                ("scaler", StandardScaler()),
                ("clf", est),
            ])

            grid = GridSearchCV(
                pipe,
                self.param_grids[model_name],
                cv=cv_inner,
                scoring=self.scorers,
                refit="F1 Score",
                n_jobs=-1,
                verbose=0,
                return_train_score=False,
                error_score=np.nan,
            )

            try:
                grid.fit(X_tr, y_tr)
            except Exception as e:
                logger.warning(f"Model {model_name} fit failed: {e}")
                continue

            # Log inner CV summary
            cvres = grid.cv_results_
            for i in range(len(cvres["params"])):
                row = {
                    "fold": fold_idx,
                    "sampler": variant,
                    "task": task_name,
                    "model_name": model_name,
                    "rank_test_f1": int(cvres.get("rank_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                    "mean_test_f1": float(cvres.get("mean_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                    "std_test_f1": float(cvres.get("std_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                }
                # Flatten clf__ params
                for k, v in cvres["params"][i].items():
                    row[k.replace("clf__", "")] = v
                var_results["inner_cv_rows"].append(row)

            # Per-split metrics for best params
            best_idx = grid.best_index_
            tm_label = f"{task_name}__{model_name} [{'SMOTE' if variant=='smote' else 'No-SMOTE'}]"
            for s in range(n_inner_total):
                rec = {"task_model": tm_label, "fold": int(s + 1)}
                ok = True
                for name in SCORER_ORDER:
                    key = f"split{s}_test_{name}"
                    if key not in cvres:
                        ok = False
                        break
                    val = cvres[key][best_idx]
                    rec[name] = float(val) if val == val else np.nan
                if ok:
                    var_results["inner_cv_split_rows"].append(rec)

            # Track best model
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model_name = model_name
                best_estimator = grid.best_estimator_

            # Evaluate on train and val
            train_metrics = compute_binary_metrics(y_tr.values, self._get_scores(grid.best_estimator_, X_tr))
            val_metrics = compute_binary_metrics(y_va.values, self._get_scores(grid.best_estimator_, X_va))

            var_results["outer_rows"].append({
                "fold": fold_idx,
                "sampler": variant,
                "phase": "train",
                "task": task_name,
                "model_name": model_name,
                **train_metrics,
            })
            var_results["outer_rows"].append({
                "fold": fold_idx,
                "sampler": variant,
                "phase": "val",
                "task": task_name,
                "model_name": model_name,
                **val_metrics,
            })

        var_results["best_model"] = best_model_name
        var_results["best_estimator"] = best_estimator
        logger.info(f"      Best model: {best_model_name} (F1={best_score:.3f})")

        return var_results

    def _make_sampler(self, variant: str):
        """Create SMOTE sampler or passthrough."""
        if variant == "smote":
            return AdaptiveSMOTE(k_max=5, random_state=self.random_state)
        return "passthrough"

    @staticmethod
    def _get_scores(pipe, X):
        """Extract scores or probabilities from pipeline."""
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        return pipe.decision_function(X)
