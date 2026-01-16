"""Hierarchical nested cross-validation with patient-level stratification."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from classiflow.config import HierarchicalConfig
from classiflow.models.torch_mlp import TorchMLPWrapper
from classiflow.models.smote import apply_smote
from classiflow.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    extract_feature_importance_mlp,
)

logger = logging.getLogger(__name__)


def get_hyperparam_candidates(base_hidden: int, base_epochs: int) -> List[Dict]:
    """
    Generate hyperparameter candidates for grid search.

    Parameters
    ----------
    base_hidden : int
        Base hidden layer size
    base_epochs : int
        Base number of epochs

    Returns
    -------
    List[Dict]
        List of hyperparameter configurations
    """
    return [
        {"hidden_dims": [base_hidden], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
        {"hidden_dims": [base_hidden * 2], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
        {"hidden_dims": [base_hidden, base_hidden // 2], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
        {"hidden_dims": [base_hidden], "lr": 5e-4, "epochs": base_epochs, "dropout": 0.2},
        {"hidden_dims": [base_hidden * 2], "lr": 5e-4, "epochs": base_epochs, "dropout": 0.4},
        {"hidden_dims": [base_hidden, base_hidden], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
    ]


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    inner_cv: StratifiedKFold,
    candidate_params: List[Dict],
    config: HierarchicalConfig,
    level_name: str = "L1",
) -> Tuple[Dict, List[Dict]]:
    """
    Tune hyperparameters using inner CV.

    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels (encoded)
    n_classes : int
        Number of classes
    inner_cv : StratifiedKFold
        Inner CV splitter
    candidate_params : List[Dict]
        Hyperparameter configurations to test
    config : HierarchicalConfig
        Training configuration
    level_name : str
        Name for logging (e.g., "L1", "L2_ClassA")

    Returns
    -------
    Tuple[Dict, List[Dict]]
        Best config and all results
    """
    n_features = X.shape[1]
    all_results = []
    best_f1 = -np.inf
    best_cfg = None

    for cfg in candidate_params:
        f1_scores = []

        if config.verbose >= 2:
            logger.debug(f"  Testing config: {cfg}")

        for fold_idx, (inner_tr_idx, inner_va_idx) in enumerate(inner_cv.split(X, y)):
            X_in_tr, X_in_va = X[inner_tr_idx], X[inner_va_idx]
            y_in_tr, y_in_va = y[inner_tr_idx], y[inner_va_idx]

            # Apply SMOTE if enabled
            if config.use_smote:
                X_in_tr, y_in_tr = apply_smote(
                    X_in_tr, y_in_tr, config.smote_k_neighbors, config.random_state + fold_idx
                )

            model = TorchMLPWrapper(
                input_dim=n_features,
                num_classes=n_classes,
                hidden_dims=cfg["hidden_dims"],
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                dropout=cfg["dropout"],
                batch_size=config.mlp_batch_size,
                early_stopping_patience=config.early_stopping_patience,
                device=config.device,
                random_state=config.random_state + fold_idx,
                verbose=0,
            )

            model.fit(X_in_tr, y_in_tr, X_in_va, y_in_va)
            y_pred = model.predict(X_in_va)

            f1_scores.append(f1_score(y_in_va, y_pred, average="macro", zero_division=0))

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))

        result = {
            "level": level_name,
            "hidden_dims": str(cfg["hidden_dims"]),
            "lr": cfg["lr"],
            "epochs": cfg["epochs"],
            "dropout": cfg["dropout"],
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
        }
        all_results.append(result)

        if config.verbose >= 2:
            logger.debug(f"    -> mean F1={mean_f1:.3f}±{std_f1:.3f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_cfg = cfg

    if config.verbose >= 1:
        logger.info(f"  Best config for {level_name}: {best_cfg}, F1={best_f1:.3f}")

    return best_cfg, all_results


def train_hierarchical(config: HierarchicalConfig) -> Dict:
    """
    Train hierarchical classifier with patient-level stratification.

    Workflow:
    1. Load data and validate
    2. Create patient-level stratified splits (no data leakage)
    3. For each outer fold:
       a. Train Level-1 classifier (primary task)
       b. If hierarchical: train Level-2 classifiers per L1 branch
       c. Evaluate on held-out fold
    4. Aggregate metrics and save artifacts

    Parameters
    ----------
    config : HierarchicalConfig
        Training configuration

    Returns
    -------
    Dict
        Training results summary

    Examples
    --------
    >>> config = HierarchicalConfig(
    ...     data_csv="data.csv",
    ...     patient_col="patient_id",
    ...     label_l1="diagnosis",
    ...     label_l2="subtype",
    ...     device="auto"
    ... )
    >>> results = train_hierarchical(config)
    """
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting hierarchical training: {config.hierarchical=}")
    logger.info(f"Device: {config.device}")

    # ========== Load and validate data ==========
    from classiflow.data import load_table

    data_path = config.resolved_data_path
    logger.info(f"Loading data from {data_path}...")
    df_all = load_table(data_path)

    # Determine if using patient stratification
    use_patient_stratification = config.patient_col is not None
    logger.info(f"Stratification mode: {'patient-level' if use_patient_stratification else 'sample-level'}")

    required_cols = [config.label_l1]
    if use_patient_stratification:
        required_cols.append(config.patient_col)

    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Column '{col}' not found in CSV")

    # Drop rows missing required columns
    df_all = df_all.dropna(subset=required_cols)
    df_all[config.label_l1] = df_all[config.label_l1].astype(str)

    # Level-1 classes
    l1_classes = sorted(df_all[config.label_l1].unique().tolist())
    if len(l1_classes) < 2:
        raise ValueError(f"Need at least 2 L1 classes, found {l1_classes}")
    logger.info(f"Level-1 classes ({len(l1_classes)}): {l1_classes}")

    # ========== Level-2 processing (if hierarchical) ==========
    branch_l2_global = {}
    if config.hierarchical:
        df_l2 = df_all[df_all[config.label_l2].notna()].copy()

        # Filter L2 classes if specified
        if config.l2_classes:
            keep_mask = df_l2[config.label_l2].isin(config.l2_classes)
            dropped = (~keep_mask).sum()
            if dropped > 0:
                logger.info(f"Dropping {dropped} rows with L2 not in {config.l2_classes}")
            df_l2 = df_l2.loc[keep_mask].copy()

        l2_all_classes = sorted(df_l2[config.label_l2].unique().tolist())
        logger.info(f"Global Level-2 classes: {l2_all_classes}")

        # Branch-specific L2 classes
        branch_l2_global = {
            l1: sorted(df_l2.loc[df_l2[config.label_l1] == l1, config.label_l2].unique().tolist())
            for l1 in l1_classes
        }
        if config.verbose >= 2:
            logger.debug("Branch-specific L2 classes:")
            for l1 in l1_classes:
                logger.debug(f"  L1={l1}: {branch_l2_global[l1]}")

    # ========== Feature columns ==========
    exclude_cols = [config.label_l1]
    if use_patient_stratification:
        exclude_cols.append(config.patient_col)
    if config.hierarchical:
        exclude_cols.append(config.label_l2)

    if config.feature_cols:
        feature_cols = config.feature_cols
    else:
        feature_cols = (
            df_all.drop(columns=[c for c in exclude_cols if c in df_all.columns])
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found")

    logger.info(f"Data: {len(df_all)} samples, {len(feature_cols)} features")

    X_all = df_all[feature_cols].values
    y_l1_all = df_all[config.label_l1].values
    y_l2_all = df_all[config.label_l2].values if config.hierarchical else None

    # Create label encoder
    le_l1 = LabelEncoder()
    le_l1.fit(l1_classes)

    # ========== Stratification setup ==========
    if use_patient_stratification:
        # Patient-level stratification: Group by patient and assign majority L1 label per patient
        patient_counts = df_all.groupby([config.patient_col, config.label_l1]).size().reset_index(name="n")
        patient_df = (
            patient_counts.sort_values("n", ascending=False)
            .drop_duplicates(subset=[config.patient_col])
            .loc[:, [config.patient_col, config.label_l1]]
            .reset_index(drop=True)
        )
        stratify_ids = patient_df[config.patient_col].values
        stratify_labels = patient_df[config.label_l1].values
        logger.info(f"Unique patients: {len(stratify_ids)}")
    else:
        # Sample-level stratification: Use sample indices directly
        stratify_ids = np.arange(len(df_all))
        stratify_labels = y_l1_all
        logger.info(f"Using sample-level stratification ({len(stratify_ids)} samples)")

    # ========== Save config ==========
    config.save(outdir / "training_config.json")

    # ========== Outer CV Setup ==========
    outer_cv = StratifiedShuffleSplit(
        n_splits=config.outer_folds,
        test_size=0.2,
        random_state=config.random_state,
    )
    candidate_params = get_hyperparam_candidates(config.mlp_hidden, config.mlp_epochs)

    inner_cv_rows = []
    outer_rows = []
    all_pipeline_metrics = [] if config.hierarchical else None

    # Storage for ROC/PR curves across folds
    all_l1_roc_data = {"fpr": [], "tpr": [], "auc": []}
    all_l1_pr_data = {"rec": [], "prec": [], "ap": []}

    all_l2_roc_data = {l1: {"fpr": [], "tpr": [], "auc": []} for l1 in l1_classes} if config.hierarchical else {}
    all_l2_pr_data = {l1: {"rec": [], "prec": [], "ap": []} for l1 in l1_classes} if config.hierarchical else {}

    # ========== Outer CV Loop ==========
    fold_id = 0
    for strat_tr_idx, strat_va_idx in tqdm(
        outer_cv.split(stratify_ids, stratify_labels),
        total=config.outer_folds,
        desc="Outer folds",
        disable=(config.verbose < 1),
    ):
        fold_id += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"OUTER FOLD {fold_id}/{config.outer_folds}")
        logger.info(f"{'='*60}")

        fold_dir = outdir / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if use_patient_stratification:
            # Patient-level: map patient indices to sample masks
            patients_train = stratify_ids[strat_tr_idx]
            patients_val = stratify_ids[strat_va_idx]

            # Save patient split
            patient_split_df = pd.DataFrame({
                config.patient_col: np.concatenate([patients_train, patients_val]),
                "phase": ["train"] * len(patients_train) + ["val"] * len(patients_val),
            })
            patient_split_df.to_csv(fold_dir / f"patient_split_fold{fold_id}.csv", index=False)

            # Sample-level splits based on patient membership
            train_mask = df_all[config.patient_col].isin(patients_train).values
            val_mask = df_all[config.patient_col].isin(patients_val).values
        else:
            # Sample-level: use indices directly
            sample_indices_train = stratify_ids[strat_tr_idx]
            sample_indices_val = stratify_ids[strat_va_idx]

            # Save sample split
            sample_split_df = pd.DataFrame({
                "sample_idx": np.concatenate([sample_indices_train, sample_indices_val]),
                "phase": ["train"] * len(sample_indices_train) + ["val"] * len(sample_indices_val),
            })
            sample_split_df.to_csv(fold_dir / f"sample_split_fold{fold_id}.csv", index=False)

            # Create boolean masks
            train_mask = np.zeros(len(df_all), dtype=bool)
            val_mask = np.zeros(len(df_all), dtype=bool)
            train_mask[sample_indices_train] = True
            val_mask[sample_indices_val] = True

        X_tr = X_all[train_mask]
        y_l1_tr = y_l1_all[train_mask]
        X_va = X_all[val_mask]
        y_l1_va = y_l1_all[val_mask]

        if config.hierarchical:
            y_l2_tr = y_l2_all[train_mask]
            y_l2_va = y_l2_all[val_mask]

        logger.info(f"Train samples: {len(X_tr)}, Val samples: {len(X_va)}")

        # Standardize
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        joblib.dump(scaler, fold_dir / "scaler.joblib")

        n_features = X_tr_scaled.shape[1]

        # Encode L1 labels
        y_l1_tr_enc = le_l1.transform(y_l1_tr)
        y_l1_va_enc = le_l1.transform(y_l1_va)
        joblib.dump(le_l1, fold_dir / "label_encoder_l1.joblib")

        # ========== Level-1 Training ==========
        logger.info("\n[Level-1] Hyperparameter tuning...")

        inner_cv_l1 = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.random_state + fold_id,
        )

        best_cfg_l1, inner_results_l1 = tune_hyperparameters(
            X_tr_scaled, y_l1_tr_enc, len(l1_classes), inner_cv_l1,
            candidate_params, config, "L1"
        )

        for res in inner_results_l1:
            res["fold"] = fold_id
            inner_cv_rows.append(res)

        # Apply SMOTE for final training if enabled
        X_tr_l1_final = X_tr_scaled
        y_tr_l1_final = y_l1_tr_enc
        if config.use_smote:
            X_tr_l1_final, y_tr_l1_final = apply_smote(
                X_tr_scaled, y_l1_tr_enc, config.smote_k_neighbors, config.random_state
            )
            if config.verbose >= 2:
                logger.debug(f"SMOTE: {len(X_tr_scaled)} → {len(X_tr_l1_final)} samples")

        # Train final L1 model
        logger.info("[Level-1] Training final model...")
        model_l1 = TorchMLPWrapper(
            input_dim=n_features,
            num_classes=len(l1_classes),
            hidden_dims=best_cfg_l1["hidden_dims"],
            lr=best_cfg_l1["lr"],
            epochs=best_cfg_l1["epochs"],
            dropout=best_cfg_l1["dropout"],
            batch_size=config.mlp_batch_size,
            early_stopping_patience=config.early_stopping_patience,
            device=config.device,
            random_state=config.random_state + fold_id,
            verbose=config.verbose,
        )
        model_l1.fit(X_tr_l1_final, y_tr_l1_final, X_va_scaled, y_l1_va_enc)

        if config.verbose >= 2:
            logger.debug(f"Best epoch: {model_l1.best_epoch}")

        # Save L1 model
        model_l1.save(fold_dir / f"model_level1_fold{fold_id}.pt")
        with open(fold_dir / f"model_config_l1_fold{fold_id}.json", "w") as f:
            json.dump(model_l1.get_config(), f, indent=2)

        # Evaluate L1
        y_l1_pred_enc = model_l1.predict(X_va_scaled)
        y_l1_proba = model_l1.predict_proba(X_va_scaled)

        l1_acc = accuracy_score(y_l1_va_enc, y_l1_pred_enc)
        l1_bal_acc = balanced_accuracy_score(y_l1_va_enc, y_l1_pred_enc)
        l1_f1 = f1_score(y_l1_va_enc, y_l1_pred_enc, average="macro", zero_division=0)

        l1_row = {
            "fold": fold_id,
            "level": "L1",
            "accuracy": l1_acc,
            "balanced_accuracy": l1_bal_acc,
            "f1_macro": l1_f1,
            "n_classes": len(l1_classes),
        }
        outer_rows.append(l1_row)

        logger.info(
            f"[Level-1] Val: acc={l1_acc:.3f}, bal_acc={l1_bal_acc:.3f}, F1={l1_f1:.3f}"
        )

        # Plot L1 ROC/PR curves
        plot_roc_curve(
            y_l1_va_enc, y_l1_proba, l1_classes,
            f"Level-1 ROC – Fold {fold_id}",
            fold_dir / f"roc_level1_fold{fold_id}.png"
        )

        plot_pr_curve(
            y_l1_va_enc, y_l1_proba, l1_classes,
            f"Level-1 PR – Fold {fold_id}",
            fold_dir / f"pr_level1_fold{fold_id}.png"
        )

        # Plot L1 confusion matrix
        y_l1_pred_enc = model_l1.predict(X_va_scaled)
        plot_confusion_matrix(
            y_l1_va_enc, y_l1_pred_enc, l1_classes,
            f"Level-1 Confusion Matrix – Fold {fold_id}",
            fold_dir / f"cm_level1_fold{fold_id}.png"
        )

        # Compute feature importance for L1
        if config.verbose >= 2:
            logger.debug("[Level-1] Computing feature importance...")
            l1_importance = extract_feature_importance_mlp(
                model_l1, X_va_scaled, y_l1_va_enc, feature_cols, n_permutations=5
            )
            plot_feature_importance(
                l1_importance, feature_cols,
                f"Level-1 Feature Importance – Fold {fold_id}",
                fold_dir / f"feature_importance_l1_fold{fold_id}.png"
            )

        # Store ROC/PR data for averaging
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        if len(l1_classes) == 2:
            y_bin = (y_l1_va_enc == 1).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_l1_proba[:, 1])
            prec, rec, _ = precision_recall_curve(y_bin, y_l1_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            ap_val = average_precision_score(y_bin, y_l1_proba[:, 1])
        else:
            y_bin = label_binarize(y_l1_va_enc, classes=list(range(len(l1_classes))))
            if y_bin.ndim == 1:
                y_bin = np.column_stack([1 - y_bin, y_bin])
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_l1_proba.ravel())
            prec, rec, _ = precision_recall_curve(y_bin.ravel(), y_l1_proba.ravel())
            roc_auc_val = auc(fpr, tpr)
            ap_val = average_precision_score(y_bin, y_l1_proba, average="micro")

        all_l1_roc_data["fpr"].append(fpr)
        all_l1_roc_data["tpr"].append(tpr)
        all_l1_roc_data["auc"].append(roc_auc_val)
        all_l1_pr_data["rec"].append(rec)
        all_l1_pr_data["prec"].append(prec)
        all_l1_pr_data["ap"].append(ap_val)

        # ========== Level-2 (if hierarchical) ==========
        if config.hierarchical:
            logger.info("\n[Level-2] Training per-branch models...")

            branch_models = {}
            branch_encoders = {}
            branch_trained = {}

            for l1_val in l1_classes:
                logger.info(f"\n  [Branch L1={l1_val}]")

                branch_train_mask = (y_l1_tr == l1_val)
                branch_val_mask = (y_l1_va == l1_val)

                X_tr_b = X_tr_scaled[branch_train_mask]
                y_l2_tr_b = y_l2_tr[branch_train_mask]

                # Keep only non-NA L2
                valid_tr = pd.notna(y_l2_tr_b)
                X_tr_b = X_tr_b[valid_tr]
                y_l2_tr_b = y_l2_tr_b[valid_tr]

                # Filter by l2_classes if specified
                if config.l2_classes:
                    keep_tr = np.isin(y_l2_tr_b, config.l2_classes)
                    X_tr_b = X_tr_b[keep_tr]
                    y_l2_tr_b = y_l2_tr_b[keep_tr]

                unique_l2_b = sorted(np.unique(y_l2_tr_b).tolist()) if len(y_l2_tr_b) > 0 else []
                n_l2_b = len(unique_l2_b)

                if n_l2_b < config.min_l2_classes_per_branch:
                    logger.info(
                        f"    Skipping: only {n_l2_b} L2 classes (min={config.min_l2_classes_per_branch})"
                    )
                    branch_trained[l1_val] = False
                    continue

                logger.info(f"    Train samples (non-NA L2): {len(X_tr_b)}, L2 classes: {unique_l2_b}")

                # Create branch encoder
                le_l2_b = LabelEncoder()
                le_l2_b.fit(unique_l2_b)
                y_l2_tr_b_enc = le_l2_b.transform(y_l2_tr_b)

                # Hyperparameter tuning
                inner_cv_l2 = StratifiedKFold(
                    n_splits=config.inner_splits,
                    shuffle=True,
                    random_state=config.random_state + fold_id + 1000,
                )

                best_cfg_l2, inner_results_l2 = tune_hyperparameters(
                    X_tr_b, y_l2_tr_b_enc, n_l2_b, inner_cv_l2,
                    candidate_params, config, f"L2_{l1_val}"
                )

                for res in inner_results_l2:
                    res["fold"] = fold_id
                    inner_cv_rows.append(res)

                # Train final L2 branch model
                X_tr_b_final = X_tr_b
                y_tr_b_final = y_l2_tr_b_enc
                if config.use_smote:
                    X_tr_b_final, y_tr_b_final = apply_smote(
                        X_tr_b, y_l2_tr_b_enc, config.smote_k_neighbors, config.random_state + 1000
                    )

                model_l2_b = TorchMLPWrapper(
                    input_dim=n_features,
                    num_classes=n_l2_b,
                    hidden_dims=best_cfg_l2["hidden_dims"],
                    lr=best_cfg_l2["lr"],
                    epochs=best_cfg_l2["epochs"],
                    dropout=best_cfg_l2["dropout"],
                    batch_size=config.mlp_batch_size,
                    early_stopping_patience=config.early_stopping_patience,
                    device=config.device,
                    random_state=config.random_state + fold_id + 1000,
                    verbose=config.verbose,
                )

                # Get val data for early stopping
                X_va_b = X_va_scaled[branch_val_mask] if branch_val_mask.sum() > 0 else None
                y_l2_va_b = y_l2_va[branch_val_mask] if branch_val_mask.sum() > 0 else None
                y_l2_va_b_enc = None

                if y_l2_va_b is not None and X_va_b is not None:
                    valid_va = pd.notna(y_l2_va_b)
                    X_va_b = X_va_b[valid_va]
                    y_l2_va_b = y_l2_va_b[valid_va]

                    if config.l2_classes:
                        keep_va = np.isin(y_l2_va_b, config.l2_classes)
                        X_va_b = X_va_b[keep_va]
                        y_l2_va_b = y_l2_va_b[keep_va]

                    if len(y_l2_va_b) > 0:
                        in_train = np.isin(y_l2_va_b, unique_l2_b)
                        if in_train.sum() > 0:
                            X_va_b = X_va_b[in_train]
                            y_l2_va_b = y_l2_va_b[in_train]
                            y_l2_va_b_enc = le_l2_b.transform(y_l2_va_b)

                model_l2_b.fit(X_tr_b_final, y_tr_b_final, X_va_b, y_l2_va_b_enc)

                # Save branch model
                safe_l1 = l1_val.replace(" ", "_")
                model_l2_b.save(fold_dir / f"model_level2_{safe_l1}_fold{fold_id}.pt")
                joblib.dump(le_l2_b, fold_dir / f"label_encoder_l2_{safe_l1}.joblib")
                with open(fold_dir / f"model_config_l2_{safe_l1}_fold{fold_id}.json", "w") as f:
                    json.dump(model_l2_b.get_config(), f, indent=2)

                branch_models[l1_val] = model_l2_b
                branch_encoders[l1_val] = le_l2_b
                branch_trained[l1_val] = True

                # Evaluate branch
                if X_va_b is not None and len(X_va_b) > 0 and y_l2_va_b_enc is not None:
                    y_l2_pred_b_enc = model_l2_b.predict(X_va_b)
                    y_l2_proba_b = model_l2_b.predict_proba(X_va_b)

                    l2_acc = accuracy_score(y_l2_va_b_enc, y_l2_pred_b_enc)
                    l2_bal_acc = balanced_accuracy_score(y_l2_va_b_enc, y_l2_pred_b_enc)
                    l2_f1 = f1_score(y_l2_va_b_enc, y_l2_pred_b_enc, average="macro", zero_division=0)

                    l2_row = {
                        "fold": fold_id,
                        "level": f"L2_{l1_val}",
                        "accuracy": l2_acc,
                        "balanced_accuracy": l2_bal_acc,
                        "f1_macro": l2_f1,
                        "n_classes": n_l2_b,
                    }
                    outer_rows.append(l2_row)

                    logger.info(f"    Val: acc={l2_acc:.3f}, F1={l2_f1:.3f}")

                    # Plot L2 branch ROC/PR curves
                    plot_roc_curve(
                        y_l2_va_b_enc, y_l2_proba_b, unique_l2_b,
                        f"Level-2 ROC ({l1_val}) – Fold {fold_id}",
                        fold_dir / f"roc_level2_{safe_l1}_fold{fold_id}.png"
                    )

                    plot_pr_curve(
                        y_l2_va_b_enc, y_l2_proba_b, unique_l2_b,
                        f"Level-2 PR ({l1_val}) – Fold {fold_id}",
                        fold_dir / f"pr_level2_{safe_l1}_fold{fold_id}.png"
                    )

                    # Plot L2 confusion matrix
                    plot_confusion_matrix(
                        y_l2_va_b_enc, y_l2_pred_b_enc, unique_l2_b,
                        f"Level-2 CM ({l1_val}) – Fold {fold_id}",
                        fold_dir / f"cm_level2_{safe_l1}_fold{fold_id}.png"
                    )

                    # Store ROC/PR data for averaging
                    if n_l2_b == 2:
                        y_bin_b = (y_l2_va_b_enc == 1).astype(int)
                        fpr_b, tpr_b, _ = roc_curve(y_bin_b, y_l2_proba_b[:, 1])
                        prec_b, rec_b, _ = precision_recall_curve(y_bin_b, y_l2_proba_b[:, 1])
                        roc_auc_b = auc(fpr_b, tpr_b)
                        ap_b = average_precision_score(y_bin_b, y_l2_proba_b[:, 1])
                    else:
                        y_bin_b = label_binarize(y_l2_va_b_enc, classes=list(range(n_l2_b)))
                        if y_bin_b.ndim == 1:
                            y_bin_b = np.column_stack([1 - y_bin_b, y_bin_b])
                        fpr_b, tpr_b, _ = roc_curve(y_bin_b.ravel(), y_l2_proba_b.ravel())
                        prec_b, rec_b, _ = precision_recall_curve(y_bin_b.ravel(), y_l2_proba_b.ravel())
                        roc_auc_b = auc(fpr_b, tpr_b)
                        ap_b = average_precision_score(y_bin_b, y_l2_proba_b, average="micro")

                    all_l2_roc_data[l1_val]["fpr"].append(fpr_b)
                    all_l2_roc_data[l1_val]["tpr"].append(tpr_b)
                    all_l2_roc_data[l1_val]["auc"].append(roc_auc_b)
                    all_l2_pr_data[l1_val]["rec"].append(rec_b)
                    all_l2_pr_data[l1_val]["prec"].append(prec_b)
                    all_l2_pr_data[l1_val]["ap"].append(ap_b)

            # ========== Pipeline evaluation ==========
            any_branch_trained = any(branch_trained.get(l1, False) for l1 in l1_classes)

            if any_branch_trained:
                logger.info("\n[Pipeline] Evaluating hierarchical L1→L2...")

                valid_pipe = pd.notna(y_l2_va)
                if valid_pipe.sum() == 0:
                    logger.info("  Pipeline: no non-NA L2 labels in val set; skipping")
                else:
                    X_va_pipe = X_va_scaled[valid_pipe]
                    y_l1_va_pipe = y_l1_va[valid_pipe]
                    y_l2_va_pipe = y_l2_va[valid_pipe]

                    # Predict L1
                    y_l1_pred_enc_pipe = model_l1.predict(X_va_pipe)
                    y_l1_pred_labels_pipe = le_l1.inverse_transform(y_l1_pred_enc_pipe)

                    hier_true = np.array([f"{l1}::{l2}" for l1, l2 in zip(y_l1_va_pipe, y_l2_va_pipe)])
                    hier_pred = []

                    for i, l1_hat in enumerate(y_l1_pred_labels_pipe):
                        if branch_trained.get(l1_hat, False):
                            model_b = branch_models[l1_hat]
                            le_b = branch_encoders[l1_hat]
                            l2_pred_enc = model_b.predict(X_va_pipe[i:i+1])[0]
                            l2_hat = le_b.inverse_transform([l2_pred_enc])[0]
                            hier_pred.append(f"{l1_hat}::{l2_hat}")
                        else:
                            hier_pred.append(None)

                    keep = np.array([p is not None for p in hier_pred], dtype=bool)
                    if keep.sum() == 0:
                        logger.info("  Pipeline: no samples routed to trained L2 branch; skipping")
                    else:
                        hier_pred = np.array([p for p in hier_pred if p is not None])
                        hier_true = hier_true[keep]

                        acc_pipe = accuracy_score(hier_true, hier_pred)
                        balacc_pipe = balanced_accuracy_score(hier_true, hier_pred)
                        f1_pipe = f1_score(hier_true, hier_pred, average="macro", zero_division=0)

                        pipeline_row = {
                            "fold": fold_id,
                            "level": "pipeline",
                            "accuracy": acc_pipe,
                            "balanced_accuracy": balacc_pipe,
                            "f1_macro": f1_pipe,
                            "n_classes": len(np.unique(hier_true)),
                            "n_eval": int(len(hier_true)),
                        }
                        outer_rows.append(pipeline_row)
                        all_pipeline_metrics.append(pipeline_row)

                        logger.info(
                            f"  Pipeline: acc={acc_pipe:.3f}, bal_acc={balacc_pipe:.3f}, "
                            f"F1={f1_pipe:.3f} (n={len(hier_true)})"
                        )

    # ========== Save metrics ==========
    logger.info(f"\n{'='*60}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*60}")

    inner_df = pd.DataFrame(inner_cv_rows)
    outer_df = pd.DataFrame(outer_rows)

    # Compute summary statistics
    summary_rows = []

    # L1 summary
    l1_outer = outer_df[outer_df["level"] == "L1"]
    if len(l1_outer) > 0:
        l1_summary = {"level": "L1"}
        for col in ["accuracy", "balanced_accuracy", "f1_macro"]:
            if col in l1_outer.columns:
                vals = l1_outer[col].dropna()
                if len(vals) > 0:
                    l1_summary[f"{col}_mean"] = vals.mean()
                    l1_summary[f"{col}_std"] = vals.std()
        summary_rows.append(l1_summary)

    # L2 summaries (if hierarchical)
    if config.hierarchical:
        for l1_val in l1_classes:
            l2_outer = outer_df[outer_df["level"] == f"L2_{l1_val}"]
            if len(l2_outer) > 0:
                l2_summary = {"level": f"L2_{l1_val}"}
                for col in ["accuracy", "balanced_accuracy", "f1_macro"]:
                    vals = l2_outer[col].dropna()
                    if len(vals) > 0:
                        l2_summary[f"{col}_mean"] = vals.mean()
                        l2_summary[f"{col}_std"] = vals.std()
                summary_rows.append(l2_summary)

        # Pipeline summary
        pipe_outer = outer_df[outer_df["level"] == "pipeline"]
        if len(pipe_outer) > 0:
            pipe_summary = {"level": "pipeline"}
            for col in ["accuracy", "balanced_accuracy", "f1_macro"]:
                vals = pipe_outer[col].dropna()
                if len(vals) > 0:
                    pipe_summary[f"{col}_mean"] = vals.mean()
                    pipe_summary[f"{col}_std"] = vals.std()
            summary_rows.append(pipe_summary)

    summary_df = pd.DataFrame(summary_rows)

    # Save files
    if config.output_format == "xlsx":
        inner_df.to_excel(outdir / "metrics_inner_cv.xlsx", index=False)
        outer_df.to_excel(outdir / "metrics_outer_eval.xlsx", index=False)
        summary_df.to_excel(outdir / "metrics_summary.xlsx", index=False)
    else:
        inner_df.to_csv(outdir / "metrics_inner_cv.csv", index=False)
        outer_df.to_csv(outdir / "metrics_outer_eval.csv", index=False)
        summary_df.to_csv(outdir / "metrics_summary.csv", index=False)

    # ========== Plot averaged curves ==========
    logger.info("\nGenerating averaged ROC/PR curves...")

    # L1 averaged curves
    if len(all_l1_roc_data["fpr"]) > 1:
        plot_averaged_roc_curves(
            all_l1_roc_data["fpr"], all_l1_roc_data["tpr"], all_l1_roc_data["auc"],
            "Level-1 ROC – Averaged Across Folds",
            outdir / "roc_level1_averaged.png",
            show_individual=(config.outer_folds <= 5)
        )
        plot_averaged_pr_curves(
            all_l1_pr_data["rec"], all_l1_pr_data["prec"], all_l1_pr_data["ap"],
            "Level-1 PR – Averaged Across Folds",
            outdir / "pr_level1_averaged.png",
            show_individual=(config.outer_folds <= 5)
        )

    # L2 averaged curves (if hierarchical)
    if config.hierarchical:
        for l1_val in l1_classes:
            if len(all_l2_roc_data[l1_val]["fpr"]) > 1:
                safe_l1 = l1_val.replace(" ", "_")
                plot_averaged_roc_curves(
                    all_l2_roc_data[l1_val]["fpr"], all_l2_roc_data[l1_val]["tpr"],
                    all_l2_roc_data[l1_val]["auc"],
                    f"Level-2 ({l1_val}) ROC – Averaged",
                    outdir / f"roc_level2_{safe_l1}_averaged.png",
                    show_individual=(config.outer_folds <= 5)
                )
                plot_averaged_pr_curves(
                    all_l2_pr_data[l1_val]["rec"], all_l2_pr_data[l1_val]["prec"],
                    all_l2_pr_data[l1_val]["ap"],
                    f"Level-2 ({l1_val}) PR – Averaged",
                    outdir / f"pr_level2_{safe_l1}_averaged.png",
                    show_individual=(config.outer_folds <= 5)
                )

    # ========== Print summary ==========
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    logger.info("\nLevel-1 Performance (mean ± std across folds):")
    if len(l1_outer) > 0:
        for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
            if metric in l1_outer.columns:
                mean_val = l1_outer[metric].mean()
                std_val = l1_outer[metric].std()
                logger.info(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")

    if config.hierarchical and all_pipeline_metrics:
        logger.info("\nPipeline Performance (mean ± std across folds):")
        pipe_df = pd.DataFrame(all_pipeline_metrics)
        for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
            mean_val = pipe_df[metric].mean()
            std_val = pipe_df[metric].std()
            logger.info(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")

    logger.info(f"\nOutputs saved to: {outdir}/")
    logger.info(f"  • metrics_*.{config.output_format}")
    logger.info(f"  • fold*/scaler.joblib, model_*.pt, label_encoder_*.joblib")

    logger.info("\nDone!")

    return {
        "n_folds": config.outer_folds,
        "l1_classes": l1_classes,
        "l2_classes_per_branch": branch_l2_global if config.hierarchical else {},
        "summary": summary_df.to_dict(orient="records"),
    }
