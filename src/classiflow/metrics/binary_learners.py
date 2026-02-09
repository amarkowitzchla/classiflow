"""Meta-mode under-the-hood binary learner diagnostics and artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


BL_RULES: Dict[str, Dict[str, float]] = {
    "BL-001_WARN_STD": {"threshold": 0.01},
    "BL-001_ERROR_STD": {"threshold": 0.005},
    "BL-002_LOW_RATE": {"threshold": 0.005},
    "BL-002_HIGH_RATE": {"threshold": 0.995},
    "BL-003_AUC": {"threshold": 0.60},
    "BL-004_RANGE": {"threshold": 0.20},
    "BL-004_STD": {"threshold": 0.10},
    "BL-005_AUC": {"threshold": 0.995},
    "BL-006_MIN_N_POS": {"threshold": 25.0},
}


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return slug or "class"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def persist_binary_ovr_fold_outputs(
    *,
    var_dir: Path,
    fold: int,
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    classes: List[str],
    base_ovr_scores: Dict[str, np.ndarray],
) -> Optional[Path]:
    """Persist per-fold OVR base learner scores as NPZ."""
    if not base_ovr_scores:
        return None
    matrix = np.full((len(sample_ids), len(classes)), np.nan, dtype=float)
    for idx, class_name in enumerate(classes):
        if class_name in base_ovr_scores:
            matrix[:, idx] = np.asarray(base_ovr_scores[class_name], dtype=float)
    path = var_dir / f"base_ovr_proba_fold{fold}.npz"
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=str),
        y_true=np.asarray(y_true, dtype=str),
        classes=np.asarray(classes, dtype=str),
        ovr_proba=matrix,
    )
    return path


def evaluate_binary_learner_health(
    *,
    run_dir: Path,
    fold_payloads: List[Dict[str, Any]],
    classes: List[str],
    feature_names: List[str],
    preferred_variant: str,
) -> Dict[str, Any]:
    """Generate binary learner diagnostics artifacts for meta-mode technical validation."""
    if not fold_payloads:
        return {}

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    class_estimator_types: Dict[str, set[str]] = {cls: set() for cls in classes}
    preferred_final: List[Dict[str, Any]] = []

    for payload in fold_payloads:
        fold = int(payload["fold"])
        sampler = str(payload["sampler"])
        y_true = np.asarray(payload["y_true"]).astype(str)
        sample_ids = np.asarray(payload["sample_id"]).astype(str)
        base_scores = payload.get("base_ovr_scores", {}) or {}
        path = payload.get("base_scores_path")
        estimators = payload.get("estimator_type_by_class", {}) or {}
        for cls, estimator_name in estimators.items():
            if estimator_name:
                class_estimator_types.setdefault(cls, set()).add(str(estimator_name))

        if path is not None:
            manifest_rows.append(
                {
                    "fold": fold,
                    "sampler": sampler,
                    "base_ovr_proba_path": _relative(Path(path), run_dir),
                    "n_samples": int(len(sample_ids)),
                }
            )

        if sampler == preferred_variant and payload.get("final_proba") is not None:
            preferred_final.append(payload)

        for class_name in classes:
            scores = np.asarray(
                base_scores.get(class_name, np.full(len(y_true), np.nan)), dtype=float
            )
            if len(scores) != len(y_true):
                continue
            y_bin = (y_true == str(class_name)).astype(int)
            n_pos = int(np.sum(y_bin == 1))
            n_neg = int(np.sum(y_bin == 0))
            pred_pos = (scores >= 0.5).astype(int)
            tp = int(np.sum((pred_pos == 1) & (y_bin == 1)))
            tn = int(np.sum((pred_pos == 0) & (y_bin == 0)))
            fp = int(np.sum((pred_pos == 1) & (y_bin == 0)))
            fn = int(np.sum((pred_pos == 0) & (y_bin == 1)))

            recall_05 = float(tp / (tp + fn)) if (tp + fn) else np.nan
            specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
            predicted_positive_rate = float(np.mean(pred_pos))
            p_mean = float(np.nanmean(scores)) if len(scores) else np.nan
            p_std = float(np.nanstd(scores)) if len(scores) else np.nan
            p_min = float(np.nanmin(scores)) if len(scores) else np.nan
            p_max = float(np.nanmax(scores)) if len(scores) else np.nan
            frac_near_zero = float(np.mean(scores <= 0.01)) if len(scores) else np.nan
            frac_near_one = float(np.mean(scores >= 0.99)) if len(scores) else np.nan

            roc_auc = np.nan
            pr_auc = np.nan
            brier = np.nan
            ll = np.nan
            if n_pos > 0 and n_neg > 0:
                try:
                    roc_auc = float(roc_auc_score(y_bin, scores))
                except Exception:
                    roc_auc = np.nan
                try:
                    pr_auc = float(average_precision_score(y_bin, scores))
                except Exception:
                    pr_auc = np.nan
                try:
                    brier = float(brier_score_loss(y_bin, scores))
                except Exception:
                    brier = np.nan
                try:
                    clipped = np.clip(scores, 1e-9, 1.0 - 1e-9)
                    ll = float(log_loss(y_bin, clipped, labels=[0, 1]))
                except Exception:
                    ll = np.nan

            rows.append(
                {
                    "fold": fold,
                    "sampler": sampler,
                    "class_name": class_name,
                    "n_samples": int(len(y_true)),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "roc_auc_ovr": roc_auc,
                    "pr_auc_ovr": pr_auc,
                    "recall_at_0_5": recall_05,
                    "recall_at_meta_threshold": recall_05,
                    "meta_threshold": 0.5,
                    "specificity": specificity,
                    "brier_binary": brier,
                    "log_loss_binary": ll,
                    "predicted_positive_rate_0_5": predicted_positive_rate,
                    "p_mean": p_mean,
                    "p_std": p_std,
                    "p_min": p_min,
                    "p_max": p_max,
                    "p_frac_near_zero": frac_near_zero,
                    "p_frac_near_one": frac_near_one,
                }
            )

    if not rows:
        return {}

    by_fold = (
        pd.DataFrame(rows).sort_values(["class_name", "fold", "sampler"]).reset_index(drop=True)
    )
    by_fold_path = run_dir / "binary_learners_metrics_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False)

    summary = _summarize_binary_metrics(by_fold)
    warnings = build_binary_learner_warnings(
        by_fold_df=by_fold,
        summary_df=summary,
        run_dir=run_dir,
        manifest_rows=manifest_rows,
        classes=classes,
    )

    status = _class_health_status(classes=classes, warnings=warnings)
    summary["health_status"] = summary["class_name"].map(status).fillna("OK")
    summary_path = run_dir / "binary_learners_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    warnings_path = run_dir / "binary_learners_warnings.json"
    warnings_path.write_text(json.dumps({"warnings": warnings}, indent=2), encoding="utf-8")

    ovr_plots = _plot_ovr_rocs(
        by_fold_df=by_fold,
        classes=classes,
        run_dir=run_dir,
    )

    ovo_csv, ovo_plot = _build_ovo_matrix(
        preferred_payloads=preferred_final,
        classes=classes,
        run_dir=run_dir,
    )

    manifest_payload = {
        "classes": list(classes),
        "estimator_type_by_class": {
            cls: sorted(class_estimator_types.get(cls, set())) for cls in classes
        },
        "input_feature_set_summary": {
            "n_features": int(len(feature_names)),
            "feature_list_path": "run.json#feature_list",
        },
        "per_fold_outputs": manifest_rows,
        "metrics_by_fold_path": _relative(by_fold_path, run_dir),
        "metrics_summary_path": _relative(summary_path, run_dir),
        "warnings_path": _relative(warnings_path, run_dir),
        "ovo_auc_matrix_path": _relative(ovo_csv, run_dir) if ovo_csv else None,
        "ovo_auc_matrix_plot_path": _relative(ovo_plot, run_dir) if ovo_plot else None,
        "ovr_roc_plot_paths": [_relative(p, run_dir) for p in ovr_plots],
    }
    manifest_path = run_dir / "binary_learners_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    _attach_to_run_manifest(
        run_manifest_path=run_dir / "run.json",
        payload={
            "manifest_path": _relative(manifest_path, run_dir),
            "metrics_by_fold_path": _relative(by_fold_path, run_dir),
            "metrics_summary_path": _relative(summary_path, run_dir),
            "warnings_path": _relative(warnings_path, run_dir),
            "ovo_auc_matrix_path": _relative(ovo_csv, run_dir) if ovo_csv else None,
            "ovo_auc_matrix_plot_path": _relative(ovo_plot, run_dir) if ovo_plot else None,
            "ovr_roc_plot_paths": [_relative(p, run_dir) for p in ovr_plots],
        },
    )
    return {
        "manifest_path": manifest_path,
        "metrics_by_fold_path": by_fold_path,
        "metrics_summary_path": summary_path,
        "warnings_path": warnings_path,
        "ovr_roc_plots": ovr_plots,
        "ovo_auc_matrix_path": ovo_csv,
        "ovo_auc_matrix_plot_path": ovo_plot,
    }


def _summarize_binary_metrics(by_fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc_ovr",
        "pr_auc_ovr",
        "recall_at_0_5",
        "specificity",
        "predicted_positive_rate_0_5",
        "p_std",
        "n_pos",
    ]
    rows: List[Dict[str, Any]] = []
    for class_name, grp in by_fold_df.groupby("class_name"):
        entry: Dict[str, Any] = {"class_name": class_name, "n_folds": int(len(grp))}
        for col in metric_cols:
            values = pd.to_numeric(grp[col], errors="coerce").dropna().astype(float)
            entry[f"{col}_mean"] = float(values.mean()) if not values.empty else np.nan
            entry[f"{col}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
        rows.append(entry)
    if not rows:
        return pd.DataFrame(columns=["class_name", "n_folds"])
    return pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)


def _class_health_status(*, classes: List[str], warnings: List[Dict[str, Any]]) -> Dict[str, str]:
    status = {cls: "OK" for cls in classes}
    for item in warnings:
        cls = str(item.get("class_name", ""))
        severity = str(item.get("severity", "INFO")).upper()
        if cls not in status:
            continue
        if severity == "ERROR":
            status[cls] = "ERROR"
        elif severity == "WARN" and status[cls] != "ERROR":
            status[cls] = "WARN"
    return status


def build_binary_learner_warnings(
    *,
    by_fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    run_dir: Path,
    manifest_rows: List[Dict[str, Any]],
    classes: List[str],
) -> List[Dict[str, Any]]:
    """Apply BL-001..BL-006 warning rules."""
    warnings: List[Dict[str, Any]] = []
    evidence_base = [
        "binary_learners_metrics_by_fold.csv",
        "binary_learners_metrics_summary.csv",
        "run.json#artifact_registry.binary_learners",
    ]
    fold_paths = {
        (int(row["fold"]), str(row["sampler"])): str(row["base_ovr_proba_path"])
        for row in manifest_rows
        if row.get("base_ovr_proba_path")
    }

    for _, row in by_fold_df.iterrows():
        cls = str(row["class_name"])
        fold = int(row["fold"])
        sampler = str(row["sampler"])
        fold_ref = f"fold{fold}:{sampler}"
        fold_evidence = list(evidence_base)
        if (fold, sampler) in fold_paths:
            fold_evidence.append(fold_paths[(fold, sampler)])

        p_std = _safe_float(row.get("p_std"))
        pred_rate = _safe_float(row.get("predicted_positive_rate_0_5"))
        auc_val = _safe_float(row.get("roc_auc_ovr"))
        n_pos = _safe_float(row.get("n_pos"))

        if p_std is not None and p_std < BL_RULES["BL-001_ERROR_STD"]["threshold"]:
            warnings.append(
                _warning(
                    severity="ERROR",
                    rule_id="BL-001",
                    class_name=cls,
                    folds=[fold_ref],
                    finding=f"Near-constant probabilities: std(p_pos)={p_std:.6f} < 0.005",
                    measured={"std_p_pos": p_std},
                    thresholds={"error_std_lt": 0.005, "warn_std_lt": 0.01},
                    evidence_paths=fold_evidence,
                    recommendation=(
                        "Investigate feature leakage removal, model degenerate fit, and class support."
                    ),
                )
            )
        elif p_std is not None and p_std < BL_RULES["BL-001_WARN_STD"]["threshold"]:
            warnings.append(
                _warning(
                    severity="WARN",
                    rule_id="BL-001",
                    class_name=cls,
                    folds=[fold_ref],
                    finding=f"Low probability variance: std(p_pos)={p_std:.6f} < 0.01",
                    measured={"std_p_pos": p_std},
                    thresholds={"warn_std_lt": 0.01},
                    evidence_paths=fold_evidence,
                    recommendation=(
                        "Investigate feature leakage removal, model degenerate fit, and class support."
                    ),
                )
            )

        if pred_rate is not None and (
            pred_rate < BL_RULES["BL-002_LOW_RATE"]["threshold"]
            or pred_rate > BL_RULES["BL-002_HIGH_RATE"]["threshold"]
        ):
            warnings.append(
                _warning(
                    severity="WARN",
                    rule_id="BL-002",
                    class_name=cls,
                    folds=[fold_ref],
                    finding=(
                        f"Degenerate positive decision rate: {pred_rate:.6f} outside "
                        "[0.005, 0.995]"
                    ),
                    measured={"predicted_positive_rate_0_5": pred_rate},
                    thresholds={"low_lt": 0.005, "high_gt": 0.995},
                    evidence_paths=fold_evidence,
                    recommendation=(
                        "Review thresholding/imbalance handling; consider calibration only if overconfident; review class counts."
                    ),
                )
            )

        if auc_val is not None and auc_val < BL_RULES["BL-003_AUC"]["threshold"]:
            low_power = (n_pos is not None) and (n_pos < BL_RULES["BL-006_MIN_N_POS"]["threshold"])
            warnings.append(
                _warning(
                    severity="INFO" if low_power else "WARN",
                    rule_id="BL-003",
                    class_name=cls,
                    folds=[fold_ref],
                    finding=(
                        f"No-signal ROC AUC: {auc_val:.4f} < 0.60"
                        + (" with low class support." if low_power else ".")
                    ),
                    measured={"roc_auc_ovr": auc_val, "n_pos": n_pos},
                    thresholds={"auc_lt": 0.60, "low_support_n_pos_lt": 25},
                    evidence_paths=fold_evidence,
                    recommendation=("Consider more data, feature engineering, or class merge."),
                )
            )

        if auc_val is not None and auc_val > BL_RULES["BL-005_AUC"]["threshold"]:
            warnings.append(
                _warning(
                    severity="WARN",
                    rule_id="BL-005",
                    class_name=cls,
                    folds=[fold_ref],
                    finding=f"Suspiciously perfect ROC AUC: {auc_val:.4f} > 0.995",
                    measured={"roc_auc_ovr": auc_val},
                    thresholds={"auc_gt": 0.995},
                    evidence_paths=fold_evidence,
                    recommendation=(
                        "Investigate leakage risk; verify feature set and splitting behavior."
                    ),
                )
            )

    for class_name, grp in by_fold_df.groupby("class_name"):
        grp_auc = pd.to_numeric(grp["roc_auc_ovr"], errors="coerce").dropna().astype(float)
        if len(grp_auc) >= 2:
            auc_range = float(grp_auc.max() - grp_auc.min())
            auc_std = float(grp_auc.std(ddof=0))
            if (
                auc_range > BL_RULES["BL-004_RANGE"]["threshold"]
                or auc_std > BL_RULES["BL-004_STD"]["threshold"]
            ):
                impacted = [
                    f"fold{int(r.fold)}:{str(r.sampler)}" for r in grp.itertuples(index=False)
                ]
                warnings.append(
                    _warning(
                        severity="WARN",
                        rule_id="BL-004",
                        class_name=str(class_name),
                        folds=impacted,
                        finding=(
                            f"High AUC variance across folds: range={auc_range:.4f}, std={auc_std:.4f}"
                        ),
                        measured={"auc_range": auc_range, "auc_std": auc_std},
                        thresholds={"range_gt": 0.20, "std_gt": 0.10},
                        evidence_paths=evidence_base,
                        recommendation=(
                            "Treat as instability; consider stronger regularization, improved stratification, or more data."
                        ),
                    )
                )

        n_pos_values = pd.to_numeric(grp["n_pos"], errors="coerce").dropna().astype(float)
        low_folds = [
            f"fold{int(r.fold)}:{str(r.sampler)}"
            for r in grp.itertuples(index=False)
            if _safe_float(getattr(r, "n_pos", None)) is not None
            and float(getattr(r, "n_pos")) < BL_RULES["BL-006_MIN_N_POS"]["threshold"]
        ]
        mean_n_pos = float(n_pos_values.mean()) if not n_pos_values.empty else np.nan
        if low_folds or (mean_n_pos == mean_n_pos and mean_n_pos < 25.0):
            warnings.append(
                _warning(
                    severity="INFO",
                    rule_id="BL-006",
                    class_name=str(class_name),
                    folds=sorted(set(low_folds)) or ["all"],
                    finding=(
                        f"Low class power detected: mean n_pos={mean_n_pos:.2f}, "
                        f"folds_below_25={len(low_folds)}"
                    ),
                    measured={
                        "mean_n_pos": mean_n_pos,
                        "low_support_folds": sorted(set(low_folds)),
                    },
                    thresholds={"n_pos_lt": 25},
                    evidence_paths=evidence_base,
                    recommendation=(
                        "Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings."
                    ),
                )
            )

    ordered = sorted(
        warnings,
        key=lambda item: (
            {"ERROR": 0, "WARN": 1, "INFO": 2}.get(str(item.get("severity")), 3),
            str(item.get("rule_id", "")),
            str(item.get("class_name", "")),
        ),
    )
    return ordered


def _warning(
    *,
    severity: str,
    rule_id: str,
    class_name: str,
    folds: List[str],
    finding: str,
    measured: Dict[str, Any],
    thresholds: Dict[str, Any],
    evidence_paths: List[str],
    recommendation: str,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "rule_id": rule_id,
        "class_name": class_name,
        "folds": folds,
        "finding": finding,
        "measured": measured,
        "thresholds": thresholds,
        "evidence_paths": evidence_paths,
        "recommended_action": recommendation,
    }


def _plot_ovr_rocs(*, by_fold_df: pd.DataFrame, classes: List[str], run_dir: Path) -> List[Path]:
    plots_dir = run_dir / "plots"
    out_paths: List[Path] = []
    mean_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for class_name in classes:
        class_rows = by_fold_df[by_fold_df["class_name"] == class_name]
        if class_rows.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
        fpr_grid = np.linspace(0.0, 1.0, 100)
        fold_tprs: List[np.ndarray] = []
        aucs: List[float] = []
        for row in class_rows.itertuples(index=False):
            fold = int(getattr(row, "fold"))
            sampler = str(getattr(row, "sampler"))
            base_path = (
                run_dir / f"fold{fold}" / f"binary_{sampler}" / f"base_ovr_proba_fold{fold}.npz"
            )
            if not base_path.exists():
                continue
            data = np.load(base_path, allow_pickle=False)
            stored_classes = [str(c) for c in data["classes"]]
            if class_name not in stored_classes:
                continue
            idx = stored_classes.index(class_name)
            y_true = np.asarray(data["y_true"]).astype(str)
            scores = np.asarray(data["ovr_proba"][:, idx], dtype=float)
            y_bin = (y_true == class_name).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, scores)
            fold_auc = auc(fpr, tpr)
            aucs.append(float(fold_auc))
            interp = np.interp(fpr_grid, fpr, tpr)
            interp[0] = 0.0
            fold_tprs.append(interp)
            ax.plot(fpr, tpr, color="#1F77B4", alpha=0.25, linewidth=1.0)
        if fold_tprs:
            mean_tpr = np.mean(np.vstack(fold_tprs), axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = float(auc(fpr_grid, mean_tpr))
            std_auc = float(np.std(aucs)) if aucs else np.nan
            ax.plot(
                fpr_grid,
                mean_tpr,
                color="#D62728",
                linewidth=2.2,
                label=f"Mean ROC AUC={mean_auc:.3f}±{std_auc:.3f}",
            )
            mean_curves[class_name] = {"fpr": fpr_grid, "tpr": mean_tpr}
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"Base OVR ROC - {class_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.25)
        path = plots_dir / f"binary_ovr_roc_{_safe_slug(class_name)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)

    if mean_curves:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
        for class_name in classes:
            curve = mean_curves.get(class_name)
            if curve is None:
                continue
            ax.plot(curve["fpr"], curve["tpr"], linewidth=2.0, label=str(class_name))
        ax.set_title("Base OVR ROC - All Classes")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        all_path = plots_dir / "binary_ovr_roc_all_classes.png"
        fig.tight_layout()
        fig.savefig(all_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(all_path)
    return out_paths


def _build_ovo_matrix(
    *,
    preferred_payloads: List[Dict[str, Any]],
    classes: List[str],
    run_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    if not preferred_payloads:
        return None, None
    y_true_all: List[str] = []
    proba_all: List[np.ndarray] = []
    for payload in preferred_payloads:
        final_classes = [str(c) for c in payload.get("final_classes", [])]
        y_true = np.asarray(payload.get("y_true", []), dtype=str)
        y_proba = payload.get("final_proba")
        if y_proba is None:
            continue
        y_proba_arr = np.asarray(y_proba, dtype=float)
        if y_proba_arr.ndim != 2:
            continue
        col_map = {cls: idx for idx, cls in enumerate(final_classes)}
        aligned = np.full((y_proba_arr.shape[0], len(classes)), np.nan, dtype=float)
        for idx, cls in enumerate(classes):
            if cls in col_map:
                aligned[:, idx] = y_proba_arr[:, col_map[cls]]
        y_true_all.extend(y_true.tolist())
        proba_all.append(aligned)

    if not y_true_all or not proba_all:
        return None, None

    y_true_arr = np.asarray(y_true_all, dtype=str)
    y_proba_arr = np.vstack(proba_all)
    matrix = np.full((len(classes), len(classes)), np.nan, dtype=float)
    for i, c_i in enumerate(classes):
        for j, c_j in enumerate(classes):
            if i == j:
                continue
            mask = np.isin(y_true_arr, [c_i, c_j])
            if np.sum(mask) < 2:
                continue
            y_pair = (y_true_arr[mask] == c_i).astype(int)
            if len(np.unique(y_pair)) < 2:
                continue
            score = y_proba_arr[mask, i]
            try:
                matrix[i, j] = float(roc_auc_score(y_pair, score))
            except Exception:
                matrix[i, j] = np.nan

    df = pd.DataFrame(matrix, index=classes, columns=classes)
    csv_path = run_dir / "ovo_auc_matrix.csv"
    df.to_csv(csv_path, index=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    heat = np.nan_to_num(matrix, nan=0.0)
    im = ax.imshow(heat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title("OVO ROC AUC Matrix (Final Meta Probabilities)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = "-" if i == j or matrix[i, j] != matrix[i, j] else f"{matrix[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plot_path = run_dir / "plots" / "ovo_auc_matrix.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, plot_path


def _attach_to_run_manifest(*, run_manifest_path: Path, payload: Dict[str, Any]) -> None:
    if not run_manifest_path.exists():
        return
    try:
        data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    artifact_registry = data.get("artifact_registry")
    if not isinstance(artifact_registry, dict):
        artifact_registry = {}
        data["artifact_registry"] = artifact_registry
    artifact_registry["binary_learners"] = payload
    run_manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
