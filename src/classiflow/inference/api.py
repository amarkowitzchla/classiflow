"""Public API for inference pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from classiflow.inference.config import InferenceConfig
from classiflow.inference.bagging import (
    get_bagging_member_count,
    iter_single_member_models,
)
from classiflow.inference.loader import ArtifactLoader
from classiflow.inference.preprocess import FeatureAligner, validate_input_data
from classiflow.inference.predict import (
    BinaryPredictor,
    MetaPredictor,
    HierarchicalPredictor,
    MulticlassPredictor,
    add_binary_prediction_columns,
)
from classiflow.inference.metrics import compute_classification_metrics
from classiflow.metrics.calibration import compute_probability_quality
from classiflow.inference.plots import generate_all_plots
from classiflow.inference.reports import InferenceReportWriter

logger = logging.getLogger(__name__)


def _infer_hierarchical_secondary_label_col(loader: ArtifactLoader) -> Optional[str]:
    """Infer the hierarchical secondary label column from training artifacts."""
    training_config = {}
    if loader.manifest and isinstance(loader.manifest.training_config, dict):
        training_config = loader.manifest.training_config

    task_config = training_config.get("task")
    if isinstance(task_config, dict):
        hierarchy_path = task_config.get("hierarchy_path")
        if hierarchy_path:
            return str(hierarchy_path)

    label_l2 = training_config.get("label_l2")
    if label_l2:
        return str(label_l2)

    for config_path in [loader.base_dir / "training_config.json", loader.fold_dir / "training_config.json"]:
        if not config_path.exists():
            continue
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_config = config_data.get("task")
        if isinstance(task_config, dict):
            hierarchy_path = task_config.get("hierarchy_path")
            if hierarchy_path:
                return str(hierarchy_path)
        label_l2 = config_data.get("label_l2")
        if label_l2:
            return str(label_l2)

    return None


def _build_true_labels(
    metadata: pd.DataFrame,
    run_type: str,
    label_col: Optional[str],
    label_col_secondary: Optional[str] = None,
) -> np.ndarray:
    """Build the evaluation truth labels, combining hierarchical levels when needed."""
    if not label_col or label_col not in metadata.columns:
        return np.full(len(metadata), np.nan, dtype=object)

    primary = metadata[label_col]
    if run_type != "hierarchical":
        return primary.to_numpy()

    primary_non_null = primary.dropna().astype(str)
    if not primary_non_null.empty and primary_non_null.str.contains("::", regex=False).any():
        return primary.to_numpy()

    if not label_col_secondary or label_col_secondary not in metadata.columns:
        return primary.to_numpy()

    secondary = metadata[label_col_secondary]
    combined = primary.astype(object).copy()
    valid_mask = ~(pd.isna(primary) | pd.isna(secondary))
    combined.loc[valid_mask] = (
        primary.loc[valid_mask].astype(str)
        + "::"
        + secondary.loc[valid_mask].astype(str)
    )
    return combined.to_numpy()


def _truth_values(predictions: pd.DataFrame, label_col: str) -> np.ndarray:
    """Return the normalized truth vector for metric and plot generation."""
    if "y_true" in predictions.columns:
        return predictions["y_true"].to_numpy()
    if label_col in predictions.columns:
        return predictions[label_col].to_numpy()
    return np.full(len(predictions), np.nan, dtype=object)


def _hierarchical_level_values(labels: np.ndarray, level_index: int) -> np.ndarray:
    """Extract one hierarchy level (0-based) from combined labels like 'L1::L2'."""
    values: list[object] = []
    for value in labels:
        if pd.isna(value):
            values.append(np.nan)
            continue
        parts = str(value).split("::")
        if level_index >= len(parts):
            values.append(np.nan)
            continue
        level_value = parts[level_index].strip()
        values.append(level_value if level_value else np.nan)
    return np.asarray(values, dtype=object)


def _hierarchical_level_payload(level_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize level metrics into a UI-friendly shape."""
    if not isinstance(level_metrics, dict):
        return {}

    summary: Dict[str, float] = {}
    for key, value in level_metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            numeric = float(value)
            if np.isfinite(numeric):
                summary[key] = numeric

    payload: Dict[str, Any] = {}
    if summary:
        payload["summary"] = summary

    for key in ("per_class", "confusion_matrix", "roc_auc", "warnings"):
        if key in level_metrics:
            payload[key] = level_metrics[key]

    if "error" in level_metrics and not payload:
        payload["error"] = level_metrics["error"]

    return payload


def _prediction_probability_columns(predictions: pd.DataFrame) -> list[str]:
    """Return direct prediction probability columns, excluding hierarchical helper outputs."""
    return [
        col
        for col in predictions.columns
        if col.startswith("predicted_proba_")
        and col != "predicted_proba"
        and not col.startswith("predicted_proba_L1_")
        and not col.startswith("predicted_proba_L2_")
        and not col.startswith("predicted_proba_L3_")
    ]


def _observed_classes(*label_arrays: np.ndarray) -> list[str]:
    """Return sorted class labels excluding missing values."""
    values = set()
    for labels in label_arrays:
        for value in labels:
            if pd.isna(value):
                continue
            values.add(str(value))
    return sorted(values)


def run_inference(config: InferenceConfig) -> Dict[str, Any]:
    """
    Run complete inference pipeline.

    This is the main entry point for inference. It:
    1. Loads trained model artifacts
    2. Loads and preprocesses input data
    3. Runs predictions (binary, meta, or hierarchical)
    4. Computes metrics (if labels provided)
    5. Generates plots
    6. Writes publication-ready reports

    Parameters
    ----------
    config : InferenceConfig
        Inference configuration

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing:
        - predictions: pd.DataFrame with predictions
        - metrics: Dict[str, Any] with computed metrics (if labels provided)
        - output_files: Dict[str, Path] with paths to generated files
        - warnings: List[str] of warnings encountered

    Examples
    --------
    >>> from classiflow.inference import run_inference, InferenceConfig
    >>> config = InferenceConfig(
    ...     run_dir="derived/fold1",
    ...     data_csv="test_data.csv",
    ...     output_dir="inference_results",
    ...     label_col="diagnosis",
    ... )
    >>> results = run_inference(config)
    >>> print(results["metrics"]["overall"]["accuracy"])
    0.8523
    """
    logger.info("=" * 60)
    logger.info("Starting inference pipeline")
    logger.info("=" * 60)
    data_path = config.resolved_data_path
    logger.info(f"  Run directory: {config.run_dir}")
    logger.info(f"  Data file: {data_path}")
    logger.info(f"  Output directory: {config.output_dir}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize collectors
    warnings = []
    results = {
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "warnings": warnings,
    }

    # Load artifacts
    logger.info("\n[1/7] Loading model artifacts...")
    loader = ArtifactLoader(config.run_dir, fold=1, verbose=config.verbose)
    run_type = loader.run_type

    logger.info(f"  Detected run type: {run_type}")

    secondary_label_col = config.label_col_secondary
    if run_type == "hierarchical" and not secondary_label_col:
        secondary_label_col = _infer_hierarchical_secondary_label_col(loader)
        if secondary_label_col:
            config.label_col_secondary = secondary_label_col
            results["config"] = config.to_dict()

    positive_class = None
    if loader.manifest and loader.manifest.task_definitions:
        task_def = loader.manifest.task_definitions.get("binary_task")
        if isinstance(task_def, str):
            prefix = "positive_class="
            if prefix in task_def:
                positive_class = task_def.split(prefix, 1)[1].strip()
        elif isinstance(task_def, dict):
            positive_class = task_def.get("positive_class") or task_def.get("pos_label")

    # Load input data (supports CSV, Parquet, and Parquet dataset directories)
    logger.info("\n[2/7] Loading and preprocessing data...")
    from classiflow.data import load_table

    df_raw = load_table(data_path)
    logger.info(f"  Loaded {len(df_raw)} samples, {len(df_raw.columns)} columns")

    # Validate input data
    input_warnings = validate_input_data(df_raw, id_col=config.id_col)
    if input_warnings:
        warnings.extend(input_warnings)
        for w in input_warnings[:5]:
            logger.warning(f"  {w}")

    # Get feature schema
    feature_schema = loader.get_feature_schema()
    required_features = feature_schema.get("feature_list")

    if required_features is None:
        # Fallback: use all numeric columns
        logger.warning("Feature list not found in artifacts; using all numeric columns")
        required_features = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID and label columns
        if config.id_col and config.id_col in required_features:
            required_features.remove(config.id_col)
        if config.label_col and config.label_col in required_features:
            required_features.remove(config.label_col)
        if secondary_label_col and secondary_label_col in required_features:
            required_features.remove(secondary_label_col)

    logger.info(f"  Required features: {len(required_features)}")

    # Align features
    aligner = FeatureAligner(
        required_features=required_features,
        strict=config.strict_features,
        fill_strategy=config.lenient_fill_strategy,
    )

    X, metadata, align_warnings = aligner.align(
        df_raw,
        id_col=config.id_col,
        label_col=config.label_col,
        label_col_secondary=secondary_label_col,
    )

    if align_warnings:
        warnings.extend(align_warnings)
        for w in align_warnings:
            logger.warning(f"  {w}")

    logger.info(f"  Aligned features: {X.shape}")

    # Run predictions
    logger.info("\n[3/7] Running predictions...")
    predictions, prediction_context = _run_predictions(loader, X, metadata, config)
    if run_type == "binary" and "predicted_label" not in predictions.columns:
        label_series = None
        if config.label_col and config.label_col in metadata.columns:
            label_series = metadata[config.label_col]
        predictions = add_binary_prediction_columns(
            predictions,
            labels=label_series,
            positive_class=positive_class,
        )

    # Add metadata columns (ID, true label)
    y_true_values = _build_true_labels(
        metadata,
        run_type=run_type,
        label_col=config.label_col,
        label_col_secondary=secondary_label_col,
    )
    sample_id_values = (
        metadata[config.id_col].to_numpy()
        if config.id_col and config.id_col in metadata.columns
        else metadata.index.astype(str).to_numpy()
    )
    derived_cols = ["y_true", "sample_id", "split", "fold_id"]
    metadata_out = metadata.drop(
        columns=[c for c in derived_cols if c in metadata.columns],
        errors="ignore",
    )
    predictions_out = predictions.drop(
        columns=[c for c in derived_cols if c in predictions.columns],
        errors="ignore",
    )
    final_predictions = pd.concat([metadata_out, predictions_out], axis=1).copy()
    final_predictions.loc[:, "y_true"] = y_true_values
    final_predictions.loc[:, "sample_id"] = sample_id_values
    final_predictions.loc[:, "split"] = "independent_test"
    final_predictions.loc[:, "fold_id"] = loader.fold

    logger.info(f"  Predictions shape: {final_predictions.shape}")

    results["predictions"] = final_predictions

    bagging = _compute_bagging_details(
        prediction_context=prediction_context,
        X=X,
        predictions=final_predictions,
        label_col=config.label_col,
        positive_class=positive_class,
    )
    if bagging:
        results["bagging"] = bagging
        logger.info(f"  Bag members available: {bagging.get('member_count', 0)}")

    # Compute metrics (if labels provided)
    metrics = None
    calibration_curves: Dict[str, pd.DataFrame] = {}
    if config.label_col and config.label_col in metadata.columns:
        logger.info("\n[4/7] Computing metrics...")
        metrics, calibration_curves = _compute_metrics(
            final_predictions,
            label_col=config.label_col,
            run_type=run_type,
            positive_class=positive_class,
        )
        results["metrics"] = metrics
        if calibration_curves:
            results["calibration_curves"] = calibration_curves
            if "top1" in calibration_curves:
                results["calibration_curve_df"] = calibration_curves["top1"]

        # Log headline metrics
        if "overall" in metrics:
            overall = metrics["overall"]
            logger.info(f"  Accuracy: {overall.get('accuracy', np.nan):.4f}")
            logger.info(f"  Balanced Accuracy: {overall.get('balanced_accuracy', np.nan):.4f}")
            logger.info(f"  F1 (Macro): {overall.get('f1_macro', np.nan):.4f}")
    else:
        logger.info("\n[4/7] Skipping metrics (no labels provided)")

    # Generate plots
    output_files = {}
    if config.include_plots and metrics is not None:
        logger.info("\n[5/7] Generating plots...")
        plot_paths = _generate_plots(
            final_predictions,
            label_col=config.label_col,
            output_dir=config.output_dir,
            max_roc_classes=config.max_roc_curves,
        )
        output_files.update(plot_paths)
        logger.info(f"  Generated {len(plot_paths)} plots")
    else:
        logger.info("\n[5/7] Skipping plots")

    # Write reports
    logger.info("\n[6/7] Writing reports...")
    writer = InferenceReportWriter(config.output_dir)

    # Predictions CSV
    pred_path = writer.write_predictions(final_predictions, "predictions.csv")
    output_files["predictions_csv"] = pred_path

    if bagging:
        bagging_files = writer.write_bagging_summary(bagging)
        output_files.update(bagging_files)
        bag_member_csv = bagging_files.get("bag_member_metrics_csv")
        if bag_member_csv is not None:
            results["bagging"]["metrics_csv_path"] = str(
                bag_member_csv.relative_to(config.output_dir)
            )

    if calibration_curves:
        written_curves = writer.write_calibration_curves(calibration_curves)
        if "top1" in calibration_curves:
            legacy_curve_path = writer.write_calibration_curve(
                calibration_curves["top1"], "calibration_curve.csv"
            )
            if legacy_curve_path is not None:
                output_files["calibration_curve_csv"] = legacy_curve_path
        if "top1" in written_curves:
            output_files["calibration_curve_top1_csv"] = written_curves["top1"]
        for name, path in written_curves.items():
            output_files[f"calibration_curve_{name}_csv"] = path

    # Metrics workbook (if applicable)
    if config.include_excel and metrics is not None:
        run_info = {
            "run_dir": str(config.run_dir),
            "data_file": str(data_path),
            "timestamp": results["timestamp"],
            "model_type": run_type,
            "fold": 1,
            "config": config.to_dict(),
        }

        workbook_path = writer.write_metrics_workbook(
            run_info=run_info,
            metrics=metrics,
            predictions=final_predictions,
            filename="metrics.xlsx",
        )
        output_files["metrics_workbook"] = workbook_path

    results["output_files"] = output_files

    # Summary
    logger.info("\n[7/7] Inference complete!")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples processed: {len(final_predictions)}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Files generated: {len(output_files)}")

    if warnings:
        logger.info(f"\n  Warnings: {len(warnings)}")
        for i, w in enumerate(warnings[:5], 1):
            logger.info(f"    {i}. {w}")
        if len(warnings) > 5:
            logger.info(f"    ... and {len(warnings) - 5} more")

    logger.info("=" * 60)

    return results


def _run_predictions(
    loader: ArtifactLoader,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    config: InferenceConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run predictions based on run type."""
    run_type = loader.run_type
    context: Dict[str, Any] = {"run_type": run_type}

    if run_type == "hierarchical":
        # Hierarchical predictions
        models, hier_config = loader.load_hierarchical_artifacts()
        predictor = HierarchicalPredictor(models, hier_config, device=config.device)
        predictions = predictor.predict(X)

    elif run_type == "meta":
        # Meta-classifier predictions
        # Try smote first, fall back to none
        try:
            pipes, best_models, _ = loader.load_binary_artifacts(variant="smote")
        except FileNotFoundError:
            pipes, best_models, _ = loader.load_binary_artifacts(variant="none")

        binary_predictor = BinaryPredictor(pipes, best_models)
        binary_predictions = binary_predictor.predict(X)

        # Then run meta-classifier
        try:
            (
                meta_model,
                meta_features,
                meta_classes,
                calibration_metadata,
            ) = loader.load_meta_artifacts(variant="smote")
        except FileNotFoundError:
            (
                meta_model,
                meta_features,
                meta_classes,
                calibration_metadata,
            ) = loader.load_meta_artifacts(variant="none")

        meta_predictor = MetaPredictor(
            meta_model,
            meta_features,
            meta_classes,
            calibration_metadata=calibration_metadata,
        )
        meta_predictions = meta_predictor.predict(binary_predictions)

        # Combine
        predictions = pd.concat([binary_predictions, meta_predictions], axis=1)
        context.update({"pipes": pipes, "best_models": best_models})

    elif run_type == "multiclass":
        try:
            model, classes, _ = loader.load_multiclass_artifacts(variant="smote")
        except FileNotFoundError:
            model, classes, _ = loader.load_multiclass_artifacts(variant="none")

        predictor = MulticlassPredictor(model, classes)
        predictions = predictor.predict(X)
        context.update({"model": model, "classes": classes})

    elif run_type in ("binary", "legacy"):
        # Binary task predictions only (or legacy format)
        # Try smote first, fall back to none
        try:
            pipes, best_models, _ = loader.load_binary_artifacts(variant="smote")
            logger.info("  Using SMOTE variant")
        except FileNotFoundError:
            try:
                pipes, best_models, _ = loader.load_binary_artifacts(variant="none")
                logger.info("  Using no-SMOTE variant")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find binary artifacts in {loader.run_dir}. "
                    "Expected binary_smote/binary_pipes.joblib or binary_none/binary_pipes.joblib"
                )

        binary_predictor = BinaryPredictor(pipes, best_models)
        predictions = binary_predictor.predict(X)

        # Check if meta-classifier exists (for legacy runs that might have meta)
        try:
            (
                meta_model,
                meta_features,
                meta_classes,
                calibration_metadata,
            ) = loader.load_meta_artifacts(variant="smote")
            logger.info("  Found meta-classifier, applying...")
            meta_predictor = MetaPredictor(
                meta_model,
                meta_features,
                meta_classes,
                calibration_metadata=calibration_metadata,
            )
            meta_predictions = meta_predictor.predict(predictions)
            predictions = pd.concat([predictions, meta_predictions], axis=1)
        except FileNotFoundError:
            # No meta-classifier, binary predictions only
            pass

        context.update({"pipes": pipes, "best_models": best_models})

    else:
        raise ValueError(f"Unsupported run type: {run_type}")

    return predictions, context


def _compute_bagging_details(
    prediction_context: Dict[str, Any],
    X: pd.DataFrame,
    predictions: pd.DataFrame,
    label_col: Optional[str],
    positive_class: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build bag-member metrics for bagged direct estimators."""
    run_type = prediction_context.get("run_type")
    label_series = (
        predictions[label_col]
        if label_col and label_col in predictions.columns
        else None
    )

    if run_type == "multiclass":
        model = prediction_context.get("model")
        classes = prediction_context.get("classes") or []
        member_count = get_bagging_member_count(model)
        if not member_count:
            return None

        members = []
        for member_index, member_model, estimator_type in iter_single_member_models(model):
            member_predictions = MulticlassPredictor(member_model, classes).predict(X)
            members.append(
                _summarize_bag_member(
                    member_index=member_index,
                    estimator_type=estimator_type,
                    member_predictions=member_predictions,
                    ensemble_predictions=predictions,
                    label_series=label_series,
                )
            )

        return {
            "strategy": "bagged",
            "member_count": member_count,
            "estimator_type": members[0]["estimator_type"] if members else None,
            "evaluation_available": label_series is not None,
            "members": members,
        }

    if run_type == "binary":
        pipes = prediction_context.get("pipes") or {}
        best_models = prediction_context.get("best_models") or {}
        if len(best_models) != 1:
            return None

        task_name, model_name = next(iter(best_models.items()))
        pipe_key = f"{task_name}__{model_name}"
        model = pipes.get(pipe_key)
        member_count = get_bagging_member_count(model)
        if not member_count:
            return None

        if label_series is None:
            return {
                "strategy": "bagged",
                "member_count": member_count,
                "task_name": task_name,
                "evaluation_available": False,
                "members": [],
            }

        members = []
        for member_index, member_model, estimator_type in iter_single_member_models(model):
            member_predictions = BinaryPredictor(
                {pipe_key: member_model},
                {task_name: model_name},
                [task_name],
            ).predict(X)
            member_predictions = add_binary_prediction_columns(
                member_predictions,
                labels=label_series,
                positive_class=positive_class,
            )
            members.append(
                _summarize_bag_member(
                    member_index=member_index,
                    estimator_type=estimator_type,
                    member_predictions=member_predictions,
                    ensemble_predictions=predictions,
                    label_series=label_series,
                )
            )

        return {
            "strategy": "bagged",
            "member_count": member_count,
            "task_name": task_name,
            "estimator_type": members[0]["estimator_type"] if members else None,
            "evaluation_available": True,
            "members": members,
        }

    return None


def _summarize_bag_member(
    member_index: int,
    estimator_type: str,
    member_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    label_series: Optional[pd.Series],
) -> Dict[str, Any]:
    """Summarize one bag member for UI display."""
    row: Dict[str, Any] = {
        "member_index": member_index,
        "estimator_type": estimator_type,
    }

    if "predicted_label" in member_predictions.columns and "predicted_label" in ensemble_predictions.columns:
        member_labels = member_predictions["predicted_label"].astype(str).to_numpy()
        ensemble_labels = ensemble_predictions["predicted_label"].astype(str).to_numpy()
        if len(member_labels) and len(member_labels) == len(ensemble_labels):
            row["agreement_with_ensemble"] = float(np.mean(member_labels == ensemble_labels))

    if label_series is None or "predicted_label" not in member_predictions.columns:
        return row

    proba_cols = [
        c
        for c in member_predictions.columns
        if c.startswith("predicted_proba_") and c != "predicted_proba"
    ]
    class_names = [c.replace("predicted_proba_", "") for c in proba_cols]
    y_proba = member_predictions[proba_cols].to_numpy() if proba_cols else None
    overall = compute_classification_metrics(
        label_series.to_numpy(),
        member_predictions["predicted_label"].to_numpy(),
        y_proba,
        class_names if class_names else None,
    )
    row.update(_flatten_bag_member_metrics(overall))
    return row


def _flatten_bag_member_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metrics into a compact row for CSV/UI rendering."""
    roc_auc = metrics.get("roc_auc") if isinstance(metrics.get("roc_auc"), dict) else {}
    return {
        "n_samples": _int_or_none(metrics.get("n_samples")),
        "accuracy": _float_or_none(metrics.get("accuracy")),
        "balanced_accuracy": _float_or_none(metrics.get("balanced_accuracy")),
        "f1_macro": _float_or_none(metrics.get("f1_macro")),
        "mcc": _float_or_none(metrics.get("mcc")),
        "log_loss": _float_or_none(metrics.get("log_loss")),
        "roc_auc_macro": _float_or_none(roc_auc.get("macro")),
        "roc_auc_micro": _float_or_none(roc_auc.get("micro")),
    }


def _float_or_none(value: Any) -> Optional[float]:
    """Convert numeric values to finite floats."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> Optional[int]:
    """Convert numeric values to ints when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_metrics(
    predictions: pd.DataFrame,
    label_col: str,
    run_type: str,
    positive_class: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Compute metrics based on available predictions."""
    metrics: Dict[str, Any] = {}
    calibration_curves: Dict[str, pd.DataFrame] = {}

    y_true = _truth_values(predictions, label_col)

    # Overall multiclass metrics (if predicted_label exists)
    if "predicted_label" in predictions.columns:
        y_pred = predictions["predicted_label"].values

        # Get probability columns
        proba_cols = _prediction_probability_columns(predictions)

        if proba_cols:
            class_names = [c.replace("predicted_proba_", "") for c in proba_cols]
            y_proba = predictions[proba_cols].values
        else:
            class_names = _observed_classes(y_true, y_pred)
            y_proba = None

        overall_metrics = compute_classification_metrics(y_true, y_pred, y_proba, class_names)
        if y_proba is not None:
            prob_metrics, cal_curve = compute_probability_quality(
                y_true.tolist(),
                y_pred.tolist(),
                y_proba,
                class_names,
                bins=10,
                mode=run_type
                if run_type in {"binary", "multiclass", "meta", "hierarchical"}
                else "multiclass",
            )
            overall_metrics["probability_quality"] = prob_metrics
            overall_metrics["brier"] = prob_metrics.get("brier")
            overall_metrics["brier_calibrated"] = prob_metrics.get("brier_recommended")
            overall_metrics["log_loss"] = prob_metrics.get("log_loss")
            overall_metrics["log_loss_calibrated"] = prob_metrics.get("log_loss")
            overall_metrics["ece"] = prob_metrics.get("ece_top1")
            overall_metrics["ece_calibrated"] = prob_metrics.get("ece_top1")
            overall_metrics["ece_top1"] = prob_metrics.get("ece_top1")
            overall_metrics["ece_binary_pos"] = prob_metrics.get("ece_binary_pos")
            overall_metrics["ece_ovr_macro"] = prob_metrics.get("ece_ovr_macro")
            overall_metrics["pred_alignment_mismatch_rate"] = prob_metrics.get(
                "pred_alignment_mismatch_rate"
            )
            overall_metrics["pred_alignment_note"] = prob_metrics.get("pred_alignment_note")
            overall_metrics["calibration_bins"] = 10
            if "top1" in cal_curve:
                metrics["calibration_curve"] = cal_curve["top1"].to_dict("records")
            metrics["calibration_curves"] = {
                name: df.to_dict("records") for name, df in cal_curve.items()
            }
            calibration_curves = cal_curve
        if "calibration_method" in predictions.columns:
            methods = predictions["calibration_method"].dropna().unique()
            if len(methods):
                overall_metrics["calibration_method"] = methods[0]
            enabled = predictions.get("calibration_enabled")
            if enabled is not None:
                overall_metrics["calibration_enabled"] = bool(enabled.any())
            bins = predictions.get("calibration_bins")
            if bins is not None:
                unique_bins = bins.dropna().unique()
                if len(unique_bins):
                    overall_metrics["calibration_bins"] = int(unique_bins[0])
        metrics["overall"] = overall_metrics
    elif run_type == "binary":
        pred_cols = [c for c in predictions.columns if c.endswith("_pred")]
        if pred_cols:
            pred_col = sorted(pred_cols)[0]
            if len(pred_cols) > 1:
                logger.warning(f"Multiple binary prediction columns found; using '{pred_col}'")
            y_pred_raw = predictions[pred_col].values
            class_names = sorted(list(pd.unique(y_true)))
            if len(class_names) == 2:
                pos_label = positive_class if positive_class in class_names else None
                if not pos_label:
                    logger.warning("Positive class not found in manifest; skipping binary metrics.")
                    return metrics, calibration_curves
                neg_label = class_names[0] if class_names[1] == pos_label else class_names[1]
                if pd.api.types.is_numeric_dtype(predictions[pred_col]):
                    y_pred = np.where(y_pred_raw == 1, pos_label, neg_label)
                else:
                    y_pred = y_pred_raw
                class_order = [neg_label, pos_label]
                y_proba = None
                score_cols = [c for c in predictions.columns if c.endswith("_score")]
                if score_cols:
                    score_col = sorted(score_cols)[0]
                    scores = pd.to_numeric(predictions[score_col], errors="coerce").values
                    if np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
                        y_proba = np.column_stack([1.0 - scores, scores])
                overall_metrics = compute_classification_metrics(
                    y_true, y_pred, y_proba, class_order
                )
                if y_proba is not None:
                    prob_metrics, cal_curve = compute_probability_quality(
                        y_true.tolist(),
                        y_pred.tolist(),
                        y_proba,
                        class_order,
                        bins=10,
                        mode="binary",
                    )
                    overall_metrics["probability_quality"] = prob_metrics
                    overall_metrics["brier"] = prob_metrics.get("brier")
                    overall_metrics["brier_calibrated"] = prob_metrics.get("brier_recommended")
                    overall_metrics["log_loss"] = prob_metrics.get("log_loss")
                    overall_metrics["log_loss_calibrated"] = prob_metrics.get("log_loss")
                    overall_metrics["ece"] = prob_metrics.get("ece_top1")
                    overall_metrics["ece_calibrated"] = prob_metrics.get("ece_top1")
                    overall_metrics["ece_top1"] = prob_metrics.get("ece_top1")
                    overall_metrics["ece_binary_pos"] = prob_metrics.get("ece_binary_pos")
                    overall_metrics["pred_alignment_mismatch_rate"] = prob_metrics.get(
                        "pred_alignment_mismatch_rate"
                    )
                    overall_metrics["pred_alignment_note"] = prob_metrics.get("pred_alignment_note")
                    overall_metrics["calibration_bins"] = 10
                    if "top1" in cal_curve:
                        metrics["calibration_curve"] = cal_curve["top1"].to_dict("records")
                    metrics["calibration_curves"] = {
                        name: df.to_dict("records") for name, df in cal_curve.items()
                    }
                    calibration_curves = cal_curve
                metrics["overall"] = overall_metrics

    # Task-level metrics (if binary task scores exist)
    task_score_cols = [c for c in predictions.columns if c.endswith("_score")]
    if task_score_cols and run_type in ["binary", "meta"]:
        task_metrics = {}

        for col in task_score_cols:
            task_name = col.replace("_score", "")
            # This would require task-specific ground truth, which we don't have
            # Skip for now
            pass

        if task_metrics:
            metrics["task_metrics"] = task_metrics

    # Hierarchical metrics (if L1/L2/L3 predictions exist)
    if run_type == "hierarchical":
        hier_metrics = {}

        # L1 metrics
        if "predicted_label_L1" in predictions.columns:
            y_true_l1 = _hierarchical_level_values(y_true, level_index=0)
            y_pred_l1 = predictions["predicted_label_L1"].values
            l1_proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_L1_")]

            if l1_proba_cols:
                l1_classes = [c.replace("predicted_proba_L1_", "") for c in l1_proba_cols]
                y_proba_l1 = predictions[l1_proba_cols].values
            else:
                y_true_l1_classes = pd.Series(y_true_l1).dropna().astype(str).unique().tolist()
                y_pred_l1_classes = pd.Series(y_pred_l1).dropna().astype(str).unique().tolist()
                l1_classes = sorted(set(y_true_l1_classes) | set(y_pred_l1_classes))
                y_proba_l1 = None

            l1_metrics = compute_classification_metrics(
                y_true_l1, y_pred_l1, y_proba_l1, l1_classes
            )
            l1_payload = _hierarchical_level_payload(l1_metrics)
            if l1_payload:
                hier_metrics["L1"] = l1_payload

        # L2 metrics
        if "predicted_label_L2" in predictions.columns:
            y_true_l2 = _hierarchical_level_values(y_true, level_index=1)
            y_pred_l2 = predictions["predicted_label_L2"].values
            l2_proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_L2_")]

            if l2_proba_cols:
                l2_classes = [c.replace("predicted_proba_L2_", "") for c in l2_proba_cols]
                y_proba_l2 = predictions[l2_proba_cols].values
            else:
                y_true_l2_classes = pd.Series(y_true_l2).dropna().astype(str).unique().tolist()
                y_pred_l2_classes = pd.Series(y_pred_l2).dropna().astype(str).unique().tolist()
                l2_classes = sorted(set(y_true_l2_classes) | set(y_pred_l2_classes))
                y_proba_l2 = None

            l2_metrics = compute_classification_metrics(
                y_true_l2, y_pred_l2, y_proba_l2, l2_classes
            )
            l2_payload = _hierarchical_level_payload(l2_metrics)
            if l2_payload:
                hier_metrics["L2"] = l2_payload

        if "overall" in metrics:
            pipeline_payload = _hierarchical_level_payload(metrics["overall"])
            if pipeline_payload:
                hier_metrics["pipeline"] = pipeline_payload

        if hier_metrics:
            metrics["hierarchical"] = hier_metrics

    return metrics, calibration_curves


def _generate_plots(
    predictions: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    max_roc_classes: int = 10,
) -> Dict[str, Path]:
    """Generate plots for predictions."""
    y_true = _truth_values(predictions, label_col)

    # Use predicted_label if available
    if "predicted_label" in predictions.columns:
        y_pred = predictions["predicted_label"].values

        # Get probabilities
        proba_cols = _prediction_probability_columns(predictions)

        if proba_cols:
            class_names = [c.replace("predicted_proba_", "") for c in proba_cols]
            y_proba = predictions[proba_cols].values
        else:
            class_names = _observed_classes(y_true, y_pred)
            y_proba = None

        plot_paths = generate_all_plots(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            class_names=class_names,
            output_dir=output_dir,
            prefix="inference",
            max_roc_classes=max_roc_classes,
        )

        return plot_paths

    return {}
