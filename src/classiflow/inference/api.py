"""Public API for inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from classiflow.inference.config import InferenceConfig
from classiflow.inference.loader import ArtifactLoader
from classiflow.inference.preprocess import FeatureAligner, validate_input_data
from classiflow.inference.predict import BinaryPredictor, MetaPredictor, HierarchicalPredictor, MulticlassPredictor
from classiflow.inference.metrics import compute_classification_metrics
from classiflow.inference.plots import generate_all_plots
from classiflow.inference.reports import InferenceReportWriter

logger = logging.getLogger(__name__)


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
    logger.info("="*60)
    logger.info("Starting inference pipeline")
    logger.info("="*60)
    logger.info(f"  Run directory: {config.run_dir}")
    logger.info(f"  Data file: {config.data_csv}")
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

    # Load input data
    logger.info("\n[2/7] Loading input data...")
    df = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    # Validate input data
    data_warnings = validate_input_data(df, id_col=config.id_col)
    if data_warnings:
        logger.warning(f"  Data validation warnings: {len(data_warnings)}")
        for w in data_warnings[:5]:
            logger.warning(f"    - {w}")
        warnings.extend(data_warnings)

    # Feature alignment
    logger.info("\n[2/7] Aligning features...")
    feature_schema = loader.get_feature_schema()
    required_features = feature_schema["feature_list"]

    if required_features is None or len(required_features) == 0:
        # Fallback: use all numeric columns
        logger.warning("Feature list not found in artifacts; using all numeric columns")
        df_raw = pd.read_csv(config.data_csv)
        required_features = df_raw.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID and label columns
        if config.id_col and config.id_col in required_features:
            required_features.remove(config.id_col)
        if config.label_col and config.label_col in required_features:
            required_features.remove(config.label_col)
    else:
        required_features = loader.get_feature_schema()["feature_list"]

    if not required_features:
        raise ValueError("Could not determine required features from run artifacts")

    logger.info(f"  Required features: {len(required_features)}")

    # Load input data
    logger.info("\n[2/7] Loading input data...")
    df_input = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df_input)} samples with {len(df_input.columns)} columns")

    # Validate input data
    input_warnings = validate_input_data(df_input, config.id_col)
    warnings.extend(input_warnings)

    for w in input_warnings[:5]:  # Show first 5 warnings
        logger.warning(f"  {w}")

    # Feature alignment
    logger.info("\n[2/7] Aligning features...")

    feature_schema = loader.get_feature_schema()
    required_features = feature_schema.get("feature_list")

    if required_features is None:
        # Fallback: use all numeric columns
        logger.warning("Feature list not found in artifacts; using all numeric columns from input data")
        df = pd.read_csv(config.data_csv)
        required_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove metadata columns
        if config.id_col and config.id_col in required_features:
            required_features.remove(config.id_col)
        if config.label_col and config.label_col in required_features:
            required_features.remove(config.label_col)
    else:
        required_features = loader.get_feature_schema()["feature_list"]

    # Load input data
    logger.info("\n[2/7] Loading input data...")
    df_raw = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df_raw)} samples, {len(df_raw.columns)} columns")

    # Validate input data
    input_warnings = validate_input_data(df_raw, id_col=config.id_col)
    warnings.extend(input_warnings)

    for w in warnings[-len(warnings):]:  # Only new warnings
        logger.warning(f"  {w}")

    # Align features
    logger.info("\n[2/7] Aligning features...")

    # Get required features from loader
    feature_schema = loader.get_feature_schema()
    required_features = feature_schema["feature_list"]

    if required_features is None:
        # Fallback: use all numeric columns
        logger.warning("Feature list not found in artifacts; using all numeric columns from input")
        required_features = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    aligner = FeatureAligner(
        required_features=required_features if required_features else [],
        strict=config.strict_features,
        fill_strategy=config.lenient_fill_strategy,
    )

    # Load data
    logger.info("\n[2/7] Loading and validating input data...")
    df_raw = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df_raw)} samples")

    # Validate input data
    input_warnings = validate_input_data(df_raw, config.id_col)
    if input_warnings:
        warnings.extend(input_warnings)
        for w in input_warnings:
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

    # Load data
    logger.info("\n[2/7] Loading and preprocessing data...")
    df_raw = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df_raw)} samples")

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
    )

    if align_warnings:
        warnings.extend(align_warnings)
        for w in align_warnings:
            logger.warning(f"  {w}")

    logger.info(f"  Aligned features: {X.shape}")

    # Run predictions
    logger.info("\n[3/7] Running predictions...")
    predictions = _run_predictions(loader, X, metadata, config)

    # Add metadata columns (ID, true label)
    final_predictions = pd.concat([metadata, predictions], axis=1)

    logger.info(f"  Predictions shape: {final_predictions.shape}")

    results["predictions"] = final_predictions

    # Compute metrics (if labels provided)
    metrics = None
    if config.label_col and config.label_col in metadata.columns:
        logger.info("\n[4/7] Computing metrics...")
        metrics = _compute_metrics(
            final_predictions,
            label_col=config.label_col,
            run_type=run_type,
        )
        results["metrics"] = metrics

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

    # Metrics workbook (if applicable)
    if config.include_excel and metrics is not None:
        run_info = {
            "run_dir": str(config.run_dir),
            "data_file": str(config.data_csv),
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
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"  Samples processed: {len(final_predictions)}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Files generated: {len(output_files)}")

    if warnings:
        logger.info(f"\n  Warnings: {len(warnings)}")
        for i, w in enumerate(warnings[:5], 1):
            logger.info(f"    {i}. {w}")
        if len(warnings) > 5:
            logger.info(f"    ... and {len(warnings) - 5} more")

    logger.info("="*60)

    return results


def _run_predictions(
    loader: ArtifactLoader,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    config: InferenceConfig,
) -> pd.DataFrame:
    """Run predictions based on run type."""
    run_type = loader.run_type

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
            meta_model, meta_features, meta_classes = loader.load_meta_artifacts(variant="smote")
        except FileNotFoundError:
            meta_model, meta_features, meta_classes = loader.load_meta_artifacts(variant="none")

        meta_predictor = MetaPredictor(meta_model, meta_features, meta_classes)
        meta_predictions = meta_predictor.predict(binary_predictions)

        # Combine
        predictions = pd.concat([binary_predictions, meta_predictions], axis=1)

    elif run_type == "multiclass":
        try:
            model, classes, _ = loader.load_multiclass_artifacts(variant="smote")
        except FileNotFoundError:
            model, classes, _ = loader.load_multiclass_artifacts(variant="none")

        predictor = MulticlassPredictor(model, classes)
        predictions = predictor.predict(X)

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
            meta_model, meta_features, meta_classes = loader.load_meta_artifacts(variant="smote")
            logger.info("  Found meta-classifier, applying...")
            meta_predictor = MetaPredictor(meta_model, meta_features, meta_classes)
            meta_predictions = meta_predictor.predict(predictions)
            predictions = pd.concat([predictions, meta_predictions], axis=1)
        except FileNotFoundError:
            # No meta-classifier, binary predictions only
            pass

    else:
        raise ValueError(f"Unsupported run type: {run_type}")

    return predictions


def _compute_metrics(
    predictions: pd.DataFrame,
    label_col: str,
    run_type: str,
) -> Dict[str, Any]:
    """Compute metrics based on available predictions."""
    metrics = {}

    y_true = predictions[label_col].values

    # Overall multiclass metrics (if predicted_label exists)
    if "predicted_label" in predictions.columns:
        y_pred = predictions["predicted_label"].values

        # Get probability columns
        proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_") and c != "predicted_proba"]

        if proba_cols:
            class_names = [c.replace("predicted_proba_", "") for c in proba_cols]
            y_proba = predictions[proba_cols].values
        else:
            class_names = sorted(list(set(y_true) | set(y_pred)))
            y_proba = None

        overall_metrics = compute_classification_metrics(
            y_true, y_pred, y_proba, class_names
        )

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
            y_pred_l1 = predictions["predicted_label_L1"].values
            l1_proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_L1_")]

            if l1_proba_cols:
                l1_classes = [c.replace("predicted_proba_L1_", "") for c in l1_proba_cols]
                y_proba_l1 = predictions[l1_proba_cols].values
            else:
                l1_classes = sorted(list(set(y_true) | set(y_pred_l1)))
                y_proba_l1 = None

            hier_metrics["L1"] = compute_classification_metrics(
                y_true, y_pred_l1, y_proba_l1, l1_classes
            )

        # L2 metrics (if applicable)
        # ... (would require L2 ground truth)

        if hier_metrics:
            metrics["hierarchical"] = hier_metrics

    return metrics


def _generate_plots(
    predictions: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    max_roc_classes: int = 10,
) -> Dict[str, Path]:
    """Generate plots for predictions."""
    y_true = predictions[label_col].values

    # Use predicted_label if available
    if "predicted_label" in predictions.columns:
        y_pred = predictions["predicted_label"].values

        # Get probabilities
        proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_") and c != "predicted_proba"]

        if proba_cols:
            class_names = [c.replace("predicted_proba_", "") for c in proba_cols]
            y_proba = predictions[proba_cols].values
        else:
            class_names = sorted(list(set(y_true) | set(y_pred)))
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
