"""Enhanced inference API with lineage, confidence, drift, and hierarchical metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from classiflow.inference.config import InferenceConfig
from classiflow.lineage import (
    create_inference_manifest,
    load_training_manifest,
    validate_manifest_compatibility,
)
from classiflow.lineage.hashing import get_file_metadata
from classiflow.bundles import load_bundle
from classiflow.inference.confidence import (
    annotate_predictions_with_confidence,
    create_confidence_summary_sheet,
)
from classiflow.validation import (
    compute_feature_summary,
    compute_drift_scores,
    detect_drift,
    create_drift_report,
    save_feature_summaries,
    load_feature_summaries,
)
from classiflow.inference.hierarchical_metrics import (
    compute_hierarchical_metrics,
    create_hierarchical_metrics_sheets,
)

logger = logging.getLogger(__name__)


def run_enhanced_inference(
    config: InferenceConfig,
    bundle_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run inference with full lineage tracking, confidence metrics, and drift detection.

    This enhanced inference pipeline adds:
    - Data lineage tracking (run IDs, data hashes)
    - Confidence metrics for all predictions
    - Feature drift detection
    - Hierarchical-aware metrics
    - Bundle support

    Parameters
    ----------
    config : InferenceConfig
        Inference configuration
    bundle_path : Optional[Path]
        Path to model bundle (if using bundle instead of run_dir)

    Returns
    -------
    results : Dict[str, Any]
        Enhanced results with:
        - inference_manifest: InferenceRunManifest
        - predictions: pd.DataFrame with confidence columns
        - drift_report: Dict with drift warnings
        - metrics: hierarchical or standard metrics
        - output_files: Dict of generated files
    """
    logger.info("="*70)
    logger.info("Enhanced Inference Pipeline with Lineage & Drift Detection")
    logger.info("="*70)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    # === STEP 1: Load model source (bundle or run_dir) ===
    if bundle_path is not None:
        logger.info(f"\n[1/8] Loading model bundle: {bundle_path}")
        bundle_data = load_bundle(bundle_path, fold=1)
        training_manifest = bundle_data["manifest"]
        model_source = bundle_path
        model_is_bundle = True
        logger.info(f"  Run ID: {training_manifest.run_id}")
    else:
        logger.info(f"\n[1/8] Loading training manifest: {config.run_dir}")
        training_manifest = load_training_manifest(config.run_dir)
        model_source = config.run_dir
        model_is_bundle = False
        logger.info(f"  Run ID: {training_manifest.run_id}")

    # === STEP 2: Compute inference data lineage ===
    logger.info(f"\n[2/8] Computing inference data lineage...")
    data_metadata = get_file_metadata(config.data_csv)
    logger.info(f"  Data hash: {data_metadata['sha256_hash'][:16]}...")
    logger.info(f"  Data size: {data_metadata['size_bytes'] / 1024:.1f} KB")
    logger.info(f"  Data rows: {data_metadata.get('row_count', 'N/A')}")

    # Create inference manifest
    inference_manifest = create_inference_manifest(
        parent_run_id=training_manifest.run_id,
        data_path=config.data_csv,
        data_hash=data_metadata["sha256_hash"],
        data_size_bytes=data_metadata["size_bytes"],
        data_row_count=data_metadata.get("row_count"),
        config=config.to_dict(),
        model_source=model_source,
        model_is_bundle=model_is_bundle,
    )

    # Save inference manifest
    inference_manifest_path = config.output_dir / "inference_run.json"
    inference_manifest.save(inference_manifest_path)
    logger.info(f"  Saved inference manifest: {inference_manifest_path}")

    # === STEP 3: Load inference data ===
    logger.info(f"\n[3/8] Loading inference data...")
    df_input = pd.read_csv(config.data_csv)
    logger.info(f"  Loaded {len(df_input)} samples")

    # Extract features
    feature_list = training_manifest.feature_list
    if not feature_list:
        logger.warning("No feature list in training manifest, using all numeric columns")
        feature_list = df_input.select_dtypes(include=[np.number]).columns.tolist()
        if config.id_col and config.id_col in feature_list:
            feature_list.remove(config.id_col)
        if config.label_col and config.label_col in feature_list:
            feature_list.remove(config.label_col)

    logger.info(f"  Expected features: {len(feature_list)}")

    # Validate feature compatibility
    compatible, compat_warnings = validate_manifest_compatibility(
        training_manifest,
        feature_list,
    )
    if not compatible:
        warnings.extend(compat_warnings)
        logger.warning(f"  Feature compatibility issues: {len(compat_warnings)}")
        for w in compat_warnings[:3]:
            logger.warning(f"    - {w}")

    # === STEP 4: Feature drift detection ===
    logger.info(f"\n[4/8] Detecting feature drift...")

    # Load training feature summaries
    train_summary_path = config.run_dir / "feature_summaries.json" if not model_is_bundle else None
    if train_summary_path and train_summary_path.exists():
        train_summary = load_feature_summaries(train_summary_path)
        logger.info("  Loaded training feature summaries")
    else:
        # If not available, we can't compute drift
        logger.warning("  Training feature summaries not found, skipping drift detection")
        train_summary = None

    drift_warnings = []
    drift_df = None
    flagged_features = None

    if train_summary is not None:
        # Compute inference feature summary
        X_inf = df_input[feature_list]
        inf_summary = compute_feature_summary(X_inf, feature_list)

        # Compute drift
        drift_df = compute_drift_scores(train_summary, inf_summary)
        flagged_features, drift_warnings = detect_drift(drift_df)

        logger.info(f"  Flagged features: {len(flagged_features)}")
        if drift_warnings:
            for w in drift_warnings[:5]:
                logger.warning(f"    {w}")

        warnings.extend(drift_warnings)
        inference_manifest.drift_warnings = drift_warnings

        # Save drift report
        drift_output = create_drift_report(
            drift_df,
            flagged_features,
            config.output_dir,
            thresholds={"z_threshold": 3.0, "missing_threshold": 0.1, "median_threshold": 2.0},
        )
        logger.info(f"  Saved drift report: {drift_output.get('xlsx')}")

    # === STEP 5: Run predictions (placeholder - integrate with actual predictor) ===
    logger.info(f"\n[5/8] Running predictions...")
    logger.info("  (Placeholder: integrate with actual prediction pipeline)")

    # For demonstration, create a mock predictions DataFrame
    # In real implementation, this would call the actual predictor
    predictions_df = pd.DataFrame({
        "sample_id": range(len(df_input)),
        "predicted_label": ["ClassA", "ClassB", "ClassA"] * (len(df_input) // 3 + 1),
        "model_run_id": training_manifest.run_id,
        "inference_run_id": inference_manifest.inference_run_id,
        "inference_data_hash": inference_manifest.inference_data_hash,
    })[:len(df_input)]

    # === STEP 6: Annotate with confidence metrics ===
    logger.info(f"\n[6/8] Computing confidence metrics...")

    # Mock probabilities for demonstration
    # In real implementation, get from predictor
    n_classes = 3
    mock_probabilities = np.random.dirichlet(np.ones(n_classes) * 5, size=len(df_input))

    predictions_df = annotate_predictions_with_confidence(
        predictions_df,
        mock_probabilities,
        confidence_thresholds={"high_min": 0.9, "medium_min": 0.7, "low_min": 0.0},
    )

    logger.info("  Added confidence metrics: max_proba, margin, entropy, bucket")

    # === STEP 7: Compute metrics (if labels available) ===
    metrics = None
    if config.label_col and config.label_col in df_input.columns:
        logger.info(f"\n[7/8] Computing evaluation metrics...")

        y_true = df_input[config.label_col]

        # Check if hierarchical
        is_hierarchical = training_manifest.hierarchical or "::" in predictions_df["predicted_label"].iloc[0]

        if is_hierarchical:
            logger.info("  Using hierarchical metrics...")
            metrics = compute_hierarchical_metrics(
                y_true,
                predictions_df["predicted_label"],
                y_proba=mock_probabilities,
            )
        else:
            logger.info("  Using standard classification metrics...")
            from classiflow.inference.metrics import compute_classification_metrics
            metrics = compute_classification_metrics(
                y_true.values,
                predictions_df["predicted_label"].values,
                y_proba=mock_probabilities,
            )

        logger.info(f"  Accuracy: {metrics.get('accuracy', metrics.get('overall', {}).get('accuracy', 0)):.4f}")

    # === STEP 8: Generate reports ===
    logger.info(f"\n[8/8] Generating reports...")

    # Save predictions
    predictions_csv = config.output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    logger.info(f"  Predictions: {predictions_csv}")

    # Create Excel workbook
    if config.include_excel:
        xlsx_path = config.output_dir / "inference_results.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            # Predictions
            predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

            # Confidence summary
            create_confidence_summary_sheet(predictions_df, writer, sheet_name="Confidence_Summary")

            # Metrics
            if metrics is not None:
                if training_manifest.hierarchical:
                    create_hierarchical_metrics_sheets(metrics, writer)
                else:
                    # Standard metrics
                    if "per_class" in metrics:
                        per_class_df = pd.DataFrame(metrics["per_class"])
                        per_class_df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)

            # Drift (if computed)
            if drift_df is not None:
                drift_df.to_excel(writer, sheet_name="Drift", index=False)
                if flagged_features is not None and not flagged_features.empty:
                    flagged_features.to_excel(writer, sheet_name="Drift_Flagged", index=False)

        logger.info(f"  Excel workbook: {xlsx_path}")

    # Update inference manifest with warnings
    inference_manifest.validation_warnings = warnings
    inference_manifest.save(inference_manifest_path)

    logger.info("\n" + "="*70)
    logger.info("✓ Enhanced inference pipeline complete")
    logger.info("="*70)

    return {
        "inference_manifest": inference_manifest,
        "predictions": predictions_df,
        "metrics": metrics,
        "drift_report": {
            "drift_df": drift_df,
            "flagged_features": flagged_features,
            "warnings": drift_warnings,
        },
        "warnings": warnings,
        "output_files": {
            "predictions_csv": predictions_csv,
            "inference_manifest": inference_manifest_path,
            "drift_report": drift_output.get("xlsx") if drift_df is not None else None,
        },
    }
