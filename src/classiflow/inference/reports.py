"""Report generation for inference results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class InferenceReportWriter:
    """
    Generate publication-ready inference reports.

    Outputs:
    - predictions.csv: Row-level predictions with scores/probabilities
    - metrics.xlsx: Multi-sheet Excel workbook with all metrics
    - metrics/: Directory with CSV versions of each sheet
    """

    def __init__(self, output_dir: Path):
        """
        Initialize report writer.

        Parameters
        ----------
        output_dir : Path
            Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv_dir = self.output_dir / "metrics"
        self.metrics_csv_dir.mkdir(exist_ok=True)

    def write_predictions(
        self,
        predictions: pd.DataFrame,
        filename: str = "predictions.csv",
    ) -> Path:
        """
        Write row-level predictions to CSV.

        Parameters
        ----------
        predictions : pd.DataFrame
            Predictions dataframe
        filename : str
            Output filename

        Returns
        -------
        output_path : Path
            Path to written file
        """
        output_path = self.output_dir / filename
        predictions.to_csv(output_path, index=False)
        logger.info(f"Wrote predictions to {output_path}")
        return output_path

    def write_calibration_curve(
        self,
        curve_df: pd.DataFrame,
        filename: str = "calibration_curve.csv",
    ) -> Optional[Path]:
        """
        Write calibration curve data to CSV if available.
        """
        if curve_df is None or curve_df.empty:
            return None
        output_path = self.output_dir / filename
        curve_df.to_csv(output_path, index=False)
        logger.info(f"Wrote calibration curve data to {output_path}")
        return output_path

    def write_calibration_curves(
        self,
        curves: Dict[str, pd.DataFrame],
    ) -> Dict[str, Path]:
        """Write all available calibration curves."""
        written: Dict[str, Path] = {}
        for name, curve_df in (curves or {}).items():
            if curve_df is None or curve_df.empty:
                continue
            filename = f"calibration_curve_{name}.csv"
            path = self.write_calibration_curve(curve_df, filename=filename)
            if path is not None:
                written[name] = path
        return written

    def write_metrics_workbook(
        self,
        run_info: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: Optional[pd.DataFrame] = None,
        filename: str = "metrics.xlsx",
    ) -> Path:
        """
        Write comprehensive metrics workbook.

        Parameters
        ----------
        run_info : Dict[str, Any]
            Run metadata (model dir, config, timestamp, etc.)
        metrics : Dict[str, Any]
            Computed metrics
        predictions : Optional[pd.DataFrame]
            Predictions dataframe (for summary stats)
        filename : str
            Output filename

        Returns
        -------
        output_path : Path
            Path to written workbook
        """
        output_path = self.output_dir / filename

        try:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                # Sheet 1: Run Manifest
                self._write_manifest_sheet(writer, run_info, predictions)

                # Sheet 2: Overall Metrics
                if "overall" in metrics:
                    self._write_overall_metrics_sheet(writer, metrics["overall"])

                # Sheet 3: Per-Class Metrics
                if "per_class" in metrics:
                    self._write_per_class_sheet(writer, metrics["per_class"])
                elif "overall" in metrics and "per_class" in metrics["overall"]:
                    self._write_per_class_sheet(writer, metrics["overall"]["per_class"])

                # Sheet 4: Confusion Matrix
                if "overall" in metrics and "confusion_matrix" in metrics["overall"]:
                    self._write_confusion_matrix_sheet(
                        writer, metrics["overall"]["confusion_matrix"]
                    )

                # Sheet 5: ROC AUC Summary
                if "overall" in metrics and "roc_auc" in metrics["overall"]:
                    self._write_roc_auc_sheet(writer, metrics["overall"]["roc_auc"])

                # Sheet 6: Task-Level Metrics (if binary tasks exist)
                if "task_metrics" in metrics:
                    self._write_task_metrics_sheet(writer, metrics["task_metrics"])

                # Hierarchical sheets (if applicable)
                if "hierarchical" in metrics:
                    self._write_hierarchical_sheets(writer, metrics["hierarchical"])

                logger.info(f"Wrote metrics workbook to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write Excel workbook: {e}")
            logger.info("Falling back to CSV output")

            # Fallback: write individual CSV files
            self._write_csv_fallback(run_info, metrics, predictions)

        return output_path

    def _write_manifest_sheet(
        self,
        writer: pd.ExcelWriter,
        run_info: Dict[str, Any],
        predictions: Optional[pd.DataFrame],
    ) -> None:
        """Write run manifest sheet."""
        rows = []

        # Basic info
        rows.append({"Field": "Run Directory", "Value": run_info.get("run_dir", "N/A")})
        rows.append({"Field": "Data File", "Value": run_info.get("data_file", "N/A")})
        rows.append({"Field": "Inference Timestamp", "Value": run_info.get("timestamp", "N/A")})
        rows.append({"Field": "Model Type", "Value": run_info.get("model_type", "N/A")})
        rows.append({"Field": "Fold", "Value": run_info.get("fold", "N/A")})

        # Data info
        if predictions is not None:
            rows.append({"Field": "Total Samples", "Value": len(predictions)})

            # Class distribution
            if "predicted_label" in predictions.columns:
                class_counts = predictions["predicted_label"].value_counts().to_dict()
                rows.append({"Field": "--- Predicted Class Counts ---", "Value": ""})
                for cls, count in sorted(class_counts.items()):
                    rows.append({"Field": f"  {cls}", "Value": count})

            if "true_label" in predictions.columns:
                true_counts = predictions["true_label"].value_counts().to_dict()
                rows.append({"Field": "--- True Class Counts ---", "Value": ""})
                for cls, count in sorted(true_counts.items()):
                    rows.append({"Field": f"  {cls}", "Value": count})

        # Config
        if "config" in run_info:
            rows.append({"Field": "--- Configuration ---", "Value": ""})
            for key, value in run_info["config"].items():
                if key not in ["run_dir", "data_csv", "output_dir"]:
                    rows.append({"Field": f"  {key}", "Value": str(value)})

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Run_Manifest", index=False)

        # Save CSV version
        df.to_csv(self.metrics_csv_dir / "run_manifest.csv", index=False)

    def _write_overall_metrics_sheet(
        self,
        writer: pd.ExcelWriter,
        metrics: Dict[str, Any],
    ) -> None:
        """Write overall metrics sheet."""
        rows = []

        metric_order = [
            ("n_samples", "Sample Count"),
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("f1_macro", "F1 (Macro)"),
            ("f1_weighted", "F1 (Weighted)"),
            ("f1_micro", "F1 (Micro)"),
            ("mcc", "Matthews Correlation Coefficient"),
            ("log_loss", "Log Loss"),
        ]

        for key, label in metric_order:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
                else:
                    value_str = str(value)

                rows.append({"Metric": label, "Value": value_str})

        # ROC AUC
        if "roc_auc" in metrics:
            roc = metrics["roc_auc"]
            if "macro" in roc:
                val = roc["macro"]
                rows.append(
                    {
                        "Metric": "ROC AUC (Macro)",
                        "Value": f"{val:.4f}" if not np.isnan(val) else "N/A",
                    }
                )
            if "micro" in roc:
                val = roc["micro"]
                rows.append(
                    {
                        "Metric": "ROC AUC (Micro)",
                        "Value": f"{val:.4f}" if not np.isnan(val) else "N/A",
                    }
                )

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Overall_Metrics", index=False)
        df.to_csv(self.metrics_csv_dir / "overall_metrics.csv", index=False)

    def _write_per_class_sheet(
        self,
        writer: pd.ExcelWriter,
        per_class_metrics: List[Dict[str, Any]],
    ) -> None:
        """Write per-class metrics sheet."""
        df = pd.DataFrame(per_class_metrics)

        # Round floats
        for col in ["precision", "recall", "f1"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

        df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)
        df.to_csv(self.metrics_csv_dir / "per_class_metrics.csv", index=False)

    def _write_confusion_matrix_sheet(
        self,
        writer: pd.ExcelWriter,
        cm_data: Dict[str, Any],
    ) -> None:
        """Write confusion matrix sheet."""
        labels = cm_data["labels"]
        matrix = np.array(cm_data["matrix"])

        # Create DataFrame with row and column labels
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.index.name = "True \\ Predicted"

        df.to_excel(writer, sheet_name="Confusion_Matrix")
        df.to_csv(self.metrics_csv_dir / "confusion_matrix.csv")

    def _write_roc_auc_sheet(
        self,
        writer: pd.ExcelWriter,
        roc_data: Dict[str, Any],
    ) -> None:
        """Write ROC AUC summary sheet."""
        rows = []

        # Per-class AUCs
        if "per_class" in roc_data:
            for item in roc_data["per_class"]:
                auc_val = item.get("auc", np.nan)
                rows.append(
                    {
                        "Class": item["class"],
                        "AUC": f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A",
                        "Note": item.get("note", ""),
                    }
                )

        # Macro/micro averages
        if "macro" in roc_data:
            val = roc_data["macro"]
            rows.append(
                {
                    "Class": "MACRO-AVERAGE",
                    "AUC": f"{val:.4f}" if not np.isnan(val) else "N/A",
                    "Note": "Average of per-class AUCs",
                }
            )

        if "micro" in roc_data:
            val = roc_data["micro"]
            rows.append(
                {
                    "Class": "MICRO-AVERAGE",
                    "AUC": f"{val:.4f}" if not np.isnan(val) else "N/A",
                    "Note": "AUC of aggregated predictions",
                }
            )

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="ROC_AUC_Summary", index=False)
        df.to_csv(self.metrics_csv_dir / "roc_auc_summary.csv", index=False)

    def _write_task_metrics_sheet(
        self,
        writer: pd.ExcelWriter,
        task_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """Write binary task-level metrics sheet."""
        rows = []

        for task_name, metrics in task_metrics.items():
            row = {"Task": task_name}
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    row[metric_name] = f"{value:.4f}" if not np.isnan(value) else "N/A"
                else:
                    row[metric_name] = value
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Task_Level_Metrics", index=False)
        df.to_csv(self.metrics_csv_dir / "task_level_metrics.csv", index=False)

    def _write_hierarchical_sheets(
        self,
        writer: pd.ExcelWriter,
        hier_metrics: Dict[str, Any],
    ) -> None:
        """Write hierarchical-specific sheets."""
        # L1 metrics
        if "L1" in hier_metrics:
            self._write_level_metrics(writer, hier_metrics["L1"], "L1")

        # L2 metrics
        if "L2" in hier_metrics:
            self._write_level_metrics(writer, hier_metrics["L2"], "L2")

        # Pipeline metrics
        if "pipeline" in hier_metrics:
            self._write_level_metrics(writer, hier_metrics["pipeline"], "Pipeline")

    def _write_level_metrics(
        self,
        writer: pd.ExcelWriter,
        metrics: Dict[str, Any],
        level: str,
    ) -> None:
        """Write metrics for a specific hierarchical level."""
        # Overall
        if "overall" in metrics:
            df = self._metrics_to_dataframe(metrics["overall"])
            sheet_name = f"{level}_Overall"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            df.to_csv(self.metrics_csv_dir / f"{level.lower()}_overall.csv", index=False)

        # Per-class
        if "per_class" in metrics:
            df = pd.DataFrame(metrics["per_class"])
            sheet_name = f"{level}_Per_Class"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            df.to_csv(self.metrics_csv_dir / f"{level.lower()}_per_class.csv", index=False)

    def _metrics_to_dataframe(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Convert metrics dict to DataFrame."""
        rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
                else:
                    value_str = str(value)
                rows.append({"Metric": key, "Value": value_str})

        return pd.DataFrame(rows)

    def _write_csv_fallback(
        self,
        run_info: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: Optional[pd.DataFrame],
    ) -> None:
        """Write CSV files when Excel fails."""
        logger.info("Writing CSV fallback files")

        # Write individual metric files
        if "overall" in metrics:
            df = self._metrics_to_dataframe(metrics["overall"])
            df.to_csv(self.metrics_csv_dir / "overall_metrics.csv", index=False)

        # ... (similar for other sections)
