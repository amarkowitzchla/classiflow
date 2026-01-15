"""
SMOTE vs No-SMOTE Comparison Analysis

This module provides comprehensive tools for comparing SMOTE and no-SMOTE variants
to assess whether SMOTE improves model performance without overfitting.

Key Features:
- Statistical comparison (paired t-tests, effect sizes, Wilcoxon tests)
- Publication-ready plots (delta charts, scatter plots, distributions)
- Overfitting detection (concurrent performance drops)
- Detailed reports for reviewers
- Support for binary, meta, and hierarchical model types

Usage:
    from classiflow.evaluation.smote_comparison import SMOTEComparison

    comparison = SMOTEComparison.from_directory("derived/results")
    report = comparison.generate_report()
    comparison.create_all_plots(outdir="smote_analysis")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from classiflow.evaluation.smote_plots import create_all_plots


@dataclass
class SMOTEComparisonResult:
    """Results from SMOTE vs no-SMOTE comparison."""

    # Dataset metadata
    model_type: Literal["binary", "meta", "hierarchical"]
    n_folds: int
    n_tasks: Optional[int] = None  # For binary/meta
    n_branches: Optional[int] = None  # For hierarchical L2

    # Per-metric comparisons
    metrics: List[str] = field(default_factory=list)
    smote_means: Dict[str, float] = field(default_factory=dict)
    no_smote_means: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)  # smote - no_smote

    # Statistical tests
    paired_t_pvalues: Dict[str, float] = field(default_factory=dict)
    wilcoxon_pvalues: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)  # Cohen's d

    # Overfitting analysis
    overfitting_detected: bool = False
    overfitting_metrics: List[str] = field(default_factory=list)
    overfitting_reason: Optional[str] = None

    # Per-task/class breakdowns
    per_task_deltas: Optional[pd.DataFrame] = None  # For binary/meta
    per_class_deltas: Optional[pd.DataFrame] = None  # For meta/hierarchical

    # Recommendation
    recommendation: Literal["use_smote", "no_smote", "equivalent", "insufficient_data"] = "insufficient_data"
    confidence: Literal["high", "medium", "low"] = "low"
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "model_type": self.model_type,
            "n_folds": self.n_folds,
            "n_tasks": self.n_tasks,
            "n_branches": self.n_branches,
            "metrics": self.metrics,
            "smote_means": self.smote_means,
            "no_smote_means": self.no_smote_means,
            "deltas": self.deltas,
            "paired_t_pvalues": self.paired_t_pvalues,
            "wilcoxon_pvalues": self.wilcoxon_pvalues,
            "effect_sizes": self.effect_sizes,
            "overfitting_detected": self.overfitting_detected,
            "overfitting_metrics": self.overfitting_metrics,
            "overfitting_reason": self.overfitting_reason,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "  SMOTE VS NO-SMOTE COMPARISON SUMMARY",
            "=" * 70,
            f"\nModel Type: {self.model_type.upper()}",
            f"Folds: {self.n_folds}",
        ]

        if self.n_tasks:
            lines.append(f"Tasks: {self.n_tasks}")
        if self.n_branches:
            lines.append(f"L2 Branches: {self.n_branches}")

        lines.append("\n" + "-" * 70)
        lines.append("PERFORMANCE COMPARISON")
        lines.append("-" * 70)

        for metric in self.metrics:
            smote = self.smote_means.get(metric, np.nan)
            no_smote = self.no_smote_means.get(metric, np.nan)
            delta = self.deltas.get(metric, np.nan)
            p_val = self.paired_t_pvalues.get(metric, np.nan)
            effect = self.effect_sizes.get(metric, np.nan)

            sig = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""

            lines.append(f"\n{metric}:")
            lines.append(f"  SMOTE:     {smote:.4f}")
            lines.append(f"  No-SMOTE:  {no_smote:.4f}")
            lines.append(f"  Δ (SMOTE - No-SMOTE): {delta:+.4f}{sig}")
            lines.append(f"  p-value:   {p_val:.4f}")
            lines.append(f"  Cohen's d: {effect:.3f}")

        lines.append("\n" + "-" * 70)
        lines.append("OVERFITTING ANALYSIS")
        lines.append("-" * 70)

        if self.overfitting_detected:
            lines.append(f"\n⚠️  OVERFITTING DETECTED in: {', '.join(self.overfitting_metrics)}")
            lines.append(f"Reason: {self.overfitting_reason}")
        else:
            lines.append("\n✓ No overfitting detected")

        lines.append("\n" + "-" * 70)
        lines.append("RECOMMENDATION")
        lines.append("-" * 70)

        rec_emoji = {
            "use_smote": "✓",
            "no_smote": "✗",
            "equivalent": "~",
            "insufficient_data": "?"
        }

        lines.append(f"\n{rec_emoji.get(self.recommendation, '?')} {self.recommendation.upper().replace('_', ' ')}")
        lines.append(f"Confidence: {self.confidence.upper()}")
        lines.append("\nReasoning:")
        for reason in self.reasoning:
            lines.append(f"  • {reason}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


class SMOTEComparison:
    """
    Comprehensive SMOTE vs no-SMOTE comparison analysis.

    Loads training results from classiflow output directories and performs:
    - Statistical comparisons (paired tests, effect sizes)
    - Overfitting detection (concurrent performance drops)
    - Publication-ready visualizations
    - Detailed text and JSON reports

    Examples:
        # From directory with fold1/, fold2/, fold3/
        comparison = SMOTEComparison.from_directory("derived/results")

        # Generate report
        result = comparison.generate_report(
            primary_metric="f1",
            overfitting_threshold=0.03
        )

        # Create plots
        comparison.create_all_plots(outdir="smote_analysis")

        # Save report
        comparison.save_report(result, "smote_analysis")
    """

    def __init__(
        self,
        smote_data: pd.DataFrame,
        no_smote_data: pd.DataFrame,
        model_type: Literal["binary", "meta", "hierarchical"],
    ):
        """
        Initialize SMOTE comparison.

        Args:
            smote_data: DataFrame with SMOTE results (must have 'fold' column)
            no_smote_data: DataFrame with no-SMOTE results (must have 'fold' column)
            model_type: Type of model (binary, meta, or hierarchical)
        """
        self.smote_data = smote_data
        self.no_smote_data = no_smote_data
        self.model_type = model_type

        # Validate
        if "fold" not in smote_data.columns or "fold" not in no_smote_data.columns:
            raise ValueError("Both SMOTE and no-SMOTE data must have 'fold' column")

        self.n_folds = len(smote_data["fold"].unique())

        # Infer metric columns (numeric columns excluding identifiers)
        exclude_cols = {"fold", "task", "model_name", "sampler", "phase", "n", "pos_rate", "variant"}
        self.metric_columns = [
            col for col in smote_data.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(smote_data[col])
        ]

    @classmethod
    def from_directory(
        cls,
        result_dir: Union[str, Path],
        model_type: Optional[Literal["binary", "meta", "hierarchical"]] = None,
        metric_file: str = "metrics_outer_meta_eval.csv",
    ) -> SMOTEComparison:
        """
        Load results from classiflow output directory.

        Supports two directory structures:
        1. Fold subdirectories: result_dir/fold1/metrics.csv, fold2/metrics.csv, ...
        2. Combined file: result_dir/metrics.csv (with 'fold' column)

        Args:
            result_dir: Directory containing fold subdirectories OR combined metrics file
            model_type: Model type (auto-detected if None)
            metric_file: Name of metrics CSV file to load

        Returns:
            SMOTEComparison instance
        """
        result_dir = Path(result_dir)

        # Check for combined metrics file at top level first
        combined_file = result_dir / metric_file
        if combined_file.exists():
            # Load combined file
            combined = pd.read_csv(combined_file)

            # Verify it has fold column
            if "fold" not in combined.columns:
                raise ValueError(
                    f"Metrics file {combined_file} exists but lacks 'fold' column. "
                    "Expected either fold subdirectories (fold1/, fold2/, ...) or a combined "
                    "CSV with 'fold' column."
                )

            # Auto-detect model type if needed
            if model_type is None:
                if (result_dir / "fold1" / "binary_smote").exists() or \
                   "binary_smote" in str(combined.get("model_dir", "").iloc[0] if "model_dir" in combined.columns else ""):
                    model_type = "meta"
                elif (result_dir / "fold1" / "hierarchical_l1").exists():
                    model_type = "hierarchical"
                else:
                    model_type = "meta"  # Default for combined files
        else:
            # Original behavior: load from fold subdirectories
            # Auto-detect model type
            if model_type is None:
                if (result_dir / "fold1" / "binary_smote").exists():
                    model_type = "meta"
                elif (result_dir / "fold1" / "hierarchical_l1").exists():
                    model_type = "hierarchical"
                else:
                    model_type = "binary"

            # Load metrics from each fold
            dfs = []
            for fold_dir in sorted(result_dir.glob("fold*")):
                if not fold_dir.is_dir():
                    continue

                metric_path = fold_dir / metric_file
                if not metric_path.exists():
                    # Try alternative names
                    for alt_name in [
                        "metrics_outer_binary_eval.csv",
                        "metrics_inner_cv.csv",
                        "metrics.csv"
                    ]:
                        alt_path = fold_dir / alt_name
                        if alt_path.exists():
                            metric_path = alt_path
                            break

                if not metric_path.exists():
                    warnings.warn(f"No metrics file found in {fold_dir}")
                    continue

                df = pd.read_csv(metric_path)
                df["fold"] = int(fold_dir.name.replace("fold", ""))
                dfs.append(df)

            if not dfs:
                raise FileNotFoundError(
                    f"No metrics files found in {result_dir}. "
                    f"Expected either:\n"
                    f"  1. Fold subdirectories: {result_dir}/fold1/{metric_file}, fold2/..., etc.\n"
                    f"  2. Combined file: {result_dir}/{metric_file} (with 'fold' column)"
                )

            combined = pd.concat(dfs, ignore_index=True)

        # Split by sampler variant
        if "sampler" in combined.columns:
            smote_data = combined[combined["sampler"] == "smote"].copy()
            no_smote_data = combined[combined["sampler"] == "none"].copy()
        else:
            raise ValueError("Could not identify SMOTE variants in data (no 'sampler' column)")

        if smote_data.empty or no_smote_data.empty:
            raise ValueError("Missing SMOTE or no-SMOTE variant data")

        return cls(smote_data, no_smote_data, model_type)

    def compute_statistics(
        self,
        metrics: Optional[List[str]] = None,
        aggregate_by: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical comparisons between SMOTE and no-SMOTE.

        Args:
            metrics: List of metrics to compare (all numeric if None)
            aggregate_by: Columns to aggregate by before comparison (e.g., ["task"])

        Returns:
            Dictionary with statistics per metric
        """
        if metrics is None:
            metrics = self.metric_columns

        results = {}

        for metric in metrics:
            if metric not in self.smote_data.columns or metric not in self.no_smote_data.columns:
                continue

            # Aggregate data if requested
            if aggregate_by:
                smote_agg = self.smote_data.groupby(aggregate_by + ["fold"])[metric].mean().reset_index()
                no_smote_agg = self.no_smote_data.groupby(aggregate_by + ["fold"])[metric].mean().reset_index()

                # Average across tasks/classes per fold for paired test
                smote_vals = smote_agg.groupby("fold")[metric].mean().values
                no_smote_vals = no_smote_agg.groupby("fold")[metric].mean().values
            else:
                # Per-fold averages
                smote_vals = self.smote_data.groupby("fold")[metric].mean().values
                no_smote_vals = self.no_smote_data.groupby("fold")[metric].mean().values

            # Remove NaN pairs
            valid = ~(np.isnan(smote_vals) | np.isnan(no_smote_vals))
            smote_vals = smote_vals[valid]
            no_smote_vals = no_smote_vals[valid]

            # Need at least 1 value for means, 2 for statistical tests
            if len(smote_vals) == 0:
                results[metric] = {
                    "smote_mean": np.nan,
                    "no_smote_mean": np.nan,
                    "delta": np.nan,
                    "paired_t_pval": np.nan,
                    "wilcoxon_pval": np.nan,
                    "cohens_d": np.nan,
                }
                continue

            # Descriptive stats
            smote_mean = float(np.mean(smote_vals))
            no_smote_mean = float(np.mean(no_smote_vals))
            delta = smote_mean - no_smote_mean

            # Statistical tests require ≥2 samples
            if len(smote_vals) < 2:
                results[metric] = {
                    "smote_mean": smote_mean,
                    "no_smote_mean": no_smote_mean,
                    "delta": delta,
                    "paired_t_pval": np.nan,
                    "wilcoxon_pval": np.nan,
                    "cohens_d": np.nan,
                }
                continue

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(smote_vals, no_smote_vals)

            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_pval = stats.wilcoxon(smote_vals, no_smote_vals)
            except ValueError:
                w_pval = np.nan

            # Cohen's d (effect size)
            diffs = smote_vals - no_smote_vals
            cohens_d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10))

            results[metric] = {
                "smote_mean": smote_mean,
                "no_smote_mean": no_smote_mean,
                "delta": delta,
                "paired_t_pval": float(t_pval),
                "wilcoxon_pval": float(w_pval) if not np.isnan(w_pval) else np.nan,
                "cohens_d": cohens_d,
            }

        return results

    def detect_overfitting(
        self,
        primary_metric: str = "f1",
        secondary_metric: str = "roc_auc",
        delta_threshold: float = 0.03,
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Detect overfitting by checking for concurrent performance drops.

        Overfitting is suspected if SMOTE causes meaningful drops in multiple metrics.

        Args:
            primary_metric: Primary metric to check (e.g., "f1")
            secondary_metric: Secondary metric to check (e.g., "roc_auc")
            delta_threshold: Minimum absolute drop to consider meaningful

        Returns:
            (overfitting_detected, affected_metrics, reason)
        """
        stats_dict = self.compute_statistics([primary_metric, secondary_metric])

        primary_delta = stats_dict.get(primary_metric, {}).get("delta", 0)
        secondary_delta = stats_dict.get(secondary_metric, {}).get("delta", 0)

        affected = []

        if primary_delta <= -delta_threshold:
            affected.append(primary_metric)
        if secondary_delta <= -delta_threshold:
            affected.append(secondary_metric)

        if len(affected) >= 2:
            reason = (
                f"Concurrent drops in {' and '.join(affected)} "
                f"(Δ{primary_metric}={primary_delta:.4f}, Δ{secondary_metric}={secondary_delta:.4f}) "
                f"suggest SMOTE may be overfitting"
            )
            return True, affected, reason

        return False, [], None

    def generate_recommendation(
        self,
        statistics: Dict[str, Dict[str, float]],
        primary_metric: str = "f1",
        significance_level: float = 0.05,
        min_effect_size: float = 0.2,
    ) -> Tuple[str, str, List[str]]:
        """
        Generate recommendation on whether to use SMOTE.

        Args:
            statistics: Output from compute_statistics()
            primary_metric: Primary metric for decision
            significance_level: p-value threshold
            min_effect_size: Minimum Cohen's d for meaningful difference

        Returns:
            (recommendation, confidence, reasoning)
        """
        if primary_metric not in statistics:
            return "insufficient_data", "low", ["Primary metric not available"]

        stats_pm = statistics[primary_metric]
        delta = stats_pm["delta"]
        pval = stats_pm["paired_t_pval"]
        cohens_d = stats_pm["cohens_d"]

        reasoning = []

        # Decision logic
        is_significant = pval < significance_level
        is_meaningful = abs(cohens_d) >= min_effect_size
        smote_better = delta > 0

        reasoning.append(f"Δ{primary_metric} = {delta:+.4f} (p={pval:.4f}, d={cohens_d:.3f})")

        if is_significant and is_meaningful:
            if smote_better:
                recommendation = "use_smote"
                confidence = "high"
                reasoning.append(f"SMOTE significantly improves {primary_metric} with meaningful effect size")
            else:
                recommendation = "no_smote"
                confidence = "high"
                reasoning.append(f"No-SMOTE significantly better with meaningful effect size")
        elif is_significant:
            if smote_better:
                recommendation = "use_smote"
                confidence = "medium"
                reasoning.append(f"SMOTE significantly improves {primary_metric} but effect size is small")
            else:
                recommendation = "no_smote"
                confidence = "medium"
                reasoning.append(f"No-SMOTE significantly better but effect size is small")
        elif is_meaningful:
            if smote_better:
                recommendation = "use_smote"
                confidence = "low"
                reasoning.append(f"SMOTE shows meaningful improvement but not statistically significant")
            else:
                recommendation = "no_smote"
                confidence = "low"
                reasoning.append(f"No-SMOTE shows meaningful improvement but not significant")
        else:
            recommendation = "equivalent"
            confidence = "high"
            reasoning.append(f"No significant or meaningful difference detected")

        # Check other metrics for consistency
        other_metrics = [m for m in statistics.keys() if m != primary_metric]
        concordant = 0
        discordant = 0

        for metric in other_metrics:
            m_delta = statistics[metric]["delta"]
            if (m_delta > 0 and delta > 0) or (m_delta < 0 and delta < 0):
                concordant += 1
            else:
                discordant += 1

        if other_metrics:
            reasoning.append(
                f"{concordant}/{len(other_metrics)} other metrics show same direction"
            )

            if discordant > concordant and confidence == "high":
                confidence = "medium"
                reasoning.append("Confidence reduced due to inconsistent results across metrics")

        return recommendation, confidence, reasoning

    def generate_report(
        self,
        primary_metric: str = "f1",
        secondary_metric: str = "roc_auc",
        overfitting_threshold: float = 0.03,
        significance_level: float = 0.05,
        min_effect_size: float = 0.2,
    ) -> SMOTEComparisonResult:
        """
        Generate comprehensive SMOTE comparison report.

        Args:
            primary_metric: Primary metric for recommendation
            secondary_metric: Secondary metric for overfitting detection
            overfitting_threshold: Minimum drop to flag overfitting
            significance_level: p-value threshold for significance
            min_effect_size: Minimum Cohen's d for meaningful effect

        Returns:
            SMOTEComparisonResult with all analyses
        """
        # Compute statistics
        stats_dict = self.compute_statistics()

        # Extract metrics
        metrics = list(stats_dict.keys())
        smote_means = {m: stats_dict[m]["smote_mean"] for m in metrics}
        no_smote_means = {m: stats_dict[m]["no_smote_mean"] for m in metrics}
        deltas = {m: stats_dict[m]["delta"] for m in metrics}
        paired_t_pvalues = {m: stats_dict[m]["paired_t_pval"] for m in metrics}
        wilcoxon_pvalues = {m: stats_dict[m]["wilcoxon_pval"] for m in metrics}
        effect_sizes = {m: stats_dict[m]["cohens_d"] for m in metrics}

        # Overfitting detection
        overfitting, overfit_metrics, overfit_reason = self.detect_overfitting(
            primary_metric, secondary_metric, overfitting_threshold
        )

        # Recommendation
        recommendation, confidence, reasoning = self.generate_recommendation(
            stats_dict, primary_metric, significance_level, min_effect_size
        )

        # Override recommendation if overfitting detected
        if overfitting:
            recommendation = "no_smote"
            confidence = "high"
            reasoning.insert(0, "OVERFITTING DETECTED - recommending no-SMOTE")

        # Count tasks/branches
        n_tasks = None
        n_branches = None
        if "task" in self.smote_data.columns:
            n_tasks = len(self.smote_data["task"].unique())
        if self.model_type == "hierarchical" and "l1_class" in self.smote_data.columns:
            n_branches = len(self.smote_data["l1_class"].unique())

        return SMOTEComparisonResult(
            model_type=self.model_type,
            n_folds=self.n_folds,
            n_tasks=n_tasks,
            n_branches=n_branches,
            metrics=metrics,
            smote_means=smote_means,
            no_smote_means=no_smote_means,
            deltas=deltas,
            paired_t_pvalues=paired_t_pvalues,
            wilcoxon_pvalues=wilcoxon_pvalues,
            effect_sizes=effect_sizes,
            overfitting_detected=overfitting,
            overfitting_metrics=overfit_metrics,
            overfitting_reason=overfit_reason,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
        )

    def save_report(
        self,
        result: SMOTEComparisonResult,
        outdir: Union[str, Path],
        prefix: str = "smote_comparison",
    ) -> Dict[str, Path]:
        """
        Save report to text and JSON files.

        Args:
            result: SMOTEComparisonResult from generate_report()
            outdir: Output directory
            prefix: File prefix

        Returns:
            Dictionary with paths to created files
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Text report
        txt_path = outdir / f"{prefix}_{timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write(result.summary_text())

        # JSON report
        json_path = outdir / f"{prefix}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # CSV summary
        summary_rows = []
        for metric in result.metrics:
            summary_rows.append({
                "metric": metric,
                "smote_mean": result.smote_means[metric],
                "no_smote_mean": result.no_smote_means[metric],
                "delta": result.deltas[metric],
                "paired_t_pvalue": result.paired_t_pvalues[metric],
                "wilcoxon_pvalue": result.wilcoxon_pvalues[metric],
                "cohens_d": result.effect_sizes[metric],
            })

        csv_path = outdir / f"{prefix}_summary_{timestamp}.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

        return {
            "txt": txt_path,
            "json": json_path,
            "csv": csv_path,
        }

    def create_all_plots(
        self,
        outdir: Union[str, Path],
        prefix: str = "smote_comparison",
    ) -> Dict[str, Path]:
        """
        Generate all publication-ready plots and save to directory.

        Args:
            outdir: Output directory
            prefix: File prefix for plots

        Returns:
            Dictionary mapping plot type to file path
        """
        # Compute statistics for plot annotations
        stats_dict = self.compute_statistics()
        deltas = {m: stats_dict[m]["delta"] for m in stats_dict.keys()}
        pvalues = {m: stats_dict[m]["paired_t_pval"] for m in stats_dict.keys()}

        # Create all plots
        return create_all_plots(
            self.smote_data,
            self.no_smote_data,
            self.metric_columns,
            outdir,
            deltas=deltas,
            pvalues=pvalues,
            prefix=prefix,
        )
