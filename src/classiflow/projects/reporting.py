"""Reporting utilities for project runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classiflow.metrics.probability_quality_checks import (
    ProbQualityRuleResult,
    build_probability_quality_next_steps,
    collect_probability_quality_plot_payload,
    evaluate_probability_quality_checks,
)


def _write_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _format_evidence_paths(report_dir: Path, result: ProbQualityRuleResult) -> str:
    seen: List[str] = []
    for item in result.evidence:
        artifact = str(item.get("artifact", "")).strip()
        if not artifact:
            continue
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = report_dir.parent / artifact_path
        try:
            display = str(artifact_path.relative_to(report_dir))
        except ValueError:
            display = str(artifact_path)
        if display not in seen:
            seen.append(display)
    return ", ".join(seen) if seen else "-"


def _generate_probability_quality_plots(
    *,
    report_dir: Path,
    run_dir: Optional[Path],
    task_mode: Optional[str],
    probability_quality_check_thresholds: Optional[Dict[str, Any]],
    independent_metrics: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    if run_dir is None or task_mode is None:
        return []
    payload = collect_probability_quality_plot_payload(
        run_dir=run_dir,
        mode=task_mode,
        thresholds=probability_quality_check_thresholds,
        independent_metrics=independent_metrics,
    )
    if not payload:
        return []

    plots_dir = report_dir / "probability_quality_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    thresholds = payload.get("thresholds", {})
    generated: List[Dict[str, str]] = []

    curve_rows = payload.get("selected_curve", [])
    curve_kind = str(payload.get("curve_kind", "top1"))
    if curve_rows:
        curve_df = pd.DataFrame(curve_rows)
        if {"bin_id", "n", "mean_pred", "frac_pos"}.issubset(set(curve_df.columns)):
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)
            main = axes[0]
            occ = axes[1]
            main.plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", linewidth=1.25, label="Ideal")
            min_n = float(thresholds.get("low_bin_min_nonzero_n", 5))
            colors = ["#C0392B" if float(n) < min_n else "#1F77B4" for n in curve_df["n"]]
            sizes = np.clip(curve_df["n"].fillna(0).astype(float).to_numpy() * 6.0, 24, 420)
            valid = curve_df["mean_pred"].notna() & curve_df["frac_pos"].notna()
            if valid.any():
                main.scatter(
                    curve_df.loc[valid, "mean_pred"],
                    curve_df.loc[valid, "frac_pos"],
                    s=sizes[valid.to_numpy()],
                    c=np.array(colors)[valid.to_numpy()],
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.4,
                    label="Bins (size~n)",
                )
            main.set_xlim(0.0, 1.0)
            main.set_ylim(0.0, 1.0)
            main.set_ylabel("Observed Fraction Positive")
            main.set_title(f"{curve_kind} Reliability (Final Variant)")
            main.grid(alpha=0.25)
            main.legend(loc="lower right", fontsize=8)

            occ.bar(curve_df["bin_id"], curve_df["n"], color=colors, alpha=0.9)
            occ.axhline(
                min_n,
                color="#C0392B",
                linestyle="--",
                linewidth=1.4,
                label=f"min bin n ({min_n:g})",
            )
            occ.set_xlabel("Calibration Bin")
            occ.set_ylabel("Bin Count (n)")
            occ.set_title("Bin Occupancy With PQ-001 Threshold")
            occ.grid(axis="y", alpha=0.25)
            occ.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            reliability_path = plots_dir / f"prob_quality_reliability_{curve_kind}.png"
            fig.savefig(reliability_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            generated.append(
                {
                    "path": str(reliability_path.relative_to(report_dir)),
                    "caption": f"{curve_kind} reliability and occupancy thresholds (PQ-001).",
                }
            )

    final_metrics = payload.get("final_metrics_mean", {})
    if final_metrics:
        accuracy = final_metrics.get("accuracy_top1")
        confidence = final_metrics.get("mean_confidence_top1")
        gap = final_metrics.get("confidence_gap_top1")
        if all(v is not None for v in (accuracy, confidence, gap)):
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
            left, right = axes

            left.bar(
                ["Accuracy Top-1", "Mean Confidence Top-1"],
                [accuracy, confidence],
                color=["#2C7FB8", "#F39C12"],
            )
            left.set_ylim(0.0, 1.05)
            left.set_ylabel("Value")
            left.set_title("Accuracy vs Confidence")
            left.grid(axis="y", alpha=0.25)

            right.axhline(
                float(thresholds.get("underconfidence_warn_gap", -0.20)),
                color="#2980B9",
                linestyle="--",
                linewidth=1.2,
                label="Underconfidence WARN",
            )
            right.axhline(
                float(thresholds.get("underconfidence_info_gap", -0.10)),
                color="#3498DB",
                linestyle="--",
                linewidth=1.2,
                label="Underconfidence INFO",
            )
            right.axhline(
                float(thresholds.get("overconfidence_warn_gap", 0.05)),
                color="#F39C12",
                linestyle="--",
                linewidth=1.2,
                label="Overconfidence WARN",
            )
            right.axhline(
                float(thresholds.get("overconfidence_error_gap", 0.10)),
                color="#C0392B",
                linestyle="--",
                linewidth=1.2,
                label="Overconfidence ERROR",
            )
            right.axhline(0.0, color="black", linewidth=1.0)
            right.scatter([0.0], [gap], s=120, color="#34495E", zorder=3)
            right.set_xlim(-1.0, 1.0)
            right.set_xticks([])
            right.set_ylabel("confidence_gap_top1")
            right.set_title("Confidence Gap With Rule Thresholds")
            right.grid(axis="y", alpha=0.25)
            right.legend(loc="lower right", fontsize=7)

            fig.tight_layout()
            gap_path = plots_dir / "prob_quality_confidence_gap.png"
            fig.savefig(gap_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            generated.append(
                {
                    "path": str(gap_path.relative_to(report_dir)),
                    "caption": "Confidence-gap diagnostics with PQ-002/PQ-003 thresholds.",
                }
            )

    uncal = payload.get("uncal_metrics_mean", {})
    cal = payload.get("cal_metrics_mean", {})
    if uncal and cal:
        labels = []
        deltas = []
        thresholds_plot = []
        metric_threshold_pairs = [
            ("brier_recommended", "calibration_worsen_brier_delta"),
            ("log_loss", "calibration_worsen_log_loss_delta"),
            ("ece_top1", "calibration_worsen_ece_top1_delta"),
            ("ece_ovr_macro", "calibration_worsen_ece_ovr_delta"),
        ]
        for metric_name, threshold_key in metric_threshold_pairs:
            if metric_name not in uncal or metric_name not in cal:
                continue
            labels.append(metric_name)
            deltas.append(float(cal[metric_name]) - float(uncal[metric_name]))
            thresholds_plot.append(float(thresholds.get(threshold_key, 0.0)))
        if labels:
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9.0, 4.5))
            ax.bar(
                x, deltas, color="#6C5CE7", alpha=0.85, label="delta (calibrated - uncalibrated)"
            )
            for idx, threshold_val in enumerate(thresholds_plot):
                ax.hlines(
                    threshold_val,
                    idx - 0.35,
                    idx + 0.35,
                    colors="#C0392B",
                    linestyles="--",
                    linewidth=1.4,
                )
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel("Delta")
            ax.set_title("Calibration Helpfulness (PQ-004) With Regression Limits")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            delta_path = plots_dir / "prob_quality_calibration_deltas.png"
            fig.savefig(delta_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            generated.append(
                {
                    "path": str(delta_path.relative_to(report_dir)),
                    "caption": "Calibration deltas against allowed regressions (PQ-004).",
                }
            )

    return generated


def _render_probability_quality_checks(
    *,
    report_dir: Path,
    run_dir: Optional[Path],
    task_mode: Optional[str],
    probability_quality_check_thresholds: Optional[Dict[str, Any]],
    independent_metrics: Optional[Dict[str, Any]],
    promotion_gate_metrics: Optional[List[str]],
) -> List[str]:
    lines: List[str] = [
        "## Probability Quality Checks (ECE/Brier)",
        (
            "These checks are diagnostic and not promotion blockers by default. "
            "`ece_top1` evaluates Top-1 confidence calibration; `ece_ovr_macro` evaluates one-vs-rest class probability calibration."
        ),
    ]
    if task_mode == "binary":
        lines[1] = (
            "These checks are diagnostic and not promotion blockers by default. "
            "Binary calibration primarily uses positive-class probability (`ece_binary_pos`/`brier_binary`). "
            "Confidence-gap rules are interpreted as decision-confidence diagnostics."
        )
    if task_mode == "hierarchical":
        lines[1] = (
            "These checks are diagnostic and not promotion blockers by default. "
            "Top-1 metrics use argmax(proba); hierarchical postprocessing may change final labels and is tracked explicitly."
        )

    results: List[ProbQualityRuleResult] = []
    plot_refs: List[Dict[str, str]] = []
    if run_dir is not None and task_mode:
        results = evaluate_probability_quality_checks(
            run_dir=run_dir,
            mode=task_mode,
            thresholds=probability_quality_check_thresholds,
            independent_metrics=independent_metrics,
        )
        plot_refs = _generate_probability_quality_plots(
            report_dir=report_dir,
            run_dir=run_dir,
            task_mode=task_mode,
            probability_quality_check_thresholds=probability_quality_check_thresholds,
            independent_metrics=independent_metrics,
        )

    if plot_refs:
        lines.append("")
        lines.append("Diagnostic plots generated:")
        for item in plot_refs:
            lines.append(f"- `{item['path']}` - {item['caption']}")

    if results:
        table_rows = []
        for result in results:
            next_action = result.recommendations[0] if result.recommendations else "-"
            table_rows.append(
                {
                    "Severity": result.severity,
                    "Rule ID": result.rule_id,
                    "Finding": result.title,
                    "Evidence": _format_evidence_paths(report_dir, result),
                    "Recommended next action": next_action,
                }
            )
        lines.append("")
        lines.append(_write_markdown_table(pd.DataFrame(table_rows)))
        lines.append("")
        for result in results:
            lines.append(f"### {result.rule_id} - {result.title} ({result.severity})")
            lines.append(f"Condition: {result.summary}")
            lines.append(f"Measured: `{result.measured}`")
            lines.append(f"Thresholds: `{result.thresholds}`")
            lines.append("Evidence:")
            if result.evidence:
                for item in result.evidence:
                    artifact = str(item.get("artifact", "")).strip()
                    field = str(item.get("field", "")).strip()
                    note = str(item.get("note", "")).strip()
                    artifact_path = Path(artifact) if artifact else Path(".")
                    if artifact and not artifact_path.is_absolute():
                        artifact_path = report_dir.parent / artifact_path
                    try:
                        artifact_display = str(artifact_path.relative_to(report_dir))
                    except ValueError:
                        artifact_display = artifact or "-"
                    lines.append(f"- `{artifact_display}` :: `{field}` ({note})")
            else:
                lines.append("- (no evidence entries)")
            lines.append("Recommendations:")
            if result.recommendations:
                for rec in result.recommendations:
                    lines.append(f"- {rec}")
            else:
                lines.append("- (no recommendations)")
            lines.append("")
    else:
        lines.extend(
            [
                "",
                "_No probability-quality checks were triggered or required artifacts were unavailable._",
                "",
            ]
        )

    helper_steps = build_probability_quality_next_steps(results)
    if helper_steps:
        lines.append("Next steps decision helper:")
        for step in helper_steps:
            lines.append(f"- {step}")
        lines.append("")

    lines.append(
        "Promotion gate impact: None (unless user explicitly gates on probability-quality metrics)."
    )
    promotion_metrics = [m.lower() for m in (promotion_gate_metrics or [])]
    uses_prob_quality_gate = any(
        ("ece" in metric) or ("brier" in metric) for metric in promotion_metrics
    )
    if uses_prob_quality_gate:
        lines.append(
            "WARN: Promotion gates reference calibration metrics; ensure you use explicit keys "
            "(`ece_top1` vs `ece_ovr_macro`) and sufficient sample size."
        )

    return lines


def _render_binary_learner_health_report(
    *,
    report_dir: Path,
    run_dir: Optional[Path],
    task_mode: Optional[str],
) -> List[str]:
    if run_dir is None or task_mode != "meta":
        return []

    summary_path = run_dir / "binary_learners_metrics_summary.csv"
    by_fold_path = run_dir / "binary_learners_metrics_by_fold.csv"
    warnings_path = run_dir / "binary_learners_warnings.json"
    manifest_path = run_dir / "binary_learners_manifest.json"
    if not summary_path.exists() or not by_fold_path.exists() or not warnings_path.exists():
        return []

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return []
    try:
        warnings_payload = json.loads(warnings_path.read_text(encoding="utf-8"))
        warning_rows = list(warnings_payload.get("warnings", []))
    except Exception:
        warning_rows = []

    lines: List[str] = [
        "## Binary Learner Health Report (Meta Under-the-Hood)",
        (
            "Base one-vs-rest learners are internal components feeding the meta model. "
            "This section is diagnostic only and is not a promotion gate."
        ),
        "",
    ]

    display_rows: List[Dict[str, Any]] = []
    for row in summary_df.to_dict("records"):
        auc_mean = row.get("roc_auc_ovr_mean")
        auc_std = row.get("roc_auc_ovr_std")
        auc_text = "-"
        if pd.notna(auc_mean):
            auc_text = (
                f"{float(auc_mean):.3f}±{float(auc_std):.3f}"
                if pd.notna(auc_std)
                else f"{float(auc_mean):.3f}"
            )
        display_rows.append(
            {
                "Class": row.get("class_name"),
                "AUC mean±std": auc_text,
                "Pred+ rate (mean)": (
                    f"{float(row.get('predicted_positive_rate_0_5_mean')):.3f}"
                    if pd.notna(row.get("predicted_positive_rate_0_5_mean"))
                    else "-"
                ),
                "std(p_pos) (mean)": (
                    f"{float(row.get('p_std_mean')):.4f}"
                    if pd.notna(row.get("p_std_mean"))
                    else "-"
                ),
                "n_pos (mean)": (
                    f"{float(row.get('n_pos_mean')):.1f}"
                    if pd.notna(row.get("n_pos_mean"))
                    else "-"
                ),
                "Health status": row.get("health_status", "OK"),
            }
        )
    lines.append(_write_markdown_table(pd.DataFrame(display_rows)))

    lines.append("")
    lines.append("Warnings:")
    if warning_rows:
        warning_table_rows = []
        for item in warning_rows:
            evidence = ", ".join([f"`{p}`" for p in item.get("evidence_paths", [])]) or "-"
            warning_table_rows.append(
                {
                    "Severity": item.get("severity", "INFO"),
                    "Rule ID": item.get("rule_id", ""),
                    "Class": item.get("class_name", ""),
                    "Fold(s)": ", ".join(item.get("folds", [])) or "-",
                    "Finding": item.get("finding", ""),
                    "Recommended action": item.get("recommended_action", ""),
                    "Evidence paths": evidence,
                }
            )
        lines.append(_write_markdown_table(pd.DataFrame(warning_table_rows)))
    else:
        lines.append("_No BL warnings were triggered._")

    lines.append("")
    lines.append("Plots and artifacts:")
    if manifest_path.exists():
        lines.append(f"- `../{manifest_path.relative_to(report_dir.parent)}`")
    if by_fold_path.exists():
        lines.append(f"- `../{by_fold_path.relative_to(report_dir.parent)}`")
    if summary_path.exists():
        lines.append(f"- `../{summary_path.relative_to(report_dir.parent)}`")
    if warnings_path.exists():
        lines.append(f"- `../{warnings_path.relative_to(report_dir.parent)}`")

    ovr_plots = sorted((run_dir / "plots").glob("binary_ovr_roc_*.png"))
    for plot_path in ovr_plots:
        lines.append(f"- `../{plot_path.relative_to(report_dir.parent)}`")
    ovo_csv_path = run_dir / "ovo_auc_matrix.csv"
    if ovo_csv_path.exists():
        lines.append(f"- `../{ovo_csv_path.relative_to(report_dir.parent)}`")
    ovo_plot_path = run_dir / "plots" / "ovo_auc_matrix.png"
    if ovo_plot_path.exists():
        lines.append(f"- `../{ovo_plot_path.relative_to(report_dir.parent)}`")

    flagged = {
        str(item.get("class_name"))
        for item in warning_rows
        if str(item.get("severity", "")).upper() in {"WARN", "ERROR"}
    }
    lines.append("")
    if flagged:
        impacted = ", ".join(sorted(flagged))
        lines.append(
            "Impact on Meta Interpretation: WARN/ERROR findings in "
            f"`{impacted}` indicate potential probability interpretability drift, "
            "rare-class instability, and increased reliance on other learners."
        )
    else:
        lines.append(
            "Impact on Meta Interpretation: No WARN/ERROR binary learner findings were detected."
        )
    return lines


def write_technical_report(
    outdir: Path,
    metrics_summary: Dict[str, float],
    per_fold: Dict[str, List[float]],
    notes: List[str],
    calibration_decision: Dict[str, object] | None = None,
    run_dir: Path | None = None,
    task_mode: str | None = None,
    probability_quality_check_thresholds: Dict[str, Any] | None = None,
    independent_metrics: Dict[str, Any] | None = None,
    promotion_gate_metrics: List[str] | None = None,
) -> Path:
    """Write a markdown technical validation report."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, value in metrics_summary.items():
        rows.append({"metric": metric, "value": value})
    summary_df = pd.DataFrame(rows)

    fold_rows = []
    for metric, values in per_fold.items():
        for idx, value in enumerate(values, start=1):
            fold_rows.append({"metric": metric, "fold": idx, "value": value})
    fold_df = pd.DataFrame(fold_rows)

    md_lines = [
        "# Technical Validation Report",
        "",
        "## Summary Metrics",
        _write_markdown_table(summary_df),
    ]
    md_lines.extend(
        [
            "",
            *(
                _render_probability_quality_checks(
                    report_dir=outdir,
                    run_dir=run_dir,
                    task_mode=task_mode,
                    probability_quality_check_thresholds=probability_quality_check_thresholds,
                    independent_metrics=independent_metrics,
                    promotion_gate_metrics=promotion_gate_metrics,
                )
            ),
            "",
            "## Per-fold Metrics",
            _write_markdown_table(fold_df),
        ]
    )
    binary_health_lines = _render_binary_learner_health_report(
        report_dir=outdir,
        run_dir=run_dir,
        task_mode=task_mode,
    )
    if binary_health_lines:
        md_lines.extend(["", *binary_health_lines])
    cal_metrics = [
        row
        for row in summary_df.to_dict("records")
        if row["metric"]
        in {
            "brier",
            "brier_recommended",
            "brier_binary",
            "brier_multiclass_sum",
            "brier_multiclass_mean",
            "brier_calibrated",
            "ece",
            "ece_top1",
            "ece_binary_pos",
            "ece_ovr_macro",
            "ece_calibrated",
            "log_loss",
            "log_loss_calibrated",
            "pred_alignment_mismatch_rate",
            "calibration_bins",
        }
        or str(row["metric"]).endswith("_uncalibrated")
        or str(row["metric"]).endswith("_calibrated")
    ]
    if cal_metrics:
        md_lines.extend(["", "## Calibration Summary"])
        cal_df = pd.DataFrame(cal_metrics)
        md_lines.append(_write_markdown_table(cal_df))
        md_lines.append(
            "_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._"
        )
    if calibration_decision:
        reasons = calibration_decision.get("reasons") or []
        metrics_compared = calibration_decision.get("metrics_compared") or {}
        final_variant = calibration_decision.get("final_variant")
        decision = calibration_decision.get("decision")
        mismatch_rate = calibration_decision.get("pred_alignment_mismatch_rate")

        md_lines.extend(["", "## Probability Calibration Decision"])
        md_lines.append(
            f"Final variant: `{final_variant}` (`{decision}`). Calibration was "
            f"{'retained' if final_variant == 'calibrated' else 'disabled'} by policy."
        )
        if mismatch_rate is not None:
            md_lines.append(f"Prediction alignment mismatch rate: `{mismatch_rate}`.")
        if reasons:
            md_lines.append("")
            md_lines.extend([f"- {reason}" for reason in reasons])
        if metrics_compared:
            rows = []
            for metric_name in ("brier_recommended", "log_loss", "ece_top1", "ece_ovr_macro"):
                entry = metrics_compared.get(metric_name)
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    {
                        "metric": metric_name,
                        "uncalibrated": entry.get("uncal"),
                        "calibrated": entry.get("cal"),
                        "delta_cal_minus_uncal": entry.get("delta"),
                    }
                )
            if rows:
                md_lines.extend(["", _write_markdown_table(pd.DataFrame(rows))])

    if notes:
        md_lines.extend(["", "## Notes"])
        md_lines.extend([f"- {note}" for note in notes])

    path = outdir / "technical_validation_report.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path


def write_test_report(
    outdir: Path,
    metrics: Dict[str, float],
    notes: List[str],
) -> Path:
    """Write a markdown independent test report."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": metric, "value": value} for metric, value in metrics.items()]
    df = pd.DataFrame(rows)

    md_lines = [
        "# Independent Test Report",
        "",
        "## Metrics",
        _write_markdown_table(df),
    ]
    cal_rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if key
        in {
            "brier",
            "brier_recommended",
            "brier_binary",
            "brier_multiclass_sum",
            "brier_multiclass_mean",
            "brier_calibrated",
            "ece",
            "ece_top1",
            "ece_binary_pos",
            "ece_ovr_macro",
            "ece_calibrated",
            "log_loss",
            "log_loss_calibrated",
            "pred_alignment_mismatch_rate",
            "calibration_bins",
        }
    ]
    if cal_rows:
        md_lines.extend(["", "## Calibration Metrics"])
        cal_df = pd.DataFrame(cal_rows)
        md_lines.append(_write_markdown_table(cal_df))
        md_lines.append(
            "_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._"
        )
    if notes:
        md_lines.extend(["", "## Notes"])
        md_lines.extend([f"- {note}" for note in notes])

    path = outdir / "independent_test_report.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path


def write_promotion_report(
    outdir: Path,
    decision: bool,
    reasons: List[str],
    gate_table: pd.DataFrame,
    gating_metrics: pd.DataFrame,
    report_only_metrics: pd.DataFrame,
    promotion_template: Dict[str, object] | None = None,
    calibration_selection: Dict[str, str] | None = None,
) -> Path:
    """Write a markdown promotion report."""
    outdir.mkdir(parents=True, exist_ok=True)
    status = "PASS" if decision else "FAIL"
    md_lines = [
        "# Promotion Recommendation",
        "",
        f"**Decision:** {status}",
        "",
        "## Promotion Gate",
    ]
    if promotion_template:
        md_lines.extend(
            [
                f"- Template: {promotion_template.get('display_name', 'n/a')} (`{promotion_template.get('template_id', 'n/a')}`)",
                f"- Version: {promotion_template.get('version', 'n/a')}",
                f"- Source: {promotion_template.get('source', 'n/a')}",
                f"- Layman explanation: {promotion_template.get('layman_explanation', '')}",
                "",
            ]
        )
    promotion_gate_cols = [
        "phase",
        "metric",
        "op",
        "threshold",
        "observed_value",
        "passed",
        "scope",
        "aggregation",
    ]
    if not gating_metrics.empty:
        available = [col for col in promotion_gate_cols if col in gating_metrics.columns]
        md_lines.extend(
            [
                "### Gate Table",
                _write_markdown_table(gating_metrics[available]),
                "",
            ]
        )
    md_lines.extend(
        [
            "## Metrics Used for Promotion Decision",
            _write_markdown_table(gating_metrics),
            "",
            "## Supporting Metrics (Reported Only)",
            _write_markdown_table(report_only_metrics),
            "",
            "## Calibration Summary",
        ]
    )
    cal_table = pd.DataFrame([])
    if not gating_metrics.empty:
        cal_table = gating_metrics[
            gating_metrics["metric"].isin(["brier_calibrated", "ece_calibrated"])
        ]
    if not cal_table.empty:
        md_lines.append(_write_markdown_table(cal_table))
    if calibration_selection:
        selected = calibration_selection.get("method_selected")
        reason = calibration_selection.get("reason")
        if selected:
            md_lines.append(f"- Selected method: {selected}")
        if reason:
            md_lines.append(f"- Rationale: {reason}")
    else:
        md_lines.append("- Selected method: unavailable")
    md_lines.extend(
        [
            "",
            "## Gate Evaluation",
            _write_markdown_table(gate_table),
        ]
    )
    if reasons:
        md_lines.extend(["", "## Reasons"])
        md_lines.extend([f"- {reason}" for reason in reasons])

    path = outdir / "promotion_report.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path
