"""Reporting utilities for project runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _write_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_technical_report(
    outdir: Path,
    metrics_summary: Dict[str, float],
    per_fold: Dict[str, List[float]],
    notes: List[str],
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
        "",
        "## Per-fold Metrics",
        _write_markdown_table(fold_df),
    ]
    cal_metrics = [
        row
        for row in summary_df.to_dict("records")
        if row["metric"] in {
            "brier",
            "brier_calibrated",
            "ece",
            "ece_calibrated",
            "log_loss",
            "log_loss_calibrated",
            "calibration_bins",
        }
    ]
    if cal_metrics:
        md_lines.extend(["", "## Calibration Summary"])
        cal_df = pd.DataFrame(cal_metrics)
        md_lines.append(_write_markdown_table(cal_df))

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
        if key in {
            "brier",
            "brier_calibrated",
            "ece",
            "ece_calibrated",
            "log_loss",
            "log_loss_calibrated",
            "calibration_bins",
        }
    ]
    if cal_rows:
        md_lines.extend(["", "## Calibration Metrics"])
        cal_df = pd.DataFrame(cal_rows)
        md_lines.append(_write_markdown_table(cal_df))
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
    promotion_gate_cols = ["phase", "metric", "op", "threshold", "observed_value", "passed", "scope", "aggregation"]
    if not gating_metrics.empty:
        available = [col for col in promotion_gate_cols if col in gating_metrics.columns]
        md_lines.extend(
            [
                "### Gate Table",
                _write_markdown_table(gating_metrics[available]),
                "",
            ]
        )
    md_lines.extend([
        "## Metrics Used for Promotion Decision",
        _write_markdown_table(gating_metrics),
        "",
        "## Supporting Metrics (Reported Only)",
        _write_markdown_table(report_only_metrics),
        "",
        "## Calibration Summary",
    ])
    cal_table = pd.DataFrame([])
    if not gating_metrics.empty:
        cal_table = gating_metrics[gating_metrics["metric"].isin(["brier_calibrated", "ece_calibrated"])]
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
    md_lines.extend([
        "",
        "## Gate Evaluation",
        _write_markdown_table(gate_table),
    ])
    if reasons:
        md_lines.extend(["", "## Reasons"])
        md_lines.extend([f"- {reason}" for reason in reasons])

    path = outdir / "promotion_report.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path
