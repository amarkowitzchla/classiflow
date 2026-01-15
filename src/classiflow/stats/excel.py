"""Excel workbook generation with publication-ready formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import xlsxwriter


def autosize_column(ws: Any, df: pd.DataFrame, col_idx: int, col_name: str, min_width: int = 12, max_width: int = 50):
    """Set column width based on content.

    Args:
        ws: xlsxwriter worksheet object
        df: DataFrame with data
        col_idx: Column index (0-based)
        col_name: Column name
        min_width: Minimum column width
        max_width: Maximum column width
    """
    # Calculate width: max of header and content lengths
    header_len = len(str(col_name))
    content_len = df[col_name].astype(str).map(len).max() if len(df) > 0 else 0
    width = max(min_width, min(max_width, max(header_len, content_len) + 2))
    ws.set_column(col_idx, col_idx, width)


def write_sheet_with_formatting(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    formats: Dict[str, Any],
    conditional_formats: list[dict] | None = None,
):
    """Write a DataFrame to Excel with formatting.

    Args:
        writer: pandas ExcelWriter object (xlsxwriter engine)
        df: DataFrame to write
        sheet_name: Name of sheet
        formats: Dictionary of xlsxwriter format objects
        conditional_formats: Optional list of conditional format specs
    """
    if df.empty:
        return

    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    workbook = writer.book

    # Freeze header row
    ws.freeze_panes(1, 0)

    # Enable autofilter
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Autosize columns
    for i, col in enumerate(df.columns):
        autosize_column(ws, df, i, col)

    # Apply numeric formatting
    for col_idx, col_name in enumerate(df.columns):
        if col_name in ["p_value", "p_adj"]:
            # Scientific notation for p-values
            ws.set_column(col_idx, col_idx, None, formats["pvalue"])
        elif col_name in [
            "log2fc",
            "cohen_d",
            "cliff_delta",
            "rank_biserial",
            "mean",
            "sd",
            "median",
            "iqr",
            "W",
            "statistic",
            "mean_diff",
        ]:
            # 3 decimal places for effect sizes and stats
            ws.set_column(col_idx, col_idx, None, formats["decimal3"])
        elif col_name in ["alpha", "fc_eps"]:
            # Scientific notation for small values
            ws.set_column(col_idx, col_idx, None, formats["sci"])

    # Apply conditional formatting if specified
    if conditional_formats:
        for fmt_spec in conditional_formats:
            ws.conditional_format(
                fmt_spec["row_start"],
                fmt_spec["col_start"],
                fmt_spec["row_end"],
                fmt_spec["col_end"],
                fmt_spec["options"],
            )


def create_formats(workbook: Any) -> Dict[str, Any]:
    """Create xlsxwriter format objects.

    Args:
        workbook: xlsxwriter Workbook object

    Returns:
        Dictionary of format objects
    """
    return {
        "pvalue": workbook.add_format({"num_format": "0.00E+00"}),
        "sci": workbook.add_format({"num_format": "0.00E+00"}),
        "decimal3": workbook.add_format({"num_format": "0.000"}),
        "decimal2": workbook.add_format({"num_format": "0.00"}),
        "header": workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1}),
        "sig_highlight": workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C5700"}),
        "warning": workbook.add_format({"bg_color": "#F4B084"}),
    }


def write_publication_workbook(
    outdir: Path,
    run_manifest: pd.DataFrame,
    descriptives_by_class: pd.DataFrame,
    normality_summary: pd.DataFrame,
    normality_by_class: pd.DataFrame,
    parametric_overall: pd.DataFrame,
    parametric_posthoc: pd.DataFrame,
    nonparametric_overall: pd.DataFrame,
    nonparametric_posthoc: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    top_features_per_pair: pd.DataFrame,
    top_features_overall: pd.DataFrame,
) -> Path:
    """Write publication-ready Excel workbook with all results.

    Args:
        outdir: Output directory
        run_manifest: Run metadata
        descriptives_by_class: Descriptive stats by class
        normality_summary: Normality summary table
        normality_by_class: Normality detail table
        parametric_overall: Parametric test results
        parametric_posthoc: Parametric post-hoc results
        nonparametric_overall: Nonparametric test results
        nonparametric_posthoc: Nonparametric post-hoc results
        pairwise_summary: Pairwise comparison summary
        top_features_per_pair: Top features per pair
        top_features_overall: Top features overall

    Returns:
        Path to created workbook
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_xlsx = outdir / "publication_stats.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        formats = create_formats(writer.book)

        # Sheet 1: Run Manifest
        write_sheet_with_formatting(writer, run_manifest, "Run_Manifest", formats)

        # Sheet 2: Descriptives by Class
        write_sheet_with_formatting(writer, descriptives_by_class, "Descriptives_By_Class", formats)

        # Sheet 3-4: Normality
        write_sheet_with_formatting(writer, normality_summary, "Normality_Summary", formats)
        write_sheet_with_formatting(writer, normality_by_class, "Normality_By_Class", formats)

        # Sheet 5-6: Parametric
        write_sheet_with_formatting(writer, parametric_overall, "Omnibus_Parametric", formats)
        write_sheet_with_formatting(writer, parametric_posthoc, "PostHoc_Tukey", formats)

        # Sheet 7-8: Nonparametric
        write_sheet_with_formatting(writer, nonparametric_overall, "Omnibus_Nonparametric", formats)
        write_sheet_with_formatting(writer, nonparametric_posthoc, "PostHoc_Dunn", formats)

        # Sheet 9: Pairwise Summary (NEW)
        write_sheet_with_formatting(writer, pairwise_summary, "Pairwise_Summary", formats)

        # Sheet 10-11: Top Features (NEW)
        write_sheet_with_formatting(writer, top_features_per_pair, "Top_Features_Per_Pair", formats)
        write_sheet_with_formatting(writer, top_features_overall, "Top_Features_Overall", formats)

    return out_xlsx


def write_legacy_workbook(
    outdir: Path,
    normality_summary: pd.DataFrame,
    normality_by_class: pd.DataFrame,
    parametric_overall: pd.DataFrame,
    parametric_posthoc: pd.DataFrame,
    nonparametric_overall: pd.DataFrame,
    nonparametric_posthoc: pd.DataFrame,
) -> Path:
    """Write legacy stats_results.xlsx workbook for backward compatibility.

    Args:
        outdir: Output directory
        normality_summary: Normality summary table
        normality_by_class: Normality detail table
        parametric_overall: Parametric test results
        parametric_posthoc: Parametric post-hoc results
        nonparametric_overall: Nonparametric test results
        nonparametric_posthoc: Nonparametric post-hoc results

    Returns:
        Path to created workbook
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_xlsx = outdir / "stats_results.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        formats = create_formats(writer.book)

        # Match legacy sheet names
        write_sheet_with_formatting(writer, normality_summary, "Normality_Summary", formats)
        write_sheet_with_formatting(writer, normality_by_class, "Normality_By_Class", formats)

        if not parametric_overall.empty:
            write_sheet_with_formatting(writer, parametric_overall, "Parametric_Overall", formats)
        if not parametric_posthoc.empty:
            write_sheet_with_formatting(writer, parametric_posthoc, "Parametric_PostHoc", formats)

        if not nonparametric_overall.empty:
            write_sheet_with_formatting(writer, nonparametric_overall, "Nonparametric_Overall", formats)
        if not nonparametric_posthoc.empty:
            write_sheet_with_formatting(writer, nonparametric_posthoc, "Nonparametric_PostHoc", formats)

    return out_xlsx


def write_legacy_csvs(
    outdir: Path,
    normality_summary: pd.DataFrame,
    normality_by_class: pd.DataFrame,
    parametric_overall: pd.DataFrame,
    parametric_posthoc: pd.DataFrame,
    nonparametric_overall: pd.DataFrame,
    nonparametric_posthoc: pd.DataFrame,
):
    """Write legacy CSV files for backward compatibility.

    Args:
        outdir: Output directory
        normality_summary: Normality summary table
        normality_by_class: Normality detail table
        parametric_overall: Parametric test results
        parametric_posthoc: Parametric post-hoc results
        nonparametric_overall: Nonparametric test results
        nonparametric_posthoc: Nonparametric post-hoc results
    """
    outdir.mkdir(parents=True, exist_ok=True)

    normality_summary.to_csv(outdir / "Normality_Summary.csv", index=False)
    normality_by_class.to_csv(outdir / "Normality_By_Class.csv", index=False)

    if not parametric_overall.empty:
        parametric_overall.to_csv(outdir / "Parametric_Overall.csv", index=False)
    if not parametric_posthoc.empty:
        parametric_posthoc.to_csv(outdir / "Parametric_PostHoc.csv", index=False)

    if not nonparametric_overall.empty:
        nonparametric_overall.to_csv(outdir / "Nonparametric_Overall.csv", index=False)
    if not nonparametric_posthoc.empty:
        nonparametric_posthoc.to_csv(outdir / "Nonparametric_PostHoc.csv", index=False)
