"""CLI commands for statistical analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import typer

stats_app = typer.Typer(
    name="stats",
    help="Statistical analysis and visualization tools.",
    add_completion=False,
)


@stats_app.command("run")
def run_stats_cmd(
    data_csv: Path = typer.Option(..., "--data-csv", help="Path to CSV with features + labels"),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Subset/order of classes"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance threshold"),
    min_n: int = typer.Option(3, "--min-n", help="Minimum n per class for Shapiro-Wilk"),
    dunn_adjust: str = typer.Option("holm", "--dunn-adjust", help="P-value adjustment for Dunn test"),
    top_n: int = typer.Option(30, "--top-n", help="Number of top features in summary"),
    no_legacy_csv: bool = typer.Option(False, "--no-legacy-csv", help="Skip legacy CSV outputs"),
    no_legacy_xlsx: bool = typer.Option(False, "--no-legacy-xlsx", help="Skip legacy xlsx output"),
):
    """
    Run complete statistical analysis pipeline.

    Performs normality testing, parametric/nonparametric tests, and generates
    publication-ready Excel workbooks with pairwise comparisons and effect sizes.

    Examples:
        # Basic analysis
        classiflow stats run --data-csv data.csv --label-col diagnosis

        # Custom parameters
        classiflow stats run \\
            --data-csv data.csv \\
            --label-col diagnosis \\
            --alpha 0.01 \\
            --dunn-adjust fdr_bh \\
            --top-n 50
    """
    from classiflow.stats import run_stats

    try:
        typer.echo(f"Running statistical analysis on {data_csv}...")

        results = run_stats(
            data_csv=data_csv,
            label_col=label_col,
            outdir=outdir,
            classes=classes,
            alpha=alpha,
            min_n=min_n,
            dunn_adjust=dunn_adjust,
            top_n_features=top_n,
            write_legacy_csv=not no_legacy_csv,
            write_legacy_xlsx=not no_legacy_xlsx,
        )

        typer.secho(f"\n✓ Analysis complete!", fg=typer.colors.GREEN)
        typer.echo(f"  Publication workbook: {results['publication_xlsx']}")
        if results['legacy_xlsx']:
            typer.echo(f"  Legacy workbook: {results['legacy_xlsx']}")
        typer.echo(f"  Stats directory: {results['stats_dir']}")
        typer.echo(f"\n  Classes: {len(results['classes'])}")
        typer.echo(f"  Features: {results['n_features']}")
        typer.echo(f"  Samples: {results['n_samples']}")

    except Exception as e:
        typer.secho(f"\n✗ Analysis failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@stats_app.command("viz")
def run_viz_cmd(
    data_csv: Path = typer.Option(..., "--data-csv", help="Path to CSV with features + labels"),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory for visualizations"),
    stats_dir: Optional[Path] = typer.Option(None, "--stats-dir", help="Directory with stats results"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Subset/order of classes"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance threshold for volcano"),
    fc_thresh: float = typer.Option(1.0, "--fc-thresh", help="|log2FC| threshold for volcano"),
    fc_center: str = typer.Option("median", "--fc-center", help="Center for fold-change (mean/median)"),
    label_topk: int = typer.Option(12, "--label-topk", help="Top features to annotate on volcano"),
    heatmap_topn: int = typer.Option(30, "--heatmap-topn", help="Top features for heatmap (0=skip)"),
    fig_dpi: int = typer.Option(160, "--fig-dpi", help="Figure DPI"),
):
    """
    Generate statistical visualizations.

    Creates boxplots, fold-change bar plots, volcano plots, heatmaps,
    and dimensionality reduction plots (UMAP, t-SNE, LDA) from statistical
    analysis results.

    Examples:
        # Basic visualization
        classiflow stats viz --data-csv data.csv --label-col diagnosis

        # With stats results
        classiflow stats viz \\
            --data-csv data.csv \\
            --label-col diagnosis \\
            --stats-dir derived/stats_results
    """
    from classiflow.stats import run_visualizations

    try:
        typer.echo(f"Creating visualizations from {data_csv}...")

        results = run_visualizations(
            data_csv=data_csv,
            label_col=label_col,
            outdir=outdir,
            stats_dir=stats_dir,
            classes=classes,
            alpha=alpha,
            fc_thresh=fc_thresh,
            fc_center=fc_center,
            label_topk=label_topk,
            heatmap_topn=heatmap_topn,
            fig_dpi=fig_dpi,
        )

        typer.secho(f"\n✓ Visualizations complete!", fg=typer.colors.GREEN)
        typer.echo(f"  Output directory: {results['viz_dir']}")
        typer.echo(f"  Boxplots: {results['boxplots']['pdf']}")
        typer.echo(f"  Fold-change plots: {len(results['foldchange'])} files")
        typer.echo(f"  Volcano plots: {len(results['volcano'])} files")
        if results['heatmap']:
            typer.echo(f"  Heatmap: {results['heatmap']}")

        # Report projections
        if results.get('projections'):
            proj = results['projections']
            typer.echo(f"\n  Dimensionality reduction:")
            for method, path in proj.items():
                if path:
                    typer.echo(f"    • {method.upper()}: {path}")
                else:
                    typer.echo(f"    • {method.upper()}: skipped")

    except Exception as e:
        typer.secho(f"\n✗ Visualization failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@stats_app.command("umap")
def run_umap_cmd(
    data_csv: Path = typer.Option(..., "--data-csv", help="Path to CSV with features + labels"),
    label_col: str = typer.Option("MOLECULAR", "--label-col", help="Name of label column"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Subset/order of classes"),
    n_neighbors: int = typer.Option(15, "--n-neighbors", help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, "--min-dist", help="UMAP min_dist"),
    metric: str = typer.Option("euclidean", "--metric", help="UMAP metric"),
    random_state: int = typer.Option(40, "--random-state", help="Random seed"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardize features"),
    supervised: bool = typer.Option(False, "--supervised/--unsupervised", help="Use supervised UMAP"),
    pca_components: Optional[int] = typer.Option(None, "--pca-components", help="Optional PCA pre-reduction"),
):
    """
    Generate UMAP visualization.

    Creates 2D UMAP embedding with optional supervision and saves plot + coordinates.

    Examples:
        # Basic UMAP
        classiflow stats umap --data-csv data.csv --label-col diagnosis

        # Supervised UMAP with PCA
        classiflow stats umap \\
            --data-csv data.csv \\
            --label-col diagnosis \\
            --supervised \\
            --pca-components 50
    """
    typer.echo("UMAP visualization not yet implemented in stats API.")
    typer.echo("Please use the standalone script: scripts/umap_plot.py")
    typer.echo("\nExample:")
    typer.echo(f"  python scripts/umap_plot.py --data-csv {data_csv} --label-col {label_col}")
