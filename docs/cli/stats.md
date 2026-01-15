# stats

Statistical analysis and visualization commands.

## Subcommands

| Command | Description |
|---------|-------------|
| `stats run` | Run statistical analysis |
| `stats viz` | Generate visualizations |

---

## stats run

Run statistical tests and generate analysis reports.

### Usage

```bash
classiflow stats run [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV with features and labels |
| `--label-col TEXT` | Label column name |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | `derived/stats` | Output directory |
| `--classes TEXT...` | All | Subset of classes to analyze |
| `--alpha FLOAT` | 0.05 | Significance level |
| `--min-n INT` | 3 | Minimum samples per group |
| `--dunn-adjust TEXT` | `bonferroni` | Dunn test adjustment |
| `--top-n INT` | 50 | Top features for reports |

### Example

```bash
classiflow stats run \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --outdir derived/stats \
  --alpha 0.05
```

### Outputs

```
outdir/
├── stats_summary.csv         # All test results
├── stats_workbook.xlsx       # Publication-ready workbook
├── normality_tests.csv       # Shapiro-Wilk results
├── pairwise_tests.csv        # Pairwise comparisons
└── effect_sizes.csv          # Effect size calculations
```

---

## stats viz

Generate statistical visualizations.

### Usage

```bash
classiflow stats viz [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV with features |
| `--label-col TEXT` | Label column name |
| `--stats-dir PATH` | Directory with stats results |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | Same as stats-dir | Output directory |
| `--dpi INT` | 300 | Figure resolution |
| `--point-size FLOAT` | 20 | Scatter point size |
| `--alpha-points FLOAT` | 0.7 | Point transparency |

### Volcano Plot Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fc-thresh FLOAT` | 1.0 | Log2 fold change threshold |
| `--label-topk INT` | 10 | Top features to label |

### Boxplot Options

| Option | Default | Description |
|--------|---------|-------------|
| `--boxplot-ncols INT` | 4 | Columns per row |

### Heatmap Options

| Option | Default | Description |
|--------|---------|-------------|
| `--heatmap-topn INT` | 50 | Features in heatmap |

### Example

```bash
classiflow stats viz \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --stats-dir derived/stats \
  --outdir derived/stats/plots \
  --dpi 300
```

### Outputs

```
outdir/
├── volcano_plot.png          # Volcano plot
├── boxplots/                 # Per-feature boxplots
│   ├── feature_1.png
│   └── ...
├── heatmap.png               # Feature heatmap
└── effect_sizes.png          # Effect size summary
```

---

## Statistical Tests Performed

### Binary Comparisons

| Test | When Used |
|------|-----------|
| Welch's t-test | Normal data |
| Mann-Whitney U | Non-normal data |

### Multiclass Comparisons

| Test | When Used |
|------|-----------|
| One-way ANOVA | Normal data |
| Kruskal-Wallis | Non-normal data |
| Tukey HSD | Post-hoc (normal) |
| Dunn's test | Post-hoc (non-normal) |

### Effect Sizes

| Measure | Type |
|---------|------|
| Cohen's d | Binary |
| Cliff's delta | Binary (non-parametric) |
| Log2 fold change | Expression data |
| Eta-squared | Multiclass |

---

## See Also

- [Statistics Module Docs](../docs/STATS_MODULE_DOCUMENTATION.md)
- [Publication Figures Tutorial](../tutorials/publication-figures.md)
