# Customize Plots

This guide covers modifying plot styles, colors, and layouts for publications.

## Default Plot Style

Classiflow generates plots using matplotlib with sensible defaults. Customize for your journal's requirements.

## Global Style Settings

```python
import matplotlib.pyplot as plt

# Publication-quality defaults
plt.rcParams.update({
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Figure settings
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "png",

    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    # Axis settings
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```

## Color Palettes

### Colorblind-Safe Palette

```python
# Wong palette (colorblind-safe)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "cyan": "#56B4E9",
    "red": "#D55E00",
    "black": "#000000",
}

# For sequential data
from matplotlib.colors import LinearSegmentedColormap
blues = plt.cm.Blues
greens = plt.cm.Greens
```

### Using Custom Colors

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# ROC curve with custom color
ax.plot(fpr, tpr, color=COLORS["blue"], linewidth=2, label="Model")
ax.plot([0, 1], [0, 1], color=COLORS["black"], linestyle="--", linewidth=1)
ax.fill_between(fpr, 0, tpr, color=COLORS["cyan"], alpha=0.2)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
```

## ROC Curve Customization

```python
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_publication(y_true, y_score, ax=None, color="#0072B2",
                         label=None, show_ci=True, n_bootstrap=100):
    """Publication-quality ROC curve with confidence interval."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Main ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Bootstrap for CI
    if show_ci:
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
            tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))

        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)

        ax.fill_between(mean_fpr,
                       np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                       np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                       color=color, alpha=0.2)

    # Plot main curve
    curve_label = f"{label} (AUC = {roc_auc:.3f})" if label else f"AUC = {roc_auc:.3f}"
    ax.plot(fpr, tpr, color=color, linewidth=2, label=curve_label)

    # Reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7)

    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)

    return ax
```

## Confusion Matrix Customization

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_publication(y_true, y_pred, labels, ax=None,
                                      cmap="Blues", normalize=False):
    """Publication-quality confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5, linecolor="white")

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    # Rotate labels if needed
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return ax
```

## Multi-Panel Figures

```python
def create_results_figure(results_dict, save_path=None):
    """Create a publication-ready multi-panel figure."""

    fig = plt.figure(figsize=(12, 8))

    # Define grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: ROC
    ax_a = fig.add_subplot(gs[0, 0])
    plot_roc_publication(
        results_dict["y_true"],
        results_dict["y_score"],
        ax=ax_a,
        label="Model"
    )
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=14)

    # Panel B: PR Curve
    ax_b = fig.add_subplot(gs[0, 1])
    # ... add PR curve
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=14)

    # Panel C: Confusion Matrix
    ax_c = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix_publication(
        results_dict["y_true"],
        results_dict["y_pred"],
        results_dict["labels"],
        ax=ax_c
    )
    ax_c.set_title("C", loc="left", fontweight="bold", fontsize=14)

    # Panel D: Metrics Bar
    ax_d = fig.add_subplot(gs[1, 1])
    # ... add metrics
    ax_d.set_title("D", loc="left", fontweight="bold", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig
```

## Export Formats

```python
# PNG for web/preview (raster)
plt.savefig("figure.png", dpi=300, bbox_inches="tight")

# PDF for publications (vector)
plt.savefig("figure.pdf", bbox_inches="tight")

# SVG for editing (vector)
plt.savefig("figure.svg", bbox_inches="tight")

# TIFF for some journals
plt.savefig("figure.tiff", dpi=300, bbox_inches="tight", format="tiff")
```

## Journal-Specific Requirements

| Journal | Format | Resolution | Max Width |
|---------|--------|------------|-----------|
| Nature | PDF/TIFF | 300 DPI | 180 mm |
| Science | PDF/EPS | 300 DPI | 227 mm |
| PLOS | TIFF/PDF | 300 DPI | 174 mm |
| IEEE | PDF/EPS | 300 DPI | 3.5 in (single) |

## Best Practices

!!! tip "Vector Formats"
    Use PDF or SVG for line plots. They scale without quality loss.

!!! tip "Font Consistency"
    Use Arial or Helvetica for journal compatibility.

!!! warning "Color in Print"
    If published in grayscale, ensure plots are distinguishable by pattern, not just color.

!!! note "Accessibility"
    Use colorblind-safe palettes and ensure sufficient contrast.
