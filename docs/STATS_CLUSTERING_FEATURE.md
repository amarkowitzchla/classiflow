# Dimensionality Reduction Visualization Feature

## Overview

The stats module now includes **dimensionality reduction visualizations** (UMAP, t-SNE, and LDA) as part of the standard visualization pipeline. These clustering plots help visualize how well the classes separate in lower-dimensional space.

---

## Features

### Three Projection Methods

1. **UMAP (Uniform Manifold Approximation and Projection)**
   - Supervised mode (default): Uses class labels to guide projection
   - Preserves both local and global structure
   - Generally faster than t-SNE
   - Requires: `pip install umap-learn`

2. **t-SNE (t-distributed Stochastic Neighbor Embedding)**
   - Excellent for visualizing local structure
   - Perplexity parameter auto-adjusted based on sample size
   - Good for exploratory analysis

3. **LDA (Linear Discriminant Analysis)**
   - Supervised projection that maximizes class separation
   - Shows variance explained by each linear discriminant
   - Limited to (n_classes - 1) dimensions
   - Works well when classes are linearly separable

---

## Automatic Integration

### Included in `classiflow stats viz`

The projection plots are **automatically generated** when you run:

```bash
classiflow stats viz --data-csv data.csv --label-col diagnosis
```

No additional flags needed - UMAP, t-SNE, and LDA are created by default.

### Output Structure

```
derived/viz/projections/
├── umap_projection.png         # UMAP plot
├── umap_projection.svg         # UMAP vector graphic
├── umap_embedding.csv          # UMAP coordinates + labels
├── tsne_projection.png         # t-SNE plot
├── tsne_projection.svg         # t-SNE vector graphic
├── tsne_embedding.csv          # t-SNE coordinates + labels
├── lda_projection.png          # LDA plot
├── lda_projection.svg          # LDA vector graphic
└── lda_embedding.csv           # LDA coordinates + labels
```

---

## Python API

### Standalone Usage

```python
from classiflow.stats.clustering import plot_all_projections
from classiflow.stats.config import VizConfig
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
features = ["feature1", "feature2", "feature3"]
classes = ["A", "B", "C"]
label_col = "diagnosis"

# Create config
config = VizConfig(
    data_csv="data.csv",
    label_col=label_col,
    outdir="output/",
    fig_dpi=160,
    point_size=48.0,
    alpha_points=0.9
)

# Generate all projections
results = plot_all_projections(
    df=df,
    features=features,
    label_col=label_col,
    classes=classes,
    outdir=Path("output/projections"),
    config=config,
    methods=["umap", "tsne", "lda"]  # Optional: specify which methods
)

# Results dict maps method -> output path
print(results)
# {'umap': Path(...), 'tsne': Path(...), 'lda': Path(...)}
```

### Integrated with `run_visualizations()`

```python
from classiflow.stats import run_visualizations

results = run_visualizations(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="derived/viz"
)

# Projection outputs in results
projections = results['projections']
print(f"UMAP: {projections['umap']}")
print(f"t-SNE: {projections['tsne']}")
print(f"LDA: {projections['lda']}")
```

---

## CLI Usage

### Basic Command

```bash
# Projections are automatically included
classiflow stats viz --data-csv data.csv --label-col diagnosis
```

### With Custom Parameters

```bash
classiflow stats viz \
    --data-csv data.csv \
    --label-col diagnosis \
    --stats-dir derived/stats_results \
    --fig-dpi 300 \
    --outdir results/
```

### Output

```
Creating visualizations from data.csv...
Loading data from data.csv...
  • Classes: 3
  • Features: 50

[1/5] Creating boxplots...
[2/5] Creating fold-change plots...
[3/5] Creating volcano plots...
[4/5] Creating heatmap...
[5/5] Creating dimensionality reduction plots...
    • Running UMAP (supervised=True)...
      ✓ Saved: derived/viz/projections/umap_projection.png
    • Running t-SNE (perplexity=30)...
      ✓ Saved: derived/viz/projections/tsne_projection.png
    • Running LDA...
      ✓ Saved: derived/viz/projections/lda_projection.png

✓ Visualizations complete!
  Output directory: derived/viz
  Boxplots: derived/viz/boxplots_all.pdf
  Fold-change plots: 15 files
  Volcano plots: 16 files
  Heatmap: derived/viz/heatmaps/top_features_heatmap.png

  Dimensionality reduction:
    • UMAP: derived/viz/projections/umap_projection.png
    • TSNE: derived/viz/projections/tsne_projection.png
    • LDA: derived/viz/projections/lda_projection.png
```

---

## Technical Details

### Data Preprocessing

All projection methods use the same preprocessing pipeline:

1. **Feature selection**: Only numeric features are used
2. **Missing value imputation**: Median imputation
3. **Standardization**: Z-score normalization (mean=0, std=1)
4. **Label encoding**: For supervised methods (UMAP, LDA)

### UMAP Configuration

```python
UMAP(
    n_neighbors=15,        # Local neighborhood size
    min_dist=0.1,          # Minimum distance in embedding
    metric="euclidean",    # Distance metric
    n_components=2,        # 2D projection
    random_state=42,       # Reproducibility
    verbose=False
)
```

**Supervised UMAP** uses encoded class labels to guide the projection, resulting in better class separation.

### t-SNE Configuration

```python
TSNE(
    n_components=2,        # 2D projection
    perplexity=30,         # Auto-adjusted if n_samples < 100
    random_state=42,       # Reproducibility
    n_iter=1000,           # Optimization iterations
    verbose=0
)
```

**Perplexity auto-adjustment**:
- If n_samples < 100: `perplexity = min(30, (n_samples - 1) // 3)`
- If perplexity < 5: t-SNE is skipped (insufficient samples)

### LDA Configuration

```python
LinearDiscriminantAnalysis(
    n_components=min(2, n_classes - 1)  # Limited by n_classes
)
```

**Variance explained** is reported in the plot title (e.g., "LD1: 92.5%, LD2: 6.8%").

**Binary classification**: For 2-class problems, LDA can only project to 1D. A dummy second dimension (zeros) is added for plotting.

---

## Plot Styling

### Colors

Classes are assigned colors using the same scheme as other stats plots:
1. First class: **Blue**
2. Second class: **Red**
3. Third class: **Yellow**
4. Fourth class: **Green**
5. Additional classes: **tab20 colormap**

### Markers

- **Size**: Configurable via `point_size` parameter (default: 48.0)
- **Alpha**: Configurable via `alpha_points` parameter (default: 0.9)
- **Edge**: Black border (linewidth=0.8) for better visibility

### Legend

- **≤15 classes**: Legend inside plot area
- **>15 classes**: Legend outside plot area (right side)

### Formats

Both PNG and SVG formats are saved for each projection:
- **PNG**: For quick viewing and presentations
- **SVG**: For publication-ready vector graphics

---

## Troubleshooting

### UMAP Not Available

**Error**: `⚠ UMAP not available (install with: pip install umap-learn)`

**Solution**:
```bash
pip install umap-learn
# OR
pip install -e ".[viz]"
```

### t-SNE Skipped

**Warning**: `⚠ Insufficient samples for t-SNE (n=10), skipping`

**Reason**: t-SNE requires perplexity < n_samples/3. With too few samples, t-SNE is unreliable.

**Solution**: Use UMAP or LDA for small datasets (n < 30).

### LDA Requires 2+ Classes

**Warning**: `⚠ LDA requires at least 2 classes, skipping`

**Reason**: LDA is a supervised method and needs multiple classes.

**Solution**: LDA is automatically skipped for single-class data.

---

## Use Cases

### 1. **Exploratory Data Analysis**

Quickly visualize class separability before training models:

```bash
classiflow stats viz --data-csv raw_data.csv --label-col diagnosis
```

Check if classes are well-separated in UMAP/LDA → expect good classification performance.

### 2. **Feature Engineering Validation**

After feature selection or transformation, visualize whether classes remain separated:

```python
# Before feature engineering
run_visualizations(data_csv="original.csv", label_col="diagnosis", outdir="viz_original/")

# After feature engineering
run_visualizations(data_csv="engineered.csv", label_col="diagnosis", outdir="viz_engineered/")

# Compare projections side-by-side
```

### 3. **Publication Figures**

Use SVG outputs for high-quality publication figures:

```bash
classiflow stats viz --data-csv data.csv --label-col diagnosis --fig-dpi 300
```

Projections are saved as both PNG and SVG in `derived/viz/projections/`.

### 4. **Class Imbalance Detection**

UMAP/t-SNE can reveal hidden clusters or class overlap:
- **Tight clusters**: Well-defined classes
- **Scattered points**: High intra-class variability
- **Overlapping clusters**: Difficult classification boundary

---

## Comparison of Methods

| Method | Speed | Local Structure | Global Structure | Supervised | Reproducible | Best For |
|--------|-------|----------------|------------------|------------|--------------|----------|
| **UMAP** | Fast | ✓✓ | ✓✓✓ | Optional | ✓ | General purpose, large datasets |
| **t-SNE** | Slow | ✓✓✓ | ✓ | No | ✓ | Exploratory analysis, small/medium datasets |
| **LDA** | Fast | ✓ | ✓✓ | Required | ✓ | Maximizing class separation, interpretability |

### When to Use Each

**UMAP**:
- Default choice for most applications
- Large datasets (n > 1000)
- When you want both local and global structure

**t-SNE**:
- When local neighborhood relationships are most important
- Exploratory analysis to find subgroups
- Medium datasets (100 < n < 10,000)

**LDA**:
- When you want maximum class separation
- Interpretable features (linear combinations)
- Benchmarking supervised separability

**Recommendation**: Always generate all three and compare. They often reveal complementary information about your data.

---

## Example Output

For the iris dataset:

```bash
classiflow stats viz --data-csv iris.csv --label-col Species
```

**UMAP**: Clear separation of all three species with some overlap between versicolor and virginica.

**t-SNE**: Tight clusters for setosa, partial overlap between versicolor and virginica.

**LDA**: Maximum separation along LD1 (92% variance), with LD2 separating versicolor and virginica.

All three methods agree: setosa is very distinct, while versicolor and virginica are similar.

---

## Future Enhancements

Potential additions (not yet implemented):

- PCA pre-reduction option
- 3D projections (interactive plotly)
- Customizable UMAP/t-SNE parameters via CLI
- Batch projection (multiple datasets)
- Projection alignment across timepoints
- Contour density plots

---

## Citation

If you use these visualizations in your research:

```
Markowitz, A. (2025). MLSubtype: Production-grade ML toolkit with integrated
dimensionality reduction visualizations. https://github.com/alexmarkowitz/classiflow
```

---

## Additional Resources

- **UMAP paper**: McInnes et al. (2018) - https://arxiv.org/abs/1802.03426
- **t-SNE paper**: van der Maaten & Hinton (2008) - https://jmlr.org/papers/v9/vandermaaten08a.html
- **LDA tutorial**: Scikit-learn documentation - https://scikit-learn.org/stable/modules/lda_qda.html
