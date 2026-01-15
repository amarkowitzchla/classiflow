# Export Results

This guide covers exporting predictions, metrics, and figures to various formats.

## Exporting Predictions

### CSV Export

```python
from classiflow.inference import run_inference, InferenceConfig

# Run inference
config = InferenceConfig(
    run_dir="derived/run",
    data_csv="data/test.csv",
    output_dir="results/inference",
)
results = run_inference(config)

# Predictions are automatically saved
# results/inference/predictions.csv

# Manual export with custom columns
predictions = results["predictions"]
predictions[["sample_id", "predicted_label", "predicted_proba"]].to_csv(
    "results/predictions_simple.csv",
    index=False
)
```

### JSON Export

```python
import json

# Export metrics as JSON
if "metrics" in results:
    with open("results/metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)
```

## Exporting Metrics

### Summary Table

```python
import pandas as pd

# Load summary metrics
summary = pd.read_csv("derived/run/summary_metrics.csv")

# Export to various formats
summary.to_csv("results/summary_metrics.csv", index=False)
summary.to_excel("results/summary_metrics.xlsx", index=False)
summary.to_latex("results/summary_metrics.tex", index=False)
```

### Formatted for Publication

```python
def format_metrics_for_pub(summary):
    """Format metrics as mean ± SD strings."""
    rows = []
    for _, row in summary.iterrows():
        formatted = f"{row['mean']:.3f} ± {row['std']:.3f}"
        rows.append({
            "Metric": row["metric"],
            "Value": formatted,
        })
    return pd.DataFrame(rows)

pub_table = format_metrics_for_pub(summary)
pub_table.to_latex("results/metrics_publication.tex", index=False)
```

## Exporting to Excel

### Multi-Sheet Workbook

```python
import pandas as pd
from pathlib import Path

def export_complete_workbook(run_dir, output_path):
    """Export all results to a single Excel workbook."""

    run_dir = Path(run_dir)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary metrics
        summary = pd.read_csv(run_dir / "summary_metrics.csv")
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # Per-fold metrics
        fold_metrics = []
        for fold_dir in sorted(run_dir.glob("fold_*")):
            metrics_file = fold_dir / "metrics_outer_eval.csv"
            if metrics_file.exists():
                fold_df = pd.read_csv(metrics_file)
                fold_df["Fold"] = fold_dir.name
                fold_metrics.append(fold_df)

        if fold_metrics:
            pd.concat(fold_metrics).to_excel(
                writer, sheet_name="Per-Fold", index=False
            )

        # Config
        import json
        with open(run_dir / "config.json") as f:
            config = json.load(f)

        config_df = pd.DataFrame([
            {"Parameter": k, "Value": str(v)}
            for k, v in config.items()
        ])
        config_df.to_excel(writer, sheet_name="Configuration", index=False)

    print(f"Workbook saved to: {output_path}")

export_complete_workbook("derived/run", "results/complete_results.xlsx")
```

## Exporting Figures

### Batch Export

```python
from pathlib import Path
import matplotlib.pyplot as plt

def export_all_figures(fig_dict, output_dir, formats=["png", "pdf"]):
    """Export figures in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in fig_dict.items():
        for fmt in formats:
            path = output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")

# Usage
figures = {
    "roc_curve": fig_roc,
    "confusion_matrix": fig_cm,
    "calibration": fig_cal,
}
export_all_figures(figures, "results/figures")
```

## Exporting Run Manifest

### For Reproducibility

```python
import json
import shutil
from pathlib import Path

def export_reproducibility_package(run_dir, output_dir):
    """Export complete reproducibility package."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifest
    shutil.copy(run_dir / "run.json", output_dir / "run_manifest.json")

    # Copy config
    shutil.copy(run_dir / "config.json", output_dir / "training_config.json")

    # Export environment
    import subprocess
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True,
        text=True
    )
    with open(output_dir / "requirements_frozen.txt", "w") as f:
        f.write(result.stdout)

    # Export Python version
    import sys
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
    }
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    print(f"Reproducibility package saved to: {output_dir}")

export_reproducibility_package("derived/run", "results/reproducibility")
```

## Using the CLI

### Export Bundle

```bash
# Create portable bundle
classiflow bundle create \
  --run-dir derived/run \
  --out results/model_bundle.zip \
  --description "Final model v1.0"

# Inspect bundle contents
classiflow bundle inspect results/model_bundle.zip
```

## Best Practices

!!! tip "Version Your Exports"
    Include timestamps or version numbers in filenames.

!!! tip "Include Metadata"
    Always export the run manifest alongside results.

!!! warning "Large File Handling"
    Model files (`.joblib`) can be large. Consider compression.

!!! note "Reproducibility"
    Export `requirements.txt` or `environment.yml` with results.
