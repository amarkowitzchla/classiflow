# Interactive ROC/PR Charts

This document describes the interactive chart system for Classiflow runs, including the plot JSON data format, how the UI renders charts, and how to backfill existing runs.

## Overview

The interactive chart system replaces static PNG images with dynamic, data-driven visualizations for ROC and PR curves. Key features:

- **Interactive tooltips** showing exact values at each point
- **Toggle curves** to show/hide individual classes
- **Confidence bands** for averaged cross-validation results
- **Graceful fallback** to PNG images for older runs

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Python Backend │ -> │  JSON Plot Data  │ -> │  React Frontend │
│  (Training/Inf) │    │  (plots/*.json)  │    │  (Recharts)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Plot JSON Data Format

### PlotCurve Schema

Each plot JSON file follows the `PlotCurve` schema:

```json
{
  "plot_type": "roc" | "pr",
  "scope": "averaged" | "fold" | "inference",
  "task": "binary" | "multiclass",
  "labels": ["ClassA", "ClassB", ...],
  "curves": [
    {
      "label": "micro" | "macro" | "<ClassName>",
      "x": [0.0, 0.1, ...],
      "y": [0.0, 0.2, ...],
      "thresholds": [1.0, 0.95, ...]  // optional
    }
  ],
  "summary": {
    "auc": {"macro": 0.93, "micro": 0.95},  // for ROC
    "ap": {"macro": 0.88}                    // for PR
  },
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "source": "internal",
    "classiflow_version": "0.1.0",
    "run_id": "abc123",
    "fold": 1  // only if scope == "fold"
  },
  "std_band": {  // for averaged plots
    "x": [...],
    "y_upper": [...],
    "y_lower": [...]
  },
  "fold_curves": [...],  // individual fold curves
  "fold_metrics": {"auc": [0.91, 0.93, 0.89]}  // per-fold values
}
```

### Plot Manifest

Each run directory contains `plots/plot_manifest.json`:

```json
{
  "available": {
    "roc_averaged": "plots/roc_averaged.json",
    "pr_averaged": "plots/pr_averaged.json",
    "roc_by_fold": "plots/roc_by_fold.json",
    "pr_by_fold": "plots/pr_by_fold.json"
  },
  "fallback_pngs": {
    "roc_averaged": "averaged_roc.png",
    "pr_averaged": "averaged_pr.png"
  },
  "generated_at": "2024-01-15T10:30:00",
  "classiflow_version": "0.1.0"
}
```

## File Locations

### Technical Validation Runs

```
runs/technical_validation/<run_id>/
├── plots/
│   ├── plot_manifest.json
│   ├── roc_averaged.json
│   ├── pr_averaged.json
│   ├── roc_by_fold.json
│   └── pr_by_fold.json
├── averaged_roc.png         # Fallback
├── averaged_pr.png          # Fallback
└── fold*/...
```

### Independent Test Runs

```
runs/independent_test/<run_id>/
├── plots/
│   ├── plot_manifest.json
│   ├── roc_inference.json
│   └── pr_inference.json
├── inference_roc_curves.png  # Fallback
└── ...
```

## Generating Plot Data

### During Training (Automatic)

When running training commands, plot JSON is generated automatically alongside PNG files:

```python
from classiflow.plots import generate_technical_validation_plots, generate_inference_plots

# Technical validation
manifest = generate_technical_validation_plots(
    run_dir=run_dir,
    run_id=run_id,
    fold_data=[
        {"y_true": y_true, "y_proba": y_proba, "fold_num": 1},
        {"y_true": y_true, "y_proba": y_proba, "fold_num": 2},
        ...
    ],
    classes=["Normal", "Tumor"],
)

# Independent test
manifest = generate_inference_plots(
    run_dir=run_dir,
    run_id=run_id,
    y_true=y_true,
    y_proba=y_proba,
    classes=["Normal", "Tumor"],
)
```

### Backfilling Existing Runs

For runs created before the plot data feature was added:

```bash
# Backfill all runs in a project directory
classiflow backfill plots personal_projects/

# Dry run to see what would be done
classiflow backfill plots personal_projects/ --dry-run

# Process specific phases only
classiflow backfill plots personal_projects/ --phase technical_validation

# Process a specific run
classiflow backfill plots personal_projects/ \
    --project my_project \
    --phase technical_validation \
    --run abc123

# Force regeneration even if manifest exists
classiflow backfill plots personal_projects/ --force
```

## UI Integration

### API Endpoints

```
GET /api/runs/{run_key}/plots/{plot_key}
GET /api/projects/{project_id}/runs/{phase}/{run_id}/plots/{plot_key}
```

Available `plot_key` values:
- `roc_averaged` - Averaged ROC (technical_validation)
- `pr_averaged` - Averaged PR (technical_validation)
- `roc_by_fold` - Per-fold ROC curves
- `pr_by_fold` - Per-fold PR curves
- `roc_inference` - Inference ROC (independent_test)
- `pr_inference` - Inference PR (independent_test)

### React Components

```tsx
import { InteractivePlotSection } from '../components/charts';

<InteractivePlotSection
  runKey={run.run_key}
  phase={run.phase}
  plotManifest={run.plot_manifest}
  artifacts={run.artifacts}
/>
```

The component automatically:
1. Checks if JSON data is available via `plot_manifest`
2. Fetches and renders interactive charts if available
3. Falls back to PNG images if JSON is not available
4. Shows a notice when using fallback images

### Fallback Behavior

The UI gracefully degrades for older runs:

1. **JSON Available**: Renders interactive Recharts chart with:
   - Hover tooltips
   - Legend with toggle capability
   - AUC/AP summary chips
   - Confidence bands for averaged plots

2. **JSON Not Available**: Falls back to PNG with:
   - Static image display
   - Notice: "Interactive chart not available for this run"

## Python API Reference

### Schemas (classiflow.plots.schemas)

```python
from classiflow.plots.schemas import (
    PlotCurve,      # Main plot data structure
    PlotManifest,   # Available plots manifest
    CurveData,      # Individual curve data
    PlotSummary,    # AUC/AP summary
    PlotMetadata,   # Generation metadata
    PlotType,       # "roc" | "pr"
    PlotScope,      # "averaged" | "fold" | "inference"
    TaskType,       # "binary" | "multiclass"
    PlotKey,        # Standard plot key constants
)
```

### Data Export Functions (classiflow.plots.data_export)

```python
from classiflow.plots import (
    compute_roc_curve_data,           # Compute ROC from predictions
    compute_pr_curve_data,            # Compute PR from predictions
    compute_averaged_roc_data,        # Average across folds
    compute_averaged_pr_data,         # Average across folds
    save_plot_data,                   # Save to JSON file
    create_plot_manifest,             # Create manifest file
    generate_technical_validation_plots,  # Generate all TV plots
    generate_inference_plots,         # Generate all inference plots
)
```

## TypeScript Types

```typescript
import type {
  PlotCurve,
  PlotManifest,
  CurveData,
  PlotType,
  PlotScope,
  PlotKeyType,
} from '../types/plots';

import { PlotKey, CURVE_COLORS, AGGREGATE_COLORS } from '../types/plots';
```

## Testing

### Python Tests

```bash
# Run plot data export tests
pytest tests/plots/test_data_export.py -v
```

### Manual Testing

1. Generate a run with plot data:
   ```bash
   classiflow project technical-validation <project_id>
   ```

2. Start the UI:
   ```bash
   classiflow ui start
   ```

3. Navigate to a run and verify:
   - Charts tab shows interactive plots
   - Hover shows tooltips
   - Legend toggles work
   - AUC/AP values displayed

4. Test fallback:
   - Remove `plots/` directory from a run
   - Verify PNG images are shown with notice

## Troubleshooting

### Charts not showing

1. Check if `plot_manifest.json` exists in the run's `plots/` directory
2. Verify the manifest lists the expected plot files
3. Check browser console for API errors

### Backfill not working

1. Ensure run has `predictions.csv` (for inference) or fold directories
2. Check that `run.json` exists with class information
3. Use `--verbose` flag to see detailed errors:
   ```bash
   classiflow backfill plots . --verbose
   ```

### Performance issues

- Plot JSON files can be large for many-class problems
- Consider reducing the number of interpolation points (default: 100)
- Browser may struggle with >20 classes; consider aggregating

## Migration Notes

Runs created before version X.Y.Z will not have plot JSON data.
Use the backfill command to generate it for existing runs.

The UI automatically detects and handles both cases, so no manual intervention is required for viewing runs.
