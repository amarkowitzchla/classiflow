# Custom Tasks Guide for MLSubtype

This guide shows how to train meta-classifiers with custom binary tasks defined in JSON.

## Your Tasks JSON

Your tasks are defined in [data/tasks.json](data/tasks.json):

```json
{
  "G3G4_vs_Rest": { "pos": ["G3", "G4"], "neg": "rest" },
  "G3SHH_vs_Rest": { "pos": ["G3", "SHH"], "neg": "rest" },
  "G4WNT_vs_Rest": { "pos": ["G4", "WNT"], "neg": "rest" },
  "G3SHH_vs_G4WNT": { "pos": ["G3", "SHH"], "neg": ["G4", "WNT"] },
  "G3_vs_Rest": { "pos": ["G3"], "neg": "rest" },
  "G4_vs_Rest": { "pos": ["G4"], "neg": "rest" },
  "SHH_vs_Rest": { "pos": ["SHH"], "neg": "rest" },
  "WNT_vs_Rest": { "pos": ["WNT"], "neg": "rest" },
  "G3_vs_G4": { "pos": ["G3"], "neg": ["G4"] },
  "G3_vs_SHH": { "pos": ["G3"], "neg": ["SHH"] },
  "G3_vs_WNT": { "pos": ["G3"], "neg": ["WNT"] },
  "G4_vs_SHH": { "pos": ["G4"], "neg": ["SHH"] },
  "G4_vs_WNT": { "pos": ["G4"], "neg": ["WNT"] },
  "SHH_vs_WNT": { "pos": ["SHH"], "neg": ["WNT"] }
}
```

## Task JSON Format

Each task defines a binary classification:
- **`pos`**: List of class labels to treat as positive (label=1)
- **`neg`**: List of class labels to treat as negative (label=0), or `"rest"` for all other classes

## Three Training Modes

### Mode 1: Auto Tasks Only (Default)

Builds One-vs-Rest (OvR) and pairwise tasks automatically:

```bash
classiflow train-meta \
  --data-csv data/MBmerged-z-scores_MLready_correction.csv \
  --label-col MOLECULAR \
  --outdir derived_auto_tasks \
  --smote both
```

For 4 classes (G3, G4, SHH, WNT), this creates:
- 4 OvR tasks (G3_vs_Rest, G4_vs_Rest, SHH_vs_Rest, WNT_vs_Rest)
- 6 pairwise tasks (G3_vs_G4, G3_vs_SHH, G3_vs_WNT, G4_vs_SHH, G4_vs_WNT, SHH_vs_WNT)
- **Total: 10 tasks**

### Mode 2: Auto + Custom Tasks

Builds auto tasks PLUS custom tasks from JSON:

```bash
classiflow train-meta \
  --data-csv data/MBmerged-z-scores_MLready_correction.csv \
  --label-col MOLECULAR \
  --tasks-json data/tasks.json \
  --outdir derived_auto_plus_custom \
  --smote both
```

This creates:
- 4 OvR tasks (auto)
- 6 pairwise tasks (auto)
- 14 custom tasks from JSON
- **Total: 24 tasks** (some may overlap/duplicate)

### Mode 3: Custom Tasks Only (NEW!)

Builds ONLY the tasks from JSON, skipping automatic OvR/pairwise:

```bash
classiflow train-meta \
  --data-csv data/MBmerged-z-scores_MLready_correction.csv \
  --label-col MOLECULAR \
  --tasks-json data/tasks.json \
  --tasks-only \
  --outdir derived_custom_tasks_only \
  --smote both
```

This creates:
- 14 custom tasks from JSON
- **Total: 14 tasks**

## Quick Start Scripts

I've created ready-to-run bash scripts:

### Run with custom tasks only:
```bash
chmod +x run_custom_tasks.sh
./run_custom_tasks.sh
```

### Run with auto + custom tasks:
```bash
chmod +x run_custom_tasks_plus_auto.sh
./run_custom_tasks_plus_auto.sh
```

## Python API Example

```python
from pathlib import Path
from classiflow.config import MetaConfig
from classiflow.training.meta import train_meta_classifier

# Custom tasks only
config = MetaConfig(
    data_csv=Path("data/MBmerged-z-scores_MLready_correction.csv"),
    label_col="MOLECULAR",
    tasks_json=Path("data/tasks.json"),
    tasks_only=True,  # Set to True to skip auto tasks
    outdir=Path("derived_custom_tasks_only"),
    outer_folds=3,
    inner_splits=5,
    inner_repeats=2,
    random_state=42,
    smote_mode="both",
    max_iter=10000,
)

results = train_meta_classifier(config)
print(f"Trained {results['n_tasks']} tasks")
```

See [example_custom_tasks.py](example_custom_tasks.py) for full examples.

## Command-Line Options

```bash
classiflow train-meta --help
```

Key options:
- `--data-csv`: Path to your CSV with features + labels
- `--label-col`: Name of label column (e.g., "MOLECULAR")
- `--tasks-json`: Path to tasks JSON file
- `--tasks-only`: If set, only use tasks from JSON (skip auto OvR/pairwise)
- `--outdir`: Output directory for results
- `--outer-folds`: Number of outer CV folds (default: 3)
- `--inner-splits`: Number of inner CV splits (default: 5)
- `--inner-repeats`: Number of inner CV repeats (default: 2)
- `--smote`: SMOTE mode: `off`, `on`, or `both` (default: `both`)
- `--random-state`: Random seed (default: 42)
- `--verbose`: Enable verbose logging

## Output Structure

After training, you'll get:

```
derived_custom_tasks_only/
├── run.json                           # Training manifest with lineage
├── metrics_inner_cv.csv               # Inner CV hyperparameter search results
├── metrics_inner_cv_splits.csv        # Per-split metrics for all tasks
├── metrics_inner_cv_splits.xlsx       # Same as above, Excel format
├── metrics_outer_binary_eval.csv      # Outer CV binary task evaluation
├── metrics_outer_meta_eval.csv        # Outer CV meta-classifier evaluation
├── roc_meta_averaged.png              # Averaged ROC curve across folds
├── pr_meta_averaged.png               # Averaged PR curve across folds
└── fold1/, fold2/, fold3/             # Per-fold artifacts
    ├── binary_smote/
    │   ├── binary_pipes.joblib        # Trained binary models
    │   ├── meta_model.joblib          # Meta-classifier
    │   ├── meta_features.csv          # Meta-feature names
    │   ├── meta_classes.csv           # Class labels
    │   ├── roc_meta_fold1.png         # ROC curve
    │   ├── pr_meta_fold1.png          # PR curve
    │   └── cm_meta_fold1.png          # Confusion matrix
    └── binary_none/                   # Same structure without SMOTE
```

## Understanding Your Tasks

Your [data/tasks.json](data/tasks.json) creates these binary tasks:

1. **Composite OvR tasks**: Combine multiple classes as positive
   - `G3G4_vs_Rest`: G3+G4 vs all others
   - `G3SHH_vs_Rest`: G3+SHH vs all others
   - `G4WNT_vs_Rest`: G4+WNT vs all others

2. **Composite pairwise task**:
   - `G3SHH_vs_G4WNT`: G3+SHH vs G4+WNT

3. **Standard OvR tasks**:
   - `G3_vs_Rest`, `G4_vs_Rest`, `SHH_vs_Rest`, `WNT_vs_Rest`

4. **Standard pairwise tasks**:
   - `G3_vs_G4`, `G3_vs_SHH`, `G3_vs_WNT`, `G4_vs_SHH`, `G4_vs_WNT`, `SHH_vs_WNT`

## Recommendations

For your use case with the medulloblastoma subtypes (G3, G4, SHH, WNT), I recommend:

**Option 1: Custom tasks only** (`--tasks-only`)
- Use this if you've carefully designed your tasks based on domain knowledge
- Avoids redundancy and focuses on biologically meaningful comparisons
- Faster training (14 tasks vs 24)

**Option 2: Auto + custom tasks** (no `--tasks-only`)
- Use this if you want maximum coverage
- Let the meta-classifier learn which task combinations work best
- More comprehensive but slower training

## Next Steps

1. Start with custom tasks only:
   ```bash
   ./run_custom_tasks.sh
   ```

2. Review results in `derived_custom_tasks_only/metrics_outer_meta_eval.csv`

3. Check fold-averaged performance in `roc_meta_averaged.png` and `pr_meta_averaged.png`

4. If needed, experiment with auto + custom tasks for comparison

## Questions?

- See the main README for general classiflow documentation
- Check [classiflow/tasks/](src/classiflow/tasks/) for task builder implementation
- Review [classiflow/training/meta.py](src/classiflow/training/meta.py) for training logic
