# Quick Start: Train with Custom Tasks

## TL;DR

Run this to train a meta-classifier with ONLY your custom tasks from `data/tasks.json`:

```bash
./run_custom_tasks.sh
```

Or use the CLI directly:

```bash
classiflow train-meta \
  --data-csv data/MBmerged-z-scores_MLready_correction.csv \
  --label-col MOLECULAR \
  --tasks-json data/tasks.json \
  --tasks-only \
  --outdir derived_custom_tasks_only \
  --smote both \
  --verbose
```

## What This Does

1. Loads your data from `data/MBmerged-z-scores_MLready_correction.csv`
2. Reads custom tasks from `data/tasks.json` (14 tasks)
3. Trains binary classifiers for each task with nested cross-validation
4. Trains a meta-classifier that combines binary task scores for multiclass prediction
5. Evaluates with SMOTE (on) and without SMOTE (none)
6. Saves results to `derived_custom_tasks_only/`

## Your 14 Custom Tasks

From `data/tasks.json`:

**Composite tasks:**
- G3G4_vs_Rest
- G3SHH_vs_Rest
- G4WNT_vs_Rest
- G3SHH_vs_G4WNT

**One-vs-Rest:**
- G3_vs_Rest, G4_vs_Rest, SHH_vs_Rest, WNT_vs_Rest

**Pairwise:**
- G3_vs_G4, G3_vs_SHH, G3_vs_WNT
- G4_vs_SHH, G4_vs_WNT, SHH_vs_WNT

## Key Files Created

```
derived_custom_tasks_only/
├── metrics_outer_meta_eval.csv     ← Main results (accuracy, F1, ROC-AUC)
├── roc_meta_averaged.png           ← ROC curve averaged across folds
├── pr_meta_averaged.png            ← Precision-recall curve
└── fold1/, fold2/, fold3/          ← Per-fold models and plots
```

## What's New?

I added the `--tasks-only` flag to classiflow, which:
- Skips automatic OvR and pairwise task generation
- Only uses tasks defined in your JSON file
- Prevents duplicate/redundant tasks

**Before**: `--tasks-json` added custom tasks ON TOP of auto tasks (10 auto + 14 custom = 24 total)
**Now**: `--tasks-only` uses ONLY custom tasks (14 total)

## Files I Created

1. **[run_custom_tasks.sh](run_custom_tasks.sh)** - Script to run with custom tasks only
2. **[run_custom_tasks_plus_auto.sh](run_custom_tasks_plus_auto.sh)** - Script to run with auto + custom tasks
3. **[example_custom_tasks.py](example_custom_tasks.py)** - Python API examples
4. **[CUSTOM_TASKS_GUIDE.md](CUSTOM_TASKS_GUIDE.md)** - Detailed documentation

## Code Changes

Modified 3 files to add `--tasks-only` support:
1. [src/classiflow/config.py](src/classiflow/config.py) - Added `tasks_only` field to `MetaConfig`
2. [src/classiflow/training/meta.py](src/classiflow/training/meta.py) - Logic to skip auto tasks when `tasks_only=True`
3. [src/classiflow/cli/main.py](src/classiflow/cli/main.py) - Added `--tasks-only` CLI flag

## Next Steps

1. **Run training**:
   ```bash
   ./run_custom_tasks.sh
   ```

2. **Monitor progress**: Look for log messages showing task training

3. **Check results**:
   ```bash
   # View meta-classifier performance
   cat derived_custom_tasks_only/metrics_outer_meta_eval.csv

   # View plots
   open derived_custom_tasks_only/roc_meta_averaged.png
   open derived_custom_tasks_only/pr_meta_averaged.png
   ```

4. **Compare with auto tasks** (optional):
   ```bash
   ./run_custom_tasks_plus_auto.sh
   ```

## Full Documentation

See [CUSTOM_TASKS_GUIDE.md](CUSTOM_TASKS_GUIDE.md) for complete details.
