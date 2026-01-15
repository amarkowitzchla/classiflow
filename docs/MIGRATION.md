# Migration Guide: From Scripts to classiflow Package

This guide helps you migrate from the old script-based workflow to the new `classiflow` package.

## Installation

### Old Way (scripts/)

```bash
cd project/
pip install -r requirements.txt
python scripts/train_binary_meta_classifier.py --data-csv ...
```

### New Way (package)

```bash
pip install classiflow[all]
classiflow train-meta --data-csv ...
```

## Training Binary Models

### Old Way

```bash
python scripts/train_binary.py \
    --data-csv data.csv \
    --label-col diagnosis \
    --pos-label positive \
    --smote on \
    --outer-folds 5 \
    --outdir derived
```

### New Way (CLI)

```bash
classiflow train-binary \
    --data-csv data.csv \
    --label-col diagnosis \
    --pos-label positive \
    --smote on \
    --outer-folds 5 \
    --outdir derived
```

### New Way (Python API)

```python
from classiflow import TrainConfig, train_binary_task
from pathlib import Path

config = TrainConfig(
    data_csv=Path("data.csv"),
    label_col="diagnosis",
    pos_label="positive",
    outdir=Path("derived"),
    outer_folds=5,
    smote_mode="on",
    random_state=42,
)

results = train_binary_task(config)
```

## Training Meta-Classifier (Multiclass)

### Old Way

```bash
python scripts/train_binary_meta_classifier.py \
    --data-csv data.csv \
    --label-col MOLECULAR \
    --smote both \
    --outer-folds 3 \
    --tasks-json tasks.json \
    --outdir derived
```

### New Way (CLI)

```bash
classiflow train-meta \
    --data-csv data.csv \
    --label-col MOLECULAR \
    --smote both \
    --outer-folds 3 \
    --tasks-json tasks.json \
    --outdir derived
```

### New Way (Python API)

```python
from classiflow import MetaConfig, train_meta_classifier
from pathlib import Path

config = MetaConfig(
    data_csv=Path("data.csv"),
    label_col="MOLECULAR",
    tasks_json=Path("tasks.json"),
    outdir=Path("derived"),
    outer_folds=3,
    smote_mode="both",
    random_state=42,
)

results = train_meta_classifier(config)
```

## Streamlit UI

### Old Way

```bash
cd project/
streamlit run app.py
```

### New Way

```bash
# After installing with [app] extras
pip install classiflow[app]
streamlit run -m classiflow.streamlit_app.app
```

## Using Legacy Wrappers (Backward Compatibility)

If you have existing Streamlit pages that import from `utils.wrappers`, you can use the compatibility layer:

```python
# Old code (still works temporarily)
from utils.wrappers import run_train_meta_classifier

run_train_meta_classifier(
    project_root=root,
    data_csv=csv_path,
    tasks_json=tasks_json_path,
    outdir=DERIVED,
    smote_mode="both",
    outer_folds=3,
    random_state=42,
    label_col="MOLECULAR",
)
```

**Recommended Migration:**

```python
# New code (preferred)
from classiflow import MetaConfig, train_meta_classifier
from pathlib import Path

config = MetaConfig(
    data_csv=Path(csv_path),
    label_col="MOLECULAR",
    tasks_json=Path(tasks_json_path) if tasks_json_path else None,
    outdir=Path(DERIVED),
    outer_folds=3,
    smote_mode="both",
    random_state=42,
)

train_meta_classifier(config)
```

## Output Structure

Both old and new produce the same output structure:

```
derived/
├── run_manifest.json              # NEW: Reproducibility metadata
├── metrics_inner_cv.csv
├── metrics_inner_cv_splits.{csv,xlsx}
├── metrics_outer_binary_eval.csv
├── metrics_outer_meta_eval.csv
└── fold{N}/
    ├── binary_smote/
    │   ├── binary_pipes.joblib
    │   ├── meta_model.joblib
    │   ├── meta_features.csv
    │   └── meta_classes.csv
    └── binary_none/  (if --smote both)
```

**New additions:**
- `run_manifest.json`: Contains config, git hash, timestamp, environment info for reproducibility

## Key Differences

### Configuration

**Old**: Scattered argparse arguments

**New**: Type-safe dataclasses with validation

```python
from classiflow import TrainConfig

config = TrainConfig(
    data_csv=Path("data.csv"),
    label_col="diagnosis",
    outer_folds=5,  # Type-checked, validated
    random_state=42,
)
```

### Logging

**Old**: Print statements

**New**: Structured logging

```python
import logging
logging.basicConfig(level=logging.INFO)
# Or use --verbose flag in CLI
```

### Testing

**Old**: Manual validation

**New**: Automated test suite

```bash
pytest tests/
pytest --cov=classiflow
```

### Imports

**Old**: Relative imports, script execution

**New**: Clean package imports

```python
# Old
from utils.wrappers import run_train_meta_classifier
from scripts.train_binary_meta_classifier import AdaptiveSMOTE

# New
from classiflow import train_meta_classifier, MetaConfig
from classiflow.models import AdaptiveSMOTE
```

## Checklist for Migration

- [ ] Install classiflow package: `pip install classiflow[all]`
- [ ] Update CLI commands from `python scripts/...` to `classiflow ...`
- [ ] Migrate Python code to use `classiflow` imports
- [ ] Update Streamlit app imports (if applicable)
- [ ] Run tests to verify behavior: `pytest`
- [ ] Update documentation and scripts
- [ ] Remove old `scripts/` folder (after verification)
- [ ] Update `requirements.txt` → `pyproject.toml` dependencies

## Need Help?

- Check the [README.md](README.md) for full documentation
- Run `classiflow --help` for CLI usage
- Open an issue at https://github.com/alexmarkowitz/classiflow/issues
