# Environment Setup Guide

## Recommended: Mamba/Conda Environment

For best compatibility with scientific Python packages, we **strongly recommend** using mamba or conda rather than venv.

### Why Mamba/Conda?

- **Better dependency resolution** for ML packages (scikit-learn, numpy, scipy, etc.)
- **Pre-compiled binaries** for faster installation
- **Handles native dependencies** automatically (BLAS, LAPACK, etc.)
- **Prevents version conflicts** between imblearn and scikit-learn

---

## Quick Start (Recommended)

### 1. Install Mamba (if not already installed)

**Miniforge (recommended - includes mamba):**
```bash
# macOS (Intel)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh

# macOS (Apple Silicon)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```

After installation, restart your terminal.

### 2. Create Environment and Install Package

**Option A: Use the automated setup script**
```bash
cd /Users/alex/Documents/project-MLSubtype
chmod +x setup_env.sh
./setup_env.sh
```

**Option B: Manual setup**
```bash
# Create environment
mamba create -n classiflow-env python=3.11 -y

# Activate environment
mamba activate classiflow-env

# Install package with all dependencies
pip install -e ".[all]"
```

### 3. Verify Installation

```bash
# Run tests
pytest -v

# Check CLI
classiflow --version

# Run quick test
classiflow train-meta --help
```

---

## Alternative: Using venv (Not Recommended)

If you must use venv instead of mamba/conda:

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install package
pip install -e ".[all]"
```

**Note:** You may encounter version conflicts between scikit-learn and imbalanced-learn. If this happens, try:

```bash
pip install --upgrade scikit-learn==1.5.2 imbalanced-learn==0.12.4
pip install -e ".[all]"
```

---

## Troubleshooting

### ImportError: cannot import name '_safe_tags' from 'sklearn.utils._tags'

**Cause:** Version mismatch between `scikit-learn` and `imbalanced-learn`

**Solution:**
```bash
# Uninstall both packages
pip uninstall scikit-learn imbalanced-learn -y

# Reinstall with compatible versions
pip install scikit-learn==1.5.2 imbalanced-learn==0.12.4

# Reinstall package
pip install -e ".[all]"
```

### ModuleNotFoundError: No module named 'classiflow'

**Cause:** Package not installed in development mode

**Solution:**
```bash
cd /Users/alex/Documents/project-MLSubtype
pip install -e ".[all]"
```

### pytest: command not found

**Cause:** pytest not installed

**Solution:**
```bash
pip install ".[dev]"
```

---

## Environment Management

### Activate Environment

```bash
# Mamba/Conda
mamba activate classiflow-env

# venv
source .venv/bin/activate
```

### Deactivate Environment

```bash
# Mamba/Conda
mamba deactivate

# venv
deactivate
```

### Delete Environment

```bash
# Mamba/Conda
mamba env remove -n classiflow-env

# venv
rm -rf .venv
```

### List Installed Packages

```bash
pip list
```

### Update Package Dependencies

```bash
pip install --upgrade -e ".[all]"
```

---

## Development Setup

For contributing to the package:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=classiflow --cov-report=html

# Run linting
ruff check src/

# Run type checking
mypy src/classiflow/

# Format code
black src/ tests/
```

---

## Running the Package

### CLI Usage

```bash
# Binary classification
classiflow train-binary \
    --data-csv data/example.csv \
    --label-col diagnosis \
    --pos-label malignant \
    --outdir results/

# Meta-classifier (multiclass)
classiflow train-meta \
    --data-csv data/example.csv \
    --label-col subtype \
    --tasks-json tasks.json \
    --smote both \
    --outdir results/
```

### Streamlit UI

```bash
# From package
streamlit run -m classiflow.streamlit_app.app

# Or from original app.py
streamlit run app.py
```

### Python API

```python
from classiflow import MetaConfig, train_meta_classifier

config = MetaConfig(
    data_csv="data/example.csv",
    label_col="subtype",
    tasks_json="tasks.json",
    outdir="results/",
    outer_folds=3,
    smote_mode="both",
)

results = train_meta_classifier(config)
```

---

## Recommended Workflow

1. **Initial setup** (once):
   ```bash
   ./setup_env.sh
   ```

2. **Start work session**:
   ```bash
   mamba activate classiflow-env
   ```

3. **Run tests** (before committing changes):
   ```bash
   pytest -v
   ```

4. **Run training**:
   ```bash
   classiflow train-meta --data-csv YOUR_DATA.csv --label-col YOUR_LABEL
   ```

5. **End work session**:
   ```bash
   mamba deactivate
   ```

---

## System Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **OS**: macOS, Linux, or Windows
- **RAM**: Minimum 8GB (16GB+ recommended for large datasets)
- **Disk**: ~2GB for dependencies + space for output artifacts

---

## Quick Reference

| Task | Command |
|------|---------|
| Create environment | `mamba create -n classiflow-env python=3.11 -y` |
| Activate environment | `mamba activate classiflow-env` |
| Install package | `pip install -e ".[all]"` |
| Run tests | `pytest -v` |
| Check version | `classiflow --version` |
| Train model | `classiflow train-meta --data-csv DATA.csv --label-col LABEL` |
| Launch UI | `streamlit run -m classiflow.streamlit_app.app` |
| Deactivate | `mamba deactivate` |

---

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Migration Guide**: See [MIGRATION.md](MIGRATION.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Issues**: https://github.com/alexmarkowitz/classiflow/issues
- **Contact**: alexmarkowitz@ucla.edu
