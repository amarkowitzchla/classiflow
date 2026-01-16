# Installation

This guide covers all installation methods for classiflow.

## Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows
- **GPU** (optional): CUDA-capable GPU or Apple Silicon for accelerated training

## Quick Install

Install classiflow from PyPI:

```bash
pip install classiflow
```

This installs the core package with all essential dependencies for training, inference, and CLI usage.

## Optional Dependencies

Classiflow provides optional dependency groups for additional functionality:

=== "All extras"

    ```bash
    pip install classiflow[all]
    ```

    Installs everything: Streamlit app, visualization tools, statistical modules, and development tools.

=== "Streamlit App"

    ```bash
    pip install classiflow[app]
    ```

    Includes: `streamlit`, `plotly`, `openpyxl`, `xlsxwriter`, `Pillow`

=== "Visualizations"

    ```bash
    pip install classiflow[viz]
    ```

    Includes: `umap-learn`, `plotly`

=== "Statistics"

    ```bash
    pip install classiflow[stats]
    ```

    Includes: `statsmodels`, `scikit-posthocs`

=== "Development"

    ```bash
    pip install classiflow[dev]
    ```

    Includes: `pytest`, `pytest-cov`, `ruff`, `mypy`, `black`

=== "Documentation"

    ```bash
    pip install classiflow[docs]
    ```

    Includes: `mkdocs-material`, `mkdocstrings`, `mike`

You can combine multiple extras:

```bash
pip install classiflow[app,stats]
```

## Project run prerequisites

`classiflow project run-feasibility` and `classiflow project run-technical` use Excel exports, stats tests, and UMAP visualizations, so install the optional extras before you execute those commands:

```bash
pip install classiflow[app,stats,viz]
```

- `app` provides `xlsxwriter`/`openpyxl` for the spreadsheets created during technical validation.
- `stats` pulls in `statsmodels` and `scikit-posthocs` for statistical tests and reports.
- `viz` brings in `umap-learn`, which now installs cleanly because the core dependency allows `scikit-learn` up through version 2.0.

## Development Installation

For contributing or modifying classiflow:

```bash
# Clone the repository
git clone https://github.com/alexmarkowitz/classiflow.git
cd classiflow

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## GPU Support

Classiflow uses PyTorch for neural network models. GPU support is included automatically:

### CUDA (NVIDIA GPUs)

PyTorch with CUDA support is included in the default installation. Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

### MPS (Apple Silicon)

For M1/M2/M3 Macs, MPS acceleration is available:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Device Selection

Classiflow automatically selects the best available device:

```python
from classiflow.models import resolve_device

device = resolve_device("auto")  # Returns "cuda", "mps", or "cpu"
```

Or specify explicitly in configs:

```python
config = HierarchicalConfig(
    device="cuda",  # or "mps", "cpu", "auto"
    # ...
)
```

## Verifying Installation

After installation, verify everything works:

### Check Version

```python
import classiflow
print(classiflow.__version__)
```

Or via CLI:

```bash
classiflow --version
```

### Test Import

```python
from classiflow import train_binary_task, TrainConfig
from classiflow.inference import run_inference, InferenceConfig
from classiflow.stats import run_stats, StatsConfig

print("All imports successful!")
```

### Run Self-Test

```bash
# If you installed with [dev], run the test suite
pytest tests/ -v --tb=short
```

## Core Dependencies

Classiflow depends on these core packages (installed automatically):

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0,<2.3.0 | Data handling |
| numpy | >=1.24.0,<2.0.0 | Numerical computing |
| scikit-learn | >=1.4.0,<2.0.0 | ML models |
| imbalanced-learn | >=0.12.0 | SMOTE |
| matplotlib | >=3.7.0 | Plotting |
| scipy | >=1.10.0 | Scientific computing |
| torch | >=2.0.0,<2.6.0 | Deep learning |
| pydantic | >=2.0.0,<3.0.0 | Data validation |
| typer | >=0.9.0 | CLI framework |
| seaborn | >=0.12 | Statistical plotting |

## Troubleshooting

### Common Issues

??? question "ImportError: No module named 'classiflow'"

    Ensure you're in the correct virtual environment:
    ```bash
    which python  # Should point to your venv
    pip list | grep classiflow
    ```

??? question "CUDA out of memory"

    Reduce batch size in your config:
    ```python
    config = HierarchicalConfig(
        mlp_batch_size=128,  # Default is 256
        # ...
    )
    ```

??? question "ModuleNotFoundError for optional packages"

    Install the required extra:
    ```bash
    pip install classiflow[stats]  # For statsmodels
    pip install classiflow[app]    # For streamlit
    ```

??? question "Slow training without GPU"

    Verify PyTorch can see your GPU:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())
    ```

    If False, reinstall PyTorch with GPU support per [pytorch.org](https://pytorch.org/get-started/locally/).

### Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md)
2. Search [existing issues](https://github.com/alexmarkowitz/classiflow/issues)
3. Open a [new issue](https://github.com/alexmarkowitz/classiflow/issues/new) with:
   - Python version (`python --version`)
   - OS and version
   - Full error traceback
   - Minimal reproducing code

## Next Steps

- [Quickstart Guide](quickstart.md) - Run your first classification pipeline
- [Tutorials](../tutorials/index.md) - Step-by-step guides for common workflows
- [CLI Reference](../cli/index.md) - Command-line interface documentation
