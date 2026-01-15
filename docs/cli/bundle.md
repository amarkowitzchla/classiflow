# bundle

Create and manage portable model bundles.

## Subcommands

| Command | Description |
|---------|-------------|
| `bundle create` | Create a portable bundle |
| `bundle inspect` | View bundle metadata |
| `bundle validate` | Validate bundle completeness |

---

## bundle create

Package a training run into a portable ZIP bundle.

### Usage

```bash
classiflow bundle create [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--run-dir PATH` | Path to training run directory |
| `--out PATH` | Output bundle path (.zip) |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fold INT` | 1 | Fold to include (or --all-folds) |
| `--all-folds` | False | Include all folds |
| `--description TEXT` | None | Bundle description |
| `--include-cv-results` | False | Include CV result files |

### Example

```bash
# Single fold
classiflow bundle create \
  --run-dir derived/binary \
  --out artifacts/model_v1.zip \
  --fold 1 \
  --description "Binary classifier v1.0"

# All folds
classiflow bundle create \
  --run-dir derived/binary \
  --out artifacts/model_v1_full.zip \
  --all-folds \
  --description "Full model with all folds"
```

### Bundle Contents

```
model_bundle.zip
├── manifest.json           # Bundle metadata
├── run.json                # Original run manifest
├── config.json             # Training config
├── fold_1/
│   ├── binary_none/
│   │   ├── binary_pipes.joblib
│   │   └── best_models.json
│   └── (or binary_smote/)
└── feature_schema.json     # Feature requirements
```

---

## bundle inspect

View bundle metadata and contents.

### Usage

```bash
classiflow bundle inspect BUNDLE_PATH
```

### Example

```bash
classiflow bundle inspect artifacts/model_v1.zip
```

### Output

```
=== Bundle Information ===
Path: artifacts/model_v1.zip
Created: 2024-01-15T10:30:00
Description: Binary classifier v1.0

=== Run Information ===
Run ID: a1b2c3d4-...
Training timestamp: 2024-01-14T09:00:00
Package version: 0.1.0

=== Contents ===
Folds: [1]
Model variant: binary_none
Features: 30

=== Data Requirements ===
Required features: 30
Label column: diagnosis
Positive class: Malignant
```

---

## bundle validate

Check bundle completeness and integrity.

### Usage

```bash
classiflow bundle validate BUNDLE_PATH
```

### Example

```bash
classiflow bundle validate artifacts/model_v1.zip
```

### Output

```
Validating bundle: artifacts/model_v1.zip

[✓] manifest.json present
[✓] run.json present
[✓] config.json present
[✓] fold_1/binary_none/binary_pipes.joblib present
[✓] All model files loadable
[✓] Feature schema valid

Bundle validation: PASSED
```

---

## Using Bundles

### Load for Inference

```bash
classiflow infer \
  --bundle artifacts/model_v1.zip \
  --data-csv data/new_samples.csv \
  --outdir inference_results
```

### Programmatic Loading

```python
from classiflow.bundles import load_bundle, BundleLoader

# Load bundle
loader = BundleLoader("artifacts/model_v1.zip")
pipes, best_models = loader.load_binary_artifacts()

# Or use the convenience function
artifacts = load_bundle("artifacts/model_v1.zip")
```

---

## Best Practices

!!! tip "Include Description"
    Always add a description for tracking:
    ```bash
    --description "Model v1.2 - added SMOTE"
    ```

!!! tip "Version Your Bundles"
    Use semantic versioning in filenames:
    ```bash
    --out model_v1.2.0.zip
    ```

!!! warning "Bundle Size"
    Bundles can be large (100s of MB). Consider:
    - Including only needed folds
    - Excluding CV results if not needed

---

## See Also

- [What is a Run](../concepts/what-is-a-run.md)
- [Export Results Guide](../how-to/export-results.md)
