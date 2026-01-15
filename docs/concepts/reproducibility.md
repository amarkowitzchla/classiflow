# Reproducibility Guarantees

Understanding what classiflow guarantees for reproducibility and its limitations.

## What Classiflow Guarantees

### 1. Deterministic Splitting

With a fixed `random_state`, train/test splits are identical:

```python
config1 = TrainConfig(random_state=42, ...)
config2 = TrainConfig(random_state=42, ...)

# Folds will contain the same samples
```

### 2. Complete Configuration Capture

Every run saves its full configuration:

```python
# Saved automatically to config.json
{
    "data_csv": "data/features.csv",
    "label_col": "diagnosis",
    "outer_folds": 5,
    "inner_splits": 5,
    "random_state": 42,
    ...
}
```

### 3. Data Integrity Verification

The run manifest includes data hash:

```python
# In run.json
{
    "training_data_hash": "sha256:a1b2c3...",
    "training_data_row_count": 569
}
```

Verify data hasn't changed:

```python
from classiflow.lineage.hashing import compute_file_hash

current_hash = compute_file_hash("data/features.csv")
stored_hash = manifest["training_data_hash"]
assert current_hash == stored_hash, "Data has changed!"
```

### 4. Environment Capture

```python
# In run.json
{
    "package_version": "0.1.0",
    "python_version": "3.10.12",
    "git_hash": "abc123...",
    "hostname": "research-server"
}
```

### 5. Feature Order Preservation

The exact feature list is stored:

```python
# In run.json
{
    "feature_list": ["feature_1", "feature_2", ...]
}
```

## What Classiflow Cannot Guarantee

### 1. Exact Numeric Reproduction

Floating-point operations can vary by:

- **Hardware**: Different CPUs/GPUs may use different instruction sets
- **BLAS libraries**: OpenBLAS vs MKL may give slightly different results
- **Parallelization**: Thread ordering affects accumulation

!!! note "Typical Variation"
    Expect differences in the 5th-6th decimal place on different machines.

### 2. Third-Party Library Changes

Even with version pinning, library behavior can differ:

```python
# Recommended: freeze exact versions
pip freeze > requirements_frozen.txt
```

### 3. OS-Level Differences

System libraries can affect results:

- Random number generator implementation
- File encoding
- Floating-point handling

## Reproducibility Levels

### Level 1: Same Machine, Same Code

**Achievable**: Exact reproduction

```python
# Run twice on same machine
results1 = train_binary_task(config)
results2 = train_binary_task(config)
# results1 == results2 (exactly)
```

### Level 2: Same Machine, Different Session

**Achievable**: Exact reproduction (with caveats)

- Requires same library versions
- May differ if CUDA toolkit changed

### Level 3: Different Machine, Same OS

**Achievable**: Highly similar results

- Minor floating-point differences possible
- Same conclusions, nearly identical metrics

### Level 4: Different OS

**Achievable**: Similar conclusions

- May have larger numeric differences
- Model rankings should be stable

## Best Practices

### 1. Always Set Random State

```python
config = TrainConfig(
    random_state=42,  # Always explicit
    ...
)
```

### 2. Document Environment

```bash
# Create environment file
pip freeze > requirements.txt

# Or with conda
conda env export > environment.yml
```

### 3. Use Version Control

```bash
git add .
git commit -m "Before training run"
# Git hash captured in manifest
```

### 4. Store Run Artifacts

Keep complete run directories:
```
archive/
├── run_2024-01-15_v1/
├── run_2024-01-20_v2/
└── run_2024-01-25_final/
```

### 5. Verify Data Integrity

```python
def verify_run(run_dir, data_path):
    """Verify run can be reproduced."""
    import json
    from classiflow.lineage.hashing import compute_file_hash

    with open(run_dir / "run.json") as f:
        manifest = json.load(f)

    current_hash = compute_file_hash(data_path)
    if current_hash != manifest["training_data_hash"]:
        raise ValueError("Data file has changed!")

    print("Data integrity verified")
```

## Troubleshooting Non-Reproducibility

### Check #1: Random State

```python
# Ensure random_state is set everywhere
config.random_state = 42
```

### Check #2: Data Changes

```python
# Verify data hash
compute_file_hash("data.csv") == manifest["training_data_hash"]
```

### Check #3: Library Versions

```python
import classiflow
import sklearn
import numpy

print(f"classiflow: {classiflow.__version__}")
print(f"sklearn: {sklearn.__version__}")
print(f"numpy: {numpy.__version__}")
```

### Check #4: Feature Order

```python
# Check feature columns
current_features = df.columns.tolist()
stored_features = manifest["feature_list"]
assert current_features == stored_features
```

## For Publications

Include in your methods:

1. **Software versions**: classiflow, Python, key libraries
2. **Random seed**: The exact seed used
3. **Hardware**: CPU/GPU, memory (if relevant)
4. **Data hash**: For verification
5. **Configuration file**: As supplementary material

Example:

> Analyses were performed using classiflow v0.1.0 (Python 3.10, scikit-learn 1.4.0) with random seed 42. Complete configuration and data checksums are provided in Supplementary File S1.
