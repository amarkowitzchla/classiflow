# Choosing Mode and Engine

Use `task.mode` for prediction objective and `execution.engine` for runtime stack.

## Supported Combinations

| task.mode | execution.engine | Supported | Notes |
|---|---|---|---|
| binary | sklearn | Yes | CPU sklearn estimators |
| binary | torch | Yes | Torch estimator registry |
| binary | hybrid | No | Use sklearn or torch |
| meta | sklearn | Yes | CPU sklearn base + meta |
| meta | torch | Yes | Torch base + torch meta |
| meta | hybrid | No | Use sklearn or torch |
| multiclass | sklearn | Yes | `multiclass.backend: sklearn_cpu` |
| multiclass | torch | Yes | `multiclass.backend: torch_<device>` |
| multiclass | hybrid | Yes | `multiclass.backend: hybrid_sklearn_meta_torch_base` |
| hierarchical | sklearn | Yes | Existing hierarchical flow |
| hierarchical | torch | No | Use sklearn |
| hierarchical | hybrid | No | Use sklearn |

## Example: Multiclass + Sklearn

```yaml
task:
  mode: multiclass
execution:
  engine: sklearn
multiclass:
  backend: sklearn_cpu
  group_stratify: true
  sklearn:
    logreg:
      solver: saga
      max_iter: 5000
```

## Example: Binary + Torch (MPS)

```yaml
task:
  mode: binary
execution:
  engine: torch
  device: mps
  model_set: torch_basic
  torch:
    dtype: float32
    num_workers: 0
    require_device: false
```

## Example: Multiclass + Hybrid

```yaml
task:
  mode: multiclass
execution:
  engine: hybrid
  device: cuda
  torch:
    dtype: float32
    num_workers: 0
    require_device: false
multiclass:
  backend: hybrid_sklearn_meta_torch_base
  torch:
    model_set: torch_basic
```

## Discoverability Commands

```bash
classiflow project bootstrap --show-options
classiflow config show --mode multiclass --engine sklearn
classiflow config explain multiclass.backend
classiflow config validate path/to/project.yaml
classiflow config normalize path/to/project.yaml --out path/to/project.normalized.yaml
```

## Cheat Sheet

Common knobs most users should touch:
- `key_columns.label`, `key_columns.patient_id`
- `task.mode`
- `execution.engine`, `execution.device` (torch/hybrid)
- `models.selection_metric`
- `validation.nested_cv.outer_folds`, `validation.nested_cv.inner_folds`

Advanced knobs most users can ignore:
- `execution.torch.num_workers`
- `execution.torch.require_device`
- `execution.model_set`
- `multiclass.torch.model_set`
