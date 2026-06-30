# splitting

## Objective
- Provide patient/group-aware splitting utilities to enforce **no leakage** across CV folds.
- Offer deterministic split iterators for outer and inner CV in training pipelines.

## Public Interfaces
- `make_group_labels(df: pd.DataFrame, patient_col: str, label_col: str) -> pd.Series` in `src/classiflow/splitting/group_stratified.py`
- `iter_outer_splits(df: pd.DataFrame, y: Iterable, patient_col: str, n_splits: int, random_state: int, mode="kfold", stratify=True) -> Iterator[(tr_idx, va_idx)]`
- `iter_inner_splits(df_tr: pd.DataFrame, y_tr: Iterable, patient_col: str, n_splits: int, n_repeats: int, random_state: int, stratify=True) -> Iterator[(tr_idx, va_idx)]`
- `assert_no_patient_leakage(df: pd.DataFrame, patient_col: str, train_idx: np.ndarray, val_idx: np.ndarray, context: str) -> None`

## Inputs
- For group-aware splitting:
  - `df` must contain `patient_col` and be index-aligned to `y`.
  - `y`/`y_tr` is an iterable of labels aligned to the same index as `df`.
- `iter_outer_splits`:
  - uses `StratifiedGroupKFold` if available, else a greedy fallback that attempts label balancing per fold.
  - supports `stratify=False` to use `GroupKFold` without label stratification.

## Outputs
- Yields `(train_idx, val_idx)` arrays that index into the original row space.
- `make_group_labels` returns one label per patient ID and raises if a patient has conflicting labels.

## Internal Workflow
- `make_group_labels`:
  - group by patient, enforce `nunique(label)==1`, return `first()` label per patient.
- Split iteration:
  - compute patient-level label series
  - split at patient level
  - expand back to row indices by selecting rows with those patient IDs.
- `assert_no_patient_leakage` checks overlap of patient IDs between train and validation indices and raises on leakage.

## Dependencies
- Upstream callers:
  - Training: `training/nested_cv.py`, `training/meta.py`, `training/multiclass.py` for patient-level CV
  - Tests: explicit split/leakage tests
- External dependencies: `numpy`, `pandas`, `sklearn`.

## Invariants & Safety Constraints
- **No patient leakage** is a hard invariant:
  - for any group-based split, no patient ID may appear in both train and validation.
- `random_state` must make split generation deterministic for a fixed dataset order.
- `make_group_labels` must fail closed on conflicting patient labels; “majority label” is not acceptable in this module (hierarchical training implements different logic explicitly).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new split modes (additive) | Medium | Add unit tests; document intended use and invariants |
| Change fallback stratification logic | High | Regression tests; validate fold label balance and determinism |
| Relax leakage checks | High | Not allowed without explicit governance; add ADR and full review |

## Testing Requirements
- Unit: `pytest tests/splitting/test_group_stratified.py`
- Training wiring: `pytest tests/training/test_patient_stratified_wiring.py`

## Common Pitfalls
- Passing a `df` whose index is not aligned to `y` (silent mis-splitting).
- Using `patient_col` with missing values; caller should drop/handle missing group IDs.
- Confusing “patient-level stratification” (group split) with “sample-level stratification”.

## Examples
```python
import pandas as pd
import numpy as np
from classiflow.splitting import iter_outer_splits

df_groups = pd.DataFrame({"patient_id": ["p1","p1","p2","p3"]})
y = np.array([0, 0, 1, 0])
splits = list(iter_outer_splits(df_groups, y, patient_col="patient_id", n_splits=2, random_state=42))
print([(len(tr), len(va)) for tr, va in splits])
```

## High-Risk Change Protocol
- Required design note (ADR):
  - State the leakage model (what constitutes a “group”), and how determinism and stratification are enforced.
- Required test additions:
  - Add a regression test demonstrating the leakage failure mode being prevented.
  - Add determinism tests (same seed ⇒ same splits).
- Required backward compatibility checks:
  - Existing training metrics baselines may shift; document expected changes.
  - Ensure configs/CLI semantics remain unchanged (`patient_col` meaning).
- Required release note items:
  - “Splitting changes” section including a clear warning about comparability of historical metrics.

