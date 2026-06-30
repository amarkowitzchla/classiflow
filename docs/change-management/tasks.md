# tasks

## Objective
- Construct binary task definitions from multiclass labels for OvR, pairwise, and composite task training.
- Provide a stable mapping from human-readable task names to labeler functions.

## Public Interfaces
- `TaskBuilder(classes: list[str])` in `src/classiflow/tasks/builder.py`
  - `.build_ovr_tasks() -> TaskBuilder`
  - `.build_pairwise_tasks() -> TaskBuilder`
  - `.build_all_auto_tasks() -> TaskBuilder`
  - `.add_composite_task(name: str, pos_classes: list[str], neg_classes: list[str]|"rest") -> TaskBuilder`
  - `.get_tasks() -> dict[str, Callable[[pd.Series], pd.Series]]`
- `load_composite_tasks(json_path: Path, builder: TaskBuilder) -> TaskBuilder` in `src/classiflow/tasks/composite.py`

## Inputs
- `classes`: ordered list of class labels (order matters downstream for reporting and meta training).
- Composite tasks JSON formats (supported by `load_composite_tasks`):
  - Dict form:
    - `{ "WNT_like": {"pos": ["WNT"], "neg": "rest"}, ... }`
  - List form:
    - `[{"name": "...", "pos": [...], "neg": ...}, ...]`
- Labeler functions operate on `pd.Series` of string labels and return:
  - OvR: `0.0/1.0`
  - Pairwise: `0.0/1.0` with `NaN` for non-pair classes (via mapping)
  - Composite: `0.0/1.0` with `NaN` for excluded classes

## Outputs
- Task dictionary `{task_name: labeler}` consumed by:
  - `training/meta.py` for base task training and meta-feature generation
  - `projects/orchestrator.py` for project workflows

## Internal Workflow
- Task names are synthesized:
  - OvR: `{class}_vs_Rest`
  - Pairwise: `{a}_vs_{b}`
- Composite tasks:
  - normalize pos/neg classes by intersecting with known classes
  - `"rest"` neg means all classes not in pos
  - invalid specifications emit warnings and skip task creation

## Dependencies
- Upstream callers:
  - `classiflow/__init__.py` re-exports `TaskBuilder`
  - Meta training: `training/meta.py`
  - Projects: `projects/orchestrator.py`
- External dependencies: `pandas`, `numpy`.

## Invariants & Safety Constraints
- Task naming is part of the artifact schema:
  - binary pipeline keys are stored as `{task}__{model_name}` (see `training/meta.py`, `inference/predict.py`).
- Labeler behavior must remain stable:
  - `NaN` indicates rows excluded from a task; changing this changes training sets and metrics.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new task constructors (additive) | Medium | Update docs/tests; ensure meta-feature naming stability |
| Change labeler outputs or NaN handling | High | Regression tests; update meta-feature building and inference expectations |
| Change task naming conventions | High | Migration for existing artifacts (`meta_features.csv`, metrics tables) |

## Testing Requirements
- Unit: `pytest tests/unit/test_tasks.py`
- Integration: `pytest tests/integration/test_meta_class_order_bug.py`

## Change Log
- **Production-readiness cleanup (2026-06-30)** — Low risk:
  - `tasks/builder.py`: Added `Literal` to typing imports to resolve F821 undefined-name for the
    `add_composite_task` type annotation. Runtime behavior is unchanged; the annotation is
    string-evaluated under `from __future__ import annotations`.

## Common Pitfalls
- Using class labels with characters that later become filenames/keys; keep names file-safe when possible.
- Accidentally creating composite tasks with empty pos/neg sets (warnings are emitted, but training may silently miss tasks if not checked).
- Changing class ordering changes task ordering and meta-class encoding.

## Examples
```python
from classiflow.tasks import TaskBuilder, load_composite_tasks
from pathlib import Path

builder = TaskBuilder(["A", "B", "C"]).build_all_auto_tasks()
builder = load_composite_tasks(Path("tasks.json"), builder)
tasks = builder.get_tasks()
print(sorted(tasks)[:3])
```

