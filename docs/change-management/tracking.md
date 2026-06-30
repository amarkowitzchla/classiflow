# Module: tracking

## Objective

Provide optional experiment tracking integration for classiflow training workflows, supporting MLflow and Weights & Biases backends with a pluggable architecture.

## Risk Level

**Medium**

- New module with no changes to existing training logic
- Optional dependencies - fails gracefully when not installed
- No impact on artifacts, bundles, or lineage (additive only)

## Public Interfaces

### Factory Function

```python
def get_tracker(
    backend: Optional[str] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> ExperimentTracker
```

Returns a tracker instance based on the backend parameter. Returns `NoOpTracker` when backend is None.

### Base Class

```python
class ExperimentTracker(ABC):
    def start_run(run_name, tags) -> self
    def end_run() -> None
    def log_params(params: Dict) -> None
    def log_metrics(metrics: Dict, step: int) -> None
    def log_artifact(path: Path) -> None
    def log_figure(name: str, figure: Figure) -> None
    def set_tags(tags: Dict) -> None
```

### Utility Functions

```python
def flatten_dict(d: Dict, sep: str = "/") -> Dict
def sanitize_metric_name(name: str) -> str
def extract_loggable_params(config) -> Dict
def summarize_metrics(metrics_dict: Dict, prefix: str = "") -> Dict[str, float]
```

## Inputs/Outputs

### Inputs

- Training configuration (TrainConfig, MetaConfig, etc.)
- Metrics dictionaries from training results
- Artifact file paths

### Outputs

- Logged parameters to tracking system
- Logged metrics to tracking system
- Logged artifacts to tracking system
- No local file outputs (tracking is remote/external)

## Invariants

1. **No-op by default**: When tracker is None, all operations are no-ops with no side effects
2. **Fail gracefully**: Import errors are caught and raised only when backend is explicitly requested
3. **Non-blocking**: Tracking failures should log warnings but not fail training
4. **Complement lineage**: Don't duplicate data already in `run.json`; add structured parameters and metrics for filtering

## Implementation Details

### Module Structure

```
src/classiflow/tracking/
├── __init__.py          # Factory function and public API
├── base.py              # ExperimentTracker abstract base class
├── noop.py              # NoOpTracker (null object pattern)
├── mlflow_tracker.py    # MLflow implementation
├── wandb_tracker.py     # Weights & Biases implementation
└── utils.py             # Helper functions
```

### Integration Points

Tracking is integrated at the training function level:
- `training/binary.py::train_binary_task()`
- `training/meta.py::train_meta_classifier()`
- `training/multiclass.py::train_multiclass_classifier()`
- `training/hierarchical_cv.py::train_hierarchical()`

Each function:
1. Creates tracker via `get_tracker(config.tracker, config.experiment_name)`
2. Starts run with `tracker.start_run(config.run_name, config.tracker_tags)`
3. Logs params after training completes
4. Logs summary metrics
5. Logs artifact files
6. Ends run

### CLI Integration

New flags added to training commands:
- `--tracker`: mlflow, wandb, or omitted for no tracking
- `--experiment-name`: Experiment/project name
- `--run-name`: Run name (auto-generated if omitted)

### Config Integration

Fields added to config dataclasses:
```python
tracker: Optional[Literal["mlflow", "wandb"]] = None
experiment_name: Optional[str] = None
run_name: Optional[str] = None
tracker_tags: Optional[Dict[str, str]] = None
```

## Testing Requirements

### Unit Tests

- `tests/tracking/test_tracker_base.py`: NoOpTracker and get_tracker factory
- `tests/tracking/test_utils.py`: Utility functions
- `tests/tracking/test_mlflow_tracker.py`: MLflow tracker (mocked)
- `tests/tracking/test_wandb_tracker.py`: W&B tracker (mocked)

### Run Tests

```bash
pytest tests/tracking/
```

### Manual Testing

```bash
# With MLflow
pip install classiflow[mlflow]
classiflow train-binary --data data.csv --label-col label --tracker mlflow
mlflow ui  # Check runs appear

# With W&B
pip install classiflow[wandb]
wandb login
classiflow train-binary --data data.csv --label-col label --tracker wandb
# Check wandb.ai dashboard
```

## Dependencies

### Optional

- `mlflow>=2.10.0` - Extra: `[mlflow]`
- `wandb>=0.16.0` - Extra: `[wandb]`
- Both: `[tracking]`

### Required (no additional)

Uses only stdlib and existing classiflow dependencies.

## Backward Compatibility

- Fully backward compatible
- No changes to existing training behavior when tracker is not specified
- Config dataclasses have new optional fields with None defaults
- CLI commands have new optional flags
