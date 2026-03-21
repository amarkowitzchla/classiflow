# Upgrade Plan: Bagged Final Estimators + Expanded MLP Tuning Grid

## Goal

Add two features:

1. A bagged-output mode where the selected estimator is bootstrap aggregated as the final estimator strategy.
2. An `--expanded-mlp-tuning-grid` option that broadens MLP search spaces to include the same hyperparameter dimensions used by the CCIX classifier MLP, plus nearby values, without pulling in the CCIX background autoencoder work.

This plan is written against the current codebase, not a greenfield design.

## Current State Observed

- Binary and meta training pull estimator sets from `src/classiflow/backends/registry.py`.
- Direct multiclass training uses a separate path through `src/classiflow/models/estimators.py` and `src/classiflow/models/torch_multiclass.py`.
- Project workflows and final-model retraining reuse those registries through `src/classiflow/projects/orchestrator.py` and `src/classiflow/projects/final_train.py`.
- There is currently no bagging strategy in the training stack.
- The current classiflow torch MLPs are simpler than the CCIX classifier MLP:
  - classiflow torch MLPs currently expose `hidden_dim`, `n_layers`, and `dropout`, and use ReLU-only modules in `src/classiflow/backends/torch/modules.py`.
  - the direct multiclass torch wrapper in `src/classiflow/models/torch_multiclass.py` currently hard-codes `dropout=0.0`.
  - the CCIX base classifier MLP is closer to:
    - hidden size `512`
    - `3` layers
    - dropout `0.2`
    - ELU activation
    - batch norm enabled
    - learning rate `1e-4`
    - weight decay `1e-6`
    - epochs `50`
    - batch size `256`

## Decisions Confirmed

- Bagging should be available everywhere except `train-meta` and the project meta workflow:
  - `train-binary`
  - `train-multiclass`
  - project technical validation and project final retraining for non-meta modes
- `train-meta` is explicitly excluded from bagging because the meta-classifier already aggregates signals from multiple models.
- The expanded option should apply to any MLP, not to linear/logistic models.
- The clearer flag name is `--expanded-mlp-tuning-grid`.
- The expanded grid should not hard-code one exact CCIX config. It should expose the same MLP hyperparameter dimensions as CCIX and include values around the CCIX defaults.
- Because CCIX exposes activation and batchnorm choices, the classiflow MLP implementation needs to expose those options too if we want the grid to cover the same space.

## Recommended Design

## Feature 1: Bagged Final Estimator Strategy

### User-Facing API

- Add a new strategy flag rather than overloading model names.
- CLI proposal:
  - `--final-estimator-strategy single|bagged`
  - `--bagging-n-estimators`
  - `--bagging-max-samples`
  - `--bagging-max-features`
  - `--bagging-bootstrap/--no-bagging-bootstrap`
  - `--bagging-bootstrap-features/--no-bagging-bootstrap-features`
- Project config proposal:
  - add a top-level `ensemble` block or equivalent config section with the same settings
  - keep defaults equivalent to current behavior (`single`)

### Implementation Shape

- Introduce a small helper module, likely `src/classiflow/models/ensemble.py`, to build bagged wrappers consistently.
- Recommended behavior:
  - keep sampler/scaler where they are today
  - wrap the classifier step only
  - use sklearn `BaggingClassifier` where possible
- Apply the wrapper in the non-meta training paths only:
  - `src/classiflow/backends/registry.py` for `train-binary`
  - `src/classiflow/models/estimators.py` for direct multiclass
  - project orchestration/final retraining for non-meta modes
- Do not apply bagging inside `train-meta`:
  - not to base binary learners
  - not to the meta/final estimator

### Parameter Grid Handling

- When bagging is enabled, base-estimator params will move under an estimator namespace, e.g. `clf__estimator__hidden_dim`.
- That means we also need to update:
  - metric logging
  - config extraction from inner-CV CSVs
  - final retraining code that reapplies saved params
- Files likely impacted:
  - `src/classiflow/training/nested_cv.py`
  - `src/classiflow/projects/final_train.py`
  - `src/classiflow/projects/orchestrator.py`

### Artifact and Metadata Updates

- Persist strategy and bagging settings into run manifests and config dumps.
- Make model naming explicit in outputs, e.g. append `_bagged` or a strategy field instead of silently replacing the estimator type.
- Ensure inference artifacts still expose stable `predict`/`predict_proba` behavior.
- Ensure reports and saved configs make it obvious when a selected model is a bagged wrapper instead of a single estimator.

## Feature 2: `--expanded-mlp-tuning-grid` for Any MLP

### User-Facing API

- Add:
  - `--expanded-mlp-tuning-grid/--no-expanded-mlp-tuning-grid`
- Carry the same option into project config so project technical-validation runs can reproduce it.

### Implementation Shape

- Create one shared MLP tuning-spec builder that both current integration paths consume.
- Recommended shape:
  - a shared helper module for MLP search-space definitions
  - one conservative/default profile
  - one expanded profile driven by `expanded_mlp_tuning_grid`
- Both consumers should adapt the same underlying spec rather than maintain separate grids:
  - binary/meta registry path via `src/classiflow/backends/registry.py`
  - direct multiclass path via `src/classiflow/models/estimators.py`
- Because the consumers expect different parameter names, the shared helper should emit normalized hyperparameter specs and the call sites should add the required prefixes:
  - pipeline path: `clf__...`
  - meta final-estimator path: bare estimator params

### Why Unifying the Grids Matters

- Today, binary/meta torch tuning and direct multiclass torch tuning are defined in different places with different defaults and different search spaces.
- If we only changed one side, users would get different MLP behavior depending on which command they ran.
- A single shared tuning spec removes that drift while still allowing each path to wrap the params in its own required naming convention.

### Torch MLP Architecture Changes Needed

- The current torch MLP implementation cannot represent the full CCIX MLP hyperparameter space because it lacks activation and batchnorm controls.
- Recommended extension:
  - add `activation` and `use_batchnorm` to the torch MLP modules in `src/classiflow/backends/torch/modules.py`
  - pass those through estimator wrappers in `src/classiflow/backends/torch/estimators.py`
  - expose the same knobs in `src/classiflow/models/torch_multiclass.py`
- This lets the expanded grid cover the same hyperparameter dimensions as CCIX instead of approximating them with a reduced model family.

### Expanded Grid Contents

- Keep the current default grid as the baseline when `--expanded-mlp-tuning-grid` is off.
- For expanded mode, include the CCIX MLP hyperparameter dimensions and values around the CCIX defaults rather than one hard-coded point:
  - hidden size around `512`
  - number of layers around `3`
  - dropout around `0.2`
  - activation options including `elu`
  - batchnorm on/off, with CCIX-style `True` represented
  - learning rate around `1e-4`
  - weight decay around `1e-6`
  - epochs around `50`
  - batch size around `256`
- Keep the expanded grid bounded; do not create an unmanageably large cartesian product.
- Apply the shared spec to any MLP exposed by classiflow:
  - torch MLPs in `src/classiflow/backends/registry.py`
  - direct multiclass torch MLPs in `src/classiflow/models/estimators.py`
  - any project/final-train path that reconstructs MLPs from selected configs

### Config Plumbing

- Add config plumbing in:
  - `src/classiflow/config.py`
  - `src/classiflow/cli/main.py`
  - `src/classiflow/projects/project_models.py`
  - `src/classiflow/projects/orchestrator.py`
  - `src/classiflow/projects/final_train.py` where final retraining should honor the richer MLP parameter set

## Work Breakdown

## Phase 1: Config and Plumbing

- Add new config fields for bagging strategy and expanded MLP grid.
- Expose them in CLI and project models.
- Ensure config serialization and manifests include the new fields.

## Phase 2: Torch MLP Capability Upgrade

- Extend torch MLP modules to support ELU and optional batchnorm.
- Update estimator wrappers and cloning behavior.
- Remove the current direct-multiclass wrapper mismatch where dropout is hard-coded to `0.0`.

## Phase 3: Shared MLP Tuning-Spec Integration

- Add one shared MLP tuning-spec builder.
- Update both torch integration paths to consume that shared spec.
- Add expanded-grid branches without duplicating the search-space definition in multiple modules.

## Phase 4: Bagging Integration

- Add bagging wrappers to the non-meta estimator factories.
- Update nested CV and non-meta project/final-train code to tolerate bagged estimator parameter namespaces.
- Exclude meta training and meta final-estimator selection from bagging.

## Phase 5: Project and Final-Train Integration

- Thread the same settings into project technical validation.
- Make final retraining capable of reconstructing bagged estimators from selected configs.
- Ensure selected-config extraction still works when best params include `estimator__*` names.

## Phase 6: Validation

- Add/extend tests for:
  - shared torch MLP tuning-spec contents
  - binary/meta registry consumption of the shared MLP grid
  - direct multiclass consumption of the shared MLP grid
  - bagged sklearn estimators
  - bagged torch estimators
  - explicit assertion that meta training does not enable bagging
  - project config round-trip
  - final retraining with bagged selected configs
- Add or modify smoke tests where user-facing entry points change:
  - CLI smoke coverage for the new `--expanded-mlp-tuning-grid` flag
  - CLI smoke coverage for bagging flags on non-meta commands
  - project/bootstrap/config smoke coverage if new config fields are surfaced there
  - ensure meta smoke tests explicitly confirm bagging flags are absent or rejected, depending on final CLI design

## Likely Files To Touch

- `src/classiflow/cli/main.py`
- `src/classiflow/config.py`
- `src/classiflow/backends/registry.py`
- `src/classiflow/backends/torch/modules.py`
- `src/classiflow/backends/torch/estimators.py`
- `src/classiflow/models/estimators.py`
- `src/classiflow/models/ensemble.py`
- `src/classiflow/models/torch_multiclass.py`
- `src/classiflow/training/nested_cv.py`
- `src/classiflow/training/meta.py`
- `src/classiflow/training/multiclass.py`
- `src/classiflow/projects/project_models.py`
- `src/classiflow/projects/orchestrator.py`
- `src/classiflow/projects/final_train.py`
- tests under `tests/unit`, `tests/training`, and `tests/integration`
- likely smoke-test files:
  - `tests/integration/test_cli_smoke.py`
  - `tests/integration/test_cli_config.py`
  - project/integration smoke tests that exercise bootstrap and end-to-end training entry points

## Main Risks

- There are currently two separate torch integration paths, so the shared tuning spec must be adopted in both or drift will continue.
- Bagging changes parameter names and cloning semantics, which can break config extraction and final retraining if not handled explicitly.
- Matching the CCIX hyperparameter space requires architecture extensions, not just a bigger grid.
- Bagged torch estimators may be materially slower, so defaults should stay conservative.

## Recommendation

Proceed in two implementation slices:

1. First land the shared MLP tuning spec plus the torch MLP capability upgrade and `--expanded-mlp-tuning-grid`.
2. Then land bagging for the non-meta paths once the MLP parameter surfaces are stable.

That sequencing reduces churn because the bagging wrapper will otherwise have to be updated twice as the torch estimator API changes.
