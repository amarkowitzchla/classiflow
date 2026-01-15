# classiflow Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                          │
├──────────────────────┬──────────────────────┬───────────────────┤
│   CLI (Typer)        │  Python API          │  Streamlit UI     │
│   classiflow train-*  │  TrainConfig         │  app.py + pages/  │
└──────────────────────┴──────────────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  train_binary_task() / train_meta_classifier()           │  │
│  │  • Load & validate data                                  │  │
│  │  • Build tasks (OvR, pairwise, composite)                │  │
│  │  • Run nested CV with orchestrator                       │  │
│  │  • Save artifacts & metrics                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Nested CV Orchestrator                         │
├─────────────────────────────────────────────────────────────────┤
│  Outer CV (Validation)          Inner CV (Hyperparameters)      │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │ For each fold:       │      │ GridSearchCV with:   │         │
│  │ • Split train/val    │──────▶│ • Multi-metric      │         │
│  │ • Train models       │      │ • SMOTE variants    │         │
│  │ • Evaluate on val    │      │ • Best param search │         │
│  └──────────────────────┘      └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Tasks      │  │   Models     │  │   Metrics    │
    ├──────────────┤  ├──────────────┤  ├──────────────┤
    │ TaskBuilder  │  │ Estimators   │  │ Binary       │
    │ • OvR        │  │ • LogReg     │  │ • Accuracy   │
    │ • Pairwise   │  │ • SVM        │  │ • F1 Score   │
    │ • Composite  │  │ • RF/GB      │  │ • ROC AUC    │
    │ JSON loading │  │ AdaptiveSMOTE│  │ • MCC        │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Module Dependency Graph

```
classiflow/
│
├── config.py ───────────────────────┐
│   (TrainConfig, MetaConfig)        │
│                                    │
├── io/ ─────────────────────────────┤
│   ├── loaders.py                   │
│   └── schema.py                    │
│                                    │
├── tasks/ ──────────────────────────┤
│   ├── builder.py                   │
│   └── composite.py                 │
│                                    │
├── models/ ─────────────────────────┤
│   ├── estimators.py                │
│   └── smote.py                     │
│                                    │
├── metrics/ ────────────────────────┤
│   ├── binary.py                    │
│   └── scorers.py                   │
│                                    │
│       ▲  ▲  ▲  ▲  ▲  ▲             │
│       │  │  │  │  │  │             │
│       └──┴──┴──┴──┴──┘             │
│                                    │
├── training/ ◀──────────────────────┘
│   ├── nested_cv.py    (uses all above)
│   ├── binary.py       (uses nested_cv)
│   └── meta.py         (uses nested_cv + tasks)
│           │
│           ▼
├── artifacts/
│   ├── saver.py
│   └── loader.py
│           │
│           ▼
├── cli/
│   └── main.py ──────────▶ training/*
│           │
│           ▼
└── streamlit_app/
    ├── app.py ───────────▶ training/*
    └── pages/
        └── 01_Train_Models.py
```

## Data Flow: Meta-Classifier Training

```
1. User Input
   ├─ data.csv (features + labels)
   ├─ tasks.json (optional composite tasks)
   └─ config (folds, SMOTE, seed, etc.)
                │
                ▼
2. Load & Validate Data
   ├─ io.loaders.load_data()
   ├─ io.loaders.validate_data()
   └─ io.schema.DataSchema.from_data()
                │
                ▼
3. Build Tasks
   ├─ tasks.builder.TaskBuilder(classes)
   │  ├─ .build_ovr_tasks()          → {A_vs_Rest, B_vs_Rest, ...}
   │  └─ .build_pairwise_tasks()     → {A_vs_B, A_vs_C, ...}
   └─ tasks.composite.load_composite_tasks(json_path)
                │                      → {Custom_Task_1, ...}
                ▼
4. Nested CV Loop (Outer Folds)
   For each outer_fold in [1, 2, 3]:
      ├─ Split: train_idx, val_idx
      │
      ├─ For each SMOTE variant in [smote, none]:
      │    │
      │    ├─ Train Binary Tasks (Inner CV)
      │    │   For each task in tasks:
      │    │      ├─ Extract binary labels: y_bin = task(y_train)
      │    │      ├─ For each model in [LogReg, SVM, RF, GB]:
      │    │      │    ├─ Build pipeline: [SMOTE?, VarThreshold, Scaler, Estimator]
      │    │      │    ├─ GridSearchCV with inner CV (RepeatedStratifiedKFold)
      │    │      │    │    └─ Multi-metric: [Acc, Prec, F1, MCC, Sens, Spec, AUC, BAcc]
      │    │      │    ├─ Select best params (refit on F1)
      │    │      │    └─ Evaluate on train & val
      │    │      └─ Select best model per task (highest F1)
      │    │
      │    ├─ Build Meta-Features
      │    │    For each task:
      │    │       ├─ Get best binary model for task
      │    │       ├─ Extract scores: scores = model.predict_proba(X)[:, 1]
      │    │       └─ Create meta-feature column: task_score
      │    │    Result: X_meta = [task1_score, task2_score, ..., taskN_score]
      │    │
      │    └─ Train Meta-Classifier
      │         ├─ Model: LogisticRegression(multi_class='multinomial')
      │         ├─ GridSearchCV for C hyperparameter
      │         ├─ Fit: meta_model.fit(X_meta_train, y_train)
      │         ├─ Predict: y_pred = meta_model.predict(X_meta_val)
      │         └─ Evaluate: accuracy, f1_macro, f1_weighted, ROC AUC
      │
      └─ Save Fold Artifacts
           ├─ fold{N}/binary_{variant}/binary_pipes.joblib
           ├─ fold{N}/binary_{variant}/meta_model.joblib
           ├─ fold{N}/binary_{variant}/meta_features.csv
           └─ fold{N}/binary_{variant}/meta_classes.csv
                │
                ▼
5. Aggregate & Export
   ├─ metrics_inner_cv.csv               (all GridSearchCV candidates)
   ├─ metrics_inner_cv_splits.{csv,xlsx} (per-split metrics for best params)
   ├─ metrics_outer_binary_eval.csv      (binary task train/val metrics)
   ├─ metrics_outer_meta_eval.csv        (meta-classifier train/val metrics)
   └─ run_manifest.json                  (config + git hash + timestamp)
                │
                ▼
6. Output
   ├─ Trained models in fold{N}/
   ├─ Metrics CSVs for analysis
   ├─ Reproducibility manifest
   └─ Ready for:
      ├─ Inference (load models + predict on new data)
      ├─ Summarization (aggregate CV metrics)
      ├─ Visualization (ROC, confusion, calibration)
      └─ Publication (tables, figures, supplementary data)
```

## Adaptive SMOTE Flow

```
Pipeline: [AdaptiveSMOTE, VarianceThreshold, StandardScaler, Estimator]
                │
                ▼
AdaptiveSMOTE.fit_resample(X_train, y_train):
    │
    ├─ Check: Is y_train binary (0/1)?
    │   ├─ No  → Pass through (X_train, y_train)
    │   └─ Yes → Continue
    │
    ├─ Count minority class: minority = min(y_train.value_counts())
    │
    ├─ Check: minority > 1?
    │   ├─ No  → Pass through (too few samples)
    │   └─ Yes → Continue
    │
    ├─ Adapt k_neighbors: k = max(1, min(k_max, minority - 1))
    │
    ├─ Apply SMOTE: sm = SMOTE(k_neighbors=k)
    │                X_res, y_res = sm.fit_resample(X_train, y_train)
    │
    └─ Return: (X_res, y_res) with balanced classes
```

## CLI → Library Call Chain

```
$ classiflow train-meta --data-csv data.csv --label-col subtype --smote both

    │
    ▼
cli/main.py:train_meta()
    │
    ├─ Parse arguments
    ├─ Build MetaConfig(data_csv, label_col, smote_mode='both', ...)
    │
    └─ Call: training.train_meta_classifier(config)
              │
              ▼
         training/meta.py:train_meta_classifier(config)
              │
              ├─ Load data: io.load_data()
              ├─ Build tasks: tasks.TaskBuilder(...).build_all_auto_tasks()
              │
              └─ Call: _run_meta_nested_cv(X, y, tasks, config)
                    │
                    ├─ Outer CV: StratifiedKFold.split(X, y)
                    │
                    └─ For each fold:
                          │
                          ├─ Train binary tasks: _train_binary_tasks(...)
                          │     │
                          │     └─ Uses: models.get_estimators()
                          │               models.AdaptiveSMOTE()
                          │               metrics.get_scorers()
                          │               sklearn.GridSearchCV
                          │
                          ├─ Build meta-features: _build_meta_features(...)
                          │
                          ├─ Train meta-model: _train_meta_model(...)
                          │
                          └─ Save artifacts: artifacts.save_*()
```

## Backward Compatibility Layer

```
Legacy Streamlit Page (pages/01_Train_Models.py)
    │
    ├─ Import: from utils.wrappers import run_train_meta_classifier
    │
    └─ Call: run_train_meta_classifier(root, csv_path, tasks_json, ...)
                │
                ▼
         utils/compat.py:run_train_meta_classifier()
                │
                ├─ Emit DeprecationWarning
                │
                ├─ Check: Is classiflow package installed?
                │   │
                │   ├─ Yes → Build MetaConfig → Call classiflow.train_meta_classifier()
                │   │
                │   └─ No  → Fallback: subprocess.run(scripts/train_binary_meta_classifier.py)
                │
                └─ Return results
```

## Key Design Patterns

### 1. **Separation of Concerns**
- **Config**: Dataclasses (config.py)
- **I/O**: Loading + validation (io/)
- **Logic**: Training orchestration (training/)
- **Persistence**: Artifacts save/load (artifacts/)
- **UI**: CLI + Streamlit (cli/, streamlit_app/)

### 2. **Dependency Injection**
- Pass `config` objects instead of scattered arguments
- Pass estimators/scorers/tasks as dictionaries
- Configurable via constructor or factory functions

### 3. **Builder Pattern**
- `TaskBuilder`: Fluent API for constructing tasks
  ```python
  builder = TaskBuilder(classes)
      .build_ovr_tasks()
      .build_pairwise_tasks()
      .add_composite_task("Custom", pos, neg)
  tasks = builder.get_tasks()
  ```

### 4. **Strategy Pattern**
- SMOTE variants: `"off"`, `"on"`, `"both"` → different samplers
- Model selection: Dictionary of estimators + param grids

### 5. **Adapter Pattern**
- `AdaptiveSMOTE`: Adapts SMOTE to work with GridSearchCV
- Implements `fit_resample()` compatible with imblearn.Pipeline

### 6. **Facade Pattern**
- `train_binary_task(config)`: High-level API hides complexity
- `train_meta_classifier(config)`: High-level API for meta pipeline

---

## Testing Strategy

```
Unit Tests (tests/unit/)
├── test_tasks.py          → TaskBuilder logic
├── test_smote.py          → AdaptiveSMOTE behavior
└── test_metrics.py        → Binary metrics computation

Integration Tests (future)
└── test_workflows.py      → Full training + inference pipelines

Fixtures (tests/conftest.py)
├── sample_binary_data     → 100 samples, 20 features, 2 classes
├── sample_multiclass_data → 150 samples, 20 features, 3 classes
└── temp_outdir            → Temporary directory for outputs
```

---

## Production Considerations

### ✅ Implemented

1. **Type Safety**: Dataclasses + type hints
2. **Logging**: Structured logging (not print)
3. **Error Handling**: Validation + meaningful errors
4. **Determinism**: Fixed seeds, run manifests
5. **Documentation**: Docstrings + README + guides
6. **Testing**: Unit tests with fixtures
7. **Packaging**: pyproject.toml + src/ layout
8. **Versioning**: Semantic versioning (0.1.0)
9. **Licensing**: MIT license
10. **Citation**: CITATION.cff for academic use

### 🔄 Future Work

1. **Inference Pipeline**: Load models + predict on new data
2. **Plotting**: ROC curves, confusion matrices, calibration
3. **Summarization**: Aggregate CV metrics across folds
4. **Export**: Best task spreadsheets
5. **Integration Tests**: End-to-end workflows
6. **CI/CD**: GitHub Actions pipeline
7. **Performance**: Parallel processing, caching
8. **Extended Docs**: Tutorials, API reference

---

## Summary

The architecture is:
- **Modular**: Clear boundaries between components
- **Testable**: Unit tests for core logic
- **Extensible**: Easy to add models, metrics, tasks
- **Maintainable**: Type hints, docstrings, logging
- **Production-Ready**: Packaging, versioning, documentation

**Ready for PyPI publication and academic citation! 🚀**
