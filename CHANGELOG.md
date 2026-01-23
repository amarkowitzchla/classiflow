# Changelog

All notable user-visible changes to `classiflow` must be documented here.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning:
- https://keepachangelog.com/en/1.0.0/
- https://semver.org/spec/v2.0.0.html

Guidelines:
- Record changes that affect CLI behavior, defaults, outputs/artifacts, bundle layouts, manifests, schemas, and APIs.
- Prefer linking changes to PRs/issues and note any migration/compatibility guidance.
- Do not invent historical entries; add entries starting from the point this file is adopted.

## [Unreleased]

### Added

- **Experiment Tracking Integration**: Optional MLflow and Weights & Biases support for all training commands
  - New `tracking` module with pluggable tracker architecture
  - CLI flags: `--tracker`, `--experiment-name`, `--run-name` for all `train-*` commands
  - Automatic logging of parameters, metrics, and artifacts
  - Project workflow integration via `project.yaml` config
  - Optional dependencies: `pip install classiflow[mlflow]` or `pip install classiflow[wandb]`
  - See `docs/TRACKING_GUIDE.md` for full documentation

### Changed

### Deprecated

### Removed

### Fixed

### Security
