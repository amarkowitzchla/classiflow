# Changelog

All notable changes to classiflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-13

### Added

- Initial production release
- Nested cross-validation for binary classification
- Meta-classifier training for multiclass problems
- Adaptive SMOTE with automatic k-neighbors adjustment
- Task builder for OvR, pairwise, and composite tasks
- CLI tools (`classiflow train-binary`, `classiflow train-meta`)
- Streamlit web UI for interactive training
- Comprehensive metrics: accuracy, F1, MCC, ROC AUC, balanced accuracy
- Multi-metric inner CV with per-split metrics export
- Run manifests for reproducibility (git hash, timestamp, config)
- Artifact management (save/load models and pipelines)
- Unit tests for core components
- Full documentation and examples
- PyPI-ready packaging with `pyproject.toml`
- Optional dependencies for app, viz, stats, dev

### Changed

- Refactored from script-based workflow to package architecture
- Migrated from `requirements.txt` to `pyproject.toml`
- Improved error handling and logging throughout
- Standardized configuration using dataclasses
- Enhanced Streamlit UI with better state management

### Fixed

- Robust inner CV split adaptation for small minority classes
- Graceful SMOTE fallback when minority count is insufficient
- Deterministic random seeds for reproducibility

## [Unreleased]

### Planned

- Inference CLI and API for predictions on new data
- Summary and export commands for aggregating CV results
- ROC and confusion matrix plotting utilities
- Feature importance extraction and visualization
- Hierarchical task support
- Model comparison and selection tools
- Extended documentation with tutorials
- Integration tests for full workflows
- GitHub Actions CI/CD pipeline
