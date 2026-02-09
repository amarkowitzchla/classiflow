# {project_name}

Test ID: `{project_id}`

## Overview

Describe the clinical test, intended use, and data provenance.

## Quick Commands

```bash
classiflow project register-dataset --type train
classiflow project run-feasibility
classiflow project run-technical
classiflow project build-bundle
classiflow project run-test
classiflow project recommend
```

## Notes

- Update `project.yaml` to reflect your manifest columns.
- Update `registry/thresholds.yaml` with promotion gates.
- Calibration gates are optional: set `promotion.calibration.brier_max` and/or
  `promotion.calibration.ece_max` only when you want them to gate promotion.
- Store reviewer feedback in `promotion/`.
- For hierarchical workflows, set `task.hierarchy_path` to the L2 label column.
