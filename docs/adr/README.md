# Architecture Decision Records (ADRs)

ADRs document **high-impact design decisions** so changes remain reviewable, reproducible, and compatible over time.

## When an ADR is required
Write an ADR for **High risk** changes as defined in `docs/change-management/README.md`, especially when touching:
- `bundles` (bundle layout/required files/version checks)
- `inference` (artifact loading, feature alignment rules, output schemas)
- `metrics` (metric definitions, scorer order, column names)
- `projects` (project filesystem layout, promotion gate semantics)
- `splitting` (group/patient split logic, determinism, leakage checks)
- `training` (nested CV semantics, artifacts/manifests, split wiring)

Also write an ADR when:
- making a backward-incompatible CLI change (command/flag removal or meaning change)
- changing any on-disk schema (`run.json`, metrics CSV schemas, `artifacts.json`, project registries)
- introducing a new optional dependency that changes runtime behavior

## Naming convention
- File name: `NNNN-short-title.md` (e.g., `0007-bundle-version-compat.md`)
- `NNNN` is a zero-padded sequence number. Pick the next available number.

## ADR template
Copy/paste this skeleton:

```md
# ADR NNNN: <Title>

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What problem are we solving? What invariants/constraints apply?
Link relevant `docs/change-management/<module>.md` and any issues/PRs.

## Decision
What are we doing, precisely?

## Consequences
### Positive
- ...
### Negative / tradeoffs
- ...

## Compatibility
- CLI flags:
- Artifacts/manifests/bundles:
- Migrations/deprecations:

## Testing plan
- Unit:
- Integration:
- Regression/golden fixtures:

## Rollout / release notes
What goes into `CHANGELOG.md` and what users must do (if anything).
```

## Review checklist (ADRs)
- Identifies risk level and affected modules
- States leakage/determinism implications (if applicable)
- Lists on-disk schema changes (filenames, keys, columns) explicitly
- Includes backward compatibility and migration plan
- Lists tests added/updated and how to run them

