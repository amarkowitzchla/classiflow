# Production Release & Distribution Guide

This document describes how to finalize `classiflow` for production, package it, and transfer the built artifact from one machine to another (along with model bundles if desired).

You should run the following steps from the repository root (`project-MLSubtype`). Start by confirming you are in the right place:

```bash
pwd  # /Users/.../project-MLSubtype
ls -la
```

The listing should include `README.md`, `pyproject.toml`, `docs/`, `src/`, `tests/`, plus directories like `derived/`, `data/`, and `tmp/`.

`docs/PRODUCTION_RELEASE.md` lives inside the `docs/` directory so it can be referenced alongside other guides.

## 1. Pre-flight Checklist

1. **Version bump**  
   - Update `pyproject.toml` `project.version` and any release notes (e.g., `CHANGELOG.md`).
   - Re-run tests (`python -m pytest`) and linters (`ruff`, `black`, `mypy` as needed) to ensure a clean slate.
2. **Documentation sanity**  
   - Confirm `README.md`, `HIERARCHICAL_TRAINING.md`, and other docs reflect the release (features, CLI usage).
   - Generate new artefact references (bundles, stats) from the release candidate run if you want to embed those links.
3. **Artifact metadata**  
   - When you run `classiflow train-*`, the package writes `run.json` and `run_manifest.json` for lineage. Keep at least one such run under `derived/` as a reference for the release.
4. **Git hygiene**  
   - `git status` should be clean.
   - Pull the latest changes: `git fetch` + `git pull` (or rebase) from your release branch.
   - Create a release branch if needed: `git checkout -b release/vX.Y.Z`.

## 2. Build & Validate

1. **Install build tools** (if not already available):

   ```bash
   pip install build twine
   ```

2. **Clean previous artifacts**:

   ```bash
   rm -rf build dist *.egg-info
   ```

3. **Build sdist + wheel**:

   ```bash
   python -m build
   ```

   Result: `dist/classiflow-<version>-py3-none-any.whl` and an sdist.

4. **Smoke test the wheel locally**:

   ```bash
   python -m venv /tmp/classiflow-test
   source /tmp/classiflow-test/bin/activate
   pip install dist/classiflow-*.whl
   classiflow --help
   deactivate
   rm -rf /tmp/classiflow-test
   ```

   This ensures the CLI entry point, dependencies, and entry scripts are preserved.

5. **Create model bundles for inference (optional)**  
   If you have production-trained models, package them with `classiflow bundle` for subsequent deployment:

   ```bash
   classiflow bundle create --run-dir derived/final_run --out artifacts/classiflow_production.zip --all-folds
   classiflow bundle validate artifacts/classiflow_production.zip
   ```

   Keep the ZIP alongside the Python wheel for a full transfer.

## 3. Uploading the Package

1. **Publish to PyPI (or private index)**:

   ```bash
   python -m twine upload dist/*
   ```

   - Use `--repository-url` or `.pypirc` to target a private repository if needed.
   - For strict environments, run `python -m twine check dist/*` before uploading.

2. **Share via file transfer**  
   In restricted environments where you cannot hit PyPI, transfer the wheel manually:

   ```bash
   scp dist/classiflow-0.1.0-py3-none-any.whl user@remote:/tmp/
   scp artifacts/classiflow_production.zip user@remote:/tmp/
   ```

   Or use `rsync`, `sftp`, or mount/network share – the goal is to make the wheel (and bundle ZIP) available on the target host.

## 4. Installing on Another Machine

1. **Prepare the target** (ensure Python 3.9+):

   ```bash
   python -m venv ~/envs/classiflow
   source ~/envs/classiflow/bin/activate
   pip install --upgrade pip
   ```

2. **Install the wheel** (from local file or PyPI):

   ```bash
   pip install /tmp/classiflow-*.whl
   ```

   Or, if you published to PyPI:

   ```bash
   pip install classiflow==0.1.0
   ```

3. **Verify the CLI**:

   ```bash
   classiflow --help
   classiflow train-binary --help
   classiflow bundle inspect /tmp/classiflow_production.zip
   ```

4. **Copy over model bundle** (if relevant):

   ```bash
   unzip /tmp/classiflow_production.zip -d ~/models/classiflow_prod
   classiflow infer --bundle ~/models/classiflow_prod/my_model.zip --data-csv data.csv
   ```

   - Keep the bundle zipped until deployment; use `bundle inspect`/`validate` to confirm integrity.

## 5. Post-release Hygiene

- Tag the release in git (`git tag -a v0.1.0 -m "Release 0.1.0"`) and push tags.
- Update `CHANGELOG.md` and `HIERARCHICAL_TRAINING.md` with any late-breaking notes.
- Archive `dist/` artifacts if you retain them for auditing, or upload to a release storage (GitHub Releases, internal bucket).
- If you have model bundles, version them alongside the wheel so both training code and inference artifacts move together.
