"""Local filesystem scanner for discovering projects, runs, and artifacts."""

from __future__ import annotations

import hashlib
import logging
import math
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numbers
from typing import Optional

from classiflow.projects.promotion import evaluate_promotion
from classiflow.ui_api.adapters.manifest import (
    RunManifestNormalized,
    parse_run_manifest,
    parse_metrics,
)
from classiflow.ui_api.adapters.project import (
    ProjectConfigNormalized,
    parse_project_config,
    parse_datasets_registry,
    parse_thresholds,
    parse_decision,
    get_project_updated_at,
    DatasetInfo,
    ThresholdsConfig,
    DecisionResult,
)
from classiflow.ui_api.models import (
    ArtifactKind,
    DecisionBadge,
    MetricsSummary,
    GateCheck,
    GateResult,
)

logger = logging.getLogger(__name__)

# Allowlisted extensions for serving
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".svg", ".gif",  # Images
    ".pdf",  # Documents
    ".html", ".htm",  # Web
    ".md", ".txt",  # Text
    ".json", ".yaml", ".yml",  # Config
    ".csv", ".xlsx", ".xls",  # Data
    ".joblib", ".pkl", ".pickle",  # Models
    ".zip",  # Archives
}

# Extensions that can be viewed inline
VIEWABLE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".svg", ".gif",  # Images
    ".pdf",  # Documents
    ".md", ".txt",  # Text
    ".json", ".yaml", ".yml",  # Config
    ".csv",  # Tabular
    ".html", ".htm",  # Web (sandboxed)
}

# Map extensions to artifact kinds
EXTENSION_KINDS = {
    ".png": ArtifactKind.IMAGE,
    ".jpg": ArtifactKind.IMAGE,
    ".jpeg": ArtifactKind.IMAGE,
    ".svg": ArtifactKind.IMAGE,
    ".gif": ArtifactKind.IMAGE,
    ".md": ArtifactKind.REPORT,
    ".html": ArtifactKind.REPORT,
    ".htm": ArtifactKind.REPORT,
    ".json": ArtifactKind.METRICS,
    ".csv": ArtifactKind.METRICS,
    ".xlsx": ArtifactKind.METRICS,
    ".xls": ArtifactKind.METRICS,
    ".yaml": ArtifactKind.CONFIG,
    ".yml": ArtifactKind.CONFIG,
    ".joblib": ArtifactKind.MODEL,
    ".pkl": ArtifactKind.MODEL,
    ".pickle": ArtifactKind.MODEL,
    ".zip": ArtifactKind.MODEL,
    ".pdf": ArtifactKind.REPORT,
    ".txt": ArtifactKind.OTHER,
}


@dataclass
class ScannedProject:
    """Scanned project with all discovered metadata."""

    id: str
    path: Path
    config: ProjectConfigNormalized
    datasets: dict[str, DatasetInfo] = field(default_factory=dict)
    thresholds: Optional[ThresholdsConfig] = None
    decision: Optional[DecisionResult] = None
    updated_at: Optional[datetime] = None
    phases: dict[str, list[str]] = field(default_factory=dict)  # phase -> [run_ids]
    run_count: int = 0


@dataclass
class ScannedRun:
    """Scanned run with manifest and metrics."""

    manifest: RunManifestNormalized
    metrics: dict
    cache_signature: int
    artifact_paths: list[str] = field(default_factory=list)
    artifact_count: int = 0


@dataclass
class ScannedArtifact:
    """Scanned artifact metadata."""

    artifact_id: str
    relative_path: str
    absolute_path: Path
    run_key: str
    phase: str
    kind: ArtifactKind
    mime_type: Optional[str]
    size_bytes: int
    created_at: Optional[datetime]
    is_viewable: bool
    is_allowed: bool
    title: str


class LocalFilesystemScanner:
    """Scanner for discovering Classiflow projects and runs on local filesystem."""

    def __init__(self, projects_root: Path):
        """
        Initialize scanner.

        Parameters
        ----------
        projects_root : Path
            Root directory containing project directories
        """
        self.projects_root = Path(projects_root).resolve()
        if not self.projects_root.is_dir():
            raise ValueError(f"Projects root is not a directory: {self.projects_root}")

        self._project_cache: dict[str, ScannedProject] = {}
        self._run_cache: dict[str, ScannedRun] = {}

    @staticmethod
    def _run_cache_signature(run_dir: Path) -> int:
        """
        Return a lightweight signature used to invalidate stale run-cache entries.

        Hash-like signature is based on mtime of core metadata/metrics files that
        affect run detail payloads shown in UI.
        """
        candidates = [
            run_dir / "run.json",
            run_dir / "lineage.json",
            run_dir / "metrics.json",
            run_dir / "metrics_summary.json",
            run_dir / "metrics_outer_binary_eval.csv",
            run_dir / "metrics_outer_meta_eval.csv",
            run_dir / "metrics_outer_multiclass_eval.csv",
            run_dir / "metrics_outer_eval.csv",
            run_dir / "metrics" / "overall_metrics.csv",
            run_dir / "metrics" / "per_class_metrics.csv",
            run_dir / "metrics" / "confusion_matrix.csv",
        ]

        mtimes: list[int] = []
        try:
            mtimes.append(run_dir.stat().st_mtime_ns)
        except OSError:
            pass
        for path in candidates:
            if not path.exists():
                continue
            try:
                mtimes.append(path.stat().st_mtime_ns)
            except OSError:
                continue
        return max(mtimes) if mtimes else 0

    def scan_projects(self, force: bool = False) -> list[ScannedProject]:
        """
        Scan for all projects.

        Parameters
        ----------
        force : bool
            Force rescan even if cached

        Returns
        -------
        list[ScannedProject]
            List of discovered projects
        """
        if self._project_cache and not force:
            return list(self._project_cache.values())

        self._project_cache.clear()
        projects = []

        for entry in self.projects_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue

            # Check if it looks like a project
            project_yaml = entry / "project.yaml"
            runs_dir = entry / "runs"

            # Accept if has project.yaml OR runs directory OR matches naming convention
            has_project_yaml = project_yaml.exists()
            has_runs = runs_dir.is_dir()
            matches_pattern = "__" in entry.name

            if not (has_project_yaml or has_runs or matches_pattern):
                continue

            try:
                project = self._scan_single_project(entry)
                projects.append(project)
                self._project_cache[project.id] = project
            except Exception as e:
                logger.warning(f"Failed to scan project {entry}: {e}")

        return projects

    def _scan_single_project(self, project_dir: Path) -> ScannedProject:
        """Scan a single project directory."""
        project_id = project_dir.name

        # Parse config
        config = parse_project_config(project_dir)

        # Parse registry
        datasets = parse_datasets_registry(project_dir)
        thresholds = parse_thresholds(project_dir)
        decision = parse_decision(project_dir)

        # Get update time
        updated_at = get_project_updated_at(project_dir)

        # Scan phases and runs
        phases: dict[str, list[str]] = {}
        run_count = 0

        runs_dir = project_dir / "runs"
        if runs_dir.is_dir():
            for phase_dir in sorted(runs_dir.iterdir()):
                if not phase_dir.is_dir():
                    continue
                if phase_dir.name.startswith("."):
                    continue

                phase_name = phase_dir.name
                run_ids = []

                for run_dir in sorted(phase_dir.iterdir(), reverse=True):
                    if not run_dir.is_dir():
                        continue
                    if run_dir.name.startswith("."):
                        continue
                    run_ids.append(run_dir.name)
                    run_count += 1

                if run_ids:
                    phases[phase_name] = run_ids

        return ScannedProject(
            id=project_id,
            path=project_dir,
            config=config,
            datasets=datasets,
            thresholds=thresholds,
            decision=decision,
            updated_at=updated_at,
            phases=phases,
            run_count=run_count,
        )

    def get_project(self, project_id: str) -> Optional[ScannedProject]:
        """Get a specific project by ID."""
        if project_id in self._project_cache:
            return self._project_cache[project_id]

        project_dir = self.projects_root / project_id
        if not project_dir.is_dir():
            return None

        try:
            project = self._scan_single_project(project_dir)
            self._project_cache[project_id] = project
            return project
        except Exception as e:
            logger.warning(f"Failed to get project {project_id}: {e}")
            return None

    def get_run(self, project_id: str, phase: str, run_id: str) -> Optional[ScannedRun]:
        """Get a specific run."""
        run_key = f"{project_id}:{phase}:{run_id}"
        run_dir = self.projects_root / project_id / "runs" / phase / run_id

        if not run_dir.is_dir():
            return None

        signature = self._run_cache_signature(run_dir)
        cached = self._run_cache.get(run_key)
        if cached and cached.cache_signature == signature:
            return cached

        try:
            scanned = self._scan_single_run(run_dir, project_id, phase, signature=signature)
            self._run_cache[run_key] = scanned
            return scanned
        except Exception as e:
            logger.warning(f"Failed to get run {run_key}: {e}")
            return None

    def get_run_by_key(self, run_key: str) -> Optional[ScannedRun]:
        """Get a run by its composite key."""
        parts = run_key.split(":")
        if len(parts) != 3:
            return None
        return self.get_run(parts[0], parts[1], parts[2])

    @staticmethod
    def _sanitize_metrics(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                cleaned_item = LocalFilesystemScanner._sanitize_metrics(item)
                if cleaned_item is None:
                    continue
                cleaned[key] = cleaned_item
            return cleaned
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                cleaned_item = LocalFilesystemScanner._sanitize_metrics(item)
                if cleaned_item is None:
                    continue
                cleaned_list.append(cleaned_item)
            return cleaned_list
        if isinstance(value, numbers.Real):
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        return value

    def _scan_single_run(
        self,
        run_dir: Path,
        project_id: str,
        phase: str,
        *,
        signature: Optional[int] = None,
    ) -> ScannedRun:
        """Scan a single run directory."""
        manifest = parse_run_manifest(run_dir, project_id, phase)
        metrics = parse_metrics(run_dir, phase)
        metrics = self._sanitize_metrics(metrics)

        # Scan artifacts
        artifact_paths = []
        for path in self._iter_artifacts(run_dir):
            rel_path = str(path.relative_to(run_dir))
            artifact_paths.append(rel_path)

        return ScannedRun(
            manifest=manifest,
            metrics=metrics,
            cache_signature=signature if signature is not None else self._run_cache_signature(run_dir),
            artifact_paths=artifact_paths,
            artifact_count=len(artifact_paths),
        )

    def _iter_artifacts(self, run_dir: Path, max_depth: int = 4) -> list[Path]:
        """Iterate over artifact files in a run directory."""
        artifacts = []

        def _scan(directory: Path, depth: int):
            if depth > max_depth:
                return

            try:
                for entry in directory.iterdir():
                    if entry.name.startswith("."):
                        continue

                    if entry.is_file():
                        ext = entry.suffix.lower()
                        if ext in ALLOWED_EXTENSIONS:
                            artifacts.append(entry)
                    elif entry.is_dir():
                        _scan(entry, depth + 1)
            except PermissionError:
                pass

        _scan(run_dir, 0)
        return artifacts

    def get_artifacts(self, project_id: str, phase: str, run_id: str) -> list[ScannedArtifact]:
        """Get all artifacts for a run."""
        run_key = f"{project_id}:{phase}:{run_id}"
        run_dir = self.projects_root / project_id / "runs" / phase / run_id

        if not run_dir.is_dir():
            return []

        artifacts = []
        for path in self._iter_artifacts(run_dir):
            artifact = self._scan_artifact(path, run_dir, run_key, phase)
            if artifact:
                artifacts.append(artifact)

        return artifacts

    def _scan_artifact(
        self, path: Path, run_dir: Path, run_key: str, phase: str
    ) -> Optional[ScannedArtifact]:
        """Scan a single artifact file."""
        try:
            rel_path = str(path.relative_to(run_dir))
            ext = path.suffix.lower()

            # Generate stable artifact ID
            artifact_id = self._generate_artifact_id(run_key, rel_path)

            # Get file stats
            stat = path.stat()
            size_bytes = stat.st_size
            created_at = datetime.fromtimestamp(stat.st_mtime)

            # Determine kind and MIME type
            kind = EXTENSION_KINDS.get(ext, ArtifactKind.OTHER)
            mime_type, _ = mimetypes.guess_type(str(path))

            # Determine if viewable
            is_viewable = ext in VIEWABLE_EXTENSIONS
            is_allowed = ext in ALLOWED_EXTENSIONS

            # Generate title
            title = path.name
            if "/" in rel_path:
                # Include parent folder for context
                parts = rel_path.rsplit("/", 1)
                if parts[0]:
                    title = f"{parts[0]}/{parts[1]}"

            return ScannedArtifact(
                artifact_id=artifact_id,
                relative_path=rel_path,
                absolute_path=path,
                run_key=run_key,
                phase=phase,
                kind=kind,
                mime_type=mime_type,
                size_bytes=size_bytes,
                created_at=created_at,
                is_viewable=is_viewable,
                is_allowed=is_allowed,
                title=title,
            )
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to scan artifact {path}: {e}")
            return None

    def _generate_artifact_id(self, run_key: str, relative_path: str) -> str:
        """Generate a stable artifact ID."""
        content = f"{run_key}:{relative_path}"
        return hashlib.sha1(content.encode()).hexdigest()[:16]

    def get_artifact_by_id(self, artifact_id: str) -> Optional[ScannedArtifact]:
        """
        Find an artifact by its ID.

        Note: This requires scanning all runs to find the artifact.
        In production, would use a database index.
        """
        # First check if we have it in any cached run
        for run_key, scanned_run in self._run_cache.items():
            parts = run_key.split(":")
            if len(parts) != 3:
                continue

            project_id, phase, run_id = parts
            run_dir = self.projects_root / project_id / "runs" / phase / run_id

            for rel_path in scanned_run.artifact_paths:
                check_id = self._generate_artifact_id(run_key, rel_path)
                if check_id == artifact_id:
                    path = run_dir / rel_path
                    return self._scan_artifact(path, run_dir, run_key, phase)

        # Not found in cache - would need full scan
        return None

    def resolve_artifact_path(
        self, project_id: str, phase: str, run_id: str, relative_path: str
    ) -> Optional[Path]:
        """
        Safely resolve an artifact path, preventing path traversal.

        Returns None if path is invalid or escapes the run directory.
        """
        run_dir = self.projects_root / project_id / "runs" / phase / run_id

        if not run_dir.is_dir():
            return None

        # Normalize and resolve path
        try:
            # Join and resolve
            target = (run_dir / relative_path).resolve()

            # Security check: must be under run_dir
            if not str(target).startswith(str(run_dir.resolve())):
                logger.warning(f"Path traversal attempt: {relative_path}")
                return None

            # Must not be a symlink pointing outside
            if target.is_symlink():
                real_target = target.resolve()
                if not str(real_target).startswith(str(run_dir.resolve())):
                    logger.warning(f"Symlink escape attempt: {relative_path}")
                    return None

            # Must exist and be a file
            if not target.is_file():
                return None

            # Extension must be allowed
            ext = target.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                logger.warning(f"Disallowed extension: {ext}")
                return None

            return target

        except (ValueError, OSError) as e:
            logger.warning(f"Path resolution failed: {e}")
            return None

    def get_decision_badge(self, project_id: str) -> DecisionBadge:
        """Get decision badge for a project."""
        project = self.get_project(project_id)
        if not project or not project.decision:
            return DecisionBadge.PENDING

        decision = project.decision.decision.upper()
        if decision == "PASS":
            return DecisionBadge.PASS
        elif decision == "FAIL":
            return DecisionBadge.FAIL
        elif project.decision.override_enabled:
            return DecisionBadge.OVERRIDE
        return DecisionBadge.PENDING

    def get_headline_metrics(
        self, project_id: str, phase: str, run_id: str
    ) -> dict[str, float]:
        """Get headline metrics for display."""
        run = self.get_run(project_id, phase, run_id)
        if not run:
            return {}

        metrics = run.metrics.get("summary", {})
        headline = {}

        # Priority order for headline metrics
        priority_keys = [
            "balanced_accuracy",
            "f1_macro",
            "f1",
            "accuracy",
            "roc_auc_macro",
            "mcc",
        ]

        for key in priority_keys:
            if key in metrics:
                headline[key] = metrics[key]
            if len(headline) >= 3:
                break

        return headline

    def clear_cache(self):
        """Clear all cached data."""
        self._project_cache.clear()
        self._run_cache.clear()

    def compute_gate_results(self, project_id: str) -> dict[str, GateResult]:
        """
        Compute detailed gate results for each phase.

        Uses thresholds.yaml and metrics from runs to determine gate status.
        """
        project = self.get_project(project_id)
        if not project:
            return {}

        thresholds = project.thresholds
        if not thresholds:
            return {}

        decision = project.decision
        results: dict[str, GateResult] = {}

        phase_labels = {
            "technical_validation": "Technical Validation",
            "independent_test": "Independent Test",
        }

        def _numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
            filtered: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, numbers.Real):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        filtered[key] = numeric
            return filtered

        tech_run_id = decision.technical_run if decision else None
        if not tech_run_id and "technical_validation" in project.phases:
            tech_run_id = (
                project.phases["technical_validation"][0]
                if project.phases["technical_validation"]
                else None
            )

        test_run_id = decision.test_run if decision else None
        if not test_run_id and "independent_test" in project.phases:
            test_run_id = (
                project.phases["independent_test"][0]
                if project.phases["independent_test"]
                else None
            )

        tech_run = (
            self.get_run(project_id, "technical_validation", tech_run_id)
            if tech_run_id
            else None
        )
        test_run = (
            self.get_run(project_id, "independent_test", test_run_id)
            if test_run_id
            else None
        )

        tech_metrics = tech_run.metrics.get("summary", {}) if tech_run else {}
        tech_per_fold = tech_run.metrics.get("per_fold", {}) if tech_run else {}
        test_metrics = test_run.metrics.get("summary", {}) if test_run else {}

        promotion_results = evaluate_promotion(
            thresholds=thresholds,
            technical_metrics=tech_metrics,
            technical_per_fold=tech_per_fold,
            test_metrics=test_metrics,
        )

        def _check_type(metric: str, op: str) -> str:
            if metric.endswith("_std"):
                return "stability_std"
            if metric.endswith("_pass_rate"):
                return "stability_pass_rate"
            if op in {"<", "<="}:
                return "safety"
            return "required"

        for phase, run_id, run_metrics in [
            ("technical_validation", tech_run_id, tech_metrics),
            ("independent_test", test_run_id, test_metrics),
        ]:
            if not run_id:
                continue
            phase_result = promotion_results[phase]
            checks = [
                GateCheck(
                    metric=gate.metric,
                    threshold=gate.threshold,
                    actual=gate.observed_value,
                    passed=gate.passed,
                    check_type=_check_type(gate.metric, gate.op),
                )
                for gate in phase_result.per_gate_results
            ]
            results[phase] = GateResult(
                phase=phase,
                phase_label=phase_labels[phase],
                passed=phase_result.passed,
                run_id=run_id,
                run_key=f"{project_id}:{phase}:{run_id}",
                checks=checks,
                metrics_available=_numeric_metrics(run_metrics),
            )

        return results
