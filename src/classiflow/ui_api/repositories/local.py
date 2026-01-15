"""Local filesystem repository implementations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from classiflow.ui_api.models import (
    Artifact,
    ArtifactKind,
    DecisionBadge,
    GateStatus,
    MetricsSummary,
    ProjectCard,
    ProjectDashboard,
    PromotionSummary,
    RegistrySummary,
    RunBrief,
    RunDetail,
)
from classiflow.ui_api.repositories.interfaces import (
    ArtifactRepository,
    ProjectRepository,
    RunRepository,
)
from classiflow.ui_api.scanner import LocalFilesystemScanner


class LocalFilesystemRepository(ProjectRepository, RunRepository, ArtifactRepository):
    """
    Combined repository implementation using local filesystem scanner.

    Implements ProjectRepository, RunRepository, and ArtifactRepository
    for local mode operation.
    """

    def __init__(self, projects_root: Path, api_base_url: str = "/api"):
        """
        Initialize repository.

        Parameters
        ----------
        projects_root : Path
            Root directory containing projects
        api_base_url : str
            Base URL for artifact serving endpoints
        """
        self.scanner = LocalFilesystemScanner(projects_root)
        self.api_base_url = api_base_url

    # -------------------------------------------------------------------------
    # ProjectRepository implementation
    # -------------------------------------------------------------------------

    def list_projects(
        self,
        query: Optional[str] = None,
        mode: Optional[str] = None,
        owner: Optional[str] = None,
        updated_after: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ProjectCard], int]:
        """List projects with optional filtering."""
        projects = self.scanner.scan_projects()

        # Apply filters
        filtered = []
        for project in projects:
            # Query filter (search in name, ID, description)
            if query:
                q = query.lower()
                searchable = f"{project.id} {project.config.name} {project.config.description or ''}".lower()
                if q not in searchable:
                    continue

            # Mode filter
            if mode and project.config.task_mode != mode:
                continue

            # Owner filter
            if owner and project.config.owner != owner:
                continue

            # Updated after filter
            if updated_after and project.updated_at:
                try:
                    after_dt = datetime.fromisoformat(updated_after)
                    if project.updated_at < after_dt:
                        continue
                except ValueError:
                    pass

            filtered.append(project)

        # Sort by updated_at descending
        filtered.sort(key=lambda p: p.updated_at or datetime.min, reverse=True)

        # Paginate
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = filtered[start:end]

        # Convert to ProjectCard
        cards = []
        for project in page_items:
            card = self._to_project_card(project)
            cards.append(card)

        return cards, total

    def _to_project_card(self, project) -> ProjectCard:
        """Convert ScannedProject to ProjectCard."""
        # Get decision badge
        badge = self.scanner.get_decision_badge(project.id)

        # Get latest runs by phase
        latest_runs = {}
        headline_metrics = {}

        for phase, run_ids in project.phases.items():
            if run_ids:
                latest_id = run_ids[0]  # Already sorted descending
                run = self.scanner.get_run(project.id, phase, latest_id)
                if run:
                    latest_runs[phase] = RunBrief(
                        run_key=run.manifest.run_key,
                        run_id=run.manifest.run_id,
                        phase=phase,
                        created_at=run.manifest.created_at,
                        task_type=run.manifest.task_type,
                        headline_metrics=self.scanner.get_headline_metrics(
                            project.id, phase, latest_id
                        ),
                    )

                    # Use independent_test metrics for project headline if available
                    if phase == "independent_test" and not headline_metrics:
                        headline_metrics = self.scanner.get_headline_metrics(
                            project.id, phase, latest_id
                        )

        # Fallback to technical_validation metrics
        if not headline_metrics and "technical_validation" in latest_runs:
            latest_runs["technical_validation"].headline_metrics

        # Compute gate status for each phase
        gate_status: dict[str, GateStatus] = {}
        gate_results = self.scanner.compute_gate_results(project.id)
        for phase_name in ["technical_validation", "independent_test"]:
            if phase_name in gate_results:
                gate = gate_results[phase_name]
                gate_status[phase_name] = GateStatus.PASS if gate.passed else GateStatus.FAIL
            elif phase_name in project.phases and project.phases[phase_name]:
                # Has runs but no gate result (no thresholds configured)
                gate_status[phase_name] = GateStatus.PENDING
            else:
                # No runs for this phase
                gate_status[phase_name] = GateStatus.PENDING

        return ProjectCard(
            id=project.id,
            name=project.config.name,
            description=project.config.description,
            owner=project.config.owner,
            task_mode=project.config.task_mode,
            updated_at=project.updated_at,
            phases_present=list(project.phases.keys()),
            decision_badge=badge,
            gate_status=gate_status,
            latest_runs_by_phase=latest_runs,
            headline_metrics=headline_metrics,
            run_count=project.run_count,
        )

    def get_project(self, project_id: str) -> Optional[ProjectDashboard]:
        """Get full project dashboard."""
        project = self.scanner.get_project(project_id)
        if not project:
            return None

        # Build registry summary
        registry = RegistrySummary()
        if "train" in project.datasets:
            ds = project.datasets["train"]
            registry.train = {
                "manifest_path": ds.manifest_path,
                "sha256": ds.sha256,
                "size_bytes": ds.size_bytes,
                "row_count": ds.row_count,
                "feature_count": len(ds.feature_columns),
                "label_distribution": ds.label_distribution,
            }
        if "test" in project.datasets:
            ds = project.datasets["test"]
            registry.test = {
                "manifest_path": ds.manifest_path,
                "sha256": ds.sha256,
                "size_bytes": ds.size_bytes,
                "row_count": ds.row_count,
            }
        if project.thresholds:
            registry.thresholds = {
                "technical_validation": project.thresholds.technical_validation,
                "independent_test": project.thresholds.independent_test,
                "promotion_logic": project.thresholds.promotion_logic,
            }

        # Build promotion summary with detailed gate results
        promotion = PromotionSummary()
        if project.decision:
            dec = project.decision
            promotion.decision = DecisionBadge(dec.decision) if dec.decision in ["PASS", "FAIL"] else DecisionBadge.PENDING
            promotion.timestamp = dec.timestamp
            promotion.technical_run = dec.technical_run
            promotion.test_run = dec.test_run
            promotion.reasons = dec.reasons
            promotion.override_enabled = dec.override_enabled
            promotion.override_comment = dec.override_comment
            promotion.override_approver = dec.override_approver

        # Compute detailed gate results
        gate_results = self.scanner.compute_gate_results(project_id)
        promotion.gates = gate_results

        # Build phases with runs
        phases: dict[str, list[RunBrief]] = {}
        for phase, run_ids in project.phases.items():
            runs = []
            for run_id in run_ids[:10]:  # Limit to 10 most recent
                run = self.scanner.get_run(project.id, phase, run_id)
                if run:
                    runs.append(RunBrief(
                        run_key=run.manifest.run_key,
                        run_id=run.manifest.run_id,
                        phase=phase,
                        created_at=run.manifest.created_at,
                        task_type=run.manifest.task_type,
                        headline_metrics=self.scanner.get_headline_metrics(
                            project.id, phase, run_id
                        ),
                    ))
            if runs:
                phases[phase] = runs

        # Artifact highlights (latest plots/reports)
        highlights = []
        for phase in ["independent_test", "technical_validation"]:
            if phase not in project.phases:
                continue
            run_ids = project.phases[phase]
            if not run_ids:
                continue

            artifacts = self.scanner.get_artifacts(project.id, phase, run_ids[0])
            for art in artifacts[:5]:
                if art.kind in [ArtifactKind.IMAGE, ArtifactKind.REPORT]:
                    highlights.append(self._to_artifact(art))
                    if len(highlights) >= 6:
                        break
            if len(highlights) >= 6:
                break

        return ProjectDashboard(
            id=project.id,
            name=project.config.name,
            description=project.config.description,
            owner=project.config.owner,
            task_mode=project.config.task_mode,
            updated_at=project.updated_at,
            registry=registry,
            promotion=promotion,
            phases=phases,
            artifact_highlights=highlights,
        )

    def get_project_runs(self, project_id: str) -> dict[str, list[RunBrief]]:
        """Get runs grouped by phase."""
        project = self.scanner.get_project(project_id)
        if not project:
            return {}

        result: dict[str, list[RunBrief]] = {}
        for phase, run_ids in project.phases.items():
            runs = []
            for run_id in run_ids:
                run = self.scanner.get_run(project_id, phase, run_id)
                if run:
                    runs.append(RunBrief(
                        run_key=run.manifest.run_key,
                        run_id=run.manifest.run_id,
                        phase=phase,
                        created_at=run.manifest.created_at,
                        task_type=run.manifest.task_type,
                        headline_metrics=self.scanner.get_headline_metrics(
                            project_id, phase, run_id
                        ),
                    ))
            if runs:
                result[phase] = runs

        return result

    # -------------------------------------------------------------------------
    # RunRepository implementation
    # -------------------------------------------------------------------------

    def get_run(self, run_key: str) -> Optional[RunDetail]:
        """Get run detail by composite key."""
        scanned = self.scanner.get_run_by_key(run_key)
        if not scanned:
            return None

        m = scanned.manifest
        metrics = scanned.metrics

        # Build metrics summary
        metrics_summary = MetricsSummary(
            primary=metrics.get("summary", {}),
            per_fold=metrics.get("per_fold", {}),
            per_class=metrics.get("per_class", []),
            confusion_matrix=metrics.get("confusion_matrix"),
            roc_auc=metrics.get("roc_auc"),
        )

        # Get artifacts
        parts = run_key.split(":")
        artifacts = []
        if len(parts) == 3:
            scanned_arts = self.scanner.get_artifacts(parts[0], parts[1], parts[2])
            artifacts = [self._to_artifact(a) for a in scanned_arts]

        return RunDetail(
            run_key=m.run_key,
            run_id=m.run_id,
            project_id=m.project_id,
            phase=m.phase,
            created_at=m.created_at,
            task_type=m.task_type,
            config=m.config,
            metrics=metrics_summary,
            feature_count=m.feature_count,
            feature_list=m.feature_list,
            lineage=m.lineage if m.lineage else None,
            artifact_count=scanned.artifact_count,
            artifacts=artifacts,
        )

    def list_runs(
        self,
        project_id: Optional[str] = None,
        phase: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[RunBrief], int]:
        """List runs with optional filtering."""
        projects = self.scanner.scan_projects()

        all_runs = []
        for project in projects:
            if project_id and project.id != project_id:
                continue

            for p, run_ids in project.phases.items():
                if phase and p != phase:
                    continue

                for run_id in run_ids:
                    run = self.scanner.get_run(project.id, p, run_id)
                    if run:
                        all_runs.append(RunBrief(
                            run_key=run.manifest.run_key,
                            run_id=run.manifest.run_id,
                            phase=p,
                            created_at=run.manifest.created_at,
                            task_type=run.manifest.task_type,
                            headline_metrics=self.scanner.get_headline_metrics(
                                project.id, p, run_id
                            ),
                        ))

        # Sort by created_at descending
        all_runs.sort(key=lambda r: r.created_at or datetime.min, reverse=True)

        # Paginate
        total = len(all_runs)
        start = (page - 1) * page_size
        end = start + page_size

        return all_runs[start:end], total

    # -------------------------------------------------------------------------
    # ArtifactRepository implementation
    # -------------------------------------------------------------------------

    def list_artifacts(
        self,
        run_key: str,
        kind: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Artifact], int]:
        """List artifacts for a run."""
        parts = run_key.split(":")
        if len(parts) != 3:
            return [], 0

        project_id, phase, run_id = parts
        scanned = self.scanner.get_artifacts(project_id, phase, run_id)

        # Filter by kind
        if kind:
            try:
                kind_enum = ArtifactKind(kind)
                scanned = [a for a in scanned if a.kind == kind_enum]
            except ValueError:
                pass

        # Convert and paginate
        total = len(scanned)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = scanned[start:end]

        artifacts = [self._to_artifact(a) for a in page_items]
        return artifacts, total

    def _to_artifact(self, scanned) -> Artifact:
        """Convert ScannedArtifact to Artifact model."""
        parts = scanned.run_key.split(":")
        project_id = parts[0] if len(parts) >= 1 else ""
        phase = parts[1] if len(parts) >= 2 else ""
        run_id = parts[2] if len(parts) >= 3 else ""

        view_url = None
        download_url = None

        if scanned.is_viewable:
            view_url = f"{self.api_base_url}/artifacts/{scanned.artifact_id}/content"
        if scanned.is_allowed:
            download_url = f"{self.api_base_url}/artifacts/{scanned.artifact_id}/content?download=true"

        return Artifact(
            artifact_id=scanned.artifact_id,
            title=scanned.title,
            relative_path=scanned.relative_path,
            kind=scanned.kind,
            mime_type=scanned.mime_type,
            size_bytes=scanned.size_bytes,
            created_at=scanned.created_at,
            run_key=scanned.run_key,
            phase=scanned.phase,
            is_viewable=scanned.is_viewable,
            view_url=view_url,
            download_url=download_url,
        )

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact by ID."""
        scanned = self.scanner.get_artifact_by_id(artifact_id)
        if not scanned:
            return None
        return self._to_artifact(scanned)

    def get_artifact_path(self, artifact_id: str) -> Optional[Path]:
        """Get filesystem path for an artifact."""
        scanned = self.scanner.get_artifact_by_id(artifact_id)
        if not scanned:
            return None
        return scanned.absolute_path

    def resolve_artifact_path(
        self,
        project_id: str,
        phase: str,
        run_id: str,
        relative_path: str,
    ) -> Optional[Path]:
        """Safely resolve an artifact path."""
        return self.scanner.resolve_artifact_path(project_id, phase, run_id, relative_path)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def refresh(self):
        """Clear caches and rescan."""
        self.scanner.clear_cache()
        self.scanner.scan_projects(force=True)
