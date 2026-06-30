"""Local filesystem repository implementations."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from classiflow.ui_api.models import (
    Artifact,
    ArtifactKind,
    BaggingDetail,
    DecisionBadge,
    GateStatus,
    MetricsSummary,
    PlotManifestResponse,
    ProjectCard,
    ProjectDashboard,
    PromotionSummary,
    RegistrySummary,
    RunBrief,
    RunDetail,
    SelectedFinalModelSummary,
    SelectedModelConfig,
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
                searchable = (
                    f"{project.id} {project.config.name} {project.config.description or ''}".lower()
                )
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
            if hasattr(project.thresholds, "model_dump"):
                registry.thresholds = project.thresholds.model_dump(mode="python")
            else:
                registry.thresholds = {
                    "technical_validation": project.thresholds.technical_validation,
                    "independent_test": project.thresholds.independent_test,
                    "promotion_logic": project.thresholds.promotion_logic,
                    "promotion": project.thresholds.promotion,
                }

        # Build promotion summary with detailed gate results
        promotion = PromotionSummary()
        if project.decision:
            dec = project.decision
            promotion.decision = (
                DecisionBadge(dec.decision)
                if dec.decision in ["PASS", "FAIL"]
                else DecisionBadge.PENDING
            )
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
                    runs.append(
                        RunBrief(
                            run_key=run.manifest.run_key,
                            run_id=run.manifest.run_id,
                            phase=phase,
                            created_at=run.manifest.created_at,
                            task_type=run.manifest.task_type,
                            headline_metrics=self.scanner.get_headline_metrics(
                                project.id, phase, run_id
                            ),
                        )
                    )
            if runs:
                phases[phase] = runs

        selected_final_model = None
        final_runs = project.phases.get("final_model", [])
        if final_runs:
            selected_final_model = self._load_selected_final_model(project.id, final_runs[0])

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
            model_settings={
                "engine": project.config.execution_engine,
                "device": project.config.execution_device,
                "model_set": project.config.model_set,
                "candidates": project.config.candidates,
                "expanded_mlp_tuning_grid": project.config.expanded_mlp_tuning_grid,
                "final_estimator_strategy": project.config.final_estimator_strategy,
                "technical_final_estimator_strategy": project.config.technical_final_estimator_strategy,
                "bagging_n_estimators": project.config.bagging_n_estimators,
                "bagging_max_samples": project.config.bagging_max_samples,
                "bagging_max_features": project.config.bagging_max_features,
                "bagging_bootstrap": project.config.bagging_bootstrap,
                "bagging_bootstrap_features": project.config.bagging_bootstrap_features,
            },
            selected_final_model=selected_final_model,
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
                    runs.append(
                        RunBrief(
                            run_key=run.manifest.run_key,
                            run_id=run.manifest.run_id,
                            phase=phase,
                            created_at=run.manifest.created_at,
                            task_type=run.manifest.task_type,
                            headline_metrics=self.scanner.get_headline_metrics(
                                project_id, phase, run_id
                            ),
                        )
                    )
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
            hierarchical=metrics.get("hierarchical", {}),
        )

        # Get artifacts
        parts = run_key.split(":")
        artifacts = []
        plot_manifest = None
        bagging = None
        selected_final_model = None
        if len(parts) == 3:
            project_id, phase, run_id = parts
            scanned_arts = self.scanner.get_artifacts(project_id, phase, run_id)
            artifacts = [self._to_artifact(a) for a in scanned_arts]

            # Load plot manifest if available
            plot_manifest = self._load_plot_manifest(project_id, phase, run_id)
            bagging = self._load_bagging_detail(project_id, phase, run_id)
            if phase == "final_model":
                selected_final_model = self._load_selected_final_model(project_id, run_id)

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
            bagging=bagging,
            selected_final_model=selected_final_model,
            plot_manifest=plot_manifest,
        )

    def _load_plot_manifest(
        self,
        project_id: str,
        phase: str,
        run_id: str,
    ) -> Optional[PlotManifestResponse]:
        """Load plot manifest for a run if available."""
        # Get run directory
        run_dir = self.scanner.resolve_artifact_path(
            project_id, phase, run_id, "plots/plot_manifest.json"
        )
        if not run_dir or not run_dir.is_file():
            return None

        try:
            with open(run_dir) as f:
                data = json.load(f)

            return PlotManifestResponse(
                available=data.get("available", {}),
                fallback_pngs=data.get("fallback_pngs", {}),
                generated_at=datetime.fromisoformat(data["generated_at"])
                if data.get("generated_at")
                else None,
                classiflow_version=data.get("classiflow_version"),
            )
        except Exception:
            return None

    def _load_bagging_detail(
        self,
        project_id: str,
        phase: str,
        run_id: str,
    ) -> Optional[BaggingDetail]:
        """Load bag-member summary for a run if available."""
        summary_path = self.scanner.resolve_artifact_path(
            project_id, phase, run_id, "bagging_summary.json"
        )
        if summary_path is None:
            return None

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return BaggingDetail.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _sanitize_json_value(value):
        """Return a JSON/API-safe copy with non-finite floats removed."""
        if isinstance(value, dict):
            return {k: LocalFilesystemRepository._sanitize_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [LocalFilesystemRepository._sanitize_json_value(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _selected_model_from_payload(
        task_name: str,
        payload: dict,
    ) -> Optional[SelectedModelConfig]:
        """Normalize selected-model registry payloads for UI responses."""
        if not isinstance(payload, dict):
            return None
        model_name = payload.get("model_name")
        if not model_name:
            return None
        mean_score = payload.get("mean_score")
        try:
            mean_score = float(mean_score) if mean_score is not None else None
            if mean_score is not None and not math.isfinite(mean_score):
                mean_score = None
        except (TypeError, ValueError):
            mean_score = None
        return SelectedModelConfig(
            task_name=str(payload.get("task_name") or task_name),
            model_name=str(model_name),
            sampler=payload.get("sampler"),
            mean_score=mean_score,
            params=LocalFilesystemRepository._sanitize_json_value(payload.get("params", {})) or {},
        )

    def _load_json_artifact(
        self,
        project_id: str,
        phase: str,
        run_id: str,
        relative_path: str,
    ) -> Optional[dict]:
        path = self.scanner.resolve_artifact_path(project_id, phase, run_id, relative_path)
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _load_selected_final_model(
        self,
        project_id: str,
        run_id: str,
    ) -> Optional[SelectedFinalModelSummary]:
        """Load or synthesize the selected final-bundle model summary."""
        explicit = self._load_json_artifact(
            project_id, "final_model", run_id, "final_model_summary.json"
        )
        if explicit:
            try:
                return SelectedFinalModelSummary.model_validate(
                    self._sanitize_json_value(explicit)
                )
            except Exception:
                pass

        run_payload = self._load_json_artifact(project_id, "final_model", run_id, "run.json")
        if not run_payload:
            return None

        run_key = f"{project_id}:final_model:{run_id}"
        config = run_payload.get("config", {})
        config = config if isinstance(config, dict) else {}
        final_model = config.get("final_model", {})
        final_model = final_model if isinstance(final_model, dict) else {}
        models = config.get("models", {})
        models = models if isinstance(models, dict) else {}
        execution = config.get("execution", {})
        execution = execution if isinstance(execution, dict) else {}

        selected_models: list[SelectedModelConfig] = []
        selected_binary = self._load_json_artifact(
            project_id, "final_model", run_id, "registry/selected_binary_configs.json"
        )
        if selected_binary:
            for task_name, payload in selected_binary.items():
                model = self._selected_model_from_payload(str(task_name), payload)
                if model is not None:
                    selected_models.append(model)

        if not selected_models:
            best_models = run_payload.get("best_models", {})
            if isinstance(best_models, dict):
                for task_name, model_name in best_models.items():
                    if model_name:
                        selected_models.append(
                            SelectedModelConfig(
                                task_name=str(task_name),
                                model_name=str(model_name),
                            )
                        )

        if not selected_models:
            model_name_path = self.scanner.resolve_artifact_path(
                project_id,
                "final_model",
                run_id,
                "fold1/multiclass_none/multiclass_model_name.txt",
            ) or self.scanner.resolve_artifact_path(
                project_id,
                "final_model",
                run_id,
                "fold1/multiclass_smote/multiclass_model_name.txt",
            )
            if model_name_path is not None:
                selected_models.append(
                    SelectedModelConfig(
                        task_name="multiclass",
                        model_name=model_name_path.read_text(encoding="utf-8").strip(),
                        sampler=final_model.get("sampler"),
                    )
                )

        meta_model = None
        selected_meta = self._load_json_artifact(
            project_id, "final_model", run_id, "registry/selected_meta_config.json"
        )
        if selected_meta:
            meta_model = self._selected_model_from_payload("meta", selected_meta)

        bundle_path = None
        if self.scanner.resolve_artifact_path(
            project_id, "final_model", run_id, "model_bundle.zip"
        ):
            bundle_path = "model_bundle.zip"

        strategy = {
            "final_estimator_strategy": models.get("final_estimator_strategy"),
            "bagging_n_estimators": models.get("bagging_n_estimators"),
            "bagging_max_samples": models.get("bagging_max_samples"),
            "bagging_max_features": models.get("bagging_max_features"),
            "bagging_bootstrap": models.get("bagging_bootstrap"),
            "bagging_bootstrap_features": models.get("bagging_bootstrap_features"),
        }
        strategy = {k: v for k, v in strategy.items() if v is not None}

        try:
            return SelectedFinalModelSummary(
                run_id=run_id,
                run_key=run_key,
                task_type=run_payload.get("task_type"),
                bundle_path=bundle_path,
                technical_run=final_model.get("technical_run"),
                sampler=final_model.get("sampler"),
                train_from_scratch=bool(final_model.get("train_from_scratch", True)),
                selection_metric=models.get("selection_metric"),
                selection_direction=models.get("selection_direction"),
                execution=self._sanitize_json_value(execution) or {},
                strategy=self._sanitize_json_value(strategy) or {},
                selected_models=selected_models,
                meta_model=meta_model,
            )
        except Exception:
            return None

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
                        all_runs.append(
                            RunBrief(
                                run_key=run.manifest.run_key,
                                run_id=run.manifest.run_id,
                                phase=p,
                                created_at=run.manifest.created_at,
                                task_type=run.manifest.task_type,
                                headline_metrics=self.scanner.get_headline_metrics(
                                    project.id, p, run_id
                                ),
                            )
                        )

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
        view_url = None
        download_url = None

        if scanned.is_viewable:
            view_url = f"{self.api_base_url}/artifacts/{scanned.artifact_id}/content"
        if scanned.is_allowed:
            download_url = (
                f"{self.api_base_url}/artifacts/{scanned.artifact_id}/content?download=true"
            )

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
