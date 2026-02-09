"""Rule-based probability quality checks for technical reporting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

import pandas as pd

Severity = Literal["INFO", "WARN", "ERROR"]


@dataclass
class ProbQualityRuleResult:
    rule_id: str
    severity: Severity
    title: str
    summary: str
    measured: Dict[str, Any]
    thresholds: Dict[str, Any]
    evidence: List[Dict[str, str]]
    recommendations: List[str]


DEFAULT_PROBABILITY_QUALITY_CHECK_THRESHOLDS: Dict[str, Any] = {
    "low_bin_min_nonzero_n": 5,
    "low_bin_max_zero_fraction": 0.30,
    "min_total_n": 200,
    "underconfidence_info_gap": -0.10,
    "underconfidence_warn_gap": -0.20,
    "overconfidence_warn_gap": 0.05,
    "overconfidence_error_gap": 0.10,
    "calibration_worsen_brier_delta": 0.002,
    "calibration_worsen_log_loss_delta": 0.01,
    "calibration_worsen_ece_top1_delta": 0.02,
    "calibration_worsen_ece_ovr_delta": 0.01,
    "shift_brier_delta": 0.01,
    "shift_log_loss_delta": 0.05,
    "weak_class_ece": 0.15,
    "weak_class_min_n": 25,
    "near_perfect_accuracy": 0.97,
    "near_perfect_high_ece": 0.15,
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_probability_quality(run_json_path: Path) -> Dict[str, Any]:
    """Load probability-quality payload from run.json artifact registry."""
    if not run_json_path.exists():
        return {}
    try:
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    registry = payload.get("artifact_registry", {})
    if not isinstance(registry, dict):
        return {}
    prob_quality = registry.get("probability_quality", {})
    if not isinstance(prob_quality, dict):
        return {}
    return prob_quality


def load_curve_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load a calibration curve CSV safely."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@dataclass
class _FoldArtifact:
    fold_key: str
    base_dir: Path
    selected_curve_path: Optional[Path]
    calibration_metadata_path: Optional[Path]
    n_samples: Optional[int]
    final_variant: Optional[str]
    curve_kind: str


@dataclass
class _ProbQualityContext:
    run_dir: Path
    mode: str
    run_json_path: Path
    thresholds: Dict[str, Any]
    folds: Dict[str, Dict[str, Any]]
    final_variant: str
    final_metrics_mean: Dict[str, float]
    uncal_metrics_mean: Dict[str, float]
    cal_metrics_mean: Dict[str, float]
    selected_curve_df: Optional[pd.DataFrame]
    selected_curve_sources: List[Path]
    selected_curve_kind: str
    bins: Optional[int]
    n_samples_total: Optional[int]
    class_counts: Dict[str, int]
    fold_artifacts: List[_FoldArtifact]
    independent_metrics: Optional[Dict[str, Any]]


def _merge_thresholds(custom: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PROBABILITY_QUALITY_CHECK_THRESHOLDS)
    if custom:
        for key, value in custom.items():
            merged[key] = value
    return merged


def _resolve_variant_dir(
    *,
    run_dir: Path,
    fold_name: str,
    variant: str,
    mode: str,
) -> Optional[Path]:
    prefixes_by_mode = {
        "binary": ("binary",),
        "meta": ("binary",),
        "multiclass": ("multiclass",),
        "hierarchical": ("hierarchical",),
    }
    fold_dir = run_dir / fold_name
    if not fold_dir.exists():
        return None
    prefixes = prefixes_by_mode.get(mode, ("binary",))
    for prefix in prefixes:
        candidate = fold_dir / f"{prefix}_{variant}"
        if candidate.exists():
            return candidate
    for prefix in prefixes:
        candidates = sorted(fold_dir.glob(f"{prefix}_*"))
        if candidates:
            return candidates[0]
    return None


def _extract_n_samples(entry: Mapping[str, Any]) -> Optional[int]:
    decision = entry.get("calibration_decision", {})
    if not isinstance(decision, dict):
        return None
    value = decision.get("n_samples")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_curve_for_fold(var_dir: Path, variant: str, mode: str) -> tuple[Optional[Path], str]:
    if mode == "binary":
        candidates = [
            (var_dir / f"calibration_curve_binary_pos_{variant}.csv", "binary_pos"),
            (var_dir / "calibration_curve_binary_pos.csv", "binary_pos"),
            (var_dir / f"calibration_curve_top1_{variant}.csv", "top1"),
            (var_dir / "calibration_curve.csv", "top1"),
            (var_dir / "calibration_curve_top1.csv", "top1"),
        ]
    else:
        candidates = [
            (var_dir / f"calibration_curve_top1_{variant}.csv", "top1"),
            (var_dir / "calibration_curve.csv", "top1"),
            (var_dir / "calibration_curve_top1.csv", "top1"),
        ]
    for candidate in candidates:
        path, curve_kind = candidate
        if path.exists():
            return path, curve_kind
    return None, "top1"


def _collect_fold_artifacts(
    run_dir: Path, folds: Mapping[str, Dict[str, Any]], default_variant: str, mode: str
) -> List[_FoldArtifact]:
    entries: List[_FoldArtifact] = []
    for fold_key, payload in folds.items():
        parts = fold_key.split("_")
        if len(parts) >= 2 and parts[0] == "fold" and parts[1].isdigit():
            fold_name = f"fold{parts[1]}"
            fold_variant = parts[2] if len(parts) >= 3 else default_variant
        else:
            fold_name = fold_key
            fold_variant = default_variant
        final_variant = str(payload.get("final_variant") or default_variant or "uncalibrated")
        var_dir = _resolve_variant_dir(
            run_dir=run_dir,
            fold_name=fold_name,
            variant=str(fold_variant),
            mode=mode,
        )
        if var_dir is None:
            continue
        curve_path, curve_kind = _resolve_curve_for_fold(var_dir, final_variant, mode)
        entries.append(
            _FoldArtifact(
                fold_key=fold_key,
                base_dir=var_dir,
                selected_curve_path=curve_path,
                calibration_metadata_path=(
                    var_dir / "calibration_metadata.json"
                    if (var_dir / "calibration_metadata.json").exists()
                    else None
                ),
                n_samples=_extract_n_samples(payload),
                final_variant=final_variant,
                curve_kind=curve_kind,
            )
        )
    return entries


def _mean_metrics(entries: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    accum: Dict[str, List[float]] = {}
    for entry in entries:
        for key, value in entry.items():
            val = _to_float(value)
            if val is None:
                continue
            accum.setdefault(key, []).append(val)
    return {key: float(sum(values) / len(values)) for key, values in accum.items() if values}


def _collect_class_counts(folds: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for payload in folds.values():
        if not isinstance(payload, dict):
            continue
        class_counts = payload.get("class_counts")
        if not isinstance(class_counts, dict):
            continue
        for key, value in class_counts.items():
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return counts


def _read_bins(fold_artifacts: Iterable[_FoldArtifact]) -> Optional[int]:
    values: List[int] = []
    for artifact in fold_artifacts:
        if artifact.calibration_metadata_path is None:
            continue
        try:
            payload = json.loads(artifact.calibration_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bins_val = payload.get("bins")
        try:
            if bins_val is not None:
                values.append(int(bins_val))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return values[0]


def _combine_top1_curves(curve_paths: Iterable[Path]) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in curve_paths:
        df = load_curve_csv(path)
        if df is None or df.empty:
            continue
        required = {"bin_id", "n", "mean_pred", "frac_pos"}
        if required.issubset(set(df.columns)):
            frames.append(df[["bin_id", "n", "mean_pred", "frac_pos"]].copy())
        elif "bin_id" in df.columns and "n" in df.columns:
            # Occupancy-only fallback for legacy/partial curve payloads.
            fallback = df[["bin_id", "n"]].copy()
            fallback["mean_pred"] = float("nan")
            fallback["frac_pos"] = float("nan")
            frames.append(fallback)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, float]] = []
    for bin_id, g in merged.groupby("bin_id"):
        n_sum = float(g["n"].sum())
        mean_pred = float("nan")
        frac_pos = float("nan")
        if n_sum > 0 and g["mean_pred"].notna().any():
            mean_pred = float((g["mean_pred"] * g["n"]).sum() / n_sum)
        if n_sum > 0 and g["frac_pos"].notna().any():
            frac_pos = float((g["frac_pos"] * g["n"]).sum() / n_sum)
        rows.append(
            {
                "bin_id": float(bin_id),
                "n": n_sum,
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
            }
        )
    combined = pd.DataFrame(rows)
    return combined.sort_values("bin_id").reset_index(drop=True)


def _build_context(
    *,
    run_dir: Path,
    mode: str,
    thresholds: Optional[Mapping[str, Any]] = None,
    independent_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[_ProbQualityContext]:
    run_json_path = run_dir / "run.json"
    prob_quality = load_probability_quality(run_json_path)
    folds = prob_quality.get("folds", {})
    if not isinstance(folds, dict) or not folds:
        return None

    default_variant = str(prob_quality.get("final_variant") or "uncalibrated")
    fold_artifacts = _collect_fold_artifacts(run_dir, folds, default_variant, mode)

    final_entries: List[Mapping[str, Any]] = []
    uncal_entries: List[Mapping[str, Any]] = []
    cal_entries: List[Mapping[str, Any]] = []
    for payload in folds.values():
        if not isinstance(payload, dict):
            continue
        uncal = payload.get("uncalibrated", {})
        cal = payload.get("calibrated", {})
        if isinstance(uncal, dict):
            uncal_entries.append(uncal)
        if isinstance(cal, dict):
            cal_entries.append(cal)
        fold_variant = str(payload.get("final_variant") or default_variant)
        final_payload = payload.get(fold_variant, {})
        if isinstance(final_payload, dict):
            final_entries.append(final_payload)

    selected_sources = [a.selected_curve_path for a in fold_artifacts if a.selected_curve_path is not None]
    selected_curve_df = _combine_top1_curves([p for p in selected_sources if p is not None])
    selected_curve_kind = fold_artifacts[0].curve_kind if fold_artifacts else "top1"
    n_samples_total = sum(a.n_samples for a in fold_artifacts if a.n_samples is not None) or None
    if n_samples_total is None and selected_curve_df is not None and "n" in selected_curve_df.columns:
        n_samples_total = int(selected_curve_df["n"].sum())

    return _ProbQualityContext(
        run_dir=run_dir,
        mode=mode,
        run_json_path=run_json_path,
        thresholds=_merge_thresholds(thresholds),
        folds={k: v for k, v in folds.items() if isinstance(v, dict)},
        final_variant=default_variant,
        final_metrics_mean=_mean_metrics(final_entries),
        uncal_metrics_mean=_mean_metrics(uncal_entries),
        cal_metrics_mean=_mean_metrics(cal_entries),
        selected_curve_df=selected_curve_df,
        selected_curve_sources=[p for p in selected_sources if p is not None],
        selected_curve_kind=selected_curve_kind,
        bins=_read_bins(fold_artifacts),
        n_samples_total=n_samples_total,
        class_counts=_collect_class_counts(folds),
        fold_artifacts=fold_artifacts,
        independent_metrics=independent_metrics,
    )


def _evidence(artifact: Path, field: str, note: str, run_dir: Path) -> Dict[str, str]:
    try:
        artifact_rel = str(artifact.relative_to(run_dir))
    except ValueError:
        artifact_rel = str(artifact)
    return {"artifact": artifact_rel, "field": field, "note": note}


def rule_pq_001_low_bin_occupancy(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    curve_df = ctx.selected_curve_df
    if curve_df is None or curve_df.empty or "n" not in curve_df.columns:
        return None
    n_values = curve_df["n"].fillna(0).astype(float)
    nonzero = n_values[n_values > 0]
    min_nonzero = float(nonzero.min()) if not nonzero.empty else 0.0
    zero_fraction = float((n_values == 0).mean()) if len(n_values) else 0.0
    total_n = ctx.n_samples_total
    if total_n is None:
        total_n = int(n_values.sum())

    triggered = (
        min_nonzero < float(ctx.thresholds["low_bin_min_nonzero_n"])
        or zero_fraction >= float(ctx.thresholds["low_bin_max_zero_fraction"])
        or total_n < int(ctx.thresholds["min_total_n"])
    )
    if not triggered:
        return None

    evidence = [
        _evidence(
            ctx.run_json_path,
            "artifact_registry.probability_quality.final_variant",
            "Final variant for curve selection",
            ctx.run_dir,
        ),
    ]
    evidence.extend(
        _evidence(
            source,
            "columns: [bin_id,n,mean_pred,frac_pos]",
            f"{ctx.selected_curve_kind} reliability curve occupancy",
            ctx.run_dir,
        )
        for source in ctx.selected_curve_sources
    )
    for artifact in ctx.fold_artifacts:
        if artifact.calibration_metadata_path is None:
            continue
        evidence.append(
            _evidence(
                artifact.calibration_metadata_path,
                "calibration_metadata.bins",
                "Configured calibration bins",
                ctx.run_dir,
            )
        )
        break

    return ProbQualityRuleResult(
        rule_id="PQ-001",
        severity="WARN",
        title="Low calibration-curve occupancy",
        summary=(
            f"{ctx.selected_curve_kind} calibration curve has sparse bins or low sample support; ECE can be unstable."
        ),
        measured={
            "min_nonzero_bin_n": min_nonzero,
            "zero_bin_fraction": zero_fraction,
            "n_samples": total_n,
            "bins": ctx.bins,
        },
        thresholds={
            "min_nonzero_bin_n": ctx.thresholds["low_bin_min_nonzero_n"],
            "max_zero_bin_fraction": ctx.thresholds["low_bin_max_zero_fraction"],
            "min_n_samples": ctx.thresholds["min_total_n"],
        },
        evidence=evidence,
        recommendations=[
            "Interpret ECE qualitatively; inspect reliability plot.",
            "Use quantile binning if not enabled.",
            "Avoid gating decisions based on ECE under low occupancy.",
        ],
    )


def rule_pq_002_underconfidence(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    gap = _to_float(ctx.final_metrics_mean.get("confidence_gap_top1"))
    if gap is None or gap > float(ctx.thresholds["underconfidence_info_gap"]):
        return None
    severity: Severity = (
        "WARN" if gap <= float(ctx.thresholds["underconfidence_warn_gap"]) else "INFO"
    )
    summary = "Predicted confidence is materially lower than observed top-1 accuracy."
    if ctx.mode == "binary":
        summary = "Decision confidence is materially lower than observed top-1 accuracy."
    return ProbQualityRuleResult(
        rule_id="PQ-002",
        severity=severity,
        title="Underconfidence behavior detected",
        summary=summary,
        measured={
            "confidence_gap_top1": gap,
            "mean_confidence_top1": _to_float(ctx.final_metrics_mean.get("mean_confidence_top1")),
            "accuracy_top1": _to_float(ctx.final_metrics_mean.get("accuracy_top1")),
            "final_variant": ctx.final_variant,
        },
        thresholds={
            "info_gap_lte": ctx.thresholds["underconfidence_info_gap"],
            "warn_gap_lte": ctx.thresholds["underconfidence_warn_gap"],
        },
        evidence=[
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.confidence_gap_top1",
                "Mean across folds for selected final variant",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.mean_confidence_top1",
                "Mean confidence context",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1",
                "Observed top-1 accuracy context",
                ctx.run_dir,
            ),
        ],
        recommendations=[
            "Model is conservative; this is usually safe.",
            "If probabilities are displayed to clinicians, consider documenting conservatism rather than calibrating.",
        ],
    )


def rule_pq_003_overconfidence(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    gap = _to_float(ctx.final_metrics_mean.get("confidence_gap_top1"))
    if gap is None or gap < float(ctx.thresholds["overconfidence_warn_gap"]):
        return None
    severity: Severity = "ERROR" if gap >= float(ctx.thresholds["overconfidence_error_gap"]) else "WARN"
    summary = "Predicted confidence exceeds observed top-1 accuracy."
    recommendations = [
        "Consider enabling calibration (sigmoid/isotonic) or tightening thresholds.",
        "Review per-class errors and reliability curve.",
    ]
    if ctx.mode == "binary":
        summary = "Decision confidence exceeds observed top-1 accuracy."
        recommendations = [
            "Consider enabling calibration (auto) or tightening the decision threshold.",
            "Review reliability curve behavior around operating thresholds.",
        ]
    return ProbQualityRuleResult(
        rule_id="PQ-003",
        severity=severity,
        title="Overconfidence behavior detected",
        summary=summary,
        measured={
            "confidence_gap_top1": gap,
            "mean_confidence_top1": _to_float(ctx.final_metrics_mean.get("mean_confidence_top1")),
            "accuracy_top1": _to_float(ctx.final_metrics_mean.get("accuracy_top1")),
            "final_variant": ctx.final_variant,
        },
        thresholds={
            "warn_gap_gte": ctx.thresholds["overconfidence_warn_gap"],
            "error_gap_gte": ctx.thresholds["overconfidence_error_gap"],
        },
        evidence=[
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.confidence_gap_top1",
                "Mean across folds for selected final variant",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.mean_confidence_top1",
                "Mean confidence context",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1",
                "Observed top-1 accuracy context",
                ctx.run_dir,
            ),
        ],
        recommendations=recommendations,
    )


def rule_pq_004_calibration_helpfulness(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    if not ctx.uncal_metrics_mean or not ctx.cal_metrics_mean:
        return None

    deltas: Dict[str, float] = {}
    metrics_to_compare = ["brier_recommended", "log_loss", "ece_top1", "ece_ovr_macro"]
    if ctx.mode == "binary":
        metrics_to_compare = ["brier_binary", "log_loss", "ece_binary_pos"]

    for key in metrics_to_compare:
        uncal = _to_float(ctx.uncal_metrics_mean.get(key))
        cal = _to_float(ctx.cal_metrics_mean.get(key))
        if uncal is None or cal is None:
            continue
        deltas[key] = cal - uncal

    checks = [("log_loss", "calibration_worsen_log_loss_delta")]
    if ctx.mode == "binary":
        checks.extend(
            [
                ("brier_binary", "calibration_worsen_brier_delta"),
                ("ece_binary_pos", "calibration_worsen_ece_top1_delta"),
            ]
        )
    else:
        checks.extend(
            [
                ("brier_recommended", "calibration_worsen_brier_delta"),
                ("ece_top1", "calibration_worsen_ece_top1_delta"),
                ("ece_ovr_macro", "calibration_worsen_ece_ovr_delta"),
            ]
        )
    worsened = {}
    for metric, threshold_key in checks:
        if metric not in deltas:
            continue
        if deltas[metric] > float(ctx.thresholds[threshold_key]):
            worsened[metric] = deltas[metric]
    if not worsened:
        return None

    return ProbQualityRuleResult(
        rule_id="PQ-004",
        severity="WARN",
        title="Calibration worsened probability quality",
        summary="Calibrated outputs regressed beyond configured tolerances relative to uncalibrated outputs.",
        measured={f"delta_{k}_cal_minus_uncal": v for k, v in deltas.items()},
        thresholds={
            "max_brier_delta": ctx.thresholds["calibration_worsen_brier_delta"],
            "max_log_loss_delta": ctx.thresholds["calibration_worsen_log_loss_delta"],
            "max_ece_top1_delta": ctx.thresholds["calibration_worsen_ece_top1_delta"],
            "max_ece_ovr_macro_delta": ctx.thresholds["calibration_worsen_ece_ovr_delta"],
        },
        evidence=[
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.uncalibrated.*",
                "Uncalibrated metrics source",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.calibrated.*",
                "Calibrated metrics source",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.calibration_decision.metrics_compared",
                "Calibration policy comparison details",
                ctx.run_dir,
            ),
        ],
        recommendations=[
            "Disable calibration or use policy auto mode.",
            "Retain uncalibrated outputs for final predictions.",
        ]
        if ctx.mode != "binary"
        else [
            "Calibration worsened binary probability quality; keep uncalibrated outputs.",
            "Prefer calibration auto mode so low-power folds can stay uncalibrated.",
        ],
    )


def rule_pq_005_distribution_shift(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    if not ctx.independent_metrics:
        return None

    brier_key = "brier_binary" if ctx.mode == "binary" else "brier_recommended"
    cv_brier = _to_float(ctx.final_metrics_mean.get(brier_key))
    cv_log_loss = _to_float(ctx.final_metrics_mean.get("log_loss"))
    test_brier = _to_float(ctx.independent_metrics.get(brier_key))
    if test_brier is None and brier_key != "brier_recommended":
        test_brier = _to_float(ctx.independent_metrics.get("brier_recommended"))
    test_log_loss = _to_float(ctx.independent_metrics.get("log_loss"))
    if cv_brier is None or cv_log_loss is None or test_brier is None or test_log_loss is None:
        return None

    brier_delta = test_brier - cv_brier
    log_loss_delta = test_log_loss - cv_log_loss
    if (
        brier_delta <= float(ctx.thresholds["shift_brier_delta"])
        and log_loss_delta <= float(ctx.thresholds["shift_log_loss_delta"])
    ):
        return None

    return ProbQualityRuleResult(
        rule_id="PQ-005",
        severity="WARN",
        title="Independent-test probability quality regression",
        summary="Independent-test calibration quality is worse than technical CV, suggesting possible distribution shift.",
        measured={
            "brier_cv": cv_brier,
            "brier_test": test_brier,
            "delta_brier_test_minus_cv": brier_delta,
            "log_loss_cv": cv_log_loss,
            "log_loss_test": test_log_loss,
            "delta_log_loss_test_minus_cv": log_loss_delta,
        },
        thresholds={
            "max_brier_test_minus_cv": ctx.thresholds["shift_brier_delta"],
            "max_log_loss_test_minus_cv": ctx.thresholds["shift_log_loss_delta"],
        },
        evidence=[
            _evidence(
                ctx.run_json_path,
                f"artifact_registry.probability_quality.folds.*.<final_variant>.{brier_key}",
                "Technical CV source",
                ctx.run_dir,
            ),
            {
                "artifact": "independent_metrics",
                "field": f"overall.probability_quality.<final_variant>.{brier_key}/log_loss",
                "note": "Independent test source",
            },
        ],
        recommendations=[
            "Potential dataset shift; review class distribution and per-class metrics on test set.",
            "Do not refit calibration on test; only document.",
        ],
    )


def rule_pq_006_weak_class_ovr(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    if ctx.mode not in {"meta", "multiclass", "hierarchical"}:
        return None

    weak_threshold = float(ctx.thresholds["weak_class_ece"])
    min_n = int(ctx.thresholds["weak_class_min_n"])
    class_hits: List[Dict[str, Any]] = []
    support_known = bool(ctx.class_counts)
    has_per_class_ovr = any(key.startswith("ece_ovr__") for key in ctx.final_metrics_mean)
    if not has_per_class_ovr:
        return None

    for key, value in ctx.final_metrics_mean.items():
        if not key.startswith("ece_ovr__"):
            continue
        class_name = key.replace("ece_ovr__", "", 1)
        ece_val = _to_float(value)
        if ece_val is None or ece_val <= weak_threshold:
            continue
        class_n = ctx.class_counts.get(class_name)
        if support_known and class_n is not None and class_n < min_n:
            continue
        class_hits.append({"class": class_name, "ece_ovr": ece_val, "class_n": class_n})

    if not class_hits:
        return None

    severity: Severity = "WARN" if support_known else "INFO"
    recommendations = []
    for hit in class_hits:
        recommendations.append(
            f"Avoid per-class probability thresholding for class {hit['class']}."
        )
    recommendations.append("Review per-class reliability curves if available.")
    if not support_known:
        recommendations.append("Class support is unavailable; interpret OVR ECE qualitatively (low power).")

    evidence = [
        _evidence(
            ctx.run_json_path,
            "artifact_registry.probability_quality.folds.*.<final_variant>.ece_ovr__<class>",
            "Per-class OVR ECE values",
            ctx.run_dir,
        )
    ]
    if support_known:
        evidence.append(
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.class_counts",
                "Class support values",
                ctx.run_dir,
            )
        )

    return ProbQualityRuleResult(
        rule_id="PQ-006",
        severity=severity,
        title="Weak per-class OVR calibration",
        summary=(
            "One or more classes have elevated OVR calibration error."
            if support_known
            else "One or more classes show elevated OVR calibration error; support is unavailable so this is low-power diagnostic evidence."
        ),
        measured={
            "classes": class_hits,
            "support_available": support_known,
        },
        thresholds={
            "ece_ovr_class_gt": weak_threshold,
            "class_n_gte": min_n,
        },
        evidence=evidence,
        recommendations=recommendations,
    )


def rule_pq_007_near_perfect_high_ece(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    accuracy = _to_float(ctx.final_metrics_mean.get("accuracy_top1"))
    ece = _to_float(ctx.final_metrics_mean.get("ece_top1"))
    if accuracy is None or ece is None:
        return None
    if accuracy < float(ctx.thresholds["near_perfect_accuracy"]):
        return None
    if ece < float(ctx.thresholds["near_perfect_high_ece"]):
        return None

    return ProbQualityRuleResult(
        rule_id="PQ-007",
        severity="INFO",
        title="Near-perfect accuracy with high ECE",
        summary="High discrimination and high ECE may reflect conservative probabilities or binning artifacts.",
        measured={
            "accuracy_top1": accuracy,
            "ece_top1": ece,
        },
        thresholds={
            "accuracy_top1_gte": ctx.thresholds["near_perfect_accuracy"],
            "ece_top1_gte": ctx.thresholds["near_perfect_high_ece"],
        },
        evidence=[
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1",
                "Top-1 accuracy source",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.ece_top1",
                "Top-1 ECE source",
                ctx.run_dir,
            ),
            *[
                _evidence(
                    source,
                    "columns: [bin_id,n,mean_pred,frac_pos]",
                    "Top-1 reliability curve occupancy context",
                    ctx.run_dir,
                )
                for source in ctx.selected_curve_sources
            ],
        ],
        recommendations=[
            "ECE may reflect conservative probabilities or binning noise; prefer confidence-gap and plots.",
        ],
    )


def rule_pq_h001_hierarchical_mismatch(ctx: _ProbQualityContext) -> Optional[ProbQualityRuleResult]:
    if ctx.mode != "hierarchical":
        return None
    mismatch = _to_float(ctx.final_metrics_mean.get("pred_alignment_mismatch_rate"))
    if mismatch is None or mismatch <= 0.0:
        return None
    severity: Severity = "WARN" if mismatch > 0.05 else "INFO"
    pct = mismatch * 100.0
    return ProbQualityRuleResult(
        rule_id="PQ-H001",
        severity=severity,
        title="Hierarchical postprocessing mismatch",
        summary=(
            f"Final hierarchical label differs from argmax(proba) for {pct:.2f}% of samples; "
            "top-1 calibration reflects argmax confidence, not postprocessed label."
        ),
        measured={"pred_alignment_mismatch_rate": mismatch, "percent": pct},
        thresholds={"warn_mismatch_gt": 0.05, "info_mismatch_gt": 0.0},
        evidence=[
            _evidence(
                ctx.run_json_path,
                "artifact_registry.probability_quality.folds.*.<final_variant>.pred_alignment_mismatch_rate",
                "Mismatch source",
                ctx.run_dir,
            ),
            _evidence(
                ctx.run_json_path,
                "config.task_mode/hierarchy settings",
                "Hierarchy configuration context",
                ctx.run_dir,
            ),
        ],
        recommendations=[
            "Review hierarchical decision rules and ensure probabilities used downstream match the decision path.",
        ],
    )


def evaluate_probability_quality_checks(
    *,
    run_dir: Path,
    mode: str,
    thresholds: Optional[Mapping[str, Any]] = None,
    independent_metrics: Optional[Dict[str, Any]] = None,
) -> List[ProbQualityRuleResult]:
    """Evaluate standardized probability-quality rules for a technical run."""
    ctx = _build_context(
        run_dir=run_dir,
        mode=mode,
        thresholds=thresholds,
        independent_metrics=independent_metrics,
    )
    if ctx is None:
        return []

    results: List[ProbQualityRuleResult] = []
    for rule in (
        rule_pq_001_low_bin_occupancy,
        rule_pq_002_underconfidence,
        rule_pq_003_overconfidence,
        rule_pq_004_calibration_helpfulness,
        rule_pq_005_distribution_shift,
        rule_pq_006_weak_class_ovr,
        rule_pq_007_near_perfect_high_ece,
        rule_pq_h001_hierarchical_mismatch,
    ):
        result = rule(ctx)
        if result is not None:
            results.append(result)
    return results


def build_probability_quality_next_steps(results: List[ProbQualityRuleResult]) -> List[str]:
    """Build a standardized decision-helper list from triggered rule IDs."""
    ids = {item.rule_id for item in results}
    next_steps: List[str] = []
    if "PQ-003" in ids or "PQ-005" in ids:
        next_steps.append(
            "Overconfidence or test-time regression detected: use calibration auto mode, collect additional data, or tighten decision thresholds."
        )
    if "PQ-002" in ids and "PQ-003" not in ids:
        next_steps.append(
            "Underconfidence with strong discrimination: keep uncalibrated outputs, document conservatism, and avoid gating on calibration alone."
        )
    if "PQ-001" in ids:
        next_steps.append(
            "Low occupancy detected: increase sample size, reduce bins, enable quantile binning, and interpret ECE qualitatively."
        )
    if "PQ-004" in ids:
        next_steps.append(
            "Calibration worsened quality: disable calibration for final predictions, confirm policy decision, and report both variants."
        )
    if "PQ-H001" in ids:
        next_steps.append(
            "Hierarchical postprocessing changed final labels relative to argmax(proba); document this when interpreting top-1 calibration."
        )
    return next_steps


def collect_probability_quality_plot_payload(
    *,
    run_dir: Path,
    mode: str,
    thresholds: Optional[Mapping[str, Any]] = None,
    independent_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect normalized payload used to build probability-quality diagnostic plots."""
    ctx = _build_context(
        run_dir=run_dir,
        mode=mode,
        thresholds=thresholds,
        independent_metrics=independent_metrics,
    )
    if ctx is None:
        return {}
    payload: Dict[str, Any] = {
        "final_variant": ctx.final_variant,
        "thresholds": dict(ctx.thresholds),
        "n_samples_total": ctx.n_samples_total,
        "bins": ctx.bins,
        "curve_kind": ctx.selected_curve_kind,
        "final_metrics_mean": dict(ctx.final_metrics_mean),
        "uncal_metrics_mean": dict(ctx.uncal_metrics_mean),
        "cal_metrics_mean": dict(ctx.cal_metrics_mean),
        "curve_sources": [str(path) for path in ctx.selected_curve_sources],
    }
    if ctx.selected_curve_df is not None and not ctx.selected_curve_df.empty:
        payload["selected_curve"] = ctx.selected_curve_df.to_dict("records")
    return payload
