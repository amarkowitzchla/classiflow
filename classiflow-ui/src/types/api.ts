// API response types matching backend models

export type DecisionBadge = 'PASS' | 'FAIL' | 'PENDING' | 'OVERRIDE';
export type GateStatus = 'PASS' | 'FAIL' | 'PENDING';
export type ArtifactKind = 'image' | 'report' | 'metrics' | 'model' | 'data' | 'config' | 'other';
export type ReviewStatus = 'pending' | 'approved' | 'rejected' | 'needs_changes';

export interface MetricsSummary {
  primary: Record<string, number | null>;
  per_fold: Record<string, Array<number | null>>;
  per_class: Array<{
    class: string;
    precision: number | null;
    recall: number | null;
    f1: number | null;
    support: number | null;
  }>;
  confusion_matrix?: {
    labels: string[];
    matrix: Array<Array<number | null>>;
  };
  roc_auc?: {
    per_class: Array<{ class: string; auc: number | null }>;
    macro: number | null;
    micro: number | null;
  };
}

export interface RunBrief {
  run_key: string;
  run_id: string;
  phase: string;
  created_at: string | null;
  task_type: string | null;
  headline_metrics: Record<string, number | null>;
}

export interface RunDetail {
  run_key: string;
  run_id: string;
  project_id: string;
  phase: string;
  created_at: string | null;
  task_type: string | null;
  config: Record<string, unknown>;
  metrics: MetricsSummary;
  feature_count: number;
  feature_list: string[];
  lineage: Record<string, unknown> | null;
  artifact_count: number;
  artifacts: Artifact[];
}

export interface Artifact {
  artifact_id: string;
  title: string;
  relative_path: string;
  kind: ArtifactKind;
  mime_type: string | null;
  size_bytes: number | null;
  created_at: string | null;
  run_key: string;
  phase: string;
  is_viewable: boolean;
  view_url: string | null;
  download_url: string | null;
}

export interface RegistrySummary {
  train?: {
    manifest_path: string;
    sha256: string;
    size_bytes: number;
    row_count: number;
    feature_count: number;
    label_distribution: Record<string, number>;
  };
  test?: {
    manifest_path: string;
    sha256: string;
    size_bytes: number;
    row_count: number;
  };
  thresholds?: Record<string, unknown>;
}

export interface GateCheck {
  metric: string;
  threshold: number;
  actual: number | null;
  passed: boolean;
  check_type: 'required' | 'stability_std' | 'stability_pass_rate' | 'safety';
}

export interface GateResult {
  phase: string;
  phase_label: string;
  passed: boolean;
  run_id: string | null;
  run_key: string | null;
  checks: GateCheck[];
  metrics_available: Record<string, number | null>;
}

export interface PromotionSummary {
  decision: DecisionBadge;
  timestamp: string | null;
  technical_run: string | null;
  test_run: string | null;
  reasons: string[];
  override_enabled: boolean;
  override_comment: string | null;
  override_approver: string | null;
  gates: Record<string, GateResult>;
}

export interface ProjectCard {
  id: string;
  name: string;
  description: string | null;
  owner: string | null;
  task_mode: string | null;
  updated_at: string | null;
  phases_present: string[];
  decision_badge: DecisionBadge;
  gate_status: Record<string, GateStatus>;
  latest_runs_by_phase: Record<string, RunBrief>;
  headline_metrics: Record<string, number | null>;
  run_count: number;
}

export interface ProjectDashboard {
  id: string;
  name: string;
  description: string | null;
  owner: string | null;
  task_mode: string | null;
  updated_at: string | null;
  registry: RegistrySummary;
  promotion: PromotionSummary;
  phases: Record<string, RunBrief[]>;
  artifact_highlights: Artifact[];
}

export interface Comment {
  id: number;
  scope_type: string;
  scope_id: string;
  author: string;
  content: string;
  created_at: string;
  updated_at: string | null;
}

export interface Review {
  id: number;
  scope_type: string;
  scope_id: string;
  reviewer: string;
  status: ReviewStatus;
  notes: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface HealthResponse {
  status: string;
  storage_mode: string;
  projects_root: string;
  db_path: string | null;
  project_count: number;
  index_status: string;
  version: string;
}
