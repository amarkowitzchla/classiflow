// Plot data types for interactive ROC/PR curve rendering

export type PlotType = 'roc' | 'pr';
export type PlotScope = 'averaged' | 'fold' | 'inference';
export type TaskType = 'binary' | 'multiclass';

/**
 * Individual curve data within a plot.
 */
export interface CurveData {
  /** Curve label: 'macro', 'micro', 'weighted', class name, or 'overall' */
  label: string;
  /** X-axis values (FPR for ROC, Recall for PR) */
  x: number[];
  /** Y-axis values (TPR for ROC, Precision for PR) */
  y: number[];
  /** Decision thresholds corresponding to each point (optional) */
  thresholds?: number[];
}

/**
 * Summary statistics for the plot.
 */
export interface PlotSummary {
  /** AUC values keyed by label (e.g., {'macro': 0.93, 'micro': 0.95}) */
  auc?: Record<string, number>;
  /** Average Precision values for PR curves */
  ap?: Record<string, number>;
}

/**
 * Metadata about how the plot was generated.
 */
export interface PlotMetadata {
  /** ISO8601 timestamp when the data was generated */
  generated_at: string;
  /** Source of the data: 'metrics.json', 'predictions.csv', 'internal' */
  source: string;
  /** Version of Classiflow that generated the data */
  classiflow_version: string;
  /** Run ID this plot belongs to */
  run_id: string;
  /** Fold number (1-indexed) if scope is 'fold' */
  fold?: number;
}

/**
 * Standard deviation band for averaged plots.
 */
export interface StdBand {
  x: number[];
  y_upper: number[];
  y_lower: number[];
}

/**
 * Complete plot curve data structure.
 * This is the main data structure for ROC and PR curve JSON files.
 */
export interface PlotCurve {
  /** Type of plot: 'roc' or 'pr' */
  plot_type: PlotType;
  /** Scope of the data: 'averaged', 'fold', or 'inference' */
  scope: PlotScope;
  /** Classification task type: 'binary' or 'multiclass' */
  task: TaskType;
  /** Class labels in order */
  labels: string[];
  /** Individual curves to plot */
  curves: CurveData[];
  /** Summary metrics (AUC, AP) */
  summary: PlotSummary;
  /** Generation metadata */
  metadata: PlotMetadata;
  /** Standard deviation band for averaged plots */
  std_band?: StdBand;
  /** Individual fold curves for averaged plots */
  fold_curves?: CurveData[];
  /** Per-fold metric values (e.g., AUC per fold) */
  fold_metrics?: Record<string, number[]>;
}

/**
 * Manifest listing available plot data files for a run.
 */
export interface PlotManifest {
  /** Mapping of plot key to relative JSON file path */
  available: Record<string, string>;
  /** Mapping of plot key to fallback PNG file path */
  fallback_pngs: Record<string, string>;
  /** When the manifest was generated */
  generated_at: string;
  /** Classiflow version that generated the manifest */
  classiflow_version: string;
}

/**
 * Standard keys for plot types in the manifest.
 */
export const PlotKey = {
  ROC_AVERAGED: 'roc_averaged',
  PR_AVERAGED: 'pr_averaged',
  ROC_BY_FOLD: 'roc_by_fold',
  PR_BY_FOLD: 'pr_by_fold',
  ROC_INFERENCE: 'roc_inference',
  PR_INFERENCE: 'pr_inference',
} as const;

export type PlotKeyType = typeof PlotKey[keyof typeof PlotKey];

/**
 * Color palette for curve visualization.
 * Uses a colorblind-friendly palette.
 */
export const CURVE_COLORS = [
  '#2563eb', // blue-600
  '#dc2626', // red-600
  '#16a34a', // green-600
  '#9333ea', // purple-600
  '#ea580c', // orange-600
  '#0891b2', // cyan-600
  '#db2777', // pink-600
  '#65a30d', // lime-600
  '#7c3aed', // violet-600
  '#f59e0b', // amber-500
] as const;

/**
 * Special colors for aggregate curves.
 */
export const AGGREGATE_COLORS = {
  macro: '#1e3a8a',    // blue-900
  micro: '#991b1b',    // red-900
  weighted: '#166534', // green-900
  mean: '#1e3a8a',     // blue-900
} as const;
