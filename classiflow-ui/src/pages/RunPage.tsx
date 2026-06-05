import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, BarChart2, FileText, GitBranch, Settings, XCircle, Clock, TrendingUp } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { clsx } from 'clsx';
import { useRun } from '../hooks/useApi';
import { FinalModelSummary, MetricValue, MetricCard } from '../components/MetricValue';
import { ArtifactList } from '../components/ArtifactViewer';
import { Comments } from '../components/Comments';
import { InteractivePlotSection } from '../components/charts';
import type { HierarchicalLevelMetrics, RunDetail, MetricsSummary } from '../types/api';

type TabId = 'metrics' | 'charts' | 'bagging' | 'artifacts' | 'config' | 'lineage';

export function RunPage() {
  const { projectId, phase, runId } = useParams<{
    projectId: string;
    phase: string;
    runId: string;
  }>();
  const [activeTab, setActiveTab] = useState<TabId>('metrics');

  const runKey = `${projectId}:${phase}:${runId}`;
  const { data: run, isLoading, error } = useRun(runKey);

  if (isLoading) {
    return <div className="text-center py-12 text-gray-500">Loading run...</div>;
  }

  if (error || !run) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 mx-auto mb-4 text-red-300" />
        <p className="text-red-500">Failed to load run</p>
        <Link
          to={`/projects/${encodeURIComponent(projectId || '')}`}
          className="text-blue-600 hover:underline mt-2 inline-block"
        >
          Back to project
        </Link>
      </div>
    );
  }

  const tabs: { id: TabId; label: string; icon: typeof BarChart2 }[] = [
    { id: 'metrics', label: 'Metrics', icon: BarChart2 },
    { id: 'charts', label: 'Charts', icon: TrendingUp },
    ...(run.bagging && run.bagging.member_count > 0
      ? [{ id: 'bagging' as const, label: `Bag Members (${run.bagging.member_count})`, icon: BarChart2 }]
      : []),
    { id: 'artifacts', label: `Artifacts (${run.artifact_count})`, icon: FileText },
    { id: 'config', label: 'Config', icon: Settings },
    { id: 'lineage', label: 'Lineage', icon: GitBranch },
  ];

  const headlineCards = buildHeadlineCards(run.metrics);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link
          to={`/projects/${encodeURIComponent(projectId || '')}`}
          className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to project
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 font-mono">{run.run_id}</h1>
            <div className="flex items-center space-x-3 mt-2">
              <span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-sm font-medium">
                {formatPhase(run.phase)}
              </span>
              {run.task_type && (
                <span className="text-sm text-gray-500">{run.task_type}</span>
              )}
            </div>
          </div>
          {run.created_at && (
            <div className="text-sm text-gray-500 flex items-center space-x-1">
              <Clock className="w-4 h-4" />
              <span>{formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}</span>
            </div>
          )}
        </div>
      </div>

      {/* Headline Metrics */}
      {headlineCards.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {headlineCards.map(({ id, label, value }) => (
            <MetricCard
              key={id}
              label={label}
              value={value}
              precision={4}
            />
          ))}
        </div>
      )}

      {run.selected_final_model && (
        <FinalModelSummary summary={run.selected_final_model} />
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={clsx(
                'flex items-center space-x-2 py-4 px-1 border-b-2 text-sm font-medium',
                activeTab === id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              )}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'metrics' && <MetricsTab metrics={run.metrics} />}
        {activeTab === 'charts' && (
          <InteractivePlotSection
            runKey={run.run_key}
            phase={run.phase}
            plotManifest={run.plot_manifest}
            artifacts={run.artifacts}
          />
        )}
        {activeTab === 'bagging' && run.bagging && (
          <BaggingTab bagging={run.bagging} artifacts={run.artifacts} />
        )}
        {activeTab === 'artifacts' && <ArtifactsTab run={run} />}
        {activeTab === 'config' && <ConfigTab config={run.config} features={run.feature_list} />}
        {activeTab === 'lineage' && <LineageTab lineage={run.lineage} />}
      </div>

      {/* Comments */}
      <section className="bg-white rounded-lg border border-gray-200 p-6">
        <Comments scopeType="run" scopeId={run.run_key} />
      </section>
    </div>
  );
}

interface MetricsTabProps {
  metrics: MetricsSummary;
}

function MetricsTab({ metrics }: MetricsTabProps) {
  const hierarchicalLevels = Object.entries(metrics.hierarchical || {}).filter(([, value]) =>
    isHierarchicalLevelMetrics(value)
  );

  return (
    <div className="space-y-8">
      {/* Primary Metrics */}
      {Object.keys(metrics.primary).length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Summary</h3>
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left">Metric</th>
                  <th className="text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.primary).map(([key, value]) => (
                  <tr key={key}>
                    <td>{formatMetricName(key)}</td>
                    <td className="text-right">
                      <MetricValue value={value} precision={4} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Hierarchical Metrics */}
      {hierarchicalLevels.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Hierarchical Metrics</h3>
          <div className="space-y-6">
            {hierarchicalLevels.map(([levelName, levelMetrics]) => (
              <HierarchicalLevelSection
                key={levelName}
                levelName={levelName}
                levelMetrics={levelMetrics}
              />
            ))}
          </div>
        </div>
      )}

      {/* Per-fold Metrics */}
      {Object.keys(metrics.per_fold).length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Per-fold Results</h3>
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left">Metric</th>
                  {Object.values(metrics.per_fold)[0]?.map((_, i) => (
                    <th key={i} className="text-right">Fold {i + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_fold).map(([key, values]) => (
                  <tr key={key}>
                    <td>{formatMetricName(key)}</td>
                    {values.map((value, i) => (
                      <td key={i} className="text-right">
                        <MetricValue value={value} precision={3} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-class Metrics */}
      {metrics.per_class && metrics.per_class.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Per-class Metrics</h3>
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left">Class</th>
                  <th className="text-right">Precision</th>
                  <th className="text-right">Recall</th>
                  <th className="text-right">F1</th>
                  <th className="text-right">Support</th>
                </tr>
              </thead>
              <tbody>
                {metrics.per_class.map((row) => (
                  <tr key={row.class}>
                    <td className="font-medium">{row.class}</td>
                    <td className="text-right">
                      <MetricValue value={row.precision} precision={3} />
                    </td>
                    <td className="text-right">
                      <MetricValue value={row.recall} precision={3} />
                    </td>
                    <td className="text-right">
                      <MetricValue value={row.f1} precision={3} />
                    </td>
                    <td className="text-right">{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ROC AUC */}
      {metrics.roc_auc && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">ROC AUC</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Macro AUC" value={metrics.roc_auc.macro} precision={4} />
            <MetricCard label="Micro AUC" value={metrics.roc_auc.micro} precision={4} />
          </div>
          {metrics.roc_auc.per_class && metrics.roc_auc.per_class.length > 0 && (
            <div className="mt-4 bg-white rounded-lg border border-gray-200 overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="text-left">Class</th>
                    <th className="text-right">AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.roc_auc.per_class.map((row) => (
                    <tr key={row.class}>
                      <td>{row.class}</td>
                      <td className="text-right">
                        <MetricValue value={row.auc} precision={4} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Confusion Matrix */}
      {metrics.confusion_matrix && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Confusion Matrix</h3>
          <div className="bg-white rounded-lg border border-gray-200 overflow-x-auto p-4">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left">Actual / Predicted</th>
                  {metrics.confusion_matrix.labels.map((label) => (
                    <th key={label} className="text-center px-3">{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.confusion_matrix.matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="font-medium">{metrics.confusion_matrix!.labels[i]}</td>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        className={clsx(
                          'text-center px-3',
                          i === j ? 'bg-green-50 font-medium' : ''
                        )}
                      >
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

interface HierarchicalLevelSectionProps {
  levelName: string;
  levelMetrics: HierarchicalLevelMetrics;
}

function HierarchicalLevelSection({ levelName, levelMetrics }: HierarchicalLevelSectionProps) {
  const summary = levelMetrics.summary || {};
  const perFold = levelMetrics.per_fold || {};
  const confusionMatrix = normalizeConfusionMatrix(levelMetrics);
  const warnings = levelMetrics.warnings || [];

  return (
    <section className="bg-white rounded-lg border border-gray-200 p-4 space-y-4">
      <h4 className="text-base font-medium text-gray-900">{formatHierarchyLevelName(levelName)}</h4>

      {levelMetrics.error && (
        <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
          {levelMetrics.error}
        </p>
      )}

      {Object.keys(summary).length > 0 && (
        <div className="overflow-hidden rounded border border-gray-200">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Metric</th>
                <th className="text-right">Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary).map(([key, value]) => (
                <tr key={`${levelName}-${key}`}>
                  <td>{formatMetricName(key)}</td>
                  <td className="text-right">
                    <MetricValue value={value} precision={4} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {Object.keys(perFold).length > 0 && (
        <div className="overflow-hidden rounded border border-gray-200">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Metric</th>
                {Object.values(perFold)[0]?.map((_, i) => (
                  <th key={`${levelName}-fold-${i}`} className="text-right">Fold {i + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(perFold).map(([key, values]) => (
                <tr key={`${levelName}-per-fold-${key}`}>
                  <td>{formatMetricName(key)}</td>
                  {values.map((value, i) => (
                    <td key={`${levelName}-${key}-${i}`} className="text-right">
                      <MetricValue value={value} precision={3} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {levelMetrics.per_class && levelMetrics.per_class.length > 0 && (
        <div className="overflow-hidden rounded border border-gray-200">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Class</th>
                <th className="text-right">Precision</th>
                <th className="text-right">Recall</th>
                <th className="text-right">F1</th>
                <th className="text-right">Support</th>
              </tr>
            </thead>
            <tbody>
              {levelMetrics.per_class.map((row) => (
                <tr key={`${levelName}-class-${row.class}`}>
                  <td className="font-medium">{row.class}</td>
                  <td className="text-right">
                    <MetricValue value={row.precision} precision={3} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={row.recall} precision={3} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={row.f1} precision={3} />
                  </td>
                  <td className="text-right">{row.support ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {confusionMatrix && (
        <div className="overflow-x-auto rounded border border-gray-200 p-3">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Actual / Predicted</th>
                {confusionMatrix.labels.map((label) => (
                  <th key={`${levelName}-cm-head-${label}`} className="text-center px-3">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.matrix.map((row, i) => (
                <tr key={`${levelName}-cm-row-${i}`}>
                  <td className="font-medium">{confusionMatrix.labels[i]}</td>
                  {row.map((value, j) => (
                    <td
                      key={`${levelName}-cm-${i}-${j}`}
                      className={clsx('text-center px-3', i === j ? 'bg-green-50 font-medium' : '')}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="space-y-1">
          {warnings.map((warning, index) => (
            <p
              key={`${levelName}-warning-${index}`}
              className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1"
            >
              {warning}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

interface ArtifactsTabProps {
  run: RunDetail;
}

function ArtifactsTab({ run }: ArtifactsTabProps) {
  if (run.artifacts.length === 0) {
    return <p className="text-gray-500">No artifacts found</p>;
  }

  return <ArtifactList artifacts={run.artifacts} columns={2} />;
}

interface BaggingTabProps {
  bagging: NonNullable<RunDetail['bagging']>;
  artifacts: RunDetail['artifacts'];
}

function BaggingTab({ bagging, artifacts }: BaggingTabProps) {
  const metricsArtifact = bagging.metrics_csv_path
    ? artifacts.find((artifact) => artifact.relative_path === bagging.metrics_csv_path)
    : undefined;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Bag Members" value={bagging.member_count} precision={0} />
        <MetricCard label="Scored Members" value={bagging.members.length} precision={0} />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-2">
        <p className="text-sm text-gray-600">
          Strategy: <span className="font-medium text-gray-900">{bagging.strategy}</span>
        </p>
        {bagging.estimator_type && (
          <p className="text-sm text-gray-600">
            Estimator: <span className="font-mono text-gray-900">{bagging.estimator_type}</span>
          </p>
        )}
        {bagging.task_name && (
          <p className="text-sm text-gray-600">
            Task: <span className="font-medium text-gray-900">{bagging.task_name}</span>
          </p>
        )}
        {metricsArtifact?.download_url && (
          <a
            href={metricsArtifact.download_url}
            className="inline-flex text-sm text-blue-600 hover:underline"
          >
            Download bag member metrics CSV
          </a>
        )}
      </div>

      {bagging.members.length > 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Member</th>
                <th className="text-right">Accuracy</th>
                <th className="text-right">Balanced Acc.</th>
                <th className="text-right">F1 Macro</th>
                <th className="text-right">MCC</th>
                <th className="text-right">ROC AUC</th>
                <th className="text-right">Agreement</th>
              </tr>
            </thead>
            <tbody>
              {bagging.members.map((member) => (
                <tr key={member.member_index}>
                  <td className="font-medium">
                    #{member.member_index}
                    {member.estimator_type && (
                      <div className="text-xs text-gray-500 font-mono">{member.estimator_type}</div>
                    )}
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.accuracy} precision={4} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.balanced_accuracy} precision={4} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.f1_macro} precision={4} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.mcc} precision={4} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.roc_auc_macro} precision={4} />
                  </td>
                  <td className="text-right">
                    <MetricValue value={member.agreement_with_ensemble} precision={4} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-gray-500">
          Member-level evaluation metrics are not available for this run.
        </p>
      )}
    </div>
  );
}

interface ConfigTabProps {
  config: Record<string, unknown>;
  features: string[];
}

function ConfigTab({ config, features }: ConfigTabProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Training Configuration</h3>
        <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
          <pre className="text-gray-100 text-sm">
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>
      </div>

      {features.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Features ({features.length})
          </h3>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex flex-wrap gap-2">
              {features.map((feature) => (
                <span
                  key={feature}
                  className="px-2 py-1 bg-gray-100 rounded text-xs font-mono"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface LineageTabProps {
  lineage: Record<string, unknown> | null;
}

function LineageTab({ lineage }: LineageTabProps) {
  if (!lineage) {
    return <p className="text-gray-500">No lineage information available</p>;
  }

  return (
    <div>
      <h3 className="text-lg font-medium text-gray-900 mb-4">Execution Lineage</h3>
      <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
        <pre className="text-gray-100 text-sm">
          {JSON.stringify(lineage, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function formatPhase(phase: string): string {
  const labels: Record<string, string> = {
    feasibility: 'Feasibility',
    technical_validation: 'Technical Validation',
    independent_test: 'Independent Test',
    final_model: 'Final Model',
  };
  return labels[phase] || phase;
}

function formatMetricName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace('F1', 'F1')
    .replace('Roc', 'ROC')
    .replace('Auc', 'AUC')
    .replace('Mcc', 'MCC');
}

function formatHierarchyLevelName(levelName: string): string {
  if (levelName === 'L1') {
    return 'Level 1';
  }
  if (levelName === 'L2') {
    return 'Level 2';
  }
  if (levelName === 'pipeline') {
    return 'Pipeline';
  }
  return formatMetricName(levelName);
}

function isHierarchicalLevelMetrics(value: unknown): value is HierarchicalLevelMetrics {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const metrics = value as Record<string, unknown>;
  return Boolean(
    metrics.summary
    || metrics.per_fold
    || metrics.per_class
    || metrics.confusion_matrix
    || metrics.roc_auc
    || metrics.error
  );
}

function normalizeConfusionMatrix(
  levelMetrics: HierarchicalLevelMetrics
): { labels: string[]; matrix: Array<Array<number | null>> } | null {
  const direct = levelMetrics.confusion_matrix as {
    labels?: string[];
    matrix?: Array<Array<number | null>>;
  } | undefined;
  if (direct && Array.isArray(direct.labels) && Array.isArray(direct.matrix)) {
    return {
      labels: direct.labels,
      matrix: direct.matrix,
    };
  }
  return null;
}

function buildHeadlineCards(metrics: MetricsSummary): Array<{ id: string; label: string; value: number | null }> {
  const cards: Array<{ id: string; label: string; value: number | null }> = [];
  const overallEntries = Object.entries(metrics.primary).slice(0, 4);
  cards.push(
    ...overallEntries.map(([key, value]) => ({
      id: `overall-${key}`,
      label: formatMetricName(key),
      value,
    }))
  );

  const l1Summary = metrics.hierarchical?.L1?.summary;
  if (l1Summary) {
    const l1MetricOrder = ['accuracy', 'balanced_accuracy', 'f1_macro'] as const;
    for (const metricName of l1MetricOrder) {
      const metricValue = l1Summary[metricName];
      if (metricValue === undefined || metricValue === null) {
        continue;
      }
      cards.push({
        id: `l1-${metricName}`,
        label: `L1 ${formatMetricName(metricName)}`,
        value: metricValue,
      });
    }
  }

  return cards;
}
