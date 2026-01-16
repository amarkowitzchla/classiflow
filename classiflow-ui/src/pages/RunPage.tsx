import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, BarChart2, FileText, GitBranch, Settings, XCircle, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { clsx } from 'clsx';
import { useRun } from '../hooks/useApi';
import { MetricValue, MetricCard } from '../components/MetricValue';
import { ArtifactList } from '../components/ArtifactViewer';
import { Comments } from '../components/Comments';
import type { RunDetail, MetricsSummary } from '../types/api';

type TabId = 'metrics' | 'artifacts' | 'config' | 'lineage';

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
    { id: 'artifacts', label: `Artifacts (${run.artifact_count})`, icon: FileText },
    { id: 'config', label: 'Config', icon: Settings },
    { id: 'lineage', label: 'Lineage', icon: GitBranch },
  ];

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
      {Object.keys(run.metrics.primary).length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(run.metrics.primary).slice(0, 4).map(([key, value]) => (
            <MetricCard
              key={key}
              label={formatMetricName(key)}
              value={value}
              precision={4}
            />
          ))}
        </div>
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

interface ArtifactsTabProps {
  run: RunDetail;
}

function ArtifactsTab({ run }: ArtifactsTabProps) {
  if (run.artifacts.length === 0) {
    return <p className="text-gray-500">No artifacts found</p>;
  }

  return <ArtifactList artifacts={run.artifacts} columns={2} />;
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
