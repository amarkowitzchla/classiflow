import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Database, XCircle, FileText, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useProject } from '../hooks/useApi';
import { DecisionBadge } from '../components/DecisionBadge';
import { MetricValue } from '../components/MetricValue';
import { ArtifactList } from '../components/ArtifactViewer';
import { Comments } from '../components/Comments';
import { PromotionGates } from '../components/PromotionGate';
import type { RunBrief } from '../types/api';

export function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: project, isLoading, error } = useProject(projectId || '');

  if (isLoading) {
    return <div className="text-center py-12 text-gray-500">Loading project...</div>;
  }

  if (error || !project) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 mx-auto mb-4 text-red-300" />
        <p className="text-red-500">Failed to load project</p>
        <Link to="/" className="text-blue-600 hover:underline mt-2 inline-block">
          Back to projects
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <Link
          to="/"
          className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to projects
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-gray-900">{project.name}</h1>
              <DecisionBadge decision={project.promotion.decision} size="lg" />
            </div>
            <p className="text-gray-500 mt-1">{project.id}</p>
            {project.description && (
              <p className="text-gray-600 mt-2">{project.description}</p>
            )}
          </div>
          <div className="text-right text-sm text-gray-500">
            {project.task_mode && (
              <span className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700">
                {project.task_mode}
              </span>
            )}
            {project.updated_at && (
              <div className="mt-2 flex items-center justify-end space-x-1">
                <Clock className="w-3 h-3" />
                <span>Updated {formatDistanceToNow(new Date(project.updated_at), { addSuffix: true })}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Promotion Gates */}
      {project.promotion && project.promotion.gates && Object.keys(project.promotion.gates).length > 0 && (
        <section>
          <PromotionGates gates={project.promotion.gates} projectId={project.id} />
          {/* Override notice */}
          {project.promotion.override_enabled && project.promotion.override_comment && (
            <div className="mt-4 p-4 rounded-lg border border-yellow-200 bg-yellow-50">
              <p className="text-sm">
                <strong>Override:</strong> {project.promotion.override_comment}
                {project.promotion.override_approver && (
                  <> by {project.promotion.override_approver}</>
                )}
              </p>
            </div>
          )}
        </section>
      )}

      {/* Registry Summary */}
      {project.registry && (project.registry.train || project.registry.test) && (
        <section>
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <Database className="w-5 h-5" />
            <span>Data Registry</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {project.registry.train && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <h3 className="font-medium text-gray-900 mb-2">Training Data</h3>
                <dl className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Rows</dt>
                    <dd className="font-mono">{project.registry.train.row_count}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Features</dt>
                    <dd className="font-mono">{project.registry.train.feature_count}</dd>
                  </div>
                  {project.registry.train.label_distribution && (
                    <div className="pt-2 mt-2 border-t border-gray-100">
                      <dt className="text-gray-500 mb-1">Class distribution</dt>
                      <dd className="flex flex-wrap gap-2">
                        {Object.entries(project.registry.train.label_distribution).map(([cls, count]) => (
                          <span key={cls} className="px-2 py-0.5 bg-gray-100 rounded text-xs">
                            {cls}: {count as number}
                          </span>
                        ))}
                      </dd>
                    </div>
                  )}
                </dl>
              </div>
            )}
            {project.registry.test && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <h3 className="font-medium text-gray-900 mb-2">Test Data</h3>
                <dl className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Rows</dt>
                    <dd className="font-mono">{project.registry.test.row_count}</dd>
                  </div>
                </dl>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Runs by Phase */}
      <section>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Runs</h2>
        <div className="space-y-6">
          {Object.entries(project.phases).map(([phase, runs]) => (
            <PhaseSection key={phase} phase={phase} runs={runs} projectId={project.id} />
          ))}
          {Object.keys(project.phases).length === 0 && (
            <p className="text-gray-500">No runs yet</p>
          )}
        </div>
      </section>

      {/* Artifact Highlights */}
      {project.artifact_highlights && project.artifact_highlights.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <FileText className="w-5 h-5" />
            <span>Recent Artifacts</span>
          </h2>
          <ArtifactList artifacts={project.artifact_highlights} columns={3} />
        </section>
      )}

      {/* Comments */}
      <section className="bg-white rounded-lg border border-gray-200 p-6">
        <Comments scopeType="project" scopeId={project.id} />
      </section>
    </div>
  );
}

interface PhaseSectionProps {
  phase: string;
  runs: RunBrief[];
  projectId: string;
}

function PhaseSection({ phase, runs, projectId }: PhaseSectionProps) {
  const phaseLabels: Record<string, string> = {
    feasibility: 'Feasibility',
    technical_validation: 'Technical Validation',
    independent_test: 'Independent Test',
    final_model: 'Final Model',
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50 rounded-t-lg">
        <h3 className="font-medium text-gray-900">{phaseLabels[phase] || phase}</h3>
      </div>
      <div className="divide-y divide-gray-100">
        {runs.map((run) => (
          <Link
            key={run.run_key}
            to={`/projects/${encodeURIComponent(projectId)}/runs/${phase}/${run.run_id}`}
            className="block px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="font-mono text-sm text-gray-900">{run.run_id}</span>
                {run.task_type && (
                  <span className="ml-2 text-xs text-gray-400">{run.task_type}</span>
                )}
              </div>
              <div className="flex items-center space-x-4">
                {Object.entries(run.headline_metrics).slice(0, 2).map(([key, value]) => (
                  <div key={key} className="text-right">
                    <span className="text-xs text-gray-400 mr-1">{formatMetricKey(key)}:</span>
                    <MetricValue value={value} precision={3} className="text-sm font-medium" />
                  </div>
                ))}
                {run.created_at && (
                  <span className="text-xs text-gray-400">
                    {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                  </span>
                )}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

function formatMetricKey(key: string): string {
  const abbrevs: Record<string, string> = {
    balanced_accuracy: 'BA',
    f1_macro: 'F1',
    accuracy: 'Acc',
    roc_auc_macro: 'AUC',
  };
  return abbrevs[key] || key.slice(0, 6);
}
