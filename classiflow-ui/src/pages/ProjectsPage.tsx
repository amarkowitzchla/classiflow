import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, RefreshCw, FolderOpen, Calendar, GitBranch } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useProjects, useReindex } from '../hooks/useApi';
import { GateBadges } from '../components/GateBadges';
import { MetricValue } from '../components/MetricValue';
import type { ProjectCard } from '../types/api';

export function ProjectsPage() {
  const [search, setSearch] = useState('');
  const [modeFilter, setModeFilter] = useState<string>('');

  const { data, isLoading, error } = useProjects({
    q: search || undefined,
    mode: modeFilter || undefined,
  });

  const reindexMutation = useReindex();

  const handleReindex = () => {
    reindexMutation.mutate();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Projects</h1>
        <button
          onClick={handleReindex}
          disabled={reindexMutation.isPending}
          className="flex items-center space-x-2 px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${reindexMutation.isPending ? 'animate-spin' : ''}`} />
          <span>Reindex</span>
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search projects..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <select
          value={modeFilter}
          onChange={(e) => setModeFilter(e.target.value)}
          className="px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All modes</option>
          <option value="binary">Binary</option>
          <option value="meta">Meta</option>
          <option value="hierarchical">Hierarchical</option>
        </select>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="text-center py-12 text-gray-500">Loading projects...</div>
      ) : error ? (
        <div className="text-center py-12 text-red-500">Failed to load projects</div>
      ) : data?.items.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <FolderOpen className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>No projects found</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {data?.items.map((project: ProjectCard) => (
            <ProjectCardComponent key={project.id} project={project} />
          ))}
        </div>
      )}

      {/* Pagination info */}
      {data && data.total > 0 && (
        <div className="text-sm text-gray-500 text-center">
          Showing {data.items.length} of {data.total} projects
        </div>
      )}
    </div>
  );
}

interface ProjectCardProps {
  project: ProjectCard;
}

function ProjectCardComponent({ project }: ProjectCardProps) {
  return (
    <Link
      to={`/projects/${encodeURIComponent(project.id)}`}
      className="block bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-sm transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-gray-900 truncate">
              {project.name}
            </h3>
            <GateBadges gateStatus={project.gate_status} />
          </div>
          <div className="mt-1 text-sm text-gray-500 truncate">
            {project.id}
          </div>
          {project.description && (
            <p className="mt-2 text-sm text-gray-600 line-clamp-2">
              {project.description}
            </p>
          )}
        </div>

        {/* Headline metrics */}
        {Object.keys(project.headline_metrics).length > 0 && (
          <div className="ml-6 flex-shrink-0">
            <div className="flex items-center space-x-4">
              {Object.entries(project.headline_metrics).slice(0, 3).map(([key, value]) => (
                <div key={key} className="text-right">
                  <div className="text-xs text-gray-400">{formatMetricName(key)}</div>
                  <div className="text-lg font-semibold">
                    <MetricValue value={value} precision={3} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center space-x-4">
          {project.task_mode && (
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs">
              {project.task_mode}
            </span>
          )}
          <span className="flex items-center space-x-1">
            <GitBranch className="w-3 h-3" />
            <span>{project.phases_present.join(', ') || 'No phases'}</span>
          </span>
          <span>{project.run_count} runs</span>
        </div>
        {project.updated_at && (
          <span className="flex items-center space-x-1">
            <Calendar className="w-3 h-3" />
            <span>
              {formatDistanceToNow(new Date(project.updated_at), { addSuffix: true })}
            </span>
          </span>
        )}
      </div>
    </Link>
  );
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
