import { clsx } from 'clsx';
import { Link } from 'react-router-dom';
import { Box, Cpu, Layers, Settings } from 'lucide-react';
import type { SelectedFinalModelSummary, SelectedModelConfig } from '../types/api';

interface MetricValueProps {
  value: number | null | undefined;
  format?: 'percent' | 'decimal' | 'integer';
  precision?: number;
  className?: string;
}

export function MetricValue({
  value,
  format = 'decimal',
  precision = 3,
  className,
}: MetricValueProps) {
  let displayValue: string;
  const isValidNumber = typeof value === 'number' && Number.isFinite(value);

  if (!isValidNumber) {
    displayValue = 'NA';
  } else if (format === 'percent') {
    displayValue = `${(value * 100).toFixed(precision - 2)}%`;
  } else if (format === 'integer') {
    displayValue = Math.round(value).toString();
  } else {
    displayValue = value.toFixed(precision);
  }

  return (
    <span className={clsx('font-mono tabular-nums', className)}>
      {displayValue}
    </span>
  );
}

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  format?: 'percent' | 'decimal' | 'integer';
  precision?: number;
  sublabel?: string;
}

export function MetricCard({
  label,
  value,
  format = 'decimal',
  precision = 3,
  sublabel,
}: MetricCardProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="text-sm text-gray-500 mb-1">{label}</div>
      <div className="text-2xl font-semibold">
        <MetricValue value={value} format={format} precision={precision} />
      </div>
      {sublabel && <div className="text-xs text-gray-400 mt-1">{sublabel}</div>}
    </div>
  );
}

interface FinalModelSummaryProps {
  summary: SelectedFinalModelSummary;
  linkToRun?: boolean;
}

export function FinalModelSummary({ summary, linkToRun = false }: FinalModelSummaryProps) {
  const strategy = stringValue(summary.strategy.final_estimator_strategy) || 'single';
  const selectedModels = summary.meta_model
    ? [...summary.selected_models, summary.meta_model]
    : summary.selected_models;

  return (
    <section className="bg-white rounded-lg border border-gray-200 p-5 space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <Box className="w-5 h-5" />
            <span>Final Bundle Selection</span>
          </h2>
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            {summary.task_type && <Badge>{summary.task_type}</Badge>}
            <Badge>{strategy}</Badge>
            {summary.sampler && <Badge>{summary.sampler}</Badge>}
            {summary.train_from_scratch && <Badge>train from scratch</Badge>}
          </div>
        </div>
        {linkToRun && (
          <Link
            to={`/projects/${encodeURIComponent(summary.run_key.split(':')[0])}/runs/final_model/${summary.run_id}`}
            className="text-sm text-blue-600 hover:underline"
          >
            Open final run
          </Link>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <InfoItem
          icon={Settings}
          label="Selection"
          value={[summary.selection_metric, summary.selection_direction].filter(Boolean).join(' / ') || 'NA'}
        />
        <InfoItem
          icon={Layers}
          label="Bagging"
          value={formatBagging(summary.strategy)}
        />
        <InfoItem
          icon={Cpu}
          label="Execution"
          value={formatExecution(summary.execution)}
        />
      </div>

      {selectedModels.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Task</th>
                <th className="text-left">Model</th>
                <th className="text-right">Score</th>
                <th className="text-left">Sampler</th>
                <th className="text-left">Selected Params</th>
              </tr>
            </thead>
            <tbody>
              {selectedModels.map((model) => (
                <SelectedModelRow key={`${model.task_name}:${model.model_name}`} model={model} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-gray-500">No selected model registry was found for this bundle.</p>
      )}
    </section>
  );
}

function SelectedModelRow({ model }: { model: SelectedModelConfig }) {
  return (
    <tr>
      <td className="font-medium">{model.task_name}</td>
      <td className="font-mono text-sm">{model.model_name}</td>
      <td className="text-right">
        <MetricValue value={model.mean_score} precision={4} />
      </td>
      <td>{model.sampler || 'NA'}</td>
      <td>
        <div className="flex flex-wrap gap-1">
          {summarizeParams(model.params).map(([key, value]) => (
            <span key={key} className="px-2 py-0.5 rounded bg-gray-100 text-xs font-mono">
              {key}={value}
            </span>
          ))}
          {Object.keys(model.params || {}).length === 0 && (
            <span className="text-xs text-gray-400">default</span>
          )}
        </div>
      </td>
    </tr>
  );
}

function InfoItem({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Settings;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded border border-gray-200 p-3">
      <div className="flex items-center space-x-2 text-gray-500 mb-1">
        <Icon className="w-4 h-4" />
        <span>{label}</span>
      </div>
      <div className="font-medium text-gray-900">{value}</div>
    </div>
  );
}

function Badge({ children }: { children: string }) {
  return (
    <span className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700">
      {children}
    </span>
  );
}

function summarizeParams(params: Record<string, unknown>): Array<[string, string]> {
  return Object.entries(params || {})
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .slice(0, 6)
    .map(([key, value]) => [key, formatParamValue(value)]);
}

function formatParamValue(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.length}]`;
  if (value && typeof value === 'object') return '{...}';
  return String(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function formatBagging(strategy: Record<string, unknown>): string {
  const kind = stringValue(strategy.final_estimator_strategy) || 'single';
  if (kind !== 'bagged') return 'single';
  const n = strategy.bagging_n_estimators;
  const maxSamples = strategy.bagging_max_samples;
  return `${n || 'NA'} estimators, max samples ${maxSamples || 'NA'}`;
}

function formatExecution(execution: Record<string, unknown>): string {
  const engine = stringValue(execution.engine) || 'unknown';
  const device = stringValue(execution.device);
  const modelSet = stringValue(execution.model_set);
  return [engine, device, modelSet].filter(Boolean).join(' / ');
}
