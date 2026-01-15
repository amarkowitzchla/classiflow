import { Link } from 'react-router-dom';
import { CheckCircle, XCircle, AlertCircle, ChevronRight, TrendingUp, Activity } from 'lucide-react';
import type { GateResult, GateCheck } from '../types/api';

interface PromotionGateProps {
  gate: GateResult;
  projectId: string;
  stepNumber: number;
}

export function PromotionGate({ gate, projectId, stepNumber }: PromotionGateProps) {
  const requiredChecks = gate.checks.filter(c => c.check_type === 'required');
  const stabilityChecks = gate.checks.filter(c =>
    c.check_type === 'stability_std' || c.check_type === 'stability_pass_rate'
  );

  return (
    <div className={`rounded-lg border-2 ${
      gate.passed
        ? 'border-green-200 bg-green-50'
        : 'border-red-200 bg-red-50'
    }`}>
      {/* Header */}
      <div className={`px-4 py-3 border-b ${
        gate.passed ? 'border-green-200 bg-green-100' : 'border-red-200 bg-red-100'
      } rounded-t-lg`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className={`flex items-center justify-center w-8 h-8 rounded-full text-white font-bold ${
              gate.passed ? 'bg-green-500' : 'bg-red-500'
            }`}>
              {stepNumber}
            </span>
            <div>
              <h3 className="font-semibold text-gray-900">{gate.phase_label}</h3>
              <p className="text-sm text-gray-600">
                {gate.phase === 'technical_validation'
                  ? 'Nested cross-validation metrics'
                  : 'Holdout test set evaluation'}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {gate.passed ? (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-500 text-white">
                <CheckCircle className="w-4 h-4 mr-1" />
                PASS
              </span>
            ) : (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-500 text-white">
                <XCircle className="w-4 h-4 mr-1" />
                FAIL
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Required Metrics Section */}
        {requiredChecks.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
              <TrendingUp className="w-4 h-4 mr-1" />
              Required Metrics
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {requiredChecks.map((check, idx) => (
                <MetricCheckRow key={idx} check={check} />
              ))}
            </div>
          </div>
        )}

        {/* Stability Section (for technical validation) */}
        {stabilityChecks.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
              <Activity className="w-4 h-4 mr-1" />
              Stability Checks
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {stabilityChecks.map((check, idx) => (
                <MetricCheckRow key={idx} check={check} isStability />
              ))}
            </div>
          </div>
        )}

        {/* Link to run */}
        {gate.run_id && (
          <div className="pt-2 border-t border-gray-200">
            <Link
              to={`/projects/${encodeURIComponent(projectId)}/runs/${gate.phase}/${gate.run_id}`}
              className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800"
            >
              View run details
              <ChevronRight className="w-4 h-4 ml-1" />
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}

interface MetricCheckRowProps {
  check: GateCheck;
  isStability?: boolean;
}

function MetricCheckRow({ check }: MetricCheckRowProps) {
  const formatMetricName = (name: string): string => {
    const labels: Record<string, string> = {
      'f1': 'F1 Macro',
      'f1_macro': 'F1 Macro',
      'balanced_accuracy': 'Balanced Accuracy',
      'accuracy': 'Accuracy',
      'roc_auc_ovr_macro': 'ROC AUC (Macro)',
      'mcc': 'MCC',
    };

    // Handle stability metric names like "f1_std" or "balanced_accuracy_pass_rate"
    if (name.endsWith('_std')) {
      const base = name.replace('_std', '');
      return `${labels[base] || base} Std Dev`;
    }
    if (name.endsWith('_pass_rate')) {
      const base = name.replace('_pass_rate', '');
      return `${labels[base] || base} Pass Rate`;
    }

    return labels[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const formatValue = (value: number | null, checkType: string): string => {
    if (value === null) return 'N/A';
    if (checkType === 'stability_pass_rate') {
      return `${(value * 100).toFixed(0)}%`;
    }
    if (checkType === 'stability_std') {
      return value.toFixed(4);
    }
    return value.toFixed(4);
  };

  const formatThreshold = (threshold: number, checkType: string): string => {
    if (checkType === 'stability_pass_rate') {
      return `≥ ${(threshold * 100).toFixed(0)}%`;
    }
    if (checkType === 'stability_std') {
      return `≤ ${threshold.toFixed(2)}`;
    }
    return `≥ ${threshold.toFixed(2)}`;
  };

  return (
    <div className={`flex items-center justify-between p-2 rounded ${
      check.passed ? 'bg-green-100' : 'bg-red-100'
    }`}>
      <div className="flex items-center space-x-2">
        {check.passed ? (
          <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />
        ) : check.actual === null ? (
          <AlertCircle className="w-4 h-4 text-yellow-600 flex-shrink-0" />
        ) : (
          <XCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
        )}
        <span className="text-sm font-medium text-gray-700">
          {formatMetricName(check.metric)}
        </span>
      </div>
      <div className="text-right">
        <span className={`text-sm font-mono ${
          check.passed ? 'text-green-700' : 'text-red-700'
        }`}>
          {formatValue(check.actual, check.check_type)}
        </span>
        <span className="text-xs text-gray-500 ml-1">
          ({formatThreshold(check.threshold, check.check_type)})
        </span>
      </div>
    </div>
  );
}

interface PromotionGatesProps {
  gates: Record<string, GateResult>;
  projectId: string;
}

export function PromotionGates({ gates, projectId }: PromotionGatesProps) {
  const orderedPhases = ['technical_validation', 'independent_test'];
  const gateEntries = orderedPhases
    .filter(phase => gates[phase])
    .map(phase => gates[phase]);

  if (gateEntries.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No promotion gates configured
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-900">Clinical Validation Gates</h2>
      <p className="text-sm text-gray-600">
        Both gates must pass for the model to be considered validated for clinical use.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {gateEntries.map((gate, idx) => (
          <PromotionGate
            key={gate.phase}
            gate={gate}
            projectId={projectId}
            stepNumber={idx + 1}
          />
        ))}
      </div>
    </div>
  );
}
