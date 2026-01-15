import { CheckCircle, XCircle, Clock } from 'lucide-react';
import { clsx } from 'clsx';
import type { GateStatus } from '../types/api';

interface GateBadgesProps {
  gateStatus: Record<string, GateStatus>;
  size?: 'sm' | 'md';
}

/**
 * Displays two badge indicators for the clinical validation gates:
 * 1. Technical Validation (TV)
 * 2. Independent Test (IT)
 *
 * Each badge shows PASS (green), FAIL (red), or PENDING (gray)
 */
export function GateBadges({ gateStatus, size = 'md' }: GateBadgesProps) {
  const techStatus = gateStatus['technical_validation'] || 'PENDING';
  const testStatus = gateStatus['independent_test'] || 'PENDING';

  return (
    <div className="flex items-center space-x-1">
      <GateBadge label="TV" status={techStatus} title="Technical Validation" size={size} />
      <GateBadge label="IT" status={testStatus} title="Independent Test" size={size} />
    </div>
  );
}

interface GateBadgeProps {
  label: string;
  status: GateStatus;
  title: string;
  size: 'sm' | 'md';
}

function GateBadge({ label, status, title, size }: GateBadgeProps) {
  const sizeClasses = {
    sm: 'px-1.5 py-0.5 text-xs',
    md: 'px-2 py-1 text-xs',
  };

  const iconSize = size === 'sm' ? 'w-3 h-3' : 'w-3.5 h-3.5';

  const statusConfig = {
    PASS: {
      bg: 'bg-green-100',
      text: 'text-green-700',
      border: 'border-green-200',
      icon: <CheckCircle className={clsx(iconSize, 'text-green-600')} />,
    },
    FAIL: {
      bg: 'bg-red-100',
      text: 'text-red-700',
      border: 'border-red-200',
      icon: <XCircle className={clsx(iconSize, 'text-red-600')} />,
    },
    PENDING: {
      bg: 'bg-gray-100',
      text: 'text-gray-500',
      border: 'border-gray-200',
      icon: <Clock className={clsx(iconSize, 'text-gray-400')} />,
    },
  };

  const config = statusConfig[status];

  return (
    <span
      title={`${title}: ${status}`}
      className={clsx(
        'inline-flex items-center space-x-1 rounded border font-medium',
        sizeClasses[size],
        config.bg,
        config.text,
        config.border
      )}
    >
      {config.icon}
      <span>{label}</span>
    </span>
  );
}
