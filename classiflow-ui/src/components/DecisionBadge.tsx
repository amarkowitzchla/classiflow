import { clsx } from 'clsx';
import type { DecisionBadge as DecisionBadgeType } from '../types/api';

interface DecisionBadgeProps {
  decision: DecisionBadgeType;
  size?: 'sm' | 'md' | 'lg';
}

export function DecisionBadge({ decision, size = 'md' }: DecisionBadgeProps) {
  const sizeClasses = {
    sm: 'px-1.5 py-0.5 text-xs',
    md: 'px-2 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm',
  };

  const colorClasses = {
    PASS: 'bg-green-100 text-green-800',
    FAIL: 'bg-red-100 text-red-800',
    PENDING: 'bg-yellow-100 text-yellow-800',
    OVERRIDE: 'bg-purple-100 text-purple-800',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full font-medium',
        sizeClasses[size],
        colorClasses[decision]
      )}
    >
      {decision}
    </span>
  );
}
