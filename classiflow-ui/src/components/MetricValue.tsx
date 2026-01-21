import { clsx } from 'clsx';

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
