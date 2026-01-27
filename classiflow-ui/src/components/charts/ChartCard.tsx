import { ReactNode } from 'react';
import { clsx } from 'clsx';
import { Info, AlertCircle } from 'lucide-react';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  loading?: boolean;
  error?: string | null;
  fallbackMessage?: string;
  actions?: ReactNode;
}

export function ChartCard({
  title,
  subtitle,
  children,
  className,
  loading = false,
  error = null,
  fallbackMessage,
  actions,
}: ChartCardProps) {
  return (
    <div
      className={clsx(
        'bg-white rounded-lg border border-gray-200 overflow-hidden',
        className
      )}
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex items-center space-x-2">{actions}</div>}
      </div>

      {/* Content */}
      <div className="p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-64 text-red-500">
            <AlertCircle className="w-10 h-10 mb-2" />
            <p className="text-sm">{error}</p>
          </div>
        ) : (
          children
        )}
      </div>

      {/* Fallback message */}
      {fallbackMessage && !loading && !error && (
        <div className="px-6 pb-4">
          <div className="flex items-start space-x-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
            <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{fallbackMessage}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChartCard;
