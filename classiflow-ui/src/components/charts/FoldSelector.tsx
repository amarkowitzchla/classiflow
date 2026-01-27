import { clsx } from 'clsx';

interface FoldSelectorProps {
  folds: number[];
  selectedFold: number | 'all';
  onSelect: (fold: number | 'all') => void;
  className?: string;
}

export function FoldSelector({
  folds,
  selectedFold,
  onSelect,
  className,
}: FoldSelectorProps) {
  return (
    <div className={clsx('flex items-center space-x-2', className)}>
      <span className="text-sm text-gray-500">View:</span>
      <div className="flex rounded-lg border border-gray-200 overflow-hidden">
        <button
          onClick={() => onSelect('all')}
          className={clsx(
            'px-3 py-1.5 text-sm font-medium transition-colors',
            selectedFold === 'all'
              ? 'bg-blue-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-50'
          )}
        >
          Averaged
        </button>
        {folds.map((fold) => (
          <button
            key={fold}
            onClick={() => onSelect(fold)}
            className={clsx(
              'px-3 py-1.5 text-sm font-medium transition-colors border-l border-gray-200',
              selectedFold === fold
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            )}
          >
            Fold {fold}
          </button>
        ))}
      </div>
    </div>
  );
}

export default FoldSelector;
