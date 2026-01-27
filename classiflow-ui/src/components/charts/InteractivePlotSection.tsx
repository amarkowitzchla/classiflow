import { useState } from 'react';
import { usePlotData } from '../../hooks/useApi';
import { RocPrChart } from './RocPrChart';
import { ChartCard } from './ChartCard';
import { FoldSelector } from './FoldSelector';
import { PlotKey, PlotKeyType } from '../../types/plots';
import type { PlotManifest, Artifact } from '../../types/api';
import { getArtifactContentUrl } from '../../api/client';

interface InteractivePlotSectionProps {
  runKey: string;
  phase: string;
  plotManifest: PlotManifest | null;
  artifacts: Artifact[];
  className?: string;
}

// Find artifact by relative path pattern
function findArtifactByPath(artifacts: Artifact[], pattern: string): Artifact | undefined {
  return artifacts.find((a) => a.relative_path.includes(pattern));
}

export function InteractivePlotSection({
  runKey,
  phase,
  plotManifest,
  artifacts,
  className,
}: InteractivePlotSectionProps) {
  const [selectedFold, setSelectedFold] = useState<number | 'all'>('all');

  // Determine which plot keys to use based on phase
  const isInference = phase === 'independent_test';
  const rocKey: PlotKeyType = isInference ? PlotKey.ROC_INFERENCE : PlotKey.ROC_AVERAGED;
  const prKey: PlotKeyType = isInference ? PlotKey.PR_INFERENCE : PlotKey.PR_AVERAGED;

  // Check if JSON data is available
  const hasRocJson = plotManifest?.available?.[rocKey];
  const hasPrJson = plotManifest?.available?.[prKey];

  // Fetch plot data if available
  const rocQuery = usePlotData(runKey, rocKey, !!hasRocJson);
  const prQuery = usePlotData(runKey, prKey, !!hasPrJson);

  // Find fallback PNG artifacts
  const rocFallbackPath = plotManifest?.fallback_pngs?.[rocKey];
  const prFallbackPath = plotManifest?.fallback_pngs?.[prKey];

  const rocFallbackArtifact = rocFallbackPath
    ? findArtifactByPath(artifacts, rocFallbackPath)
    : findArtifactByPath(artifacts, isInference ? 'inference_roc' : 'averaged_roc');

  const prFallbackArtifact = prFallbackPath
    ? findArtifactByPath(artifacts, prFallbackPath)
    : findArtifactByPath(artifacts, isInference ? 'inference_pr' : 'averaged_pr');

  // For technical validation, get fold count from plot data
  const foldCount = rocQuery.data?.fold_metrics
    ? Object.values(rocQuery.data.fold_metrics)[0]?.length ?? 0
    : 0;
  const folds = Array.from({ length: foldCount }, (_, i) => i + 1);

  const showFoldSelector = !isInference && folds.length > 0;

  return (
    <div className={className}>
      {/* Fold selector for technical validation */}
      {showFoldSelector && (
        <div className="mb-4">
          <FoldSelector
            folds={folds}
            selectedFold={selectedFold}
            onSelect={setSelectedFold}
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <ChartCard
          title="ROC Curve"
          subtitle={isInference ? 'Inference Results' : 'Cross-Validation Average'}
          loading={rocQuery.isLoading}
          error={rocQuery.error ? 'Failed to load ROC data' : null}
          fallbackMessage={
            !hasRocJson && rocFallbackArtifact
              ? 'Interactive chart not available for this run. Showing static image.'
              : undefined
          }
        >
          {rocQuery.data ? (
            <RocPrChart data={rocQuery.data} height={350} />
          ) : rocFallbackArtifact ? (
            <img
              src={getArtifactContentUrl(rocFallbackArtifact.artifact_id)}
              alt="ROC Curve"
              className="w-full h-auto"
            />
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400">
              No ROC curve available
            </div>
          )}
        </ChartCard>

        {/* PR Curve */}
        <ChartCard
          title="Precision-Recall Curve"
          subtitle={isInference ? 'Inference Results' : 'Cross-Validation Average'}
          loading={prQuery.isLoading}
          error={prQuery.error ? 'Failed to load PR data' : null}
          fallbackMessage={
            !hasPrJson && prFallbackArtifact
              ? 'Interactive chart not available for this run. Showing static image.'
              : undefined
          }
        >
          {prQuery.data ? (
            <RocPrChart data={prQuery.data} height={350} showDiagonal={false} />
          ) : prFallbackArtifact ? (
            <img
              src={getArtifactContentUrl(prFallbackArtifact.artifact_id)}
              alt="PR Curve"
              className="w-full h-auto"
            />
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400">
              No PR curve available
            </div>
          )}
        </ChartCard>
      </div>
    </div>
  );
}

export default InteractivePlotSection;
