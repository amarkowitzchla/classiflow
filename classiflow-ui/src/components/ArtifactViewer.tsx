import { Download, ExternalLink, FileText, Image, FileJson, Table } from 'lucide-react';
import { clsx } from 'clsx';
import type { Artifact, ArtifactKind } from '../types/api';
import { getArtifactContentUrl } from '../api/client';

interface ArtifactViewerProps {
  artifact: Artifact;
}

export function ArtifactViewer({ artifact }: ArtifactViewerProps) {
  const viewUrl = artifact.view_url || getArtifactContentUrl(artifact.artifact_id);
  const downloadUrl = artifact.download_url || getArtifactContentUrl(artifact.artifact_id, true);

  // Render based on kind
  if (artifact.kind === 'image') {
    return (
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <img
          src={viewUrl}
          alt={artifact.title}
          className="w-full h-auto"
          loading="lazy"
        />
        <div className="p-3 border-t border-gray-100 flex items-center justify-between">
          <span className="text-sm text-gray-600 truncate">{artifact.title}</span>
          <a
            href={downloadUrl}
            download
            className="text-blue-600 hover:text-blue-800"
          >
            <Download className="w-4 h-4" />
          </a>
        </div>
      </div>
    );
  }

  // For other types, show preview link
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center space-x-3">
        <ArtifactIcon kind={artifact.kind} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-gray-900 truncate">
            {artifact.title}
          </div>
          <div className="text-xs text-gray-500">
            {formatBytes(artifact.size_bytes)}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {artifact.is_viewable && (
            <a
              href={viewUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-blue-600"
              title="View"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
          <a
            href={downloadUrl}
            download
            className="text-gray-500 hover:text-blue-600"
            title="Download"
          >
            <Download className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
}

interface ArtifactIconProps {
  kind: ArtifactKind;
  className?: string;
}

export function ArtifactIcon({ kind, className }: ArtifactIconProps) {
  const iconProps = { className: clsx('w-5 h-5', className) };

  switch (kind) {
    case 'image':
      return <Image {...iconProps} className={clsx(iconProps.className, 'text-purple-500')} />;
    case 'report':
      return <FileText {...iconProps} className={clsx(iconProps.className, 'text-blue-500')} />;
    case 'metrics':
      return <Table {...iconProps} className={clsx(iconProps.className, 'text-green-500')} />;
    case 'config':
      return <FileJson {...iconProps} className={clsx(iconProps.className, 'text-yellow-500')} />;
    default:
      return <FileText {...iconProps} className={clsx(iconProps.className, 'text-gray-500')} />;
  }
}

interface ArtifactListProps {
  artifacts: Artifact[];
  columns?: 1 | 2 | 3;
}

export function ArtifactList({ artifacts, columns = 2 }: ArtifactListProps) {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
  };

  // Group by kind
  const imageArtifacts = artifacts.filter((a) => a.kind === 'image');
  const otherArtifacts = artifacts.filter((a) => a.kind !== 'image');

  return (
    <div className="space-y-6">
      {imageArtifacts.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Images</h4>
          <div className={clsx('grid gap-4', gridCols[columns])}>
            {imageArtifacts.map((artifact) => (
              <ArtifactViewer key={artifact.artifact_id} artifact={artifact} />
            ))}
          </div>
        </div>
      )}

      {otherArtifacts.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Files</h4>
          <div className="space-y-2">
            {otherArtifacts.map((artifact) => (
              <ArtifactViewer key={artifact.artifact_id} artifact={artifact} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatBytes(bytes: number | null): string {
  if (bytes === null) return 'Unknown size';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
