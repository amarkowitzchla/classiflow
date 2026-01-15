// React Query hooks for API calls

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as api from '../api/client';
import type { ReviewStatus } from '../types/api';

// Query keys
export const queryKeys = {
  health: ['health'] as const,
  projects: (params?: api.ListProjectsParams) => ['projects', params] as const,
  project: (id: string) => ['project', id] as const,
  projectRuns: (id: string) => ['project', id, 'runs'] as const,
  run: (key: string) => ['run', key] as const,
  runArtifacts: (key: string, params?: api.ListArtifactsParams) => ['run', key, 'artifacts', params] as const,
  artifact: (id: string) => ['artifact', id] as const,
  comments: (scopeType: string, scopeId: string) => ['comments', scopeType, scopeId] as const,
  reviews: (scopeType: string, scopeId: string) => ['reviews', scopeType, scopeId] as const,
};

// Health
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
    staleTime: 30000,
  });
}

// Projects
export function useProjects(params?: api.ListProjectsParams) {
  return useQuery({
    queryKey: queryKeys.projects(params),
    queryFn: () => api.listProjects(params),
  });
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: queryKeys.project(projectId),
    queryFn: () => api.getProject(projectId),
    enabled: !!projectId,
  });
}

export function useProjectRuns(projectId: string) {
  return useQuery({
    queryKey: queryKeys.projectRuns(projectId),
    queryFn: () => api.getProjectRuns(projectId),
    enabled: !!projectId,
  });
}

// Runs
export function useRun(runKey: string) {
  return useQuery({
    queryKey: queryKeys.run(runKey),
    queryFn: () => api.getRun(runKey),
    enabled: !!runKey,
  });
}

export function useRunArtifacts(runKey: string, params?: api.ListArtifactsParams) {
  return useQuery({
    queryKey: queryKeys.runArtifacts(runKey, params),
    queryFn: () => api.listRunArtifacts(runKey, params),
    enabled: !!runKey,
  });
}

// Artifacts
export function useArtifact(artifactId: string) {
  return useQuery({
    queryKey: queryKeys.artifact(artifactId),
    queryFn: () => api.getArtifact(artifactId),
    enabled: !!artifactId,
  });
}

// Comments
export function useComments(scopeType: string, scopeId: string) {
  return useQuery({
    queryKey: queryKeys.comments(scopeType, scopeId),
    queryFn: () => api.listComments({ scope_type: scopeType, scope_id: scopeId }),
    enabled: !!scopeType && !!scopeId,
  });
}

export function useCreateComment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.createComment,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.comments(variables.scope_type, variables.scope_id),
      });
    },
  });
}

export function useDeleteComment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.deleteComment,
    onSuccess: () => {
      // Invalidate all comment queries
      queryClient.invalidateQueries({ queryKey: ['comments'] });
    },
  });
}

// Reviews
export function useReviews(scopeType: string, scopeId: string) {
  return useQuery({
    queryKey: queryKeys.reviews(scopeType, scopeId),
    queryFn: () => api.listReviews({ scope_type: scopeType, scope_id: scopeId }),
    enabled: !!scopeType && !!scopeId,
  });
}

export function useCreateReview() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.createReview,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.reviews(variables.scope_type, variables.scope_id),
      });
    },
  });
}

export function useUpdateReview() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ reviewId, status, notes }: { reviewId: number; status: ReviewStatus; notes?: string }) =>
      api.updateReview(reviewId, status, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reviews'] });
    },
  });
}

// Reindex
export function useReindex() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.reindex,
    onSuccess: () => {
      // Invalidate all project-related queries
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['project'] });
      queryClient.invalidateQueries({ queryKey: ['health'] });
    },
  });
}
