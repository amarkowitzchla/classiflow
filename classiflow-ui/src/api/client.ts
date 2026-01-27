// API client for Classiflow backend

import type {
  ProjectCard,
  ProjectDashboard,
  RunDetail,
  Artifact,
  Comment,
  Review,
  PaginatedResponse,
  HealthResponse,
  RunBrief,
  ReviewStatus,
} from '../types/api';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Health
export async function getHealth(): Promise<HealthResponse> {
  return fetchJson(`${API_BASE}/health`);
}

// Projects
export interface ListProjectsParams {
  q?: string;
  mode?: string;
  owner?: string;
  updated_after?: string;
  page?: number;
  page_size?: number;
}

export async function listProjects(params: ListProjectsParams = {}): Promise<PaginatedResponse<ProjectCard>> {
  const searchParams = new URLSearchParams();
  if (params.q) searchParams.set('q', params.q);
  if (params.mode) searchParams.set('mode', params.mode);
  if (params.owner) searchParams.set('owner', params.owner);
  if (params.updated_after) searchParams.set('updated_after', params.updated_after);
  if (params.page) searchParams.set('page', params.page.toString());
  if (params.page_size) searchParams.set('page_size', params.page_size.toString());

  const url = `${API_BASE}/projects${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
  return fetchJson(url);
}

export async function getProject(projectId: string): Promise<ProjectDashboard> {
  return fetchJson(`${API_BASE}/projects/${encodeURIComponent(projectId)}`);
}

export async function getProjectRuns(projectId: string): Promise<Record<string, RunBrief[]>> {
  return fetchJson(`${API_BASE}/projects/${encodeURIComponent(projectId)}/runs`);
}

// Runs
export async function getRun(runKey: string): Promise<RunDetail> {
  return fetchJson(`${API_BASE}/runs/${encodeURIComponent(runKey)}`);
}

export interface ListArtifactsParams {
  kind?: string;
  page?: number;
  page_size?: number;
}

export async function listRunArtifacts(
  runKey: string,
  params: ListArtifactsParams = {}
): Promise<PaginatedResponse<Artifact>> {
  const searchParams = new URLSearchParams();
  if (params.kind) searchParams.set('kind', params.kind);
  if (params.page) searchParams.set('page', params.page.toString());
  if (params.page_size) searchParams.set('page_size', params.page_size.toString());

  const url = `${API_BASE}/runs/${encodeURIComponent(runKey)}/artifacts${
    searchParams.toString() ? '?' + searchParams.toString() : ''
  }`;
  return fetchJson(url);
}

// Artifacts
export async function getArtifact(artifactId: string): Promise<Artifact> {
  return fetchJson(`${API_BASE}/artifacts/${encodeURIComponent(artifactId)}`);
}

export function getArtifactContentUrl(artifactId: string, download = false): string {
  return `${API_BASE}/artifacts/${encodeURIComponent(artifactId)}/content${download ? '?download=true' : ''}`;
}

// Comments
export interface ListCommentsParams {
  scope_type: string;
  scope_id: string;
  page?: number;
  page_size?: number;
}

export async function listComments(params: ListCommentsParams): Promise<PaginatedResponse<Comment>> {
  const searchParams = new URLSearchParams();
  searchParams.set('scope_type', params.scope_type);
  searchParams.set('scope_id', params.scope_id);
  if (params.page) searchParams.set('page', params.page.toString());
  if (params.page_size) searchParams.set('page_size', params.page_size.toString());

  return fetchJson(`${API_BASE}/comments?${searchParams.toString()}`);
}

export interface CreateCommentData {
  scope_type: string;
  scope_id: string;
  author: string;
  content: string;
}

export async function createComment(data: CreateCommentData): Promise<Comment> {
  return fetchJson(`${API_BASE}/comments`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteComment(commentId: number): Promise<void> {
  await fetchJson(`${API_BASE}/comments/${commentId}`, { method: 'DELETE' });
}

// Reviews
export interface ListReviewsParams {
  scope_type: string;
  scope_id: string;
  page?: number;
  page_size?: number;
}

export async function listReviews(params: ListReviewsParams): Promise<PaginatedResponse<Review>> {
  const searchParams = new URLSearchParams();
  searchParams.set('scope_type', params.scope_type);
  searchParams.set('scope_id', params.scope_id);
  if (params.page) searchParams.set('page', params.page.toString());
  if (params.page_size) searchParams.set('page_size', params.page_size.toString());

  return fetchJson(`${API_BASE}/reviews?${searchParams.toString()}`);
}

export interface CreateReviewData {
  scope_type: string;
  scope_id: string;
  reviewer: string;
  status: ReviewStatus;
  notes?: string;
}

export async function createReview(data: CreateReviewData): Promise<Review> {
  return fetchJson(`${API_BASE}/reviews`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateReview(
  reviewId: number,
  status: ReviewStatus,
  notes?: string
): Promise<Review> {
  const searchParams = new URLSearchParams();
  searchParams.set('status', status);
  if (notes) searchParams.set('notes', notes);

  return fetchJson(`${API_BASE}/reviews/${reviewId}?${searchParams.toString()}`, {
    method: 'PATCH',
  });
}

// Reindex
export async function reindex(): Promise<{ status: string }> {
  return fetchJson(`${API_BASE}/reindex`, { method: 'POST' });
}

// Plot data
import type { PlotCurve, PlotKeyType } from '../types/plots';

export async function getPlotData(runKey: string, plotKey: PlotKeyType): Promise<PlotCurve> {
  return fetchJson(`${API_BASE}/runs/${encodeURIComponent(runKey)}/plots/${plotKey}`);
}

export function getPlotDataUrl(runKey: string, plotKey: PlotKeyType): string {
  return `${API_BASE}/runs/${encodeURIComponent(runKey)}/plots/${plotKey}`;
}
