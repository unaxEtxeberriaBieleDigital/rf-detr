import type {
  CancelSemanticSearchResponse,
  CheckDatasetResponse,
  CreateJobRequest,
  DimensionalityReductionStatusResponse,
  EmbeddingRecordDTO,
  EvaluationMetricsResponse,
  ImagePathPageResponse,
  JobStatusResponse,
  OptimalThresholdResponse,
  SemanticSearchStatusResponse,
} from "../types";

// During development the FastAPI backend is started separately (see visualizer/run_visualizer.*).
// Production Tauri builds start the packaged backend on this loopback-only endpoint.
export const API_BASE_URL = "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${response.status}): ${body}`);
  }
  return response.json() as Promise<T>;
}

/**
 * Check whether the backend is already accepting requests.
 *
 * The packaged backend needs several seconds to import torch and the model
 * registry before uvicorn starts serving, so the frontend polls this endpoint
 * before showing the setup form.
 *
 * @param timeoutMs Maximum time to wait for a single health probe.
 * @returns True when the backend answered the health endpoint successfully.
 */
export async function checkBackendHealth(timeoutMs = 2000): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeoutId);
  }
}

export function getModelTypes(): Promise<string[]> {
  return request<string[]>("/api/v1/model-types");
}

export function getDatasetTypes(): Promise<string[]> {
  return request<string[]>("/api/v1/dataset-types");
}

export function createJob(payload: CreateJobRequest): Promise<JobStatusResponse> {
  return request<JobStatusResponse>("/api/v1/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getJob(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/v1/jobs/${jobId}`);
}

export interface GetJobRecordsParams {
  split?: string;
  status?: string;
  classId?: number;
  limit?: number;
  offset?: number;
}

export function getJobRecords(jobId: string, params: GetJobRecordsParams = {}, signal?: AbortSignal): Promise<EmbeddingRecordDTO[]> {
  const search = new URLSearchParams();
  if (params.split) search.set("split", params.split);
  if (params.status) search.set("status", params.status);
  if (params.classId !== undefined) search.set("class_id", String(params.classId));
  search.set("limit", String(params.limit ?? 2000));
  search.set("offset", String(params.offset ?? 0));

  return request<EmbeddingRecordDTO[]>(`/api/v1/jobs/${jobId}/records?${search.toString()}`, { signal });
}

export interface GetJobImagePathsParams {
  split?: string;
  limit?: number;
  offset?: number;
}

export function getJobImagePaths(jobId: string, params: GetJobImagePathsParams = {}): Promise<ImagePathPageResponse> {
  const search = new URLSearchParams();
  if (params.split) search.set("split", params.split);
  search.set("limit", String(params.limit ?? 60));
  search.set("offset", String(params.offset ?? 0));
  return request<ImagePathPageResponse>(`/api/v1/jobs/${jobId}/image-paths?${search.toString()}`);
}

export interface RecordsByImagePathsRequest {
  image_paths: string[];
  split?: string;
}

export function getJobRecordsByImagePaths(
  jobId: string,
  payload: RecordsByImagePathsRequest,
): Promise<EmbeddingRecordDTO[]> {
  return request<EmbeddingRecordDTO[]>(`/api/v1/jobs/${jobId}/records/by-image-paths`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getRecordImageUrl(jobId: string, recordId: string): string {
  return `${API_BASE_URL}/api/v1/jobs/${jobId}/images/${encodeURIComponent(recordId)}`;
}

export function checkDataset(datasetPath: string): Promise<CheckDatasetResponse> {
  const search = new URLSearchParams({ path: datasetPath });
  return request<CheckDatasetResponse>(`/api/v1/check-dataset?${search.toString()}`);
}

export function loadJob(datasetPath: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>("/api/v1/jobs/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_path: datasetPath }),
  });
}

export type ReductionAlgorithm = "pca" | "tsne" | "umap";

export function computeReduction(
  jobId: string,
  components: 2 | 3,
  algorithm: ReductionAlgorithm,
  recordIds?: string[],
  params?: { perplexity?: number; n_neighbors?: number; min_dist?: number },
): Promise<DimensionalityReductionStatusResponse> {
  return request<DimensionalityReductionStatusResponse>(
    `/api/v1/jobs/${jobId}/dimensionality_reduction?components=${components}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        record_ids: recordIds ?? null,
        algorithm,
        perplexity: params?.perplexity ?? 30.0,
        n_neighbors: params?.n_neighbors ?? 15,
        min_dist: params?.min_dist ?? 0.1,
      }),
    },
  );
}

// -----------------------------------------------------------------------
// Semantic search
// -----------------------------------------------------------------------

export interface StartSemanticSearchRequest {
  query_record_id: string;
  search_path: string;
  k: number;
  model_path: string;
  model_type: string;
  source_type?: string;
}

export function startSemanticSearch(
  jobId: string,
  payload: StartSemanticSearchRequest,
): Promise<SemanticSearchStatusResponse> {
  return request<SemanticSearchStatusResponse>(`/api/v1/jobs/${jobId}/semantic-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function listSemanticSearches(jobId: string): Promise<SemanticSearchStatusResponse[]> {
  return request<SemanticSearchStatusResponse[]>(`/api/v1/jobs/${jobId}/semantic-search`);
}

export function getSemanticSearch(jobId: string, searchId: string): Promise<SemanticSearchStatusResponse> {
  return request<SemanticSearchStatusResponse>(`/api/v1/jobs/${jobId}/semantic-search/${searchId}`);
}

export function cancelSemanticSearch(jobId: string, searchId: string): Promise<CancelSemanticSearchResponse> {
  return request<CancelSemanticSearchResponse>(
    `/api/v1/jobs/${jobId}/semantic-search/${searchId}`,
    {
      method: "DELETE",
    }
  );
}

export function getJobEvaluation(
  jobId: string,
  classThresholds?: Record<number, number>,
  recordIds?: string[],
): Promise<EvaluationMetricsResponse> {
  if (recordIds) {
    return request<EvaluationMetricsResponse>(`/api/v1/jobs/${jobId}/evaluation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_thresholds: classThresholds ?? null,
        record_ids: recordIds,
      }),
    });
  }
  const search = new URLSearchParams();
  if (classThresholds) search.set("class_thresholds", JSON.stringify(classThresholds));
  if (recordIds) search.set("record_ids", JSON.stringify(recordIds));
  const suffix = search.toString() ? `?${search.toString()}` : "";
  return request<EvaluationMetricsResponse>(`/api/v1/jobs/${jobId}/evaluation${suffix}`);
}

export function getJobOptimalThreshold(
  jobId: string,
  metricName: string,
  numThresholds = 100,
  classId?: number,
  signal?: AbortSignal,
): Promise<OptimalThresholdResponse> {
  const search = new URLSearchParams({
    metric: metricName,
    num_thresholds: String(numThresholds),
  });
  if (classId !== undefined) search.set("class_id", String(classId));
  return request<OptimalThresholdResponse>(
    `/api/v1/jobs/${jobId}/optimal-threshold?${search.toString()}`,
    { signal }
  );
}
