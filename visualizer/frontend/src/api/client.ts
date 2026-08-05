import type { CreateJobRequest, EmbeddingRecordDTO, JobStatusResponse } from "../types";

// The FastAPI backend is started separately (see visualizer/run_visualizer.*) and listens
// on port 8000 by default. CORS is open on the backend, so this works from both the Vite
// dev server and the packaged Tauri webview.
export const API_BASE_URL = "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${response.status}): ${body}`);
  }
  return response.json() as Promise<T>;
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

export function getJobRecords(jobId: string, params: GetJobRecordsParams = {}): Promise<EmbeddingRecordDTO[]> {
  const search = new URLSearchParams();
  if (params.split) search.set("split", params.split);
  if (params.status) search.set("status", params.status);
  if (params.classId !== undefined) search.set("class_id", String(params.classId));
  search.set("limit", String(params.limit ?? 2000));
  search.set("offset", String(params.offset ?? 0));

  return request<EmbeddingRecordDTO[]>(`/api/v1/jobs/${jobId}/records?${search.toString()}`);
}

export function getRecordImageUrl(jobId: string, recordId: string): string {
  return `${API_BASE_URL}/api/v1/jobs/${jobId}/images/${encodeURIComponent(recordId)}`;
}
