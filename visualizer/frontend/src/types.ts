export interface PredictionDTO {
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number] | null;
}

export type RecordStatus = "tp" | "fp" | "fn" | "misclassified" | "correct" | "incorrect";

export interface EmbeddingRecordDTO {
  id: string;
  image_path: string;
  split: string;
  embedding: number[] | null;
  prediction: PredictionDTO | null;
  ground_truth: PredictionDTO | null;
  status: RecordStatus;
}

export interface JobStatusResponse {
  id: string;
  status: "pending" | "running" | "done" | "error";
  error: string | null;
  num_records: number;
  categories: Record<number, string>;
  num_images_total: number;
  num_images_processed: number;
  has_dimensionality_reduction: boolean;
  dimensionality_reduction_components: number | null;
}

export interface ImagePathPageResponse {
  image_paths: string[];
  total_images: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface CreateJobRequest {
  dataset_path: string;
  dataset_type: string;
  model_path: string;
  model_type: string;
  splits?: string[] | null;
  batch_size?: number;
  iou_threshold?: number;
}

export interface CheckDatasetResponse {
  has_db: boolean;
  num_records: number;
  has_dimensionality_reduction: boolean;
  dimensionality_reduction_components: number | null;
  status: string | null;
}

export interface DimensionalityReductionStatusResponse {
  updated: number;
  components: number;
}

export interface SemanticSearchResultDTO {
  image_path: string;
  bbox: [number, number, number, number] | null;
  confidence: number;
  class_id: number;
  distance: number;
  /** Preview of the matching detection, already rendered/cropped in-memory by the backend and
   *  base64-encoded as a data URL (e.g. "data:image/jpeg;base64,..."), ready to use as an
   *  <img> src with no extra HTTP request. */
  preview_data_url: string;
}

export interface SemanticSearchStatusResponse {
  id: string;
  parent_job_id: string;
  query_record_id: string;
  query_image_path: string;
  search_path: string;
  k: number;
  status: "pending" | "running" | "done" | "error";
  error: string | null;
  num_images_total: number;
  num_images_processed: number;
  results: SemanticSearchResultDTO[] | null;
}
