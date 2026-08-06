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
  has_pca: boolean;
  pca_components: number | null;
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
  has_pca: boolean;
  pca_components: number | null;
  status: string | null;
}

export interface PcaStatusResponse {
  updated: number;
  components: number;
}
