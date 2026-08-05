import { useState } from "react";
import { getRecordImageUrl } from "../api/client";
import type { EmbeddingRecordDTO } from "../types";

interface ImageCardProps {
  jobId: string;
  imagePath: string;
  records: EmbeddingRecordDTO[];
  minConfidence: number;
  categories: Record<number, string>;
  isSelected: boolean;
  onSelect: () => void;
}

const GT_COLOR = "#1565c0";
const STATUS_COLOR: Record<string, string> = {
  tp: "#2e7d32",
  correct: "#2e7d32",
  fp: "#c62828",
  incorrect: "#c62828",
  misclassified: "#6a1b9a",
};

/** Shows one dataset image with its ground-truth boxes and (confidence-filtered) predictions overlaid. */
export default function ImageCard({
  jobId,
  imagePath,
  records,
  minConfidence,
  categories,
  isSelected,
  onSelect,
}: ImageCardProps) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  const anyRecordId = records[0]?.id;
  const imageUrl = anyRecordId ? getRecordImageUrl(jobId, anyRecordId) : undefined;

  const visiblePredictions = records.filter(
    (r) => r.prediction?.bbox && r.prediction.confidence >= minConfidence,
  );
  const groundTruths = records.filter(
    (r, index, all) => r.ground_truth?.bbox && all.findIndex((o) => o.ground_truth === r.ground_truth) === index,
  );

  function classLabel(classId: number | undefined): string {
    if (classId === undefined) return "?";
    return categories[classId] ?? `clase ${classId}`;
  }

  return (
    <div
      className={`image-card ${isSelected ? "image-card-selected" : ""}`}
      onClick={onSelect}
      title={imagePath}
    >
      <div className="image-card-frame">
        {imageUrl && (
          <img
            src={imageUrl}
            alt={imagePath}
            onLoad={(e) => {
              const img = e.currentTarget;
              setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />
        )}
        {naturalSize && (
          <svg
            className="image-card-overlay"
            viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
            preserveAspectRatio="none"
          >
            {groundTruths.map((r, i) => {
              const [x1, y1, x2, y2] = r.ground_truth!.bbox!;
              return (
                <rect
                  key={`gt-${i}`}
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke={GT_COLOR}
                  strokeWidth={Math.max(naturalSize.width, naturalSize.height) / 250}
                  strokeDasharray="6 4"
                />
              );
            })}
            {visiblePredictions.map((r) => {
              const [x1, y1, x2, y2] = r.prediction!.bbox!;
              const color = STATUS_COLOR[r.status] ?? "#ef6c00";
              return (
                <rect
                  key={r.id}
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke={color}
                  strokeWidth={Math.max(naturalSize.width, naturalSize.height) / 250}
                />
              );
            })}
          </svg>
        )}
      </div>
      <div className="image-card-caption">
        <span className="image-card-name">{imagePath.split(/[\\/]/).pop()}</span>
        <span className="image-card-preds">
          {visiblePredictions.map((r) => `${classLabel(r.prediction!.class_id)} (${r.prediction!.confidence.toFixed(2)})`).join(", ") ||
            "sin predicciones"}
        </span>
      </div>
    </div>
  );
}
