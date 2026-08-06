import { useMemo, useState } from "react";
import type { EmbeddingRecordDTO } from "../types";

const GT_COLOR = "#1565c0";
const STATUS_COLOR: Record<string, string> = {
  tp: "#2e7d32",
  correct: "#2e7d32",
  fp: "#c62828",
  incorrect: "#c62828",
  misclassified: "#6a1b9a",
};

interface ImageWithBoxesProps {
  imageUrl: string;
  imagePath: string;
  records: EmbeddingRecordDTO[];
  minConfidence?: number;
  showGroundTruths?: boolean;
  showPredictions?: boolean;
}

/** Draws GT and prediction bounding boxes on top of an image using the image natural coordinates. */
export default function ImageWithBoxes({
  imageUrl,
  imagePath,
  records,
  minConfidence = 0,
  showGroundTruths = true,
  showPredictions = true,
}: ImageWithBoxesProps) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  const visiblePredictions = useMemo(
    () => records.filter((r) => r.prediction?.bbox && r.prediction.confidence >= minConfidence),
    [records, minConfidence],
  );

  const groundTruths = useMemo(() => records.filter((record) => record.ground_truth?.bbox), [records]);

  return (
    <div className="image-with-boxes">
      <img
        src={imageUrl}
        alt={imagePath}
        draggable={false}
        onLoad={(e) => {
          const img = e.currentTarget;
          setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
        }}
      />
      {naturalSize && (
        <svg
          className="image-with-boxes-overlay"
          viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {showPredictions &&
            visiblePredictions.map((r) => {
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
          {showGroundTruths &&
            groundTruths.map((r, i) => {
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
                  strokeWidth={Math.max(naturalSize.width, naturalSize.height) / 220}
                  strokeDasharray="6 4"
                />
              );
            })}
        </svg>
      )}
    </div>
  );
}
