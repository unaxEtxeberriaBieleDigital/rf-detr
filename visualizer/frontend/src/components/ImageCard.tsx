import { getRecordImageUrl } from "../api/client";
import type { EmbeddingRecordDTO } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";

interface ImageCardProps {
  jobId: string;
  imagePath: string;
  records: EmbeddingRecordDTO[];
  minConfidence: number;
  isSelected: boolean;
  onOpen: () => void;
}

/** Shows one dataset image tile with GT/prediction overlays. */
export default function ImageCard({
  jobId,
  imagePath,
  records,
  minConfidence,
  isSelected,
  onOpen,
}: ImageCardProps) {
  const anyRecordId = records[0]?.id;
  const imageUrl = anyRecordId ? getRecordImageUrl(jobId, anyRecordId) : undefined;
  const fileName = imagePath.split(/[\\/]/).pop() ?? imagePath;
  // The same file name can appear in several splits, so the split is part of the tile identity.
  const split = records[0]?.split ?? "";
  const gtCount = records.filter((r) => r.ground_truth?.bbox).length;

  return (
    <div
      className={`image-tile ${isSelected ? "image-tile-selected" : ""}`}
      onClick={onOpen}
      title={`${imagePath}\nGT: ${gtCount} · records: ${records.length}`}
    >
      <div className="image-tile-frame">
        {imageUrl && (
          <ImageWithBoxes
            imageUrl={imageUrl}
            imagePath={imagePath}
            records={records}
            minConfidence={minConfidence}
          />
        )}
        <span className="image-tile-name">
          {split && <span className="image-tile-split">{split}</span>}
          {fileName}
        </span>
      </div>
    </div>
  );
}
