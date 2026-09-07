import type { ReactNode } from "react";
import type { ClassThresholds, EmbeddingRecordDTO } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";

interface ImageTileProps {
  /** Resolved image source. When omitted the tile renders an empty frame. */
  imageUrl?: string;
  imagePath: string;
  records: EmbeddingRecordDTO[];
  minConfidence?: number;
  classThresholds?: ClassThresholds;
  showGroundTruths?: boolean;
  showPredictions?: boolean;
  isSelected?: boolean;
  /** Native tooltip shown on the whole tile. */
  title?: string;
  /** Overlay content revealed when the tile is hovered. */
  caption?: ReactNode;
  /** Extra class applied to the caption overlay, e.g. to lay it out as a split row. */
  captionClassName?: string;
  onOpen?: () => void;
}

/** Auto-sized image tile with bounding-box overlays and a hover-revealed caption.
 *
 *  The tile has no intrinsic size: it fills its grid cell and derives its height from the
 *  image aspect ratio, so callers only need to define the surrounding grid template.
 */
export default function ImageTile({
  imageUrl,
  imagePath,
  records,
  minConfidence = 0,
  classThresholds,
  showGroundTruths = true,
  showPredictions = true,
  isSelected = false,
  title,
  caption,
  captionClassName,
  onOpen,
}: ImageTileProps) {
  return (
    <div
      className={`image-tile ${isSelected ? "image-tile-selected" : ""}`}
      onClick={onOpen}
      title={title}
    >
      <div className="image-tile-frame">
        {imageUrl && (
          <ImageWithBoxes
            imageUrl={imageUrl}
            imagePath={imagePath}
            records={records}
            minConfidence={minConfidence}
            classThresholds={classThresholds}
            showGroundTruths={showGroundTruths}
            showPredictions={showPredictions}
          />
        )}
        {caption !== undefined && (
          <span className={`image-tile-name ${captionClassName ?? ""}`}>{caption}</span>
        )}
      </div>
    </div>
  );
}
