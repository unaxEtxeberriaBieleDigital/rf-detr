import { useMemo } from "react";
import ImageCard from "./ImageCard";
import type { EmbeddingRecordDTO } from "../types";

interface ImageGalleryProps {
  jobId: string;
  records: EmbeddingRecordDTO[];
  minConfidence: number;
  categories: Record<number, string>;
  selectedImagePath: string | null;
  onSelectImage: (imagePath: string) => void;
}

/** Groups the flat list of per-detection records by image and renders one card per image. */
export default function ImageGallery({
  jobId,
  records,
  minConfidence,
  categories,
  selectedImagePath,
  onSelectImage,
}: ImageGalleryProps) {
  const recordsByImage = useMemo(() => {
    const grouped = new Map<string, EmbeddingRecordDTO[]>();
    for (const record of records) {
      const bucket = grouped.get(record.image_path) ?? [];
      bucket.push(record);
      grouped.set(record.image_path, bucket);
    }
    return grouped;
  }, [records]);

  return (
    <div className="image-gallery">
      {Array.from(recordsByImage.entries()).map(([imagePath, imageRecords]) => (
        <ImageCard
          key={imagePath}
          jobId={jobId}
          imagePath={imagePath}
          records={imageRecords}
          minConfidence={minConfidence}
          categories={categories}
          isSelected={imagePath === selectedImagePath}
          onSelect={() => onSelectImage(imagePath)}
        />
      ))}
      {recordsByImage.size === 0 && <p className="image-gallery-empty">No hay imágenes que coincidan con los filtros.</p>}
    </div>
  );
}
