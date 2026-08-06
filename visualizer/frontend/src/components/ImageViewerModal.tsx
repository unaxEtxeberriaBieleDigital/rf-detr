import { useEffect, useMemo, useRef, useState } from "react";
import { getRecordImageUrl } from "../api/client";
import type { EmbeddingRecordDTO } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";

interface ImageViewerModalProps {
  isOpen: boolean;
  jobId: string;
  imagePath: string | null;
  records: EmbeddingRecordDTO[];
  minConfidence: number;
  onClose: () => void;
}

/** Full-screen image viewer with wheel zoom, drag pan and GT/Prediction visibility toggles. */
export default function ImageViewerModal({
  isOpen,
  jobId,
  imagePath,
  records,
  minConfidence,
  onClose,
}: ImageViewerModalProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef({
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });

  const [showGroundTruths, setShowGroundTruths] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  const anyRecordId = records[0]?.id;
  const imageUrl = useMemo(
    () => (anyRecordId ? getRecordImageUrl(jobId, anyRecordId) : null),
    [jobId, anyRecordId],
  );
  const fileName = imagePath?.split(/[\\/]/).pop() ?? "";
  const split = records[0]?.split ?? "";

  // The same file name can exist in several splits with different (or no) annotations, so we
  // surface the split and the actual GT/prediction counts to make the overlay state unambiguous.
  const gtCount = useMemo(() => records.filter((r) => r.ground_truth?.bbox).length, [records]);
  const predCount = useMemo(
    () => records.filter((r) => r.prediction?.bbox && r.prediction.confidence >= minConfidence).length,
    [records, minConfidence],
  );

  useEffect(() => {
    if (!isOpen) return;
    setScale(1);
    setOffset({ x: 0, y: 0 });
    setShowGroundTruths(true);
    setShowPredictions(true);
  }, [isOpen, imagePath]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !imagePath || !imageUrl) return null;

  function clampScale(value: number): number {
    return Math.min(8, Math.max(1, value));
  }

  return (
    <div className="image-viewer-backdrop" onClick={onClose}>
      <div className="image-viewer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="image-viewer-toolbar">
          <span className="image-viewer-name" title={imagePath}>
            {split && <span className="image-viewer-split">{split}</span>}
            {fileName}
          </span>
          <label className="image-viewer-toggle">
            <input
              type="checkbox"
              checked={showGroundTruths}
              onChange={(e) => setShowGroundTruths(e.currentTarget.checked)}
            />
            GT ({gtCount})
          </label>
          <label className="image-viewer-toggle">
            <input
              type="checkbox"
              checked={showPredictions}
              onChange={(e) => setShowPredictions(e.currentTarget.checked)}
            />
            Pred ({predCount})
          </label>
          {gtCount === 0 && (
            <span
              className="image-viewer-warning"
              title={`Esta imagen no tiene anotaciones en el split "${split}". Puede existir un archivo con el mismo nombre en otro split que sí las tenga.`}
            >
              Sin GT en este split
            </span>
          )}
          <button
            type="button"
            className="image-viewer-btn"
            onClick={() => {
              setScale(1);
              setOffset({ x: 0, y: 0 });
            }}
          >
            Reset
          </button>
          <button type="button" className="image-viewer-btn" onClick={onClose}>
            Cerrar
          </button>
        </div>

        <div
          ref={viewportRef}
          className={`image-viewer-viewport ${isDragging ? "dragging" : ""}`}
          onWheel={(event) => {
            event.preventDefault();
            const viewport = viewportRef.current;
            if (!viewport) return;

            const rect = viewport.getBoundingClientRect();
            const pointerX = event.clientX - rect.left;
            const pointerY = event.clientY - rect.top;
            const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
            const nextScale = clampScale(scale * zoomFactor);
            if (nextScale === scale) return;

            const worldX = (pointerX - offset.x) / scale;
            const worldY = (pointerY - offset.y) / scale;
            setOffset({
              x: pointerX - worldX * nextScale,
              y: pointerY - worldY * nextScale,
            });
            setScale(nextScale);
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            const target = event.currentTarget;
            dragRef.current = {
              active: true,
              pointerId: event.pointerId,
              startX: event.clientX,
              startY: event.clientY,
              originX: offset.x,
              originY: offset.y,
            };
            target.setPointerCapture(event.pointerId);
            setIsDragging(true);
          }}
          onPointerMove={(event) => {
            if (!dragRef.current.active) return;
            const deltaX = event.clientX - dragRef.current.startX;
            const deltaY = event.clientY - dragRef.current.startY;
            setOffset({
              x: dragRef.current.originX + deltaX,
              y: dragRef.current.originY + deltaY,
            });
          }}
          onPointerUp={(event) => {
            if (dragRef.current.active && event.currentTarget.hasPointerCapture(dragRef.current.pointerId)) {
              event.currentTarget.releasePointerCapture(dragRef.current.pointerId);
            }
            dragRef.current.active = false;
            setIsDragging(false);
          }}
          onPointerLeave={() => {
            if (!dragRef.current.active) return;
            dragRef.current.active = false;
            setIsDragging(false);
          }}
        >
          <div
            className="image-viewer-stage"
            style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }}
          >
            <ImageWithBoxes
              imageUrl={imageUrl}
              imagePath={imagePath}
              records={records}
              minConfidence={minConfidence}
              showGroundTruths={showGroundTruths}
              showPredictions={showPredictions}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
