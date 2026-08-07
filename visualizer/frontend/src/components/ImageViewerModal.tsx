import { useEffect, useMemo, useRef, useState } from "react";
import { getRecordImageUrl } from "../api/client";
import type { EmbeddingRecordDTO } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";

interface ImageViewerModalProps {
  isOpen: boolean;
  jobId: string;
  imagePath: string | null;
  records: EmbeddingRecordDTO[];
  /** Global confidence threshold coming from the sidebar filters. */
  minConfidence: number;
  /** Class id → human readable name, used to label GTs and defects in the side panel. */
  categories: Record<number, string>;
  onClose: () => void;
}

const GT_COLOR = "#1565c0";

const STATUS_LABELS: Record<string, string> = {
  tp: "TP",
  fp: "FP",
  fn: "FN",
  misclassified: "Misclas.",
  correct: "Correcto",
  incorrect: "Incorrecto",
};

const STATUS_COLORS: Record<string, string> = {
  tp: "#2e7d32",
  correct: "#2e7d32",
  fp: "#c62828",
  incorrect: "#c62828",
  fn: "#1565c0",
  misclassified: "#6a1b9a",
};

/** Full-screen image viewer with wheel zoom, drag pan and GT/Prediction visibility toggles. */
export default function ImageViewerModal({
  isOpen,
  jobId,
  imagePath,
  records,
  minConfidence,
  categories,
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

  // Internal side-panel filter: by default it mirrors the gallery's global confidence filter,
  // but it can be decoupled to explore this single image at a different threshold.
  const [useGlobalFilters, setUseGlobalFilters] = useState(true);
  const [localMinConfidence, setLocalMinConfidence] = useState(minConfidence);
  const effectiveMinConfidence = useGlobalFilters ? minConfidence : localMinConfidence;

  const anyRecordId = records[0]?.id;
  const imageUrl = useMemo(
    () => (anyRecordId ? getRecordImageUrl(jobId, anyRecordId) : null),
    [jobId, anyRecordId],
  );
  const fileName = imagePath?.split(/[\\/]/).pop() ?? "";
  const split = records[0]?.split ?? "";

  // The same file name can exist in several splits with different (or no) annotations, so we
  // surface the split and the actual GT/prediction counts to make the overlay state unambiguous.
  const groundTruthRecords = useMemo(() => records.filter((r) => r.ground_truth?.bbox), [records]);
  const defectRecords = useMemo(
    () =>
      records.filter((r) => r.prediction?.bbox && r.prediction.confidence >= effectiveMinConfidence),
    [records, effectiveMinConfidence],
  );
  const gtCount = groundTruthRecords.length;
  const predCount = defectRecords.length;

  useEffect(() => {
    if (!isOpen) return;
    setScale(1);
    setOffset({ x: 0, y: 0 });
    setShowGroundTruths(true);
    setShowPredictions(true);
    setUseGlobalFilters(true);
    setLocalMinConfidence(minConfidence);
  // eslint-disable-next-line react-hooks/exhaustive-deps
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

        <div className="image-viewer-body">
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
              minConfidence={effectiveMinConfidence}
              showGroundTruths={showGroundTruths}
              showPredictions={showPredictions}
            />
          </div>
        </div>

        <aside className="iv-sidebar">
          <div className="iv-sidebar-section">
            <div className="iv-sidebar-title">Filtro interno</div>
            <label className="iv-check-row">
              <input
                type="checkbox"
                checked={useGlobalFilters}
                onChange={(e) => setUseGlobalFilters(e.currentTarget.checked)}
              />
              Usar filtros actuales ({minConfidence.toFixed(2)})
            </label>
            <div className="iv-conf-row">
              <label htmlFor="iv-local-conf" className="iv-conf-label">
                Confianza mínima: {effectiveMinConfidence.toFixed(2)}
              </label>
              <input
                id="iv-local-conf"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={effectiveMinConfidence}
                disabled={useGlobalFilters}
                onChange={(e) => setLocalMinConfidence(Number(e.currentTarget.value))}
                className="iv-slider"
              />
            </div>
          </div>

          <div className="iv-sidebar-section">
            <div className="iv-sidebar-title">Ground Truth ({gtCount})</div>
            {gtCount === 0 && <p className="iv-empty">Sin anotaciones GT.</p>}
            <ul className="iv-list">
              {groundTruthRecords.map((r, i) => {
                const classId = r.ground_truth?.class_id;
                const label = classId !== undefined ? categories[classId] ?? `Clase ${classId}` : "—";
                return (
                  <li key={`gt-${r.id}-${i}`} className="iv-list-item">
                    <span className="iv-dot" style={{ background: GT_COLOR }} />
                    <span className="iv-list-label" title={label}>
                      {label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="iv-sidebar-section">
            <div className="iv-sidebar-title">Defectos ({predCount})</div>
            {predCount === 0 && <p className="iv-empty">Sin predicciones por encima del umbral.</p>}
            <ul className="iv-list">
              {defectRecords.map((r) => {
                const classId = r.prediction?.class_id;
                const label = classId !== undefined ? categories[classId] ?? `Clase ${classId}` : "—";
                const color = STATUS_COLORS[r.status] ?? "#ef6c00";
                return (
                  <li key={`pred-${r.id}`} className="iv-list-item">
                    <span className="iv-dot" style={{ background: color }} />
                    <span className="iv-list-label" title={label}>
                      {label}
                    </span>
                    <span className="iv-list-meta">{r.prediction!.confidence.toFixed(2)}</span>
                    <span className="iv-status-badge" style={{ color }}>
                      {STATUS_LABELS[r.status] ?? r.status}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        </aside>
        </div>
      </div>
    </div>
  );
}
