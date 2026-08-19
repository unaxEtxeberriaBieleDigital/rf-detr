import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { getRecordImageUrl, startSemanticSearch } from "../api/client";
import type { ClassThresholds, EmbeddingRecordDTO } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";
import PanZoomViewport, { type PanZoomHandle } from "./PanZoomViewport";

interface ImageViewerModalProps {
  isOpen: boolean;
  jobId: string;
  imagePath: string | null;
  records: EmbeddingRecordDTO[];
  /** Global confidence threshold coming from the sidebar filters. */
  minConfidence: number;
  classThresholds?: ClassThresholds;
  /** Class id → human readable name, used to label GTs and defects in the side panel. */
  categories: Record<number, string>;
  /** Model info, forwarded to the "search similar" feature so it can re-run the model. */
  modelPath: string;
  modelType: string;
  /** Overrides the image URL instead of resolving it from `jobId`/records. Used when opening
   *  images that don't belong to this job's dataset (e.g. semantic-search neighbour results). */
  imageUrlOverride?: string;
  /** Whether predictions can be used to start a new "search similar" job. Disabled for images
   *  that aren't tracked in this job's store (e.g. neighbour results), since there's no
   *  record id to fetch a raw embedding for. */
  allowSearch?: boolean;
  /** Called right after a semantic search job is successfully created, so the caller can show
   *  its progress/results (e.g. in a panel next to the embedding plot). */
  onSearchStarted?: (searchId: string) => void;
  /** Called when the user navigates to the previous image (arrow button or ←). */
  onPrev?: () => void;
  /** Called when the user navigates to the next image (arrow button or →). */
  onNext?: () => void;
  onClose: () => void;
}

const DEFAULT_SEARCH_K = 20;

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
  classThresholds,
  categories,
  modelPath,
  modelType,
  imageUrlOverride,
  allowSearch = true,
  onSearchStarted,
  onPrev,
  onNext,
  onClose,
}: ImageViewerModalProps) {
  const panZoomRef = useRef<PanZoomHandle | null>(null);

  const [showGroundTruths, setShowGroundTruths] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);

  // Internal side-panel filter: by default it mirrors the gallery's global confidence filter,
  // but it can be decoupled to explore this single image at a different threshold.
  const [useGlobalFilters, setUseGlobalFilters] = useState(true);
  const [localMinConfidence, setLocalMinConfidence] = useState(minConfidence);
  const effectiveMinConfidence = useGlobalFilters ? minConfidence : localMinConfidence;

  // "Search similar" panel: opened by clicking a prediction in the defects list. Searches for
  // embeddings close to that single detection's embedding, over an arbitrary folder. Progress
  // and results are shown by the caller (next to the embedding plot), not in this modal.
  const [searchRecordId, setSearchRecordId] = useState<string | null>(null);
  const [searchFolder, setSearchFolder] = useState("");
  const [searchK, setSearchK] = useState(DEFAULT_SEARCH_K);
  const [searchSourceType, setSearchSourceType] = useState<"default" | "tiled">("default");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const searchRecord = useMemo(
    () => records.find((r) => r.id === searchRecordId) ?? null,
    [records, searchRecordId],
  );

  async function handleStartSearch(): Promise<void> {
    if (!searchRecordId || !searchFolder.trim() || starting) return;
    setStarting(true);
    setStartError(null);
    try {
      const created = await startSemanticSearch(jobId, {
        query_record_id: searchRecordId,
        search_path: searchFolder.trim(),
        k: searchK,
        model_path: modelPath,
        model_type: modelType,
        source_type: searchSourceType,
      });
      onSearchStarted?.(created.id);
      onClose();
    } catch (e) {
      setStartError(String(e instanceof Error ? e.message : e));
    } finally {
      setStarting(false);
    }
  }

  async function pickSearchFolder(): Promise<void> {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Selecciona la carpeta donde buscar",
      });
      if (typeof selected === "string" && selected.trim().length > 0) {
        setSearchFolder(selected);
      }
    } catch (e) {
      setStartError(`No se pudo abrir el selector de carpetas: ${String(e instanceof Error ? e.message : e)}`);
    }
  }

  const anyRecordId = records[0]?.id;
  const imageUrl = useMemo(
    () => imageUrlOverride ?? (anyRecordId ? getRecordImageUrl(jobId, anyRecordId) : null),
    [imageUrlOverride, jobId, anyRecordId],
  );
  const fileName = imagePath?.split(/[\\/]/).pop() ?? "";
  const split = records[0]?.split ?? "";

  // The same file name can exist in several splits with different (or no) annotations, so we
  // surface the split and the actual GT/prediction counts to make the overlay state unambiguous.
  const groundTruthRecords = useMemo(() => records.filter((r) => r.ground_truth?.bbox), [records]);
  const defectRecords = useMemo(
    () =>
      records.filter((r) => {
        if (!r.prediction?.bbox) return false;
        const threshold = useGlobalFilters
          ? classThresholds?.[r.prediction.class_id] ?? effectiveMinConfidence
          : effectiveMinConfidence;
        return r.prediction.confidence >= threshold;
      }),
    [records, effectiveMinConfidence, classThresholds, useGlobalFilters],
  );
  const gtCount = groundTruthRecords.length;
  const predCount = defectRecords.length;

  useEffect(() => {
    if (!isOpen) return;
    setShowGroundTruths(true);
    setShowPredictions(true);
    setUseGlobalFilters(true);
    setLocalMinConfidence(minConfidence);
    setSearchRecordId(null);
    setSearchFolder("");
    setSearchK(DEFAULT_SEARCH_K);
    setSearchSourceType("default");
    setStartError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, imagePath]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft") onPrev?.();
      if (event.key === "ArrowRight") onNext?.();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose, onPrev, onNext]);

  if (!isOpen || !imagePath || !imageUrl) return null;

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
            onClick={() => panZoomRef.current?.reset()}
          >
            Reset
          </button>
          <button type="button" className="image-viewer-btn" onClick={onClose}>
            Cerrar
          </button>
        </div>

        <div className="image-viewer-body">
          {onPrev && (
            <button
              type="button"
              className="iv-nav-arrow iv-nav-arrow-prev"
              onClick={onPrev}
              aria-label="Imagen anterior"
            >
              ‹
            </button>
          )}
          <PanZoomViewport ref={panZoomRef} resetKey={imagePath}>
            <ImageWithBoxes
              imageUrl={imageUrl}
              imagePath={imagePath}
              records={records}
              minConfidence={effectiveMinConfidence}
              classThresholds={useGlobalFilters ? classThresholds : undefined}
              showGroundTruths={showGroundTruths}
              showPredictions={showPredictions}
            />
          </PanZoomViewport>
          {onNext && (
            <button
              type="button"
              className="iv-nav-arrow iv-nav-arrow-next"
              onClick={onNext}
              aria-label="Imagen siguiente"
            >
              ›
            </button>
          )}

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
                  const selected = r.id === searchRecordId;
                  if (!allowSearch) {
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
                  }
                  return (
                    <li key={`pred-${r.id}`}>
                      <button
                        type="button"
                        className={`iv-list-item iv-list-item-btn ${selected ? "iv-list-item-selected" : ""}`}
                        title="Buscar detecciones similares a esta"
                        onClick={() =>
                          setSearchRecordId((current) => (current === r.id ? null : r.id))
                        }
                      >
                        <span className="iv-dot" style={{ background: color }} />
                        <span className="iv-list-label" title={label}>
                          {label}
                        </span>
                        <span className="iv-list-meta">{r.prediction!.confidence.toFixed(2)}</span>
                        <span className="iv-status-badge" style={{ color }}>
                          {STATUS_LABELS[r.status] ?? r.status}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>

            {allowSearch && searchRecord && (
              <div className="iv-sidebar-section ss-sidebar-section">
                <div className="iv-sidebar-title">Buscar similares</div>
                <p className="iv-empty">
                  Predicción:{" "}
                  {searchRecord.prediction?.class_id !== undefined
                    ? categories[searchRecord.prediction.class_id] ?? `Clase ${searchRecord.prediction.class_id}`
                    : "—"}{" "}
                  ({searchRecord.prediction?.confidence.toFixed(2)})
                </p>
                <label className="ss-field">
                  Carpeta donde buscar
                  <div className="ss-folder-row">
                    <input
                      type="text"
                      placeholder="C:\ruta\a\una\carpeta"
                      value={searchFolder}
                      onChange={(e) => setSearchFolder(e.currentTarget.value)}
                    />
                    <button type="button" className="image-viewer-btn" onClick={pickSearchFolder}>
                      Elegir...
                    </button>
                  </div>
                </label>
                <label className="ss-field">
                  Nº de vecinos más cercanos (k)
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={searchK}
                    onChange={(e) => setSearchK(Math.max(1, Number(e.currentTarget.value)))}
                  />
                </label>
                <label className="ss-field">
                  Tipo de búsqueda
                  <select
                    value={searchSourceType}
                    onChange={(e) => setSearchSourceType(e.currentTarget.value as "default" | "tiled")}
                  >
                    <option value="default">Default — una imagen completa por unidad</option>
                    <option value="tiled">Tiled — divide imágenes grandes en teselas</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="pca-btn"
                  disabled={!searchFolder.trim() || starting}
                  onClick={handleStartSearch}
                >
                  {starting ? "Iniciando..." : "Buscar"}
                </button>
                {startError && <p className="setup-error">{startError}</p>}
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
