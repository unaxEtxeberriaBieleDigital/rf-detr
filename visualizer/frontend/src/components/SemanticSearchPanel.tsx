import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { getSemanticSearch } from "../api/client";
import type { SemanticSearchResultDTO, SemanticSearchStatusResponse } from "../types";
import ImageWithBoxes from "./ImageWithBoxes";

interface SemanticSearchPanelProps {
  jobId: string;
  searchId: string;
  categories: Record<number, string>;
  onOpenResult: (result: SemanticSearchResultDTO, imageUrl: string) => void;
}

const POLL_INTERVAL_MS = 1500;
const MIN_TILE_SIZE = 90;
const MAX_TILE_SIZE = 320;
const DEFAULT_TILE_SIZE = 160;

export default function SemanticSearchPanel({
  jobId,
  searchId,
  categories,
  onOpenResult,
}: SemanticSearchPanelProps) {
  const [status, setStatus] = useState<SemanticSearchStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tileSize, setTileSize] = useState(DEFAULT_TILE_SIZE);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus(null);
    setError(null);

    function poll(): void {
      getSemanticSearch(jobId, searchId)
        .then((updated) => {
          if (cancelled) return;
          setStatus(updated);
        })
        .catch((e) => {
          if (!cancelled) setError(String(e instanceof Error ? e.message : e));
        });
    }

    poll();
    pollRef.current = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobId, searchId]);

  useEffect(() => {
    if (!status) return;
    if (status.status === "done" || status.status === "error") {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }
  }, [status]);

  const progressPct = useMemo(() => {
    if (!status || status.num_images_total === 0) return 0;
    return Math.round((status.num_images_processed / status.num_images_total) * 100);
  }, [status]);

  return (
    <section className="visualizer-search-panel">

      {error && <p className="setup-error">{error}</p>}

      {!error && !status && 
        <>
          <p className="image-gallery-loading">Cargando...</p>
          <div className="loader" />
        </>
      }

      {status && (status.status === "pending" || status.status === "running") && (
        <div className="search-panel-progress">
          <div className="ss-progress-bar">
            <div className="ss-progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <p className="ss-progress-label">
            {status.num_images_processed.toLocaleString()} / {status.num_images_total.toLocaleString()}{" "}
            imágenes escaneadas
          </p>
        </div>
      )}

      {status && status.status === "error" && (
        <p className="setup-error">{status.error ?? "Error desconocido"}</p>
      )}

      {status && status.status === "done" && (
        <>
          <p className="search-panel-summary">
            {status.results?.length ?? 0} vecino(s) encontrado(s)
          </p>
          <div className="search-panel-toolbar">
            <label htmlFor="ss-zoom-slider" className="search-panel-zoom-label">
              Tamaño: {tileSize}px
            </label>
            <input
              id="ss-zoom-slider"
              type="range"
              min={MIN_TILE_SIZE}
              max={MAX_TILE_SIZE}
              step={10}
              value={tileSize}
              onChange={(e) => setTileSize(Number(e.currentTarget.value))}
              className="search-panel-zoom-slider"
            />
          </div>
          <div
            className="search-panel-results"
            style={{ "--ss-tile-size": `${tileSize}px` } as CSSProperties}
          >
            {(status.results ?? []).map((r, i) => {
              const label = categories[r.class_id] ?? `Clase ${r.class_id}`;
              const imageUrl = r.preview_data_url;
              return (
                <div
                  key={i}
                  className="ss-result-card"
                  title={r.image_path}
                  onClick={() => onOpenResult(r, imageUrl)}
                >
                  <div className="ss-result-frame">
                    <ImageWithBoxes
                      imageUrl={imageUrl}
                      imagePath={r.image_path}
                      records={[
                        {
                          id: `${searchId}-${i}`,
                          image_path: r.image_path,
                          split: "",
                          embedding: null,
                          prediction: { class_id: r.class_id, confidence: r.confidence, bbox: r.bbox },
                          ground_truth: null,
                          status: "tp",
                        },
                      ]}
                      minConfidence={0}
                      showGroundTruths={false}
                      showPredictions
                    />
                  </div>
                  <div className="ss-result-meta">
                    <span>{label}</span>
                    <span>dist. {r.distance.toFixed(3)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}