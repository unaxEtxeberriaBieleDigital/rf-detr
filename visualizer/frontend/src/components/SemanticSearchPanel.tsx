import { useEffect, useMemo, useRef, useState } from "react";
import {
  cancelSemanticSearch,
  getJobRecordsByImagePaths,
  getRecordImageUrl,
  getSemanticSearch,
} from "../api/client";
import type {
  EmbeddingRecordDTO,
  SemanticSearchResultDTO,
  SemanticSearchStatusResponse,
} from "../types";
import ImageTile from "./ImageTile";
import ImageWithBoxes from "./ImageWithBoxes";
import LoadingDiv from "./LoadingDiv";
import { Trans, useTranslation } from "react-i18next";

interface SemanticSearchPanelProps {
  jobId: string;
  searchId: string | null;
  categories: Record<number, string>;
  initialStatus: SemanticSearchStatusResponse | null;
  onOpenResult: (result: SemanticSearchResultDTO, imageUrl: string) => void;
}

interface SemanticSearchResultsProps extends Omit<SemanticSearchPanelProps, "searchId"> {
  searchId: string;
}

const POLL_INTERVAL_MS = 1500;
const RESULTS_PAGE_SIZE = 40;
const searchStatusCache = new Map<string, SemanticSearchStatusResponse>();

export default function SemanticSearchPanel({
  jobId,
  searchId,
  categories,
  initialStatus,
  onOpenResult,
}: SemanticSearchPanelProps) {
  const { t } = useTranslation();
  if (!searchId) {
    return (
      <section className="visualizer-search-panel">
        <div className="search-panel-empty">
          <h2>{t("noSemanticSearchInProgress")}</h2>
          <p>
            <Trans
              i18nKey="howToStartASemanticSearch"
              components={{
                gallery: <strong>{t("imageGallery")}</strong>,
                search: <strong>{t("searchSimilar")}</strong>,
              }}
            />
          </p>
        </div>
      </section>
    );
  }

  return (
    <SemanticSearchResults
      jobId={jobId}
      searchId={searchId}
      categories={categories}
      initialStatus={initialStatus}
      onOpenResult={onOpenResult}
    />
  );
}

function SemanticSearchResults({
  jobId,
  searchId,
  categories,
  initialStatus,
  onOpenResult,
}: SemanticSearchResultsProps) {
  const cachedStatus = searchStatusCache.get(searchId);
  const [status, setStatus] = useState<SemanticSearchStatusResponse | null>(
    cachedStatus ?? initialStatus,
  );
  const [error, setError] = useState<string | null>(null);
  const [queryRecord, setQueryRecord] = useState<EmbeddingRecordDTO | null>(null);
  const [revealedCount, setRevealedCount] = useState(RESULTS_PAGE_SIZE);
  const pollRef = useRef<number | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const { t } = useTranslation();

  useEffect(() => {
    if (initialStatus?.id !== searchId) return;
    setStatus((current) => current ?? initialStatus);
  }, [initialStatus, searchId]);

  useEffect(() => {
    let cancelled = false;
    setStatus(searchStatusCache.get(searchId) ?? initialStatus);
    setError(null);

    function poll(): void {
      getSemanticSearch(jobId, searchId)
        .then((updated) => {
          if (cancelled) return;
          searchStatusCache.set(searchId, updated);
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
  }, [jobId, searchId, initialStatus]);

  // Reveal results incrementally instead of all at once: a new search starts back at one page.
  useEffect(() => {
    setRevealedCount(RESULTS_PAGE_SIZE);
  }, [searchId]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    const root = resultsRef.current;
    if (!sentinel || !root) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry?.isIntersecting) {
          setRevealedCount((current) => current + RESULTS_PAGE_SIZE);
        }
      },
      { root, rootMargin: "200px", threshold: 0 },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [status]);

  useEffect(() => {
    if (!status) return;
    if (status.status === "done" || status.status === "error" || status.status === "cancelled") {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }
  }, [status]);

  const queryRecordId = status?.query_record_id ?? null;
  const queryImagePath = status?.query_image_path ?? null;

  // The status payload identifies the query detection but carries no boxes, so the record
  // backing it is fetched once to draw the bbox on the preview.
  useEffect(() => {
    if (!queryRecordId || !queryImagePath) {
      setQueryRecord(null);
      return;
    }
    let cancelled = false;
    getJobRecordsByImagePaths(jobId, { image_paths: [queryImagePath] })
      .then((records) => {
        if (cancelled) return;
        setQueryRecord(records.find((record) => record.id === queryRecordId) ?? null);
      })
      .catch(() => {
        // The preview is best-effort: a failure here must not hide search progress.
        if (!cancelled) setQueryRecord(null);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, queryRecordId, queryImagePath]);

  const progressPct = useMemo(() => {
    if (!status || status.num_images_total === 0) return 0;
    return Math.min(100, Math.round((status.num_images_processed / status.num_images_total) * 100));
  }, [status]);

  const isActive = status?.status === "pending" || status?.status === "running";

  return (
    <section className="visualizer-search-panel">

      {error && <p className="setup-error">{error}</p>}

      {!error && !status &&
        <LoadingDiv />
      }

      {status && status.status !== "error" && (
        <div className="search-panel-progress">
          {queryRecordId && (
            <div className="ss-query-preview" title={queryImagePath ?? undefined}>
              <ImageWithBoxes
                imageUrl={getRecordImageUrl(jobId, queryRecordId)}
                imagePath={queryImagePath ?? ""}
                records={queryRecord ? [queryRecord] : []}
                minConfidence={0}
                showGroundTruths={false}
                showPredictions
              />
            </div>
          )}
          <div className="ss-progress-body">
            <div className="ss-progress-bar">
              <div
                className={`ss-progress-fill ss-progress-fill-${status.status}`}
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <p className="ss-progress-label">
              {status.status === "pending" && status.num_images_total === 0 && "Preparando búsqueda..."}
              {(status.status === "running" || (status.status === "pending" && status.num_images_total > 0)) &&
                `${status.num_images_processed.toLocaleString()} / ${status.num_images_total.toLocaleString()} ${t("scannedImages")} (${progressPct}%)`}
              {status.status === "done" &&
                `Búsqueda completada: ${status.num_images_processed.toLocaleString()} / ${status.num_images_total.toLocaleString()} ${t("scannedImages")}`}
              {status.status === "cancelled" &&
                `Búsqueda cancelada tras escanear ${status.num_images_processed.toLocaleString()} / ${status.num_images_total.toLocaleString()} imágenes`}
            </p>
          </div>
        </div>
      )}

      {status && status.status === "error" && (
        <p className="setup-error">{status.error ?? "Error desconocido"}</p>
      )}

      {status && status.status === "done" && (
        <p className="ss-status-done search-panel-status-message">✓ Búsqueda finalizada correctamente.</p>
      )}

      {status && status.status === "cancelled" && (
        <p className="setup-error">Búsqueda cancelada por el usuario.</p>
      )}

      {status && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0px 10px' }}>
            <p className="search-panel-summary">
              {t("showingNeighbours", { count1: Math.min(revealedCount, status.results?.length ?? 0), count2: status.results?.length ?? 0 })}
            </p>
            {isActive &&
              <button onClick={() => cancelSemanticSearch(jobId, searchId)}>Cancel search</button>
            }
          </div>
          <div className="search-panel-results" ref={resultsRef}>
            {(status.results ?? []).slice(0, revealedCount).map((r, i) => {
              const label = categories[r.class_id] ?? `Clase ${r.class_id}`;
              const imageUrl = r.preview_data_url;
              return (
                <ImageTile
                  key={i}
                  imageUrl={imageUrl}
                  imagePath={r.image_path}
                  title={r.image_path}
                  records={[
                    {
                      id: `${searchId}-${i}`,
                      image_path: r.image_path,
                      split: "",
                      embedding: null,
                      prediction: { class_id: r.class_id, confidence: r.confidence, bbox: r.bbox },
                      ground_truth: null,
                      status: "tp",
                      iou: null,
                    },
                  ]}
                  minConfidence={0}
                  showGroundTruths={false}
                  showPredictions
                  caption={
                    <>
                      <span className="image-tile-caption-text">{label}</span>
                      <span>dist. {r.distance.toFixed(3)}</span>
                    </>
                  }
                  captionClassName="image-tile-name-row"
                  onOpen={() => onOpenResult(r, imageUrl)}
                />
              );
            })}
            <div ref={sentinelRef} className="search-panel-results-sentinel" />
          </div>
        </>
      )}
    </section>
  );
}