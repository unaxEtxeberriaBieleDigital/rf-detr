import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJobImagePaths, getJobRecordsByImagePaths } from "../api/client";
import ImageCard from "./ImageCard";
import ImageViewerModal from "./ImageViewerModal";
import type { ClassThresholds, EmbeddingRecordDTO } from "../types";

interface ImageGalleryProps {
  jobId: string;
  splitFilter: string;
  minConfidence: number;
  classThresholds?: ClassThresholds;
  /** Class id → human readable name, forwarded to the image viewer's side panel. */
  categories: Record<number, string>;
  /** Model info, forwarded to the image viewer's "search similar" feature. */
  modelPath: string;
  modelType: string;
  /** Filtered subset of all records. Only images that have at least one record in this set
   *  are shown. Pass the full record list to show everything. */
  filteredRecords: EmbeddingRecordDTO[];
  selectedImagePath: string | null;
  onSelectImage: (imagePath: string) => void;
  /** Called when a "search similar" job is started from the image viewer, so the caller can
   *  show its progress/results (e.g. in a panel next to the embedding plot). */
  onSearchStarted?: (searchId: string) => void;
}

const IMAGE_PAGE_SIZE = 60;

/** Renders images with incremental loading and lazy record fetching per visible image page.
 *
 *  The server is asked for records split-filtered only (fast, indexed).  Client-side,
 *  images whose raw record set has no record matching `filteredRecords` are hidden so that
 *  all sidebar filters (quality, class, confidence) also affect the gallery in real time
 *  without extra network round-trips.
 */
export default function ImageGallery({
  jobId,
  splitFilter,
  minConfidence,
  classThresholds,
  categories,
  modelPath,
  modelType,
  filteredRecords,
  selectedImagePath,
  onSelectImage,
  onSearchStarted,
}: ImageGalleryProps) {
  const [openImagePath, setOpenImagePath] = useState<string | null>(null);
  const [imagePaths, setImagePaths] = useState<string[]>([]);
  const [recordsByImage, setRecordsByImage] = useState<Map<string, EmbeddingRecordDTO[]>>(new Map());
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const isFetchingRef = useRef(false);
  const split = splitFilter === "all" ? undefined : splitFilter;

  // Build a set of record ids that pass the current filters for fast O(1) membership checks.
  const filteredRecordIds = useMemo(
    () => new Set(filteredRecords.map((r) => r.id)),
    [filteredRecords],
  );

  const loadMore = useCallback(async () => {
    if (isFetchingRef.current || !hasMore) return;
    isFetchingRef.current = true;
    setLoadingMore(true);

    try {
      const page = await getJobImagePaths(jobId, {
        split,
        offset,
        limit: IMAGE_PAGE_SIZE,
      });
      if (page.image_paths.length === 0) {
        setHasMore(false);
        return;
      }

      const pageRecords = await getJobRecordsByImagePaths(jobId, {
        image_paths: page.image_paths,
        split,
      });

      const grouped = new Map<string, EmbeddingRecordDTO[]>();
      for (const imagePath of page.image_paths) {
        grouped.set(imagePath, []);
      }
      for (const record of pageRecords) {
        const bucket = grouped.get(record.image_path) ?? [];
        bucket.push(record);
        grouped.set(record.image_path, bucket);
      }

      setImagePaths((current) => {
        const existing = new Set(current);
        const uniqueNewPaths = page.image_paths.filter((p) => !existing.has(p));
        return [...current, ...uniqueNewPaths];
      });
      setRecordsByImage((current) => {
        const next = new Map(current);
        for (const [imagePath, imageRecords] of grouped.entries()) {
          next.set(imagePath, imageRecords);
        }
        return next;
      });
      setOffset((current) => current + page.image_paths.length);
      setHasMore(page.has_more);
    } finally {
      setLoadingMore(false);
      isFetchingRef.current = false;
    }
  }, [jobId, split, offset, hasMore]);

  useEffect(() => {
    setImagePaths([]);
    setRecordsByImage(new Map());
    setOffset(0);
    setHasMore(true);
  }, [jobId, split]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const scrollRoot = sentinel.closest(".visualizer-gallery");

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry?.isIntersecting) {
          void loadMore();
        }
      },
      { root: scrollRoot, rootMargin: "200px", threshold: 0 },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMore]);

  // All (image, rawRecords) pairs loaded from the server so far.
  const imageEntries = useMemo(
    () => imagePaths.map((imagePath) => [imagePath, recordsByImage.get(imagePath) ?? []] as const),
    [imagePaths, recordsByImage],
  );

  // Only show images that have at least one record surviving the active filters.
  const visibleEntries = useMemo(
    () =>
      imageEntries.filter(([, imageRecords]) =>
        imageRecords.some((r) => filteredRecordIds.has(r.id)),
      ),
    [imageEntries, filteredRecordIds],
  );

  const openImageRecords = openImagePath ? recordsByImage.get(openImagePath) ?? [] : [];

  const currentVisibleIndex = openImagePath
    ? visibleEntries.findIndex(([p]) => p === openImagePath)
    : -1;

  function navigateTo(delta: 1 | -1): void {
    if (currentVisibleIndex < 0) return;
    const nextIndex = currentVisibleIndex + delta;
    if (nextIndex < 0 || nextIndex >= visibleEntries.length) return;
    const [nextPath] = visibleEntries[nextIndex];
    onSelectImage(nextPath);
    setOpenImagePath(nextPath);
  }

  const hasPrev = currentVisibleIndex > 0;
  const hasNext = currentVisibleIndex >= 0 && currentVisibleIndex < visibleEntries.length - 1;

  return (
    <>
      <div className="image-gallery">
        {visibleEntries.map(([imagePath, imageRecords]) => (
          <ImageCard
            key={imagePath}
            jobId={jobId}
            imagePath={imagePath}
            records={imageRecords}
            minConfidence={minConfidence}
            classThresholds={classThresholds}
            isSelected={imagePath === selectedImagePath}
            onOpen={() => {
              onSelectImage(imagePath);
              setOpenImagePath(imagePath);
            }}
          />
        ))}
        {visibleEntries.length === 0 && !loadingMore && (
          <p className="image-gallery-empty">No hay imágenes que coincidan con los filtros.</p>
        )}
        <div ref={sentinelRef} className="image-gallery-sentinel" />
        {loadingMore && <p className="image-gallery-loading">Cargando más imágenes...</p>}
      </div>
      <ImageViewerModal
        isOpen={!!openImagePath}
        jobId={jobId}
        imagePath={openImagePath}
        records={openImageRecords}
        minConfidence={minConfidence}
        classThresholds={classThresholds}
        categories={categories}
        modelPath={modelPath}
        modelType={modelType}
        onSearchStarted={onSearchStarted}
        onPrev={hasPrev ? () => navigateTo(-1) : undefined}
        onNext={hasNext ? () => navigateTo(1) : undefined}
        onClose={() => setOpenImagePath(null)}
      />
    </>
  );
}
