import { useEffect, useMemo, useState } from "react";
import { type ReductionAlgorithm, computeReduction, getJobRecords } from "../api/client";
import EmbeddingPlot from "../components/EmbeddingPlot";
import FilterSidebar, { applyFilters, defaultFilterState } from "../components/FilterSidebar";
import type { FilterState } from "../components/FilterSidebar";
import ImageGallery from "../components/ImageGallery";
import ImageViewerModal from "../components/ImageViewerModal";
import SemanticSearchPanel from "../components/SemanticSearchPanel";
import { useAppConfig } from "../context/AppContext";
import type { EmbeddingRecordDTO, SemanticSearchResultDTO } from "../types";
import LoadingDiv from "../components/LoadingDiv";

export default function VisualizerPage() {
  const { config, setConfig, reset } = useAppConfig();

  const [records, setRecords] = useState<EmbeddingRecordDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  // Semantic search: id of the currently-open search panel (next to the embedding plot),
  // and the neighbour result being inspected in the full-screen viewer (if any).
  const [activeSearchId, setActiveSearchId] = useState<string | null>(null);
  const [openResult, setOpenResult] = useState<{ result: SemanticSearchResultDTO; imageUrl: string } | null>(
    null,
  );

  const [filters, setFilters] = useState<FilterState>(defaultFilterState());
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Dimensionality-reduction panel state
  const [pcaDims, setPcaDims] = useState<2 | 3>(2);
  const [algorithm, setAlgorithm] = useState<ReductionAlgorithm>("pca");
  const [reductionRunning, setReductionRunning] = useState(false);
  const [reductionError, setReductionError] = useState<string | null>(null);

  // plotRecords – only updated when "Recalcular" is pressed, never on filter changes.
  const [plotRecords, setPlotRecords] = useState<EmbeddingRecordDTO[]>([]);

  // Cluster selection: record IDs selected via lasso/box tool in the scatter plot.
  // null = no selection active (gallery shows everything that passes the other filters).
  const [clusterSelection, setClusterSelection] = useState<Set<string> | null>(null);

  // Only re-run when the job changes (jobId), not on cosmetic config field updates.
  useEffect(() => {
    if (!config) return;
    if (
      config.dimensionalityReductionComponents === 2
      || config.dimensionalityReductionComponents === 3
    ) {
      setPcaDims(config.dimensionalityReductionComponents);
    }
    setLoading(true);
    getAllRecords(config.jobId)
      .then((all) => {
        setRecords(all);
        setPlotRecords(all);
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config?.jobId]);

  async function getAllRecords(jobId: string): Promise<EmbeddingRecordDTO[]> {
    const pageSize = 2000;
    let offset = 0;
    const all: EmbeddingRecordDTO[] = [];
    while (true) {
      const page = await getJobRecords(jobId, { limit: pageSize, offset });
      all.push(...page);
      if (page.length < pageSize) break;
      offset += page.length;
    }
    return all;
  }

  async function handleComputeReduction(): Promise<void> {
    if (!config || reductionRunning) return;
    setReductionRunning(true);
    setReductionError(null);
    try {
      const ids = filteredRecords.map((r) => r.id);
      await computeReduction(config.jobId, pcaDims, algorithm, ids);
      setConfig({
        ...config,
        hasDimensionalityReduction: true,
        dimensionalityReductionComponents: pcaDims,
      });
      const refreshed = await getAllRecords(config.jobId);
      setRecords(refreshed);
      const refreshedIds = new Set(ids);
      setPlotRecords(refreshed.filter((r) => refreshedIds.has(r.id)));
      // Clear any cluster selection – coords have changed so old selection is stale.
      setClusterSelection(null);
    } catch (e) {
      setReductionError(String(e instanceof Error ? e.message : e));
    } finally {
      setReductionRunning(false);
    }
  }

  // filteredRecords: sidebar filters + cluster selection (if active).
  const filteredRecords = useMemo(() => {
    const base = applyFilters(records, filters);
    if (!clusterSelection) return base;
    return base.filter((r) => clusterSelection.has(r.id));
  }, [records, filters, clusterSelection]);

  const gallerySplitFilter = useMemo(() => {
    if (filters.visibleSplits.size === 1) return Array.from(filters.visibleSplits)[0];
    return "all";
  }, [filters.visibleSplits]);

  const selectedImagePath = useMemo(
    () => records.find((r) => r.id === selectedRecordId)?.image_path ?? null,
    [records, selectedRecordId],
  );

  const activePcaDims = useMemo((): number | null => {
    const sample = plotRecords.find((r) => r.embedding && r.embedding.length <= 3);
    return sample?.embedding?.length ?? null;
  }, [plotRecords]);

  if (!config) return null;

  return (
    <div className="visualizer-container">
      {/* ── Collapsible sidebar ── */}
      <aside className={`visualizer-sidebar${sidebarOpen ? " visualizer-sidebar--open" : ""}`}>
        <div className="visualizer-sidebar-content">
          <FilterSidebar
            records={records}
            categories={config.categories}
            filters={filters}
            onChange={(next) => {
              setFilters(next);
              setClusterSelection(null);
            }}
          />
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className={`visualizer-main-wrapper${sidebarOpen ? " visualizer-main-wrapper--shifted" : ""}`}>
        <header className="visualizer-header">
          <div className="visualizer-header-left">
            <button
              type="button"
              className="visualizer-sidebar-toggle"
              onClick={() => setSidebarOpen((v) => !v)}
              title={sidebarOpen ? "Cerrar filtros" : "Abrir filtros"}
            >
              ☰
            </button>
            <div>
              <h1>RF-DETR Visualizer</h1>
              <p className="visualizer-subtitle">
                Dataset: <code>{config.datasetPath}</code> · Modelo: <code>{config.modelPath}</code>
              </p>
            </div>
          </div>
          <div className="visualizer-header-right">
            <button className="secondary" onClick={reset}>
              Nueva investigación
            </button>
          </div>
        </header>

        {loading && <LoadingDiv />}
        {error && <p className="setup-error">{error}</p>}

        {!loading && !error && (
          <div className="visualizer-content">
            <div className="visualizer-record-count">
              {filteredRecords.length.toLocaleString()} de {records.length.toLocaleString()} registros
              {clusterSelection && (
                <span className="cluster-sel-badge">
                  {" "}· {clusterSelection.size} seleccionados en gráfico
                </span>
              )}
            </div>

            <div className="visualizer-body">
              <section className="visualizer-gallery">
                <ImageGallery
                  jobId={config.jobId}
                  splitFilter={gallerySplitFilter}
                  filteredRecords={filteredRecords}
                  minConfidence={filters.minConfidence}
                  categories={config.categories}
                  modelPath={config.modelPath}
                  modelType={config.modelType}
                  selectedImagePath={selectedImagePath}
                  onSelectImage={(imagePath) => {
                    const first = filteredRecords.find((r) => r.image_path === imagePath);
                    setSelectedRecordId(first?.id ?? null);
                  }}
                  onSearchStarted={setActiveSearchId}
                />
              </section>

              <section className="visualizer-plot">
                <EmbeddingPlot
                  records={plotRecords}
                  dimensions={pcaDims}
                  categories={config.categories}
                  selectedRecordId={selectedRecordId}
                  clusterSelection={clusterSelection}
                  onSelectRecord={setSelectedRecordId}
                  onClusterSelectionChange={setClusterSelection}
                  activePcaDims={activePcaDims}
                  algorithm={algorithm}
                  pcaDims={pcaDims}
                  reductionRunning={reductionRunning}
                  reductionError={reductionError}
                  onAlgorithmChange={setAlgorithm}
                  onPcaDimsChange={setPcaDims}
                  onComputeReduction={handleComputeReduction}
                  onClearClusterSelection={() => setClusterSelection(null)}
                />
              </section>

              {activeSearchId && (
                <SemanticSearchPanel
                  jobId={config.jobId}
                  searchId={activeSearchId}
                  categories={config.categories}
                  onClose={() => setActiveSearchId(null)}
                  onOpenResult={(result, imageUrl) => setOpenResult({ result, imageUrl })}
                />
              )}
            </div>
          </div>
        )}
      </div>

      <ImageViewerModal
        isOpen={!!openResult}
        jobId={config.jobId}
        imagePath={openResult?.result.image_path ?? null}
        imageUrlOverride={openResult?.imageUrl}
        records={
          openResult
            ? [
                {
                  id: "search-result",
                  image_path: openResult.result.image_path,
                  split: "",
                  embedding: null,
                  prediction: {
                    class_id: openResult.result.class_id,
                    confidence: openResult.result.confidence,
                    bbox: openResult.result.bbox,
                  },
                  ground_truth: null,
                  status: "tp",
                },
              ]
            : []
        }
        minConfidence={0}
        categories={config.categories}
        modelPath={config.modelPath}
        modelType={config.modelType}
        allowSearch={false}
        onClose={() => setOpenResult(null)}
      />
    </div>
  );
}
