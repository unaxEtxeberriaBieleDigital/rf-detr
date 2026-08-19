import { useEffect, useMemo, useState } from "react";
import {
  type ReductionAlgorithm,
  computeReduction,
  getJobOptimalThreshold,
  getJobRecords,
  loadJob,
} from "../api/client";
import EmbeddingPlot from "../components/EmbeddingPlot";
import EvaluationPanel from "../components/EvaluationPanel";
import FilterSidebar, { applyFilters, defaultFilterState } from "../components/FilterSidebar";
import type { FilterState } from "../components/FilterSidebar";
import ImageGallery from "../components/ImageGallery";
import ImageViewerModal from "../components/ImageViewerModal";
import SemanticSearchPanel from "../components/SemanticSearchPanel";
import MultiPanelLayout, { type PanelDefinition } from "../components/MultiPanelLayout";
import { useAppConfig } from "../context/AppContext";
import type { ClassThresholds, EmbeddingRecordDTO, SemanticSearchResultDTO } from "../types";
import LoadingDiv from "../components/LoadingDiv";
import { PanelLeftOpen, Funnel } from "lucide-react";
import bieleLogo from "../assets/logos/biele-logo.png"

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
  const [classThresholds, setClassThresholds] = useState<ClassThresholds>({});
  const [thresholdsLoading, setThresholdsLoading] = useState(true);
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

  function isJobNotFoundError(err: unknown): boolean {
    const text = String(err instanceof Error ? err.message : err).toLowerCase();
    return text.includes("job not found") || text.includes("failed (404)");
  }

  async function recoverJobForDataset(): Promise<string> {
    if (!config) throw new Error("No config available to recover job.");
    const recovered = await loadJob(config.datasetPath);
    setConfig({
      ...config,
      jobId: recovered.id,
      categories: recovered.categories,
      hasDimensionalityReduction: recovered.has_dimensionality_reduction,
      dimensionalityReductionComponents: recovered.dimensionality_reduction_components,
    });
    return recovered.id;
  }

  // Only re-run when the job changes (jobId), not on cosmetic config field updates.
  useEffect(() => {
    if (!config) return;
    setThresholdsLoading(true);
    setClassThresholds({});
    setFilters(defaultFilterState());
    if (
      config.dimensionalityReductionComponents === 2
      || config.dimensionalityReductionComponents === 3
    ) {
      setPcaDims(config.dimensionalityReductionComponents);
    }
    setLoading(true);
    getAllRecords(config.jobId)
      .then(async (all) => {
        setRecords(all);
        setPlotRecords(all);
        await calculateOptimalClassThresholds(config.jobId, all);
      })
      .catch(async (e) => {
        if (!isJobNotFoundError(e)) {
          throw e;
        }
        const recoveredJobId = await recoverJobForDataset();
        const all = await getAllRecords(recoveredJobId);
        setRecords(all);
        setPlotRecords(all);
        await calculateOptimalClassThresholds(recoveredJobId, all);
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)))
      .finally(() => {
        setLoading(false);
        setThresholdsLoading(false);
      });
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

  async function calculateOptimalClassThresholds(
    jobId: string,
    allRecords: EmbeddingRecordDTO[],
  ): Promise<void> {
    const classIds = [...new Set(
      allRecords
        .map((record) => record.prediction?.class_id)
        .filter((classId): classId is number => classId !== undefined),
    )];
    const results = await Promise.all(
      classIds.map(async (classId) => {
        const optimal = await getJobOptimalThreshold(jobId, "f1", 120, classId);
        return [classId, optimal.threshold] as const;
      }),
    );
    const nextThresholds = Object.fromEntries(results);
    setClassThresholds(nextThresholds);
    setFilters((current) => ({
      ...current,
      minConfidence: 0,
      perClassConfidence: new Map(
        Object.entries(nextThresholds).map(([classId, threshold]) => [Number(classId), threshold]),
      ),
    }));
  }

  async function handleComputeReduction(): Promise<void> {
    if (!config || reductionRunning) return;
    setReductionRunning(true);
    setReductionError(null);
    try {
      const ids = filteredRecords.map((r) => r.id);
      let activeJobId = config.jobId;
      try {
        await computeReduction(activeJobId, pcaDims, algorithm, ids);
      } catch (e) {
        if (!isJobNotFoundError(e)) throw e;
        activeJobId = await recoverJobForDataset();
        await computeReduction(activeJobId, pcaDims, algorithm, ids);
      }
      setConfig({
        ...config,
        jobId: activeJobId,
        hasDimensionalityReduction: true,
        dimensionalityReductionComponents: pcaDims,
      });
      const refreshed = await getAllRecords(activeJobId);
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

  const filteredRecordIds = useMemo(
    () => filteredRecords.map((record) => record.id),
    [filteredRecords],
  );

  const evaluationThresholds = useMemo<ClassThresholds>(() => {
    const thresholds: ClassThresholds = {};
    for (const classId of Object.keys(config?.categories ?? {})) {
      const numericClassId = Number(classId);
      thresholds[numericClassId] =
        filters.perClassConfidence.get(numericClassId) ?? filters.minConfidence;
    }
    return thresholds;
  }, [config?.categories, filters.minConfidence, filters.perClassConfidence]);

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

  // Panel definitions for the multi-panel layout
  const panelDefinitions = useMemo((): PanelDefinition[] => {
    if (!config) return [];
    
    return [
      {
        id: "gallery",
        title: "Galería de Imágenes",
        component: () => (
          <ImageGallery
            jobId={config.jobId}
            splitFilter={gallerySplitFilter}
            filteredRecords={filteredRecords}
            minConfidence={filters.minConfidence}
            classThresholds={classThresholds}
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
        ),
      },
      {
        id: "embedding",
        title: "Gráfico de Embeddings",
        component: () => (
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
        ),
      },
      {
        id: "evaluation",
        title: "Evaluación del Modelo",
        component: () => (
          <EvaluationPanel
            jobId={config.jobId}
            classThresholds={classThresholds}
            evaluationThresholds={evaluationThresholds}
            recordIds={filteredRecordIds}
            categories={config.categories}
          />
        ),
      },
    ];
  }, [
    config,
    gallerySplitFilter,
    filteredRecords,
    filters.minConfidence,
    selectedImagePath,
    plotRecords,
    pcaDims,
    selectedRecordId,
    clusterSelection,
    activePcaDims,
    algorithm,
    reductionRunning,
    reductionError,
    classThresholds,
    evaluationThresholds,
    filteredRecordIds,
  ]);

  if (!config) return null;

  const activeFilterCount =
    filters.qualities.size +
    filters.visibleClasses.size +
    filters.visibleSplits.size +
    (filters.minConfidence > 0 ? 1 : 0) +
    filters.perClassConfidence.size;

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
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
          />
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className={`visualizer-main-wrapper${sidebarOpen ? " visualizer-main-wrapper--shifted" : ""}`}>
        <header className="visualizer-header">
          <div className="visualizer-header-left">
            <img src={bieleLogo} alt="Biele Logo" className="fsb-logo" />
            {!sidebarOpen && (
              <button
                className="btn-open-filters"
                onClick={() => setSidebarOpen(true)}
              >
                <Funnel size={18} />
                Filtros
                {activeFilterCount > 0 && (
                  <span className="filter-badge">{activeFilterCount}</span>
                )}
              </button>
            )}
          </div>

          <div>
            <h1>RF-DETR Visualizer</h1>
            <p className="visualizer-subtitle">
              Dataset: <code>{config.datasetPath}</code> · Modelo: <code>{config.modelPath}</code>
            </p>
          </div>

          <div className="visualizer-header-right">
            <button className="secondary" onClick={reset}>
              Nueva investigación
            </button>
          </div>
        </header>

        {loading && <LoadingDiv />}
        {error && <p className="setup-error">{error}</p>}

        {!loading && !thresholdsLoading && !error && (
          <div className="visualizer-content">
            <div className="visualizer-body">
              <MultiPanelLayout
                initialVisiblePanels={[panelDefinitions[0]]}
                availablePanels={panelDefinitions}
              />

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
        classThresholds={classThresholds}
        categories={config.categories}
        modelPath={config.modelPath}
        modelType={config.modelType}
        allowSearch={false}
        onClose={() => setOpenResult(null)}
      />
    </div>
  );
}
