import { useEffect, useMemo, useState } from "react";
import { getJobRecords } from "../api/client";
import ConfidenceFilter from "../components/ConfidenceFilter";
import EmbeddingPlot from "../components/EmbeddingPlot";
import ImageGallery from "../components/ImageGallery";
import { useAppConfig } from "../context/AppContext";
import type { EmbeddingRecordDTO } from "../types";

export default function VisualizerPage() {
  const { config, reset } = useAppConfig();

  const [records, setRecords] = useState<EmbeddingRecordDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [minConfidence, setMinConfidence] = useState(0);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);
  const [splitFilter, setSplitFilter] = useState<string>("all");

  useEffect(() => {
    if (!config) return;
    setLoading(true);
    getJobRecords(config.jobId)
      .then(setRecords)
      .catch((e) => setError(String(e instanceof Error ? e.message : e)))
      .finally(() => setLoading(false));
  }, [config]);

  const availableSplits = useMemo(() => Array.from(new Set(records.map((r) => r.split))).sort(), [records]);

  const filteredRecords = useMemo(() => {
    return records.filter((r) => {
      if (splitFilter !== "all" && r.split !== splitFilter) return false;
      // Ground-truth-only rows (false negatives) have no prediction/confidence to filter on,
      // so they stay visible regardless of the confidence threshold.
      if (r.prediction && r.prediction.confidence < minConfidence) return false;
      return true;
    });
  }, [records, splitFilter, minConfidence]);

  const selectedImagePath = useMemo(
    () => filteredRecords.find((r) => r.id === selectedRecordId)?.image_path ?? null,
    [filteredRecords, selectedRecordId],
  );

  if (!config) return null;

  return (
    <div className="visualizer-container">
      <header className="visualizer-header">
        <div>
          <h1>RF-DETR Visualizer</h1>
          <p className="visualizer-subtitle">
            Dataset: <code>{config.datasetPath}</code> · Modelo: <code>{config.modelPath}</code>
          </p>
        </div>
        <button className="secondary" onClick={reset}>
          Nueva investigación
        </button>
      </header>

      {loading && <p>Cargando registros...</p>}
      {error && <p className="setup-error">{error}</p>}

      {!loading && !error && (
        <>
          <div className="visualizer-toolbar">
            <ConfidenceFilter minConfidence={minConfidence} onChange={setMinConfidence} />
            <label className="split-filter">
              <span>Split</span>
              <select value={splitFilter} onChange={(e) => setSplitFilter(e.currentTarget.value)}>
                <option value="all">Todos</option>
                {availableSplits.map((split) => (
                  <option key={split} value={split}>
                    {split}
                  </option>
                ))}
              </select>
            </label>
            <span className="record-count">
              {filteredRecords.length} de {records.length} registros
            </span>
          </div>

          <div className="visualizer-body">
            <section className="visualizer-plot">
              <EmbeddingPlot
                records={filteredRecords}
                dimensions={config.dimensions}
                categories={config.categories}
                selectedRecordId={selectedRecordId}
                onSelectRecord={setSelectedRecordId}
              />
            </section>

            <section className="visualizer-gallery">
              <ImageGallery
                jobId={config.jobId}
                records={filteredRecords}
                minConfidence={minConfidence}
                categories={config.categories}
                selectedImagePath={selectedImagePath}
                onSelectImage={(imagePath) => {
                  const first = filteredRecords.find((r) => r.image_path === imagePath);
                  setSelectedRecordId(first?.id ?? null);
                }}
              />
            </section>
          </div>
        </>
      )}
    </div>
  );
}
