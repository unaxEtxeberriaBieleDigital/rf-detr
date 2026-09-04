import Plotly from "plotly.js-dist-min";
import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Data } from "plotly.js";
import type { EmbeddingRecordDTO } from "../types";
import type { ReductionAlgorithm } from "../api/client";
import Beams from "./DefaultBackground";
import SegmentedControl from "./SegmentedControl";
import "../styles/embeddingsPlot.css";
import LoadingDiv from "./LoadingDiv";

const Plot = createPlotlyComponent(Plotly);

// Marker symbol per prediction-quality status.
// 2D (scattergl) supports a richer symbol vocabulary than 3D (scatter3d).
const STATUS_SYMBOL_2D: Record<string, string> = {
  tp: "circle",
  fp: "x",
  fn: "x",
  misclassified: "x",
};

const STATUS_SYMBOL_3D: Record<string, string> = {
  tp: "circle",
  fp: "x",
  fn: "x",
  misclassified: "x",
};

interface EmbeddingPlotProps {
  records: EmbeddingRecordDTO[];
  dimensions: 2 | 3;
  categories: Record<number, string>;
  selectedRecordId: string | null;
  /** Record IDs currently highlighted via lasso/box selection; null = none. */
  clusterSelection: Set<string> | null;
  onSelectRecord: (recordId: string) => void;
  /** Called with the new selection set after a lasso/box selection,
   *  or null when the user clears the selection. */
  onClusterSelectionChange: (selection: Set<string> | null) => void;
  // ── PCA panel props ──
  activePcaDims: number | null;
  algorithm: ReductionAlgorithm;
  pcaDims: 2 | 3;
  reductionRunning: boolean;
  reductionError: string | null;
  onAlgorithmChange: (a: ReductionAlgorithm) => void;
  onPcaDimsChange: (d: 2 | 3) => void;
  onComputeReduction: () => void;
  onClearClusterSelection: () => void;
}

/** Renders the already-reduced embeddings as an interactive 2D/3D scatter plot.
 *
 *  Points are coloured by class and shaped by prediction-quality status
 *  (TP = circle, FP = ×, FN = diamond, Misclassified = triangle / cross).
 *  The lasso and box-select tools in the mode bar let the user select a
 *  cluster; the selection is propagated via `onClusterSelectionChange` and
 *  used by `VisualizerPage` to filter the image gallery.
 */
export default function EmbeddingPlot({
  records,
  dimensions,
  categories,
  selectedRecordId,
  clusterSelection,
  onSelectRecord,
  onClusterSelectionChange,
  activePcaDims,
  algorithm,
  pcaDims,
  reductionRunning,
  reductionError,
  onAlgorithmChange,
  onPcaDimsChange,
  onComputeReduction,
  onClearClusterSelection,
}: EmbeddingPlotProps) {
  function resolveClassId(record: EmbeddingRecordDTO): number | null {
    if (record.ground_truth) return record.ground_truth.class_id;
    if (record.prediction) return record.prediction.class_id;
    return null;
  }

  // Generates a deterministic random-looking color for each class. 
  // HSV gives better visual separation than generating RGB channels independently. 
  function classColor(classId: number | null): string {
    // Golden-ratio distribution gives well-spaced hues while remaining deterministic. 
    const seed = classId === null ? 0 : Math.abs(classId);
    const hue = (seed * 0.618033988749895) % 1;
    // Small deterministic variation in saturation/value. 
    const saturation = 0.65 + ((seed * 0.17) % 0.25);
    const value = 0.75 + ((seed * 0.13) % 0.2);
    const h = hue * 6; const i = Math.floor(h);
    const f = h - i;
    const p = value * (1 - saturation);
    const q = value * (1 - saturation * f);
    const t = value * (1 - saturation * (1 - f));
    let r: number; let g: number; let b: number;
    switch (i % 6) {
      case 0: r = value; g = t; b = p; break;
      case 1: r = q; g = value; b = p; break;
      case 2: r = p; g = value; b = t; break;
      case 3: r = p; g = q; b = value; break;
      case 4: r = t; g = p; b = value; break;
      default: r = value; g = p; b = q; break;
    }
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255,)})`;
  }

  function classLabel(classId: number | null): string {
    if (classId === null) return "sin clase";
    return categories[classId] ?? `clase ${classId}`;
  }

  const withEmbedding = useMemo(
    () => records.filter((r) => r.embedding && r.embedding.length >= 2 && r.embedding.length <= 3),
    [records],
  );

  const traces = useMemo<Data[]>(() => {
    if (withEmbedding.length === 0) return [];

    const symbolMap = dimensions === 3 ? STATUS_SYMBOL_3D : STATUS_SYMBOL_2D;

    // Group by class for colour; within each group, symbols encode status.
    const byClass = new Map<string, EmbeddingRecordDTO[]>();
    for (const record of withEmbedding) {
      const classId = resolveClassId(record);
      const key = classId === null ? "none" : String(classId);
      const bucket = byClass.get(key) ?? [];
      bucket.push(record);
      byClass.set(key, bucket);
    }

    return Array.from(byClass.entries()).map(([classKey, groupRecords], _) => {
      const classId = classKey === "none" ? null : Number(classKey);
      const x = groupRecords.map((r) => r.embedding![0]);
      const y = groupRecords.map((r) => r.embedding![1]);
      const z = dimensions === 3 ? groupRecords.map((r) => r.embedding![2]) : undefined;

      const text = groupRecords.map((r) => {
        const gtClass =
          r.ground_truth?.class_id !== undefined
            ? (categories[r.ground_truth.class_id] ?? `clase ${r.ground_truth.class_id}`)
            : "-";
        const predClass =
          r.prediction?.class_id !== undefined
            ? (categories[r.prediction.class_id] ?? `clase ${r.prediction.class_id}`)
            : "-";
        return `${classLabel(resolveClassId(r))} · ${r.status} · gt: ${gtClass} · pred: ${predClass}`;
      });

      // Dimmed (greyed out) when a cluster selection is active and this point
      // is not in it — gives visual feedback similar to FiftyOne.
      const baseColor = classColor(classId);
      const colors = clusterSelection
        ? groupRecords.map((r) => (clusterSelection.has(r.id) ? baseColor : "rgba(180,180,180,0.25)"))
        : baseColor;

      const sizes = groupRecords.map((r) => (r.id === selectedRecordId ? 14 : 7));
      const symbols = groupRecords.map((r) => symbolMap[r.status] ?? "circle");

      const marker = {
        size: sizes,
        color: colors,
        symbol: symbols,
        line: {
          width: groupRecords.map((r) => (r.id === selectedRecordId ? 2 : 0)),
          color: "#f3efef00",
        },
      };

      const base = {
        name: classLabel(classId),
        text,
        customdata: groupRecords.map((r) => r.id),
        hovertemplate: "%{text}<extra></extra>",
        marker,
      };

      return dimensions === 3
        ? ({ ...base, type: "scatter3d", mode: "markers", x, y, z } as Data)
        : ({ ...base, type: "scattergl", mode: "markers", x, y } as Data);
    });
  }, [withEmbedding, dimensions, categories, selectedRecordId, clusterSelection]);

  const pcaPanel = (
    <div className="pca-panel">
      <select
        className="pca-algo-select"
        value={algorithm}
        disabled={reductionRunning}
        onChange={(e) => onAlgorithmChange(e.currentTarget.value as ReductionAlgorithm)}
        title="Algoritmo de reducción de dimensionalidad"
      >
        <option value="pca">PCA</option>
        <option value="tsne">t-SNE</option>
        <option value="umap">UMAP</option>
      </select>

      <SegmentedControl
        options={[
          { value: 2, label: "2D" },
          { value: 3, label: "3D" },
        ]}
        value={pcaDims}
        onChange={onPcaDimsChange}
        disabled={reductionRunning || algorithm === "tsne"}
      />

      <button
        type="button"
        className="pca-btn"
        onClick={onComputeReduction}
        disabled={reductionRunning}
      >
        {reductionRunning ? "Computing..." : activePcaDims ? "Compute again" : "Compute"}
      </button>
      {reductionError && <span className="pca-error">{reductionError}</span>}

      {clusterSelection && (
        <button
          type="button"
          className="pca-btn pca-btn-clear"
          onClick={onClearClusterSelection}
          title="Borrar selección del gráfico"
        >
          ✕ Selección ({clusterSelection.size})
        </button>
      )}
    </div>
  );


  if (withEmbedding.length === 0) {
    const centralContent = (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
        }}
      >
        {pcaPanel}
        <div
          style={{
            backgroundColor: "#000000",
            border: "var(--border-subtle) 2px solid",
            borderRadius: "var(--radius-lg)"
          }}
        >
          <p>
            Use the panel <strong>Compute</strong> to visualize the embeddings in the scatter plot.
          </p>
        </div>
      </div>
    );

    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          width: "100%",
          color: "#ff0000",
          fontSize: "0.9rem",
          textAlign: "center",
        }}
      >
        <Beams
          beamWidth={3}
          beamHeight={30}
          beamNumber={20}
          lightColor="#ffffff"
          speed={2}
          noiseIntensity={1.75}
          scale={0.2}
          rotation={30}
          centralContent={centralContent}
        />
      </div>
    );
  }


  return (
    <div className="embedding-plot-wrapper">
      {/* key forces a full remount when switching between 2D and 3D because the trace
          type changes (scattergl ↔ scatter3d) and Plotly cannot morph between them in-place. */}
      {reductionRunning ? (
        <LoadingDiv />
      ) : (
        <div
          className="embedding-plot-canvas"
          onPointerDownCapture={(e) => {
            try {
              (e.target as Element).setPointerCapture(e.pointerId);
            } catch (err) {
              console.warn("No se pudo capturar el puntero", err);
            }
          }}
          onPointerUpCapture={(e) => {
            try {
              const target = e.target as Element;
              if (target.hasPointerCapture(e.pointerId)) {
                target.releasePointerCapture(e.pointerId);
              }
            } catch (err) {
              console.warn("No se pudo liberar el puntero", err);
            }
          }}
        >
          <Plot
            key={`plot-${dimensions}`}
            data={traces}
            layout={{
              autosize: true,
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              margin: { l: 0, r: 0, t: 0, b: 0 },
              legend: {
                orientation: "h",
                x: 0,
                y: 0,
                xanchor: "left",
                yanchor: "bottom",
                bgcolor: "rgba(0,0,0,0.45)",
                font: { color: "#fff", size: 11 },
              },
              uirevision: `dims-${dimensions}`,
              dragmode: dimensions === 2 ? "pan" : "orbital rotation",
              ...(dimensions === 2
                ? { xaxis: { visible: false }, yaxis: { visible: false } }
                : { scene: { xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false } } }),
            }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            config={{
              displaylogo: false,
              scrollZoom: true,
              responsive: true,
              modeBarButtonsToAdd: ["lasso2d"] as unknown as Plotly.ModeBarDefaultButtons[],
              modeBarButtonsToRemove: [
                "zoom2d",
                "zoom3d",
                "select2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toImage",
                "resetCameraLastSave3d",
              ]
            }}
            onClick={(event) => {
              const point = event.points?.[0];
              const recordId = (point as unknown as { customdata?: string })?.customdata;
              if (recordId) onSelectRecord(recordId);
            }}
            onSelected={(event) => {
              if (!event || event.points.length === 0) {
                onClusterSelectionChange(null);
                return;
              }
              const ids = event.points
                .map((p: unknown) => ((p as { customdata?: string }).customdata) ?? "")
                .filter(Boolean);
              onClusterSelectionChange(new Set(ids));
            }}
            onDeselect={() => onClusterSelectionChange(null)}
          />
        </div>
      )}
      {pcaPanel}
    </div>
  );
}
