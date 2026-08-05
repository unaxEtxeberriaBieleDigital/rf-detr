import Plotly from "plotly.js-dist-min";
import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Data } from "plotly.js";
import type { EmbeddingRecordDTO } from "../types";

const Plot = createPlotlyComponent(Plotly);

const STATUS_COLORS: Record<string, string> = {
  tp: "#2e7d32",
  correct: "#2e7d32",
  fp: "#c62828",
  incorrect: "#c62828",
  fn: "#f9a825",
  misclassified: "#6a1b9a",
};

interface EmbeddingPlotProps {
  records: EmbeddingRecordDTO[];
  dimensions: 2 | 3;
  categories: Record<number, string>;
  selectedRecordId: string | null;
  onSelectRecord: (recordId: string) => void;
}

/** Renders the (already PCA-reduced) embeddings as an interactive 2D/3D scatter plot. */
export default function EmbeddingPlot({
  records,
  dimensions,
  categories,
  selectedRecordId,
  onSelectRecord,
}: EmbeddingPlotProps) {
  const traces = useMemo<Data[]>(() => {
    const withEmbedding = records.filter((r) => r.embedding && r.embedding.length >= dimensions);
    const byStatus = new Map<string, EmbeddingRecordDTO[]>();
    for (const record of withEmbedding) {
      const bucket = byStatus.get(record.status) ?? [];
      bucket.push(record);
      byStatus.set(record.status, bucket);
    }

    return Array.from(byStatus.entries()).map(([status, groupRecords]) => {
      const x = groupRecords.map((r) => r.embedding![0]);
      const y = groupRecords.map((r) => r.embedding![1]);
      const z = dimensions === 3 ? groupRecords.map((r) => r.embedding![2]) : undefined;
      const text = groupRecords.map((r) => {
        const classId = (r.prediction ?? r.ground_truth)?.class_id;
        const className = classId !== undefined ? (categories[classId] ?? `clase ${classId}`) : "?";
        return `${status} · ${className}`;
      });
      const sizes = groupRecords.map((r) => (r.id === selectedRecordId ? 14 : 7));

      const marker = {
        size: sizes,
        color: STATUS_COLORS[status] ?? "#1565c0",
        line: groupRecords.map((r) => (r.id === selectedRecordId ? { width: 2, color: "#000" } : { width: 0 })),
      };

      const base = {
        name: status,
        text,
        customdata: groupRecords.map((r) => r.id),
        hovertemplate: "%{text}<extra></extra>",
        marker,
      };

      return dimensions === 3
        ? ({ ...base, type: "scatter3d", mode: "markers", x, y, z } as Data)
        : ({ ...base, type: "scattergl", mode: "markers", x, y } as Data);
    });
  }, [records, dimensions, categories, selectedRecordId]);

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        margin: { l: 30, r: 10, t: 10, b: 30 },
        legend: { orientation: "h" },
        uirevision: "embeddings",
      }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
      config={{ displaylogo: false, responsive: true }}
      onClick={(event) => {
        const point = event.points?.[0];
        const recordId = (point as unknown as { customdata?: string })?.customdata;
        if (recordId) onSelectRecord(recordId);
      }}
    />
  );
}
