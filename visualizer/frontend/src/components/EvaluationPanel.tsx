import { useEffect, useMemo, useState } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Data } from "plotly.js";
import { getJobEvaluation, getJobOptimalThreshold } from "../api/client";
import type {
  ClassThresholds,
  EvaluationMetricsResponse,
  MetricDefinitionDTO,
  OptimalThresholdResponse,
} from "../types";
import LoadingDiv from "./LoadingDiv";
import "../styles/evaluationPanel.css";

const Plot = createPlotlyComponent(Plotly);

interface EvaluationPanelProps {
  jobId: string;
  classThresholds: ClassThresholds;
  evaluationThresholds: ClassThresholds;
  recordIds: string[];
  categories: Record<number, string>;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asCurve(value: unknown): [number, number][] {
  if (!Array.isArray(value)) return [];
  const points: [number, number][] = [];
  for (const point of value) {
    if (!Array.isArray(point) || point.length < 2) continue;
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push([x, y]);
    }
  }
  return points;
}

function asMatrix(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((cell) => Number(cell)));
}

export default function EvaluationPanel({
  jobId,
  classThresholds,
  evaluationThresholds,
  recordIds,
  categories,
}: EvaluationPanelProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationMetricsResponse | null>(null);
  const [optimal, setOptimal] = useState<OptimalThresholdResponse | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState("f1");

  useEffect(() => {
    setLoading(true);
    setError(null);
    getJobEvaluation(jobId, evaluationThresholds, recordIds)
      .then((resp) => {
        setEvaluation(resp);
      })
      .catch((e) => {
        setError(String(e instanceof Error ? e.message : e));
      })
      .finally(() => setLoading(false));
  }, [jobId]);

  async function handleRecalculateWithFilters(): Promise<void> {
    setRefreshing(true);
    setError(null);
    try {
      const response = await getJobEvaluation(jobId, evaluationThresholds, recordIds);
      setEvaluation(response);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setRefreshing(false);
    }
  }

  const metricByName = useMemo(() => {
    const map = new Map<string, MetricDefinitionDTO>();
    for (const item of evaluation?.metric_definitions ?? []) map.set(item.name, item);
    return map;
  }, [evaluation]);

  const scalarMetrics = useMemo(() => {
    if (!evaluation) return [];
    return evaluation.metric_definitions
      .filter((m) => m.metric_type === "scalar")
      .map((m) => {
        const value = asNumber(evaluation.metrics[m.name]);
        return value === null ? null : { ...m, value };
      })
      .filter((m): m is MetricDefinitionDTO & { value: number } => m !== null);
  }, [evaluation]);

  const optimizableMetricNames = useMemo(
    () => scalarMetrics.map((m) => m.name).filter((name) => name !== "roc_auc"),
    [scalarMetrics],
  );

  async function handleFindOptimalThreshold(): Promise<void> {
    setOptimizing(true);
    setError(null);
    try {
      const response = await getJobOptimalThreshold(jobId, selectedMetric, 120);
      setOptimal(response);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setOptimizing(false);
    }
  }

  const rocCurve = asCurve(evaluation?.metrics.roc_curve);
  const confusionMatrix = asMatrix(evaluation?.metrics.confusion_matrix);

  const rocTrace: Data[] = [
    {
      x: rocCurve.map((p) => p[0]),
      y: rocCurve.map((p) => p[1]),
      type: "scatter",
      mode: "lines+markers",
      name: "ROC",
      line: { color: "#4a9eff", width: 2 },
      marker: { size: 6 },
    },
  ];

  const cmTrace: Data[] = [
    {
      z: confusionMatrix,
      type: "heatmap",
      colorscale: "Viridis",
      showscale: true,
    },
  ];

  if (loading) return <LoadingDiv />;
  if (error) return <p className="setup-error">{error}</p>;
  if (!evaluation) return <p className="evaluation-empty">Sin datos de evaluación.</p>;

  return (
    <div className="evaluation-panel">
      <div className="evaluation-panel__meta">
        <span>Dataset: <code>{evaluation.dataset_type}</code></span>
        <span>{evaluation.cached ? "cacheado" : "calculado ahora"}</span>
        {evaluation.calculated_at && <span>{evaluation.calculated_at}</span>}
        {evaluation.applied_record_count !== null && (
          <span>Registros filtrados: {evaluation.applied_record_count}</span>
        )}
      </div>

      <div className="evaluation-panel__cards">
        {scalarMetrics.map((metric) => (
          <div className="evaluation-card" key={metric.name} title={metric.description}>
            <div className="evaluation-card__label">{metric.display_name}</div>
            <div className="evaluation-card__value">{metric.value.toFixed(4)}</div>
          </div>
        ))}
      </div>

      <div className="evaluation-panel__optimize">
        <button
          className="secondary"
          onClick={() => void handleRecalculateWithFilters()}
          disabled={refreshing}
        >
          {refreshing ? "Recalculando..." : "Recalcular con filtros aplicados"}
        </button>
        <label>Umbrales óptimos por clase (F1)</label>
        <div className="evaluation-panel__thresholds">
          {Object.entries(classThresholds).map(([classId, threshold]) => (
            <span key={classId}>
              {categories[Number(classId)] ?? `Clase ${classId}`}: {threshold.toFixed(3)}
            </span>
          ))}
          {Object.keys(classThresholds).length === 0 && <span>Sin predicciones por clase</span>}
        </div>
        <label htmlFor="opt-metric">Métrica para umbral manual</label>
        <select
          id="opt-metric"
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
        >
          {optimizableMetricNames.map((name) => {
            const m = metricByName.get(name);
            return (
              <option key={name} value={name}>
                {m?.display_name ?? name}
              </option>
            );
          })}
        </select>
        <button className="secondary" onClick={() => void handleFindOptimalThreshold()} disabled={optimizing}>
          {optimizing ? "Calculando..." : "Calcular umbral óptimo"}
        </button>
        {optimal && (
          <span className="evaluation-panel__optimal">
            threshold={optimal.threshold.toFixed(3)} · {optimal.metric_name}={optimal.metric_value.toFixed(4)}
          </span>
        )}
      </div>

      <div className="evaluation-panel__plots">
        <div className="evaluation-panel__plot">
          <h4>ROC Curve</h4>
          <Plot
            data={rocTrace}
            layout={{
              autosize: true,
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#e6e6e6" },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              xaxis: { title: "False Positive Rate" },
              yaxis: { title: "True Positive Rate" },
            }}
            style={{ width: "100%", height: "100%" }}
            config={{ displaylogo: false, responsive: true }}
            useResizeHandler
          />
        </div>
        <div className="evaluation-panel__plot">
          <h4>Matriz de Confusión</h4>
          <Plot
            data={cmTrace}
            layout={{
              autosize: true,
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#e6e6e6" },
              margin: { l: 50, r: 20, t: 20, b: 50 },
            }}
            style={{ width: "100%", height: "100%" }}
            config={{ displaylogo: false, responsive: true }}
            useResizeHandler
          />
        </div>
      </div>
    </div>
  );
}
