import { useEffect, useState } from "react";
import { createJob, getDatasetTypes, getJob, getModelTypes } from "../api/client";
import { useAppConfig } from "../context/AppContext";

const POLL_INTERVAL_MS = 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function SetupPage() {
  const { setConfig } = useAppConfig();

  const [datasetPath, setDatasetPath] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [datasetTypes, setDatasetTypes] = useState<string[]>([]);
  const [modelTypes, setModelTypes] = useState<string[]>([]);
  const [datasetType, setDatasetType] = useState("");
  const [modelType, setModelType] = useState("");
  const [dimensions, setDimensions] = useState<2 | 3>(2);

  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [progressFraction, setProgressFraction] = useState<number | null>(null);

  useEffect(() => {
    getDatasetTypes()
      .then((types) => {
        setDatasetTypes(types);
        setDatasetType((current) => current || types[0] || "");
      })
      .catch((e) => setErrorMessage(`No se pudo conectar con el backend: ${String(e)}`));

    getModelTypes()
      .then((types) => {
        setModelTypes(types);
        setModelType((current) => current || types[0] || "");
      })
      .catch((e) => setErrorMessage(`No se pudo conectar con el backend: ${String(e)}`));
  }, []);

  const canSubmit =
    datasetPath.trim().length > 0 &&
    modelPath.trim().length > 0 &&
    datasetType.length > 0 &&
    modelType.length > 0 &&
    !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Creando trabajo de inferencia...");
    setProgressFraction(null);

    try {
      const job = await createJob({
        dataset_path: datasetPath.trim(),
        dataset_type: datasetType,
        model_path: modelPath.trim(),
        model_type: modelType,
        pca_components: dimensions,
      });

      let latest = job;
      while (latest.status === "pending" || latest.status === "running") {
        if (latest.status === "pending") {
          setStatusMessage("Esperando a que empiece el trabajo...");
          setProgressFraction(null);
        } else if (latest.num_images_total > 0) {
          setStatusMessage(
            `Calculando embeddings y predicciones: ${latest.num_images_processed}/${latest.num_images_total} imágenes ` +
              `(${latest.num_records} registros)`,
          );
          setProgressFraction(latest.num_images_processed / latest.num_images_total);
        } else {
          setStatusMessage(`Calculando embeddings y predicciones... (${latest.num_records} registros)`);
          setProgressFraction(null);
        }
        await sleep(POLL_INTERVAL_MS);
        latest = await getJob(job.id);
      }

      if (latest.status === "error") {
        throw new Error(latest.error ?? "El trabajo ha fallado por un motivo desconocido");
      }

      setConfig({
        datasetPath: datasetPath.trim(),
        datasetType,
        modelPath: modelPath.trim(),
        modelType,
        dimensions,
        jobId: latest.id,
        categories: latest.categories,
      });
    } catch (e) {
      setErrorMessage(String(e instanceof Error ? e.message : e));
      setSubmitting(false);
      setStatusMessage(null);
      setProgressFraction(null);
    }
  }

  return (
    <main className="setup-container">
      <h1>RF-DETR Visualizer</h1>
      <p className="setup-subtitle">
        Introduce el dataset y el modelo que quieres investigar. Se calcularán los embeddings y las predicciones
        antes de pasar a la visualización.
      </p>

      <form className="setup-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Ruta del dataset</span>
          <input
            type="text"
            placeholder="C:\datasets\mi-dataset"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.currentTarget.value)}
            disabled={submitting}
          />
        </label>

        <label className="field">
          <span>Tipo de dataset</span>
          <select value={datasetType} onChange={(e) => setDatasetType(e.currentTarget.value)} disabled={submitting}>
            {datasetTypes.length === 0 && <option value="">Cargando...</option>}
            {datasetTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Ruta del modelo</span>
          <input
            type="text"
            placeholder="C:\modelos\checkpoint.pth"
            value={modelPath}
            onChange={(e) => setModelPath(e.currentTarget.value)}
            disabled={submitting}
          />
        </label>

        <label className="field">
          <span>Tipo de modelo</span>
          <select value={modelType} onChange={(e) => setModelType(e.currentTarget.value)} disabled={submitting}>
            {modelTypes.length === 0 && <option value="">Cargando...</option>}
            {modelTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="field" disabled={submitting}>
          <legend>Visualización de embeddings</legend>
          <label className="radio">
            <input
              type="radio"
              name="dimensions"
              checked={dimensions === 2}
              onChange={() => setDimensions(2)}
            />
            2D
          </label>
          <label className="radio">
            <input
              type="radio"
              name="dimensions"
              checked={dimensions === 3}
              onChange={() => setDimensions(3)}
            />
            3D
          </label>
        </fieldset>

        <button type="submit" disabled={!canSubmit}>
          {submitting ? "Procesando..." : "Visualizar"}
        </button>
      </form>

      {statusMessage && <p className="setup-status">{statusMessage}</p>}
      {progressFraction !== null && (
        <div className="progress-bar" role="progressbar" aria-valuenow={Math.round(progressFraction * 100)}>
          <div className="progress-bar-fill" style={{ width: `${Math.min(progressFraction, 1) * 100}%` }} />
        </div>
      )}
      {errorMessage && <p className="setup-error">{errorMessage}</p>}
    </main>
  );
}
