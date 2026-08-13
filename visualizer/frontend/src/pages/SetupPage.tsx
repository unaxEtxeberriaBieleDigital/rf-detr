import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { checkDataset, createJob, getDatasetTypes, getJob, getModelTypes, loadJob } from "../api/client";
import { useAppConfig } from "../context/AppContext";
import type { CheckDatasetResponse } from "../types";
import bieleDigitalVideo from "../assets/BIELE_DIGITAL.mp4";
import Beams from "../components/SetUpBackground";
import ParticleText from "../components/ParticleText";

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

  // DB detection state
  const [dbCheck, setDbCheck] = useState<CheckDatasetResponse | null>(null);
  const [dbCheckLoading, setDbCheckLoading] = useState(false);
  // "none" = user hasn't chosen yet | "load" = load existing | "recalculate" = re-run inference
  const [dbChoice, setDbChoice] = useState<"none" | "load" | "recalculate">("none");

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

  // Check for an existing DB whenever datasetPath changes (debounced via blur / explicit set).
  async function handleDatasetPathCommit(path: string): Promise<void> {
    if (!path.trim()) {
      setDbCheck(null);
      setDbChoice("none");
      return;
    }
    setDbCheckLoading(true);
    setDbCheck(null);
    setDbChoice("none");
    try {
      const result = await checkDataset(path.trim());
      setDbCheck(result);
      // Auto-select "load" if the DB looks healthy; user can override.
      if (result.has_db && result.status === "done") {
        setDbChoice("load");
      } else {
        setDbChoice("recalculate");
      }
    } catch {
      // If backend is not reachable yet, silently ignore.
    } finally {
      setDbCheckLoading(false);
    }
  }

  const canSubmit =
    datasetPath.trim().length > 0 &&
    (dbChoice === "load" || modelPath.trim().length > 0) &&
    datasetType.length > 0 &&
    modelType.length > 0 &&
    !submitting;

  async function pickDatasetDirectory(): Promise<void> {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Selecciona la carpeta del dataset",
      });
      if (typeof selected === "string" && selected.trim().length > 0) {
        setDatasetPath(selected);
        await handleDatasetPathCommit(selected);
      }
    } catch (e) {
      setErrorMessage(`No se pudo abrir el selector de carpetas: ${String(e instanceof Error ? e.message : e)}`);
    }
  }

  async function pickModelFile(): Promise<void> {
    try {
      const selected = await open({
        directory: false,
        multiple: false,
        title: "Selecciona el archivo de modelo",
      });
      if (typeof selected === "string" && selected.trim().length > 0) {
        setModelPath(selected);
      }
    } catch (e) {
      setErrorMessage(`No se pudo abrir el selector de archivo: ${String(e instanceof Error ? e.message : e)}`);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setErrorMessage(null);
    setProgressFraction(null);

    try {
      // --- Load existing DB (skip inference) ---
      if (dbChoice === "load") {
        setStatusMessage("Cargando embeddings existentes...");
        const loaded = await loadJob(datasetPath.trim());
        if (loaded.status === "error") {
          throw new Error(loaded.error ?? "Error al cargar la base de datos");
        }
        setConfig({
          datasetPath: datasetPath.trim(),
          datasetType,
          modelPath: modelPath.trim(),
          modelType,
          jobId: loaded.id,
          categories: loaded.categories,
          hasDimensionalityReduction: loaded.has_dimensionality_reduction,
          dimensionalityReductionComponents: loaded.dimensionality_reduction_components,
        });
        return;
      }

      // --- Create new inference job ---
      setStatusMessage("Creando trabajo de inferencia...");
      const job = await createJob({
        dataset_path: datasetPath.trim(),
        dataset_type: datasetType,
        model_path: modelPath.trim(),
        model_type: modelType,
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
        jobId: latest.id,
        categories: latest.categories,
        hasDimensionalityReduction: false,
        dimensionalityReductionComponents: null,
      });
    } catch (e) {
      setErrorMessage(String(e instanceof Error ? e.message : e));
      setSubmitting(false);
      setStatusMessage(null);
      setProgressFraction(null);
    }
  }

  const showModelFields = dbChoice !== "load";

  return (
    <div className="setup-page">
      <Beams
        beamWidth={3}
        beamHeight={30}
        beamNumber={20}
        lightColor="#ffffff"
        speed={2}
        noiseIntensity={1.75}
        scale={0.2}
        rotation={30}
      />
      <main className="setup-container">
        <h1>
          <ParticleText
            text="Visualizer"
            particleSize={2.2}
            density={6}
            color="#f8fafc"
            highlightColor="#474649"
            scatter={190}
            gatherDuration={1600}
            stagger={420}
            pointerRepel={42}
            repelRadius={120}
            idleDrift={0.8}
            trigger="mount"
            fontSize="clamp(3.5rem, 13vw, 9rem)"
            fontWeight={800}
            fontFamily="inherit"
            glow
          />
        </h1>
        <p className="setup-subtitle">
          Introduce el dataset y el modelo que quieres investigar. Se calcularán los embeddings y las predicciones
          antes de pasar a la visualización.
        </p>

        <form className="setup-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Ruta del dataset</span>
          <div className="path-field">
            <input
              type="text"
              placeholder="C:\datasets\mi-dataset"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.currentTarget.value)}
              onBlur={(e) => handleDatasetPathCommit(e.currentTarget.value)}
              disabled={submitting}
            />
            <button type="button" onClick={pickDatasetDirectory} disabled={submitting}>
              Seleccionar carpeta
            </button>
          </div>
        </label>

        {/* DB detection banner */}
        {dbCheckLoading && (
          <p className="setup-db-checking">Comprobando base de datos existente...</p>
        )}
        {dbCheck?.has_db && !dbCheckLoading && (
          <div className="setup-db-banner">
            <p className="setup-db-found">
              <strong>Base de datos existente encontrada</strong>
              {" — "}
              {dbCheck.num_records.toLocaleString()} registros
              {dbCheck.has_dimensionality_reduction && dbCheck.dimensionality_reduction_components
                ? `, reducción ${dbCheck.dimensionality_reduction_components}D calculada`
                : ", sin reducción calculada"}
              {dbCheck.status === "error" && (
                <span className="setup-db-warn"> (la inferencia anterior fue interrumpida)</span>
              )}
            </p>
            <div className="setup-db-actions">
              <button
                type="button"
                className={`setup-db-btn ${dbChoice === "load" ? "setup-db-btn-active" : ""}`}
                onClick={() => setDbChoice("load")}
                disabled={submitting || dbCheck.status === "error"}
              >
                Cargar existente
              </button>
              <button
                type="button"
                className={`setup-db-btn ${dbChoice === "recalculate" ? "setup-db-btn-active" : ""}`}
                onClick={() => setDbChoice("recalculate")}
                disabled={submitting}
              >
                Recalcular (sobreescribir)
              </button>
            </div>
          </div>
        )}

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

        {showModelFields && (
          <>
            <label className="field">
              <span>Ruta del modelo</span>
              <div className="path-field">
                <input
                  type="text"
                  placeholder="C:\modelos\checkpoint.pth"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.currentTarget.value)}
                  disabled={submitting}
                />
                <button type="button" onClick={pickModelFile} disabled={submitting}>
                  Seleccionar archivo
                </button>
              </div>
            </label>

            <label className="field">
              <span>Tipo de modelo</span>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.currentTarget.value)}
                disabled={submitting}
              >
                {modelTypes.length === 0 && <option value="">Cargando...</option>}
                {modelTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}

        <button type="submit" disabled={!canSubmit}>
          {submitting ? "Procesando..." : dbChoice === "load" ? "Cargar" : "Visualizar"}
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
    </div>
  );
}
