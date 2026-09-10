import { ChangeEvent, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { checkDataset, createJob, getDatasetTypes, getJob, getModelTypes, loadJob } from "../api/client";
import { useAppConfig } from "../context/AppContext";
import type { CheckDatasetResponse } from "../types";
import Beams from "../components/DefaultBackground";
import ParticleText from "../components/ParticleText";
import { EtaEstimator, formatDuration } from "../utils/eta";
import bieleLogo from "../assets/logos/biele-logo.png";
import { useTranslation } from "react-i18next";
import { getStored, setStored } from "../utils/store";

const POLL_INTERVAL_MS = 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function SetupPage() {
  const { setConfig } = useAppConfig();
  const { t, i18n } = useTranslation();

  const [datasetPath, setDatasetPath] = useState("");
  const [modelPath, setModelPath] = useState("");

  const [datasetTypes, setDatasetTypes] = useState<string[]>([]);
  const [modelTypes, setModelTypes] = useState<string[]>([]);
  const [datasetType, setDatasetType] = useState("");
  const [modelType, setModelType] = useState("");

  // DB detection state
  const [dbCheck, setDbCheck] = useState<CheckDatasetResponse | null>(null);
  const [dbCheckLoading, setDbCheckLoading] = useState(false);
  // "none" = user hasn't chosen yet | "load" = load existing |
  // "resume" = continue interrupted inference | "recalculate" = re-run inference
  const [dbChoice, setDbChoice] = useState<"none" | "load" | "resume" | "recalculate">("none");

  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [progressFraction, setProgressFraction] = useState<number | null>(null);
  const [etaMessage, setEtaMessage] = useState<string | null>(null);

  const handleLanguageChange = (e: ChangeEvent<HTMLSelectElement>) => {
    i18n.changeLanguage(e.target.value);
  };

  // Cargar rutas persistidas desde el store de Tauri al montar
  useEffect(() => {
    let mounted = true;

    async function loadSavedPaths() {
      const [savedDataset, savedModel] = await Promise.all([
        getStored("datasetPath"),
        getStored("modelPath"),
      ]);

      if (mounted) {
        if (savedDataset) {
          setDatasetPath(savedDataset);
          void handleDatasetPathCommit(savedDataset);
        }
        if (savedModel) {
          setModelPath(savedModel);
        }
      }
    }

    void loadSavedPaths();

    return () => {
      mounted = false;
    };
  }, []);

  const handleDatasetChange = (newPath: string) => {
    setDatasetPath(newPath);
    void setStored("datasetPath", newPath);
  };

  const handleModelChange = (newPath: string) => {
    setModelPath(newPath);
    void setStored("modelPath", newPath);
  };

  useEffect(() => {
    getDatasetTypes()
      .then((types) => {
        setDatasetTypes(types);
        setDatasetType((current) => current || types[0] || "");
      })
      .catch((e) => setErrorMessage(t("errors.connectionToBackendFail", "Could not connect to the backend") + `: ${String(e)}`));

    getModelTypes()
      .then((types) => {
        setModelTypes(types);
        setModelType((current) => current || types[0] || "");
      })
      .catch((e) => setErrorMessage(t("errors.connectionToBackendFail", "Could not connect to the backend") + `: ${String(e)}`));
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
      // Completed DBs load directly; interrupted DBs resume by default.
      if (result.has_db && result.status === "done") {
        setDbChoice("load");
      } else if (result.has_db && result.can_resume) {
        setDbChoice("resume");
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
    modelPath.trim().length > 0 &&
    datasetType.length > 0 &&
    modelType.length > 0 &&
    !submitting &&
    !dbCheckLoading;

  async function pickDatasetDirectory(): Promise<void> {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Selecciona la carpeta del dataset",
      });
      if (typeof selected === "string" && selected.trim().length > 0) {
        handleDatasetChange(selected);
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
        title: t("selectModelFile"),
        filters: [
          {
            name: t("models", { modelType: "PyTorch" }),
            extensions: ['pth']
          },
          {
            name: t("allFiles"),
            extensions: ['*']
          }
        ]
      });
      if (typeof selected === "string" && selected.trim().length > 0) {
        handleModelChange(selected);
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
    setEtaMessage(null);
    const etaEstimator = new EtaEstimator();

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

      // --- Resume or create a new inference job ---
      setStatusMessage("Creando trabajo de inferencia...");
      const job = await createJob({
        dataset_path: datasetPath.trim(),
        dataset_type: datasetType,
        model_path: modelPath.trim(),
        model_type: modelType,
        resume: dbChoice === "resume",
      });

      let latest = job;
      while (latest.status === "pending" || latest.status === "running") {
        if (latest.status === "pending") {
          setStatusMessage("Esperando a que empiece el trabajo...");
          setProgressFraction(null);
          setEtaMessage(null);
        } else if (latest.num_images_total > 0) {
          setStatusMessage(
            t("computingEmbeddingsAndPredictions", { processedCount: latest.num_images_processed, totalCount: latest.num_images_total }) +
            `(${latest.num_records} registros)`,
          );
          setProgressFraction(latest.num_images_processed / latest.num_images_total);
          const remainingSeconds = etaEstimator.update(latest.num_images_processed, latest.num_images_total);
          setEtaMessage(
            remainingSeconds == null
              ? "Estimating remaining time..."
              : remainingSeconds === 0
                ? t("estimatingRemainingTime")
                : t("estimatedRemainingTime", { remainingSeconds: formatDuration(remainingSeconds, t) }),
          );
        } else {
          setStatusMessage(t("computingEmbeddingsAndPredictionsRecords_one", { count: latest.num_records }));
          setProgressFraction(null);
          setEtaMessage(null);
        }
        await sleep(POLL_INTERVAL_MS);
        latest = await getJob(job.id);
      }

      if (latest.status === "error") {
        throw new Error(latest.error ?? t("datasetInferenceJobFail"));
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
      setEtaMessage(null);
    }
  }

  const LANGUAGES = [
    { code: "es_ES", label: "Español" },
    { code: "en_US", label: "English (US)" }
  ]

  return (
    <div className="setup-page">
      <Beams
        beamWidth={3}
        beamHeight={30}
        beamNumber={20}
        lightColor="#ffffff"
        speed={5}
        noiseIntensity={1.75}
        scale={0.2}
        rotation={30}
        centralContent={<img src={bieleLogo} alt="Biele" className="beams-logo" />}
      />
      <main className="setup-container">
        <label htmlFor="language-select" style={{ marginRight: "0.5rem" }}>
          {t("select_language", "Idioma:")}
        </label>

        <select
          id="language-select"
          value={i18n.resolvedLanguage}
          onChange={handleLanguageChange}
        >
          {LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.label}
            </option>
          ))}
        </select>
        <h1 style={{ margin: '0px' }}>
          <ParticleText
            text="Model tweaker"
            particleSize={2.2}
            density={6}
            color="#fcfcfc"
            highlightColor="#7a7a7a"
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
          {t("setupPageText")}
        </p>

        {progressFraction == null ? (
          <form className="setup-form" onSubmit={handleSubmit}>
            <label className="field">
              <span>{t("datasetPath")}</span>
              <div className="path-field">
                <input
                  type="text"
                  placeholder="C:\datasets\mi-dataset"
                  value={datasetPath}
                  onChange={(e) => handleDatasetChange(e.currentTarget.value)}
                  onBlur={(e) => handleDatasetPathCommit(e.currentTarget.value)}
                  disabled={submitting}
                />
                <button type="button" onClick={pickDatasetDirectory} disabled={submitting} >
                  {t("selectFolder")}
                </button>
              </div>
            </label>

            {/* DB detection banner */}
            {dbCheckLoading && (
              <p className="setup-db-checking">{t("searchingExistingDatabase")}</p>
            )}
            {dbCheck?.has_db && !dbCheckLoading && (
              <div className="setup-db-banner">
                <p className="setup-db-found">
                  <strong>{t("existingDatabaseFound")}</strong>
                  {" — "}
                  {dbCheck.num_records.toLocaleString()} {t("records")}
                  {dbCheck.has_dimensionality_reduction && dbCheck.dimensionality_reduction_components
                    ? `, reducción components ${dbCheck.dimensionality_reduction_components}`
                    : ", without computed reduction"}
                  {dbCheck.can_resume && (
                    <span className="setup-db-warn">
                      {" "}
                      ({t("interruptedInference", { count: dbCheck.num_images_remaining.toLocaleString() })})
                    </span>
                  )}
                </p>
                <div className="setup-db-actions">
                  <button
                    type="button"
                    className={`setup-db-btn ${dbChoice === "load" ? "setup-db-btn-active" : ""}`}
                    onClick={() => setDbChoice("load")}
                    disabled={submitting || dbCheck.can_resume}
                  >
                    {t("loadExistingDatabase")}
                  </button>
                  {dbCheck.can_resume && (
                    <button
                      type="button"
                      className={`setup-db-btn ${dbChoice === "resume" ? "setup-db-btn-active" : ""}`}
                      onClick={() => setDbChoice("resume")}
                      disabled={submitting}
                    >
                      {t("resumeInference")}
                    </button>
                  )}
                  <button
                    type="button"
                    className={`setup-db-btn ${dbChoice === "recalculate" ? "setup-db-btn-active" : ""}`}
                    onClick={() => setDbChoice("recalculate")}
                    disabled={submitting}
                  >
                    {t("recalculateAndOverride")}
                  </button>
                </div>
              </div>
            )}

            <label className="field">
              <span>{t("datasetType")}</span>
              <select value={datasetType} onChange={(e) => setDatasetType(e.currentTarget.value)} disabled={submitting}>
                {datasetTypes.length === 0 && <option value="">{t("loading")}</option>}
                {datasetTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>{t("modelPath")}</span>
              <div className="path-field">
                <input
                  type="text"
                  placeholder="C:\modelos\checkpoint.pth"
                  value={modelPath}
                  onChange={(e) => handleModelChange(e.currentTarget.value)}
                  disabled={submitting}
                />
                <button type="button" onClick={pickModelFile} disabled={submitting}>
                  {t("selectFile")}
                </button>
              </div>
            </label>

            <label className="field">
              <span>{t("modelType")}</span>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.currentTarget.value)}
                disabled={submitting}
              >
                {modelTypes.length === 0 && <option value="">{t("loading")}</option>}
                {modelTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>

            <button type="submit" disabled={!canSubmit}>
              {submitting ? t("processing") : dbChoice === "load" ? t("load") : dbChoice === "resume" ? t("resume") : t("view")}
            </button>
          </form>
        ) : (
          <>
            <p className="setup-status">{statusMessage}</p>
            <div className="progress-bar" role="progressbar" aria-valuenow={Math.round(progressFraction * 100)}>
              <div className="progress-bar-fill" style={{ width: `${Math.min(progressFraction, 1) * 100}%` }} />
            </div>
            {etaMessage && <p className="setup-eta">{etaMessage}</p>}
          </>
        )}

        {errorMessage && <p className="setup-error">{errorMessage}</p>}
      </main>
    </div>
  );
}
