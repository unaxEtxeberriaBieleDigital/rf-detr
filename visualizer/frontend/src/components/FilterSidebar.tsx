import React, { useMemo, useState } from "react";
import type { EmbeddingRecordDTO } from "../types";
import {
  PanelLeftClose,
  Funnel,
  ChartPie,
  SlidersHorizontal,
  Tags,
  Layers,
  ChevronRight,
  LucideIcon
} from 'lucide-react';
import SegmentedControl from "./SegmentedControl";

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

export type PredQuality = "tp" | "fp" | "fn" | "misclassified";

export interface ClassConfidence {
  classId: number;
  minConfidence: number;
}

export interface FilterState {
  /** Which prediction-quality categories are shown (empty = all). */
  qualities: Set<PredQuality>;
  /** Global minimum confidence (0–1). Applied when perClassConfidence is empty. */
  minConfidence: number;
  /** Per-class confidence overrides (classId → min confidence). */
  perClassConfidence: Map<number, number>;
  /** Which class ids are shown (empty set = all). */
  visibleClasses: Set<number>;
  /** Which splits are shown (empty set = all). */
  visibleSplits: Set<string>;
}

export function defaultFilterState(): FilterState {
  return {
    qualities: new Set(),
    minConfidence: 0,
    perClassConfidence: new Map(),
    visibleClasses: new Set(),
    visibleSplits: new Set(),
  };
}

/** Returns the confidence threshold that applies to *classId* (per-class override, or the
 *  global minimum confidence when there's no override). */
function getConfidenceThreshold(classId: number | null, filters: FilterState): number {
  const perClass = classId !== null ? filters.perClassConfidence.get(classId) : undefined;
  return perClass !== undefined ? perClass : filters.minConfidence;
}

/** Recomputes a record's prediction-quality status after applying the confidence threshold.
 *
 *  The backend matches every prediction to ground truth regardless of confidence (see
 *  ``evaluator.match_detections``), so a low-confidence detection can still "claim" a
 *  ground-truth box as tp/misclassified. Once the confidence threshold rises above that
 *  detection's own confidence, it effectively stops existing: any ground truth it had
 *  claimed becomes a miss ("fn"), while a plain false positive (no ground truth) simply
 *  disappears -- it doesn't count as anything.
 */
function getEffectiveStatus(r: EmbeddingRecordDTO, filters: FilterState): PredQuality | null {
  if (!r.prediction) {
    return r.status as PredQuality;
  }
  const classId = r.ground_truth?.class_id ?? r.prediction.class_id ?? null;
  const threshold = getConfidenceThreshold(classId, filters);
  if (r.prediction.confidence >= threshold) {
    return r.status as PredQuality;
  }
  return r.ground_truth ? "fn" : null;
}

/** Apply FilterState to a list of records and return the matching subset. */
export function applyFilters(
  records: EmbeddingRecordDTO[],
  filters: FilterState,
): EmbeddingRecordDTO[] {
  return records.filter((r) => {
    // Split filter
    if (filters.visibleSplits.size > 0 && !filters.visibleSplits.has(r.split)) return false;

    // Status / prediction-quality filter
    if (filters.qualities.size > 0) {
      const status = r.status as PredQuality;
      if (!filters.qualities.has(status)) return false;
    }

    // Class filter
    const classId = r.prediction?.class_id ?? r.ground_truth?.class_id ?? null;
    if (filters.visibleClasses.size > 0 && classId !== null && !filters.visibleClasses.has(classId)) return false;

    // Confidence filter
    if (r.prediction) {
      const threshold = getConfidenceThreshold(classId, filters);
      if (r.prediction.confidence < threshold) return false;
    }

    return true;
  });
}

// -----------------------------------------------------------------------
// Props
// -----------------------------------------------------------------------

interface FilterSidebarProps {
  records: EmbeddingRecordDTO[];
  categories: Record<number, string>;
  filters: FilterState;
  onChange: (next: FilterState) => void;
  sidebarOpen: boolean,
  setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>,
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

const QUALITY_LABELS: Record<PredQuality, string> = {
  tp: "TP",
  fp: "FP",
  fn: "FN",
  misclassified: "Misclas.",
};

const QUALITY_COLORS: Record<PredQuality, string> = {
  tp: "#2e7d32",
  fp: "#c62828",
  fn: "#1565c0",
  misclassified: "#6a1b9a",
};

const ALL_QUALITIES: PredQuality[] = ["tp", "fp", "fn", "misclassified"];

function Section({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: LucideIcon;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className={`fsb-section ${open ? "fsb-section-open" : ""}`}>
      <button
        type="button"
        className="fsb-section-header"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="fsb-section-header-left">
          <Icon size={18} className="fsb-section-icon" />
          <span>{title}</span>
        </div>
        <ChevronRight size={18} className={`fsb-chevron ${open ? "open" : ""}`} />
      </button>
      {open && <div className="fsb-section-body">{children}</div>}
    </div>
  );
}

// -----------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------

/** Left-side panel with all filtering controls for the visualizer. */
export default function FilterSidebar({
  records,
  categories,
  filters,
  onChange,
  sidebarOpen,
  setSidebarOpen,
}: FilterSidebarProps) {
  const [confMode, setConfMode] = useState<"global" | "perclass">("global");
  // Derived collections
  const availableSplits = useMemo(
    () => Array.from(new Set(records.map((r) => r.split))).sort(),
    [records],
  );

  const availableClasses = useMemo(() => {
    const ids = new Set<number>();
    for (const r of records) {
      if (r.ground_truth?.class_id !== undefined) ids.add(r.ground_truth.class_id);
      if (r.prediction?.class_id !== undefined) ids.add(r.prediction.class_id);
    }
    return Array.from(ids).sort((a, b) => {
      const la = categories[a] ?? String(a);
      const lb = categories[b] ?? String(b);
      return la.localeCompare(lb);
    });
  }, [records, categories]);

  // Count per quality (for badges), respecting every other active filter (confidence,
  // classes, splits) so the pills reflect what the current confidence threshold would
  // actually show -- only the quality filter itself is excluded from this pass. Uses
  // getEffectiveStatus so a tp/misclassified record whose prediction falls below the
  // confidence threshold is recounted as "fn" (its ground truth is now unmatched) instead
  // of just vanishing, which is why the fn badge grows as the threshold rises.
  const qualityCounts = useMemo(() => {
    const counts: Record<string, number> = { tp: 0, fp: 0, fn: 0, misclassified: 0 };
    for (const r of records) {
      if (filters.visibleSplits.size > 0 && !filters.visibleSplits.has(r.split)) continue;

      const classId = r.prediction?.class_id ?? r.ground_truth?.class_id ?? null;
      if (filters.visibleClasses.size > 0 && classId !== null && !filters.visibleClasses.has(classId)) {
        continue;
      }

      const effectiveStatus = getEffectiveStatus(r, filters);
      if (effectiveStatus && effectiveStatus in counts) counts[effectiveStatus]++;
    }
    return counts;
  }, [records, filters]);

  // ---- Handlers --------------------------------------------------------

  function toggleQuality(q: PredQuality) {
    const next = new Set(filters.qualities);
    if (next.has(q)) next.delete(q);
    else next.add(q);
    onChange({ ...filters, qualities: next });
  }

  function setGlobalConfidence(value: number) {
    onChange({ ...filters, minConfidence: value });
  }

  function setPerClassConfidence(classId: number, value: number) {
    const next = new Map(filters.perClassConfidence);
    // If back to global value, remove override to keep state clean.
    if (value === filters.minConfidence) next.delete(classId);
    else next.set(classId, value);
    onChange({ ...filters, perClassConfidence: next });
  }

  function toggleClass(classId: number) {
    const next = new Set(filters.visibleClasses);
    if (next.has(classId)) next.delete(classId);
    else next.add(classId);
    onChange({ ...filters, visibleClasses: next });
  }

  function toggleSplit(split: string) {
    const next = new Set(filters.visibleSplits);
    if (next.has(split)) next.delete(split);
    else next.add(split);
    onChange({ ...filters, visibleSplits: next });
  }

  const activeFilterCount =
    filters.qualities.size +
    filters.visibleClasses.size +
    filters.visibleSplits.size +
    (filters.minConfidence > 0 ? 1 : 0) +
    filters.perClassConfidence.size;

  function resetAll() {
    onChange(defaultFilterState());
  }

  // ---- Render ----------------------------------------------------------

  return (
    <aside className="fsb-sidebar">
      {/* Cabecera del sidebar: Logo y botón de cerrar */}
      <div className="fsb-top-bar">
        <button
          type="button"
          className="visualizer-sidebar-close"
          onClick={() => setSidebarOpen(false)}
          title="Cerrar filtros"
        >
          <PanelLeftClose size={20} />
        </button>
      </div>

      <div className="fsb-header">
        <span className="fsb-title">Filtros <Funnel size={18} /></span>
        {activeFilterCount > 0 && (
          <button type="button" className="fsb-reset" onClick={resetAll}>
            Borrar filtros ({activeFilterCount})
          </button>
        )}
      </div>

      {/* ── Prediction quality ── */}
      <Section title="Calidad de predicción" icon={ChartPie}>
        <div className="fsb-quality-pills">
          {ALL_QUALITIES.map((q) => {
            const active = filters.qualities.size === 0 || filters.qualities.has(q);
            return (
              <button
                key={q}
                type="button"
                className={`fsb-pill ${active ? "fsb-pill-active" : "fsb-pill-inactive"}`}
                style={active ? { background: QUALITY_COLORS[q], borderColor: QUALITY_COLORS[q] } : {}}
                onClick={() => toggleQuality(q)}
                title={`${QUALITY_LABELS[q]}: ${qualityCounts[q]} registros`}
              >
                {QUALITY_LABELS[q]}
                <span className="fsb-pill-count">{qualityCounts[q]}</span>
              </button>
            );
          })}
        </div>
      </Section>

      {/* ── Confidence ── */}
      <Section title="Confianza" icon={SlidersHorizontal}>
        <div className="conf-type-selector">
          <SegmentedControl
            options={[
              { value: "global", label: "Global" },
              { value: "perclass", label: "Por clase" },
            ]}
            value={confMode}
            onChange={setConfMode}
          />
        </div>

        {confMode === "global" && (
          <div className="fsb-conf-row">
            <label className="fsb-conf-label" htmlFor="fsb-global-conf">
              Global: {filters.minConfidence.toFixed(2)}
            </label>
            <input
              id="fsb-global-conf"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={filters.minConfidence}
              onChange={(e) => setGlobalConfidence(Number(e.currentTarget.value))}
              className="fsb-slider"
            />
          </div>
        )}

        {confMode === "perclass" && availableClasses.length > 0 && (
          <div className="fsb-per-class-conf">
            {availableClasses.map((id) => {
              const override = filters.perClassConfidence.get(id);
              const value = override !== undefined ? override : filters.minConfidence;
              const hasOverride = override !== undefined;
              return (
                <div key={id} className="fsb-conf-row fsb-conf-row-class">
                  <span
                    className={`fsb-conf-label fsb-class-label ${hasOverride ? "fsb-conf-override" : ""}`}
                    title={categories[id] ?? `Clase ${id}`}
                  >
                    {categories[id] ?? `Clase ${id}`}
                    {hasOverride && <span className="fsb-override-dot" />}
                  </span>
                  <span className="fsb-conf-value">{value.toFixed(2)}</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={value}
                    onChange={(e) => setPerClassConfidence(id, Number(e.currentTarget.value))}
                    className="fsb-slider"
                  />
                </div>
              );
            })}
          </div>
        )}

        {confMode === "perclass" && availableClasses.length === 0 && (
          <p className="fsb-conf-sub">No hay clases disponibles.</p>
        )}
      </Section>

      {/* ── Classes ── */}
      <Section title="Clases" icon={Tags}>
        <div className="fsb-class-list">
          {availableClasses.map((id) => {
            const checked = filters.visibleClasses.size === 0 || filters.visibleClasses.has(id);
            return (
              <label key={id} className="fsb-check-row">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleClass(id)}
                />
                <span className="fsb-check-label" title={categories[id] ?? `Clase ${id}`}>
                  {categories[id] ?? `Clase ${id}`}
                </span>
              </label>
            );
          })}
        </div>
      </Section>

      {/* ── Split ── */}
      <Section title="Split" icon={Layers}>
        <div className="fsb-class-list">
          {availableSplits.map((split) => {
            const checked = filters.visibleSplits.size === 0 || filters.visibleSplits.has(split);
            return (
              <label key={split} className="fsb-check-row">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleSplit(split)}
                />
                <span className="fsb-check-label">{split}</span>
              </label>
            );
          })}
        </div>
      </Section>
    </aside>
  );
}
