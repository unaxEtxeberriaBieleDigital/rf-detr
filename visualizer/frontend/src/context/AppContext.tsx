import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

/**
 * Global, app-wide configuration decided on the setup screen. Kept in a single React
 * context so both the setup page and the visualizer page share the same source of truth
 * without prop drilling or re-fetching the user's choices.
 */
export interface AppConfig {
  datasetPath: string;
  datasetType: string;
  modelPath: string;
  modelType: string;
  jobId: string;
  categories: Record<number, string>;
  /** Whether dimensionality reduction has already been computed for this job. */
  hasDimensionalityReduction: boolean;
  /**
   * Number of reduced dimensions (2 or 3), or null if dimensionality
   * reduction hasn't been computed.
   */
  dimensionalityReductionComponents: number | null;
}

interface AppContextValue {
  config: AppConfig | null;
  setConfig: (config: AppConfig) => void;
  reset: () => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [config, setConfigState] = useState<AppConfig | null>(null);

  const value = useMemo<AppContextValue>(
    () => ({
      config,
      setConfig: (next: AppConfig) => setConfigState(next),
      reset: () => setConfigState(null),
    }),
    [config],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppConfig(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error("useAppConfig must be used within an AppProvider");
  }
  return ctx;
}
