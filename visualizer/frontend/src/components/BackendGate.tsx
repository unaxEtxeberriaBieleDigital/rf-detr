import { useEffect, useState, type ReactNode } from "react";
import { checkBackendHealth } from "../api/client";
import Beams from "./DefaultBackground";
import bieleLogo from "../assets/logos/biele-logo.png";

const HEALTH_POLL_INTERVAL_MS = 600;
/** After this long the message tells the user that the startup is taking a while. */
const SLOW_STARTUP_THRESHOLD_MS = 15_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Block the UI until the backend answers its health endpoint.
 *
 * The Tauri window opens long before the Python backend has finished importing its
 * dependencies, so without this gate the user sees a fully interactive setup form
 * whose requests silently fail.
 *
 * @param children Application UI rendered once the backend is ready.
 */
export default function BackendGate({ children }: { children: ReactNode }) {
  const [isReady, setIsReady] = useState(false);
  const [waitedMs, setWaitedMs] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const startedAtMs = Date.now();

    async function pollUntilReady(): Promise<void> {
      while (!cancelled) {
        const healthy = await checkBackendHealth();
        if (cancelled) return;
        if (healthy) {
          setIsReady(true);
          return;
        }
        setWaitedMs(Date.now() - startedAtMs);
        await sleep(HEALTH_POLL_INTERVAL_MS);
      }
    }

    void pollUntilReady();
    return () => {
      cancelled = true;
    };
  }, []);

  if (isReady) return <>{children}</>;

  const isSlow = waitedMs >= SLOW_STARTUP_THRESHOLD_MS;

  return (
    <div className="backend-gate" role="status" aria-live="polite" aria-busy="true">
      <Beams
        beamWidth={3}
        beamHeight={30}
        beamNumber={20}
        lightColor="#ffffff"
        speed={2}
        noiseIntensity={1.75}
        scale={0.2}
        rotation={30}
        centralContent={
          <div className="backend-gate-center">
            <img src={bieleLogo} alt="Biele" className="beams-logo" />
            <div className="loader" />
            <p className="backend-gate-title">Starting application...</p>
            <p className="backend-gate-subtitle">
              {isSlow
                ? `Preparing the model and the dependencies. Still starting up (${Math.round(waitedMs / 1000)} s).`
                : "Preparing the model and the dependencies. This might take a few seconds."}
            </p>
          </div>
        }
      />
    </div>
  );
}
