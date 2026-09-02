/**
 * Helpers to estimate the remaining time of a long running, image-by-image job
 * (e.g. embedding extraction) from the progress samples the frontend polls.
 */

interface EtaSample {
  atMs: number;
  processed: number;
}

/** Minimum observation span before an estimate is considered meaningful. */
const MIN_OBSERVATION_MS = 3000;

/**
 * Sliding-window throughput estimator.
 *
 * Keeps the progress samples observed during the last `windowMs` milliseconds and
 * derives the remaining time from the average throughput inside that window, so the
 * estimate adapts when the processing speed changes.
 */
export class EtaEstimator {
  private samples: EtaSample[] = [];

  constructor(private readonly windowMs: number = 60_000) {}

  /** Drop every recorded sample, e.g. when a new job starts. */
  reset(): void {
    this.samples = [];
  }

  /**
   * Record a new progress observation and return the estimated remaining seconds.
   *
   * @param processed Number of items already processed.
   * @param total Total number of items to process.
   * @param atMs Observation timestamp in milliseconds.
   * @returns Estimated remaining seconds, or null when there is not enough data yet.
   */
  update(processed: number, total: number, atMs: number = Date.now()): number | null {
    this.samples.push({ atMs, processed });
    this.samples = this.samples.filter((sample) => atMs - sample.atMs <= this.windowMs);
    if (this.samples.length < 2) return null;

    const oldest = this.samples[0];
    const elapsedMs = atMs - oldest.atMs;
    const processedDelta = processed - oldest.processed;
    if (elapsedMs < MIN_OBSERVATION_MS || processedDelta <= 0) return null;

    const remaining = Math.max(total - processed, 0);
    if (remaining === 0) return 0;

    const itemsPerMs = processedDelta / elapsedMs;
    return remaining / itemsPerMs / 1000;
  }
}

/**
 * Format a duration in seconds as a short, human-readable Spanish label.
 *
 * @param seconds Duration to format.
 * @returns Localized duration label (e.g. "2 min 05 s").
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "unos segundos";

  const totalSeconds = Math.round(seconds);
  if (totalSeconds < 60) return `${Math.max(totalSeconds, 1)} s`;

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours > 0) return `${hours} h ${String(minutes).padStart(2, "0")} min`;

  const remainderSeconds = totalSeconds % 60;
  return `${minutes} min ${String(remainderSeconds).padStart(2, "0")} s`;
}
