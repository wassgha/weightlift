/** High-level load lifecycle. */
export type LoadStatus = "idle" | "loading" | "ready" | "error";

/** Per-file download / cache-read progress. */
export interface FileProgress {
  loaded: number;
  /** `null` when the server omitted Content-Length. */
  total: number | null;
  status: "pending" | "downloading" | "done";
}

/**
 * Aggregated model-load progress.
 *
 * `percent` is `null` (and `indeterminate` is true) when no usable byte total
 * is known yet — files can download in parallel and some servers omit
 * Content-Length, so a single 0..1 bar is not always available.
 *
 * UI copy ("Downloading…") is intentionally not part of this state — compose
 * labels in the app from `status` / `fromCache` on {@link ModelRecord}.
 */
export interface WeightliftState {
  status: LoadStatus;
  files: Record<string, FileProgress>;
  loadedBytes: number;
  /** Sum of known file totals; `null` if none reported a total. */
  totalBytes: number | null;
  /** 0..1 progress, or `null` when indeterminate. */
  percent: number | null;
  indeterminate: boolean;
  error: Error | null;
}

/**
 * Normalized progress events. Map your runtime's callback into these and
 * `dispatch` them — Transformers.js, WebLLM, whisper.cpp, custom fetch, etc.
 */
export type WeightliftEvent =
  | { type: "start" }
  | { type: "initiate"; file: string }
  | {
      type: "progress";
      file: string;
      loaded: number;
      /** Omit or pass `undefined` when Content-Length is missing. */
      total?: number;
    }
  /**
   * Runtime-provided aggregate (e.g. transformers.js `progress_total`).
   * Preferred over summing per-file events when available — totals are often
   * pre-seeded so the bar does not jump to 100% after the first file finishes.
   */
  | { type: "progress_total"; loaded: number; total: number }
  | { type: "done"; file: string }
  | { type: "ready" }
  | { type: "error"; error: Error }
  | { type: "reset" };

export type WeightliftListener = (state: WeightliftState) => void;

export const INITIAL_STATE: WeightliftState = {
  status: "idle",
  files: {},
  loadedBytes: 0,
  totalBytes: null,
  percent: null,
  indeterminate: true,
  error: null,
};
