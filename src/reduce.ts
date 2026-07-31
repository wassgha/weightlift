import {
  INITIAL_STATE,
  type FileProgress,
  type WeightliftEvent,
  type WeightliftState,
} from "./types.js";

interface InternalTotals {
  /** Prefer runtime aggregate events when the caller emits them. */
  useTotalEvents: boolean;
  /** Highest percent observed; only moves forward except when totals grow. */
  best: number;
  lastTotal: number;
}

export interface ReduceContext {
  totals: InternalTotals;
}

export function createReduceContext(): ReduceContext {
  return {
    totals: { useTotalEvents: false, best: 0, lastTotal: 0 },
  };
}

function cloneFiles(
  files: Record<string, FileProgress>
): Record<string, FileProgress> {
  const next: Record<string, FileProgress> = {};
  for (const [k, v] of Object.entries(files)) {
    next[k] = { ...v };
  }
  return next;
}

function sumFiles(files: Record<string, FileProgress>): {
  loadedBytes: number;
  totalBytes: number | null;
} {
  // Only files that reported a Content-Length contribute to the percentage.
  // Files without a total are ignored for the bar — if *no* file has a total,
  // the store stays indeterminate.
  let loadedBytes = 0;
  let totalBytes = 0;
  let anyTotal = false;

  for (const f of Object.values(files)) {
    if (f.total != null && f.total > 0) {
      anyTotal = true;
      loadedBytes += Math.min(f.loaded, f.total);
      totalBytes += f.total;
    } else if (f.status === "done") {
      loadedBytes += f.loaded;
      totalBytes += f.loaded;
      if (f.loaded > 0) anyTotal = true;
    }
  }

  return {
    loadedBytes,
    totalBytes: anyTotal ? totalBytes : null,
  };
}

/**
 * Apply a monotonic percent update. When the known byte total grows mid-load
 * (common when more files are discovered), rescale `best` so the bar can move
 * backwards honestly instead of sticking at an inflated value.
 */
function reportPercent(
  ctx: ReduceContext,
  loaded: number,
  total: number
): { percent: number | null; indeterminate: boolean } {
  if (!(total > 0)) {
    return { percent: null, indeterminate: true };
  }
  const { totals } = ctx;
  if (total > totals.lastTotal && totals.lastTotal > 0 && totals.best > 0) {
    totals.best *= totals.lastTotal / total;
  }
  totals.lastTotal = total;
  totals.best = Math.max(totals.best, Math.min(1, loaded / total));
  return { percent: totals.best, indeterminate: false };
}

function withDerived(
  state: WeightliftState,
  ctx: ReduceContext,
  patch: Partial<WeightliftState>
): WeightliftState {
  const next: WeightliftState = { ...state, ...patch };

  if (!ctx.totals.useTotalEvents) {
    const { loadedBytes, totalBytes } = sumFiles(next.files);
    next.loadedBytes = loadedBytes;
    next.totalBytes = totalBytes;
    if (totalBytes != null && totalBytes > 0) {
      const reported = reportPercent(ctx, loadedBytes, totalBytes);
      next.percent = reported.percent;
      next.indeterminate = reported.indeterminate;
    } else {
      next.percent = null;
      next.indeterminate = true;
    }
  }

  return next;
}

/** Pure reducer: previous state + event → next state. */
export function reduce(
  state: WeightliftState,
  event: WeightliftEvent,
  ctx: ReduceContext
): WeightliftState {
  switch (event.type) {
    case "reset":
      ctx.totals = { useTotalEvents: false, best: 0, lastTotal: 0 };
      return { ...INITIAL_STATE, files: {} };

    case "start":
      return withDerived(state, ctx, {
        status: "loading",
        error: null,
      });

    case "initiate": {
      const files = cloneFiles(state.files);
      if (!files[event.file]) {
        files[event.file] = { loaded: 0, total: null, status: "pending" };
      }
      return withDerived(state, ctx, {
        status: state.status === "idle" ? "loading" : state.status,
        files,
        error: null,
      });
    }

    case "progress": {
      // Once the runtime is emitting aggregate totals, ignore per-file noise.
      if (ctx.totals.useTotalEvents) {
        return state;
      }
      const files = cloneFiles(state.files);
      const prev = files[event.file];
      const total =
        event.total != null && event.total > 0
          ? event.total
          : (prev?.total ?? null);
      files[event.file] = {
        loaded: event.loaded,
        total,
        status: "downloading",
      };
      return withDerived(state, ctx, {
        status: "loading",
        files,
        error: null,
      });
    }

    case "progress_total": {
      ctx.totals.useTotalEvents = true;
      const reported = reportPercent(ctx, event.loaded, event.total);
      return {
        ...state,
        status: "loading",
        loadedBytes: event.loaded,
        totalBytes: event.total,
        percent: reported.percent,
        indeterminate: reported.indeterminate,
        error: null,
      };
    }

    case "done": {
      const files = cloneFiles(state.files);
      const prev = files[event.file];
      files[event.file] = {
        loaded: prev?.total ?? prev?.loaded ?? 0,
        total: prev?.total ?? prev?.loaded ?? null,
        status: "done",
      };
      return withDerived(state, ctx, { files });
    }

    case "ready":
      return {
        ...state,
        status: "ready",
        percent: state.percent ?? 1,
        indeterminate: false,
        error: null,
      };

    case "error":
      return {
        ...state,
        status: "error",
        error: event.error,
      };

    default:
      return state;
  }
}
