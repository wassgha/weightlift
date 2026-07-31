import type { Weightlift } from "./store.js";
import type { WeightliftEvent, WeightliftState } from "./types.js";

/** Discriminator so host apps can multiplex other worker messages safely. */
export const WEIGHTLIFT_MESSAGE_TYPE = "weightlift" as const;

export type WeightliftWorkerMessage =
  | { type: typeof WEIGHTLIFT_MESSAGE_TYPE; kind: "event"; event: WeightliftEvent }
  | {
      type: typeof WEIGHTLIFT_MESSAGE_TYPE;
      kind: "state";
      state: WeightliftState;
    };

/** Minimal sink that accepts normalized events (store or reporter). */
export interface EventSink {
  dispatch(event: WeightliftEvent): void;
}

function isWeightliftMessage(data: unknown): data is WeightliftWorkerMessage {
  return (
    typeof data === "object" &&
    data !== null &&
    (data as { type?: unknown }).type === WEIGHTLIFT_MESSAGE_TYPE
  );
}

export interface PostMessageTarget {
  postMessage(message: unknown, transfer?: Transferable[]): void;
}

export interface WorkerReporter {
  /** Dispatch into the local store and forward across the worker boundary. */
  dispatch(event: WeightliftEvent): void;
  /** Stop forwarding (store continues to work locally). */
  unsubscribe: () => void;
}

/**
 * Worker-side helper: every update is forwarded to the main thread as a
 * serializable {@link WeightliftWorkerMessage}.
 *
 * Default `mode: "state"` posts snapshots. `mode: "event"` forwards raw
 * events so the main thread can rehydrate via {@link attachWorker}.
 */
export function createWorkerReporter(
  post: PostMessageTarget | ((message: unknown) => void),
  weightlift: Weightlift,
  options?: { mode?: "state" | "event" }
): WorkerReporter {
  const mode = options?.mode ?? "state";
  const postMessage =
    typeof post === "function" ? post : (msg: unknown) => post.postMessage(msg);

  if (mode === "event") {
    return {
      dispatch(event: WeightliftEvent) {
        weightlift.dispatch(event);
        const msg: WeightliftWorkerMessage = {
          type: WEIGHTLIFT_MESSAGE_TYPE,
          kind: "event",
          event,
        };
        postMessage(msg);
      },
      unsubscribe: () => {
        /* nothing to detach in event mode */
      },
    };
  }

  const unsubscribe = weightlift.subscribe((state) => {
    const msg: WeightliftWorkerMessage = {
      type: WEIGHTLIFT_MESSAGE_TYPE,
      kind: "state",
      state,
    };
    postMessage(msg);
  });

  return {
    dispatch(event: WeightliftEvent) {
      weightlift.dispatch(event);
    },
    unsubscribe,
  };
}

/**
 * Main-thread helper: listen for weightlift messages from a worker and apply
 * them onto a local {@link Weightlift} store.
 */
export function attachWorker(
  worker: {
    addEventListener: (
      type: "message",
      listener: (ev: MessageEvent) => void
    ) => void;
    removeEventListener: (
      type: "message",
      listener: (ev: MessageEvent) => void
    ) => void;
  },
  weightlift: Weightlift,
  options?: {
    onState?: (state: WeightliftState) => void;
  }
): () => void {
  const onMessage = (ev: MessageEvent) => {
    if (!isWeightliftMessage(ev.data)) return;
    if (ev.data.kind === "event") {
      weightlift.dispatch(ev.data.event);
      return;
    }
    options?.onState?.(ev.data.state);
    mirrorState(weightlift, ev.data.state);
  };

  worker.addEventListener("message", onMessage);
  return () => worker.removeEventListener("message", onMessage);
}

/** Best-effort projection of a remote snapshot onto a local store. */
function mirrorState(weightlift: Weightlift, state: WeightliftState): void {
  weightlift.reset();
  if (state.status === "idle") return;

  weightlift.start();

  if (state.totalBytes != null && state.totalBytes > 0) {
    weightlift.dispatch({
      type: "progress_total",
      loaded: state.loadedBytes,
      total: state.totalBytes,
    });
  } else {
    for (const [file, fp] of Object.entries(state.files)) {
      weightlift.dispatch({ type: "initiate", file });
      weightlift.dispatch({
        type: "progress",
        file,
        loaded: fp.loaded,
        total: fp.total ?? undefined,
      });
      if (fp.status === "done") {
        weightlift.dispatch({ type: "done", file });
      }
    }
  }

  if (state.status === "ready") weightlift.ready();
  if (state.status === "error") {
    weightlift.fail(state.error ?? new Error("load failed"));
  }
}

export { isWeightliftMessage };
