import { Weightlift, type Unsubscribe } from "./store.js";
import type { LoadStatus, WeightliftState } from "./types.js";

/** Context passed to a model definition's `load` function. */
export interface LoadContext {
  /** Stable model key passed to {@link ModelManager.load}. */
  id: string;
  /**
   * Per-model progress store. Map your runtime's progress callback into
   * `progress.dispatch({ type: "progress_total" | "progress" | … })`.
   */
  progress: Weightlift;
  /**
   * Result of `isCached()` when provided; `null` if not checked.
   * Compose UI copy from this in the app — weightlift does not own strings.
   */
  fromCache: boolean | null;
  /** Optional abort signal if the manager starts supporting cancellation. */
  signal?: AbortSignal;
}

/** How to materialize (and optionally dispose) one model. */
export interface ModelDefinition<T = unknown> {
  /** Fetch / construct the model. Called at most once until unloaded. */
  load: (ctx: LoadContext) => Promise<T>;
  /**
   * Detect whether weights are already in a browser cache (Cache Storage,
   * IndexedDB, etc.). Exposed on {@link ModelRecord.fromCache} so the app
   * can choose its own labeling; does not affect load behavior.
   */
  isCached?: () => boolean | Promise<boolean>;
  /** Release GPU/WASM resources when {@link ModelManager.unload} is called. */
  dispose?: (value: T) => void | Promise<void>;
}

/** Initial registry passed to {@link ModelManager}'s constructor. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ModelRegistryMap = Record<string, ModelDefinition<any>>;

export interface ModelManagerOptions {
  /** Models to register up front. Prefer this over defining at `load()` time. */
  models?: ModelRegistryMap;
}

/** UI-facing per-model record (no heavyweight model handle). */
export interface ModelRecord {
  id: string;
  status: LoadStatus;
  /** `true` / `false` after cache check; `null` if not checked or idle. */
  fromCache: boolean | null;
  percent: number | null;
  indeterminate: boolean;
  loadedBytes: number;
  totalBytes: number | null;
  error: Error | null;
  /** Raw progress file map when you need a detailed breakdown. */
  files: WeightliftState["files"];
}

/** Aggregated manager snapshot for `useSyncExternalStore`. */
export interface ManagerSnapshot {
  models: Record<string, ModelRecord>;
  /** Ids currently in `loading`. */
  loading: string[];
  /** Ids in `ready`. */
  ready: string[];
  /** Ids in `error`. */
  errors: string[];
}

export type ManagerListener = (snapshot: ManagerSnapshot) => void;

interface Entry<T = unknown> {
  id: string;
  definition: ModelDefinition<T>;
  progress: Weightlift;
  fromCache: boolean | null;
  promise: Promise<T> | null;
  value: T | undefined;
  unsubProgress: Unsubscribe;
}

function recordFrom(entry: Entry): ModelRecord {
  const p = entry.progress.getSnapshot();
  return {
    id: entry.id,
    status: p.status,
    fromCache: entry.fromCache,
    percent: p.percent,
    indeterminate: p.indeterminate,
    loadedBytes: p.loadedBytes,
    totalBytes: p.totalBytes,
    error: p.error,
    files: p.files,
  };
}

function idleRecord(id: string): ModelRecord {
  return {
    id,
    status: "idle",
    fromCache: null,
    percent: null,
    indeterminate: true,
    loadedBytes: 0,
    totalBytes: null,
    error: null,
    files: {},
  };
}

function buildSnapshot(entries: Map<string, Entry>): ManagerSnapshot {
  const models: Record<string, ModelRecord> = {};
  const loading: string[] = [];
  const ready: string[] = [];
  const errors: string[] = [];
  for (const [id, entry] of entries) {
    const rec = recordFrom(entry);
    models[id] = rec;
    if (rec.status === "loading") loading.push(id);
    else if (rec.status === "ready") ready.push(id);
    else if (rec.status === "error") errors.push(id);
  }
  return { models, loading, ready, errors };
}

const EMPTY_SNAPSHOT: ManagerSnapshot = {
  models: {},
  loading: [],
  ready: [],
  errors: [],
};

/**
 * In-browser ML **model registry**: register loaders up front (or via
 * {@link ModelManager.define} / {@link ModelManager.remove}), then
 * `load` / `get` / `unload` by id.
 *
 * Runtime-agnostic — `T` can be a Whisper pipeline, CLIP/SigLIP encoder,
 * WebLLM engine, or any other handle. weightlift only owns lifecycle + bytes.
 *
 * ```ts
 * const models = new ModelManager({
 *   models: {
 *     siglip: {
 *       load: async ({ progress }) =>
 *         pipeline("zero-shot-image-classification", modelId, {
 *           progress_callback: (p) => {
 *             // map runtime events → progress.dispatch(...)
 *           },
 *         }),
 *     },
 *   },
 * });
 *
 * const clip = await models.load("siglip");
 * ```
 */
export class ModelManager {
  #entries = new Map<string, Entry>();
  #listeners = new Set<ManagerListener>();
  #snapshot: ManagerSnapshot = EMPTY_SNAPSHOT;

  constructor(options?: ModelManagerOptions) {
    if (options?.models) {
      for (const [id, definition] of Object.entries(options.models)) {
        this.define(id, definition);
      }
    }
  }

  /** Register (or replace) how a model id is loaded. */
  define<T>(id: string, definition: ModelDefinition<T>): this {
    const existing = this.#entries.get(id);
    if (existing?.promise && existing.progress.status === "loading") {
      throw new Error(
        `weightlift: cannot redefine "${id}" while a load is in flight`
      );
    }
    if (existing) {
      existing.unsubProgress();
      void this.#disposeEntry(existing);
    }

    const progress = new Weightlift();
    const entry: Entry<T> = {
      id,
      definition,
      progress,
      fromCache: null,
      promise: null,
      value: undefined,
      unsubProgress: progress.subscribe(() => this.#emit()),
    };
    this.#entries.set(id, entry as Entry);
    this.#emit();
    return this;
  }

  /**
   * Remove a model from the registry (disposes a ready instance if any).
   * Different from {@link unload}, which keeps the definition for a later load.
   */
  async remove(id: string): Promise<void> {
    const entry = this.#entries.get(id);
    if (!entry) return;
    if (entry.promise && entry.progress.isLoading) {
      throw new Error(
        `weightlift: cannot remove "${id}" while a load is in flight`
      );
    }
    entry.unsubProgress();
    await this.#disposeEntry(entry);
    this.#entries.delete(id);
    this.#emit();
  }

  /** Registered model ids. */
  ids(): string[] {
    return [...this.#entries.keys()];
  }

  /**
   * Load a registered model by id. Concurrent callers share one promise.
   * The model must already be in the registry (`define` or constructor).
   */
  load<T = unknown>(id: string): Promise<T> {
    const entry = this.#entries.get(id) as Entry<T> | undefined;
    if (!entry) {
      return Promise.reject(
        new Error(
          `weightlift: unknown model "${id}". Register it in the constructor or via define().`
        )
      );
    }
    if (entry.value !== undefined && entry.progress.isReady) {
      return Promise.resolve(entry.value);
    }
    if (entry.promise) return entry.promise;

    entry.promise = this.#runLoad(entry);
    return entry.promise;
  }

  async #runLoad<T>(entry: Entry<T>): Promise<T> {
    const { definition, progress, id } = entry;
    progress.reset();
    entry.fromCache = null;

    let fromCache: boolean | null = null;
    if (definition.isCached) {
      try {
        fromCache = Boolean(await definition.isCached());
      } catch {
        fromCache = null;
      }
    }
    entry.fromCache = fromCache;
    progress.start();
    this.#emit();

    try {
      const value = await definition.load({
        id,
        progress,
        fromCache,
      });
      entry.value = value;
      progress.ready();
      this.#emit();
      return value;
    } catch (err) {
      entry.promise = null;
      entry.value = undefined;
      progress.fail(err);
      this.#emit();
      throw err;
    }
  }

  /** Synchronously return a ready model, or `undefined`. */
  get<T = unknown>(id: string): T | undefined {
    const entry = this.#entries.get(id);
    if (!entry || !entry.progress.isReady) return undefined;
    return entry.value as T | undefined;
  }

  /** Whether a definition exists for this id. */
  has(id: string): boolean {
    return this.#entries.has(id);
  }

  isReady(id: string): boolean {
    return this.#entries.get(id)?.progress.isReady ?? false;
  }

  isLoading(id: string): boolean {
    return this.#entries.get(id)?.progress.isLoading ?? false;
  }

  /** Per-model UI record (idle stub if unknown). */
  status(id: string): ModelRecord {
    const entry = this.#entries.get(id);
    return entry ? recordFrom(entry) : idleRecord(id);
  }

  /** Underlying progress store for advanced use. */
  progress(id: string): Weightlift | undefined {
    return this.#entries.get(id)?.progress;
  }

  /**
   * Drop a cached instance (and call `dispose` if provided) so the next
   * `load()` runs the definition again. Keeps the model in the registry.
   */
  async unload(id: string): Promise<void> {
    const entry = this.#entries.get(id);
    if (!entry) return;
    await this.#disposeEntry(entry);
    entry.promise = null;
    entry.value = undefined;
    entry.fromCache = null;
    entry.progress.reset();
    this.#emit();
  }

  /** Unload every cached instance; definitions stay registered. */
  async unloadAll(): Promise<void> {
    await Promise.all(this.ids().map((id) => this.unload(id)));
  }

  /** Unload and remove every model from the registry. */
  async clear(): Promise<void> {
    const ids = [...this.#entries.keys()];
    for (const id of ids) {
      const entry = this.#entries.get(id);
      if (!entry) continue;
      entry.unsubProgress();
      await this.#disposeEntry(entry);
    }
    this.#entries.clear();
    this.#emit();
  }

  /** Warm multiple registered models (settles when all have finished or failed). */
  async preload(ids: string[]): Promise<void> {
    await Promise.all(ids.map((id) => this.load(id)));
  }

  getSnapshot = (): ManagerSnapshot => this.#snapshot;

  getServerSnapshot = (): ManagerSnapshot => EMPTY_SNAPSHOT;

  subscribe = (listener: ManagerListener): Unsubscribe => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };

  async #disposeEntry(entry: Entry): Promise<void> {
    if (entry.value !== undefined && entry.definition.dispose) {
      try {
        await entry.definition.dispose(entry.value);
      } catch {
        // dispose errors should not block unload
      }
    }
  }

  #emit(): void {
    this.#snapshot = buildSnapshot(this.#entries);
    for (const listener of this.#listeners) {
      listener(this.#snapshot);
    }
  }
}

export { idleRecord };
