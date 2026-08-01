import type { ModelDefinition } from "./manager.js";
import type { WeightliftEvent } from "./types.js";

/** Anything that can accept normalized events (store or worker reporter). */
export interface EventSink {
  dispatch(event: WeightliftEvent): void;
}

/**
 * Structural subset of transformers.js `ProgressInfo`.
 * No runtime dependency on `@huggingface/transformers`.
 *
 * @see https://huggingface.co/docs/transformers.js/guides/progress
 */
export interface TransformersProgressInfo {
  status?: string;
  file?: string;
  name?: string;
  loaded?: number;
  total?: number;
  /** 0..100 when present on `progress_total` events. */
  progress?: number;
}

/**
 * Convert a transformers.js `progress_callback` argument into a normalized
 * {@link WeightliftEvent}, or `null` if the event should be ignored.
 */
export function transformersEvent(
  p: TransformersProgressInfo
): WeightliftEvent | null {
  const status = p.status;

  if (status === "initiate" && p.file) {
    return { type: "initiate", file: p.file };
  }

  if (status === "progress_total") {
    const total = p.total ?? 0;
    if (!(total > 0)) return null;
    const loaded =
      typeof p.progress === "number"
        ? (p.progress / 100) * total
        : (p.loaded ?? 0);
    return { type: "progress_total", loaded, total };
  }

  if (status === "progress" && p.file) {
    return {
      type: "progress",
      file: p.file,
      loaded: p.loaded ?? 0,
      total: p.total,
    };
  }

  if (status === "done" && p.file) {
    return { type: "done", file: p.file };
  }

  if (status === "ready") {
    return { type: "ready" };
  }

  return null;
}

/**
 * Build a `progress_callback` for `pipeline()` / `from_pretrained()`.
 *
 * Prefer {@link transformersModel} when you want the full load path
 * (device pick + WebGPU→WASM retry). This is the lower-level binder.
 */
export function transformersProgress(
  sink: EventSink
): (progress: TransformersProgressInfo) => void {
  return (progress) => {
    const event = transformersEvent(progress);
    if (event) sink.dispatch(event);
  };
}

export interface TransformersCacheOptions {
  /**
   * Cache Storage name used by transformers.js (`env.cacheKey`).
   * Defaults to `"transformers-cache"`.
   */
  cacheKey?: string;
  /**
   * Substring that must appear in a cached request URL to count as a hit.
   * Defaults to `".onnx"` so tokenizer-only cache entries do not count.
   */
  fileMarker?: string;
  /**
   * Minimum total `Content-Length` of matching OK responses to count as cached.
   * Filters out redirect stubs and empty placeholder entries.
   * Defaults to `1_000_000` (1 MB).
   */
  minBytes?: number;
}

/**
 * Whether a model's ONNX weights look present in the transformers.js
 * browser cache. Useful for {@link ModelDefinition.isCached}.
 *
 * Returns `false` when Cache Storage is unavailable (SSR, private mode, etc.).
 * Requires real OK responses with enough `Content-Length` — a bare URL match
 * (e.g. a cached 302 to the Hub) is not enough.
 */
export async function isTransformersModelCached(
  modelId: string,
  options?: TransformersCacheOptions
): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  const cacheKey = options?.cacheKey ?? "transformers-cache";
  const fileMarker = options?.fileMarker ?? ".onnx";
  const minBytes = options?.minBytes ?? 1_000_000;
  try {
    const cache = await caches.open(cacheKey);
    const keys = await cache.keys();
    let total = 0;
    for (const req of keys) {
      if (!req.url.includes(modelId) || !req.url.includes(fileMarker)) {
        continue;
      }
      const res = await cache.match(req);
      if (!res || !res.ok) continue;

      const headerLen = Number(res.headers.get("content-length"));
      if (!Number.isFinite(headerLen) || headerLen <= 0) continue;

      total += headerLen;
      if (total >= minBytes) return true;
    }
    return false;
  } catch {
    return false;
  }
}

export type TransformersDevice = "webgpu" | "wasm";

/**
 * Shared device policy across models in one app / worker.
 * After a WebGPU failure (or an explicit {@link TransformersDevicePolicy.preferWasm}),
 * every subsequent pick returns `"wasm"`.
 */
export interface TransformersDevicePolicy {
  /** When true, {@link TransformersDevicePolicy.pickDevice} always returns `"wasm"`. */
  readonly forceWasm: boolean;
  pickDevice(): Promise<TransformersDevice>;
  /** Stick to WASM for the rest of this page / worker lifetime. */
  preferWasm(): void;
}

function createDevicePolicy(): TransformersDevicePolicy {
  let forceWasm = false;
  return {
    get forceWasm() {
      return forceWasm;
    },
    preferWasm() {
      forceWasm = true;
    },
    async pickDevice() {
      if (forceWasm) return "wasm";
      try {
        const gpu = (
          globalThis as typeof globalThis & {
            navigator?: {
              gpu?: { requestAdapter: () => Promise<unknown | null> };
            };
          }
        ).navigator?.gpu;
        if (gpu && (await gpu.requestAdapter())) return "webgpu";
      } catch {
        // fall through
      }
      return "wasm";
    },
  };
}

/**
 * Default shared device policy used by {@link transformersModel}.
 * Prefers WebGPU when available; call `preferWasm()` after a GPU loss so
 * every model sticks to WASM for the rest of the page / worker lifetime.
 */
export const fallbackDevicePolicy: TransformersDevicePolicy =
  createDevicePolicy();

/**
 * Injected `pipeline` from `@huggingface/transformers`.
 * Typed loosely so weightlift does not depend on that package — real
 * `pipeline` overloads are accepted without casts.
 */
export type TransformersPipelineFn = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  task: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelId: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  options?: any
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
) => Promise<any>;

export interface TransformersModelOptions<T = unknown> {
  /** Pass `pipeline` from `@huggingface/transformers`. */
  pipeline: TransformersPipelineFn;
  /** Task name, e.g. `"automatic-speech-recognition"`. */
  task: string;
  /** Hugging Face model id. */
  modelId: string;
  /**
   * dtype (or any per-device options) keyed by device. Merged into the
   * `pipeline()` call as `dtype: dtype[device]`.
   */
  dtype?: {
    webgpu?: unknown;
    wasm?: unknown;
  };
  /** Extra options merged into every `pipeline()` call. */
  pipelineOptions?: Record<string, unknown>;
  /**
   * Device policy. Defaults to the shared {@link fallbackDevicePolicy} —
   * you usually do not need to pass this.
   */
  devicePolicy?: TransformersDevicePolicy;
  /** Called whenever the resolved device is chosen (including WASM fallback). */
  onDevice?: (device: TransformersDevice) => void;
  /** Cache Storage name for {@link isTransformersModelCached}. */
  cacheKey?: string;
  /**
   * Try WebGPU first and retry once on WASM if `pipeline()` throws.
   * Default `true`.
   */
  webgpuFallback?: boolean;
  /** Optional dispose hook forwarded to the {@link ModelDefinition}. */
  dispose?: (value: T) => void | Promise<void>;
}

/**
 * Build a {@link ModelDefinition} that loads a transformers.js model:
 * device pick → `pipeline()` with progress wiring → WebGPU→WASM retry.
 *
 * Uses {@link fallbackDevicePolicy} by default (no need to pass one).
 * Inject `pipeline` from `@huggingface/transformers` so weightlift stays
 * dependency-free.
 *
 * ```ts
 * import { pipeline } from "@huggingface/transformers";
 * import { ModelManager } from "weightlift";
 * import { transformersModel } from "weightlift/transformers";
 *
 * const models = new ModelManager({
 *   models: {
 *     whisper: transformersModel({
 *       pipeline,
 *       task: "automatic-speech-recognition",
 *       modelId: "onnx-community/whisper-base_timestamped",
 *       dtype: { webgpu: {…}, wasm: {…} },
 *     }),
 *   },
 * });
 * ```
 */
export function transformersModel<T = unknown>(
  options: TransformersModelOptions<T>
): ModelDefinition<T> {
  const {
    pipeline: pipelineFn,
    task,
    modelId,
    dtype,
    pipelineOptions,
    onDevice,
    cacheKey,
    webgpuFallback = true,
    dispose,
  } = options;
  const devicePolicy = options.devicePolicy ?? fallbackDevicePolicy;

  return {
    isCached: () => isTransformersModelCached(modelId, { cacheKey }),
    dispose,
    load: async ({ progress }) => {
      const onProgress = transformersProgress(progress);
      const device = await devicePolicy.pickDevice();
      onDevice?.(device);

      const run = (dev: TransformersDevice) =>
        pipelineFn(task, modelId, {
          ...pipelineOptions,
          ...(dtype?.[dev] !== undefined ? { dtype: dtype[dev] } : {}),
          device: dev,
          progress_callback: onProgress,
        }) as Promise<T>;

      try {
        return await run(device);
      } catch (err) {
        if (webgpuFallback && device === "webgpu") {
          devicePolicy.preferWasm();
          onDevice?.("wasm");
          return await run("wasm");
        }
        throw err;
      }
    },
  };
}
