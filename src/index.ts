/**
 * weightlift — in-browser ML model manager with download progress.
 *
 * @packageDocumentation
 *
 * Runtime-agnostic: Whisper, CLIP, SigLIP, WebLLM, custom ONNX — anything
 * that can resolve a promise and optionally report byte progress.
 *
 * ```ts
 * import { ModelManager } from "weightlift";
 *
 * const models = new ModelManager();
 * models.define("siglip", {
 *   load: async ({ progress }) => {
 *     return pipeline("zero-shot-image-classification", modelId, {
 *       progress_callback: (p) => {
 *         if (p.status === "progress_total" && p.total) {
 *           progress.dispatch({
 *             type: "progress_total",
 *             loaded: ((p.progress ?? 0) / 100) * p.total,
 *             total: p.total,
 *           });
 *         }
 *       },
 *     });
 *   },
 * });
 *
 * const clip = await models.load("siglip");
 * ```
 */

export {
  ModelManager,
  type LoadContext,
  type ModelDefinition,
  type ModelRecord,
  type ManagerSnapshot,
  type ManagerListener,
} from "./manager.js";

export {
  Weightlift,
  createModelLoader,
  createModelRegistry,
  type Unsubscribe,
} from "./store.js";

export {
  reduce,
  createReduceContext,
  type ReduceContext,
} from "./reduce.js";

export {
  INITIAL_STATE,
  type LoadStatus,
  type FileProgress,
  type WeightliftState,
  type WeightliftEvent,
  type WeightliftListener,
} from "./types.js";
