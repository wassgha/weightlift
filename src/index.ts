/**
 * weightlift — in-browser ML model registry with download progress.
 *
 * @packageDocumentation
 *
 * Runtime-agnostic: Whisper, CLIP, SigLIP, WebLLM, custom ONNX — anything
 * that can resolve a promise and optionally report byte progress.
 *
 * ```ts
 * import { ModelManager } from "weightlift";
 *
 * import { transformersModel } from "weightlift/transformers";
 *
 * const models = new ModelManager({
 *   models: {
 *     siglip: transformersModel({
 *       pipeline,
 *       task: "zero-shot-image-classification",
 *       modelId,
 *     }),
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
  type ModelRegistryMap,
  type ModelManagerOptions,
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
