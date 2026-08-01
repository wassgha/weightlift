"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import {
  ModelManager,
  idleRecord,
  type ManagerSnapshot,
  type ModelManagerOptions,
  type ModelRecord,
} from "./manager.js";
import { Weightlift } from "./store.js";
import type { WeightliftState } from "./types.js";

export type {
  WeightliftState,
  LoadStatus,
  FileProgress,
} from "./types.js";
export type {
  ModelRecord,
  ManagerSnapshot,
  ModelDefinition,
  ModelRegistryMap,
  ModelManagerOptions,
  LoadContext,
} from "./manager.js";
export {
  Weightlift,
  createModelLoader,
  createModelRegistry,
} from "./store.js";
export { ModelManager } from "./manager.js";

/** Subscribe to the full {@link ModelManager} snapshot. */
export function useModelManager(manager: ModelManager): ManagerSnapshot {
  return useSyncExternalStore(
    manager.subscribe,
    manager.getSnapshot,
    manager.getServerSnapshot
  );
}

/**
 * Subscribe to one model id on a {@link ModelManager}.
 *
 * Compose UI labels from `status` / `fromCache` in the component — the
 * package does not ship copy.
 */
export function useModel<T = unknown>(
  manager: ModelManager,
  id: string
): ModelRecord & {
  isLoading: boolean;
  isReady: boolean;
  isIdle: boolean;
  hasError: boolean;
  value: T | undefined;
  load: () => Promise<T>;
  unload: () => Promise<void>;
} {
  const snapshot = useModelManager(manager);
  const record = snapshot.models[id] ?? idleRecord(id);

  const load = useCallback(() => manager.load<T>(id), [manager, id]);
  const unload = useCallback(() => manager.unload(id), [manager, id]);

  return {
    ...record,
    isLoading: record.status === "loading",
    isReady: record.status === "ready",
    isIdle: record.status === "idle",
    hasError: record.status === "error",
    value: manager.get<T>(id),
    load,
    unload,
  };
}

/**
 * Stable {@link ModelManager} for the lifetime of the component.
 * `options` are read once on mount (seed the registry there).
 */
export function useModelManagerStore(options?: ModelManagerOptions): ModelManager {
  return useMemo(() => new ModelManager(options), []);
}

/** Low-level progress store subscription. Prefer {@link useModel}. */
export function useWeightlift(weightlift: Weightlift): WeightliftState & {
  isLoading: boolean;
  isReady: boolean;
  isIdle: boolean;
  hasError: boolean;
} {
  const state = useSyncExternalStore(
    weightlift.subscribe,
    weightlift.getSnapshot,
    weightlift.getServerSnapshot
  );

  return {
    ...state,
    isLoading: state.status === "loading",
    isReady: state.status === "ready",
    isIdle: state.status === "idle",
    hasError: state.status === "error",
  };
}

export function useWeightliftStore(key?: string): Weightlift {
  return useMemo(() => new Weightlift(), [key]);
}
