# weightlift

Load and cache ML models in the browser, with download progress you can show in the UI.

**[Live demo](https://wassgha.github.io/weightlift/)** · `npm install weightlift`

```bash
npm install weightlift
```

## Quick start

### Transformers.js

```ts
import { pipeline } from "@huggingface/transformers";
import { ModelManager } from "weightlift";
import { transformersModel } from "weightlift/transformers";

const models = new ModelManager({
  models: {
    siglip: transformersModel({
      pipeline,
      task: "zero-shot-image-classification",
      modelId: "Xenova/clip-vit-base-patch16",
      dtype: { webgpu: "fp16", wasm: "fp32" },
    }),
  },
});

const clip = await models.load("siglip");
```

`transformersModel` picks WebGPU when available (falls back to WASM), wires progress callbacks, and detects cache hits. After a GPU device loss:

```ts
import { fallbackDevicePolicy } from "weightlift/transformers";

fallbackDevicePolicy.preferWasm();
await models.unloadAll();
```

### Any other runtime

Register a `ModelDefinition` that loads your model and reports progress:

```ts
const models = new ModelManager({
  models: {
    whisper: {
      load: async ({ progress }) => {
        progress.dispatch({ type: "start" });
        // …fetch / init your model, dispatch progress events…
        progress.dispatch({ type: "ready" });
        return myModel;
      },
      isCached: () => /* optional */,
      dispose: (model) => model.destroy?.(),
    },
  },
});

const whisper = await models.load("whisper");
```

## Packages

| Import | Exports |
| --- | --- |
| `weightlift` | `ModelManager`, `Weightlift` |
| `weightlift/react` | `useModel`, `useModelManager`, … |
| `weightlift/worker` | `createWorkerReporter`, `attachWorker` |
| `weightlift/transformers` | `transformersModel`, `fallbackDevicePolicy`, … |

React is an optional peer dependency. `@huggingface/transformers` is not a dependency — pass `pipeline` in yourself.

## ModelManager

```ts
const models = new ModelManager({ models: { /* … */ } });

await models.load("siglip");   // dedupes concurrent callers
models.get("siglip");          // sync access once ready
models.status("siglip");       // { status, percent, fromCache, … }
await models.unload("siglip");
await models.preload(["a", "b"]);
```

| Method | Purpose |
| --- | --- |
| `define(id, definition)` | Add or replace a model |
| `remove(id)` | Drop definition and dispose instance |
| `load(id)` | Load (or return in-flight promise) |
| `get(id)` | Sync access to a ready instance |
| `status(id)` | Progress + lifecycle for one model |
| `isReady` / `isLoading` / `has` / `ids` | Queries |
| `unload(id)` / `unloadAll()` | Dispose instances; keep definitions |
| `clear()` | Unload and remove all definitions |
| `preload([ids])` | Warm several models |
| `subscribe` / `getSnapshot` | Subscribe to manager-wide state |

### Progress in the UI

```ts
const { status, fromCache, percent } = models.status("siglip");

const label =
  status === "loading"
    ? fromCache
      ? "Loading from cache…"
      : "Downloading…"
    : "";
```

## React

```tsx
import { useModel } from "weightlift/react";

function ProgressBar({ manager }: { manager: ModelManager }) {
  const { percent, fromCache, isLoading, load } = useModel(manager, "siglip");

  return (
    <button onClick={() => load()} disabled={isLoading}>
      {isLoading
        ? fromCache
          ? "Loading from cache…"
          : `Downloading… ${Math.round((percent ?? 0) * 100)}%`
        : "Load model"}
    </button>
  );
}
```

`useModelManager(manager)` subscribes to the full snapshot. `useModelManagerStore(options)` creates a manager that lives for the component’s lifetime.

## Workers

Load models off the main thread and mirror progress back:

```ts
// worker.ts
import { Weightlift } from "weightlift";
import { createWorkerReporter } from "weightlift/worker";

const progress = new Weightlift();
const reporter = createWorkerReporter(self, progress);

// main thread
import { Weightlift } from "weightlift";
import { attachWorker } from "weightlift/worker";

const progress = new Weightlift();
const detach = attachWorker(worker, progress);
```

## Develop

```bash
npm install
npm test
npm run build
npm run demo:dev   # local Vite demo
```

## License

MIT
