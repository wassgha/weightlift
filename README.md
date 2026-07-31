# weightlift

In-browser **ML model manager** with download progress tracking.

```ts
import { ModelManager } from "weightlift";

const models = new ModelManager();

models.define("siglip", {
  isCached: () => isOnnxCached(modelId),
  load: async ({ progress }) =>
    pipeline("zero-shot-image-classification", modelId, {
      progress_callback: (p) => {
        if (p.status === "progress_total" && p.total) {
          progress.dispatch({
            type: "progress_total",
            loaded: ((p.progress ?? 0) / 100) * p.total,
            total: p.total,
          });
        } else if (p.status === "progress" && p.file) {
          progress.dispatch({
            type: "progress",
            file: p.file,
            loaded: p.loaded ?? 0,
            total: p.total,
          });
        }
      },
    }),
});

const clip = await models.load("siglip");
```

Works the same for Whisper, CLIP, SigLIP, WebLLM, custom ONNX — `T` is whatever your `load` returns.

## Install

```bash
npm install weightlift
```

## Exports

| Import | What |
| --- | --- |
| `weightlift` | `ModelManager`, low-level `Weightlift` progress store |
| `weightlift/react` | `useModel` / `useModelManager` |
| `weightlift/worker` | `createWorkerReporter` / `attachWorker` |

Map your runtime’s progress callback into `progress.dispatch(...)` at the call site. UI strings stay in the app: compose from `status` + `fromCache`.

## ModelManager

| Method | Purpose |
| --- | --- |
| `define(id, definition)` | Register how to load (and optionally dispose) a model |
| `load(id)` / `load(id, definition)` | Load once; dedupe concurrent callers |
| `get(id)` | Sync access to a ready instance |
| `status(id)` | `{ status, percent, fromCache, loadedBytes, … }` |
| `isReady` / `isLoading` / `has` | Quick queries |
| `unload(id)` | Drop instance (calls `dispose`) |
| `preload([ids])` | Warm several models |
| `subscribe` / `getSnapshot` | Manager-wide snapshot |

### Definition

```ts
interface ModelDefinition<T> {
  load: (ctx: {
    id: string;
    progress: Weightlift;
    fromCache: boolean | null;
  }) => Promise<T>;
  isCached?: () => boolean | Promise<boolean>;  // → ModelRecord.fromCache
  dispose?: (value: T) => void | Promise<void>;
}
```

### UI-side Progress Display

```ts
const { status, fromCache, percent } = models.status("siglip");
const label =
  status === "loading"
    ? fromCache
      ? "Loading vision model from cache…"
      : "Downloading vision model…"
    : "";
```

## React

```tsx
import { useModel } from "weightlift/react";

function Bar({ manager }: { manager: ModelManager }) {
  const { percent, fromCache, isLoading, load } = useModel(manager, "siglip");
  const label = fromCache ? "Loading from cache…" : "Downloading…";
  // …
}
```

## Develop

```bash
npm install
npm test
npm run build
```

Publish:

```bash
npm publish
```

## License

MIT
