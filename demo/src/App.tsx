import { useMemo, useState } from "react";
import { ModelManager } from "weightlift";
import { useModel } from "weightlift/react";

const MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

type Sentiment = { label: string; score: number };
type Classifier = (input: string) => Promise<Sentiment[]>;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

async function createSentimentDefinition() {
  const [{ env, pipeline }, { transformersModel }] = await Promise.all([
    import("@huggingface/transformers"),
    import("weightlift/transformers"),
  ]);

  env.allowLocalModels = false;
  env.useBrowserCache = true;
  if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.proxy = false;
  }

  return transformersModel<Classifier>({
    pipeline,
    task: "sentiment-analysis",
    modelId: MODEL_ID,
    dtype: { webgpu: "q8", wasm: "q8" },
  });
}

export function App() {
  const manager = useMemo(() => new ModelManager(), []);
  const model = useModel<Classifier>(manager, "sentiment");

  const [text, setText] = useState(
    "Weightlifting models in the browser feels light."
  );
  const [result, setResult] = useState<Sentiment | null>(null);
  const [running, setRunning] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(false);

  const busy = model.isLoading || bootstrapping;
  const percent = model.percent != null ? Math.round(model.percent * 100) : null;
  const statusLabel = bootstrapping
    ? "Preparing runtime"
    : model.isLoading
      ? model.fromCache
        ? "Loading from cache"
        : "Downloading weights"
      : model.isReady
        ? "Ready"
        : model.hasError
          ? "Failed"
          : "Idle";

  async function onLoad() {
    setResult(null);
    setBootstrapping(true);
    try {
      if (!manager.has("sentiment")) {
        manager.define("sentiment", await createSentimentDefinition());
      }
      setBootstrapping(false);
      await model.load();
    } catch {
      setBootstrapping(false);
    }
  }

  async function onClassify() {
    if (!model.value) return;
    setRunning(true);
    setResult(null);
    try {
      const out = await model.value(text);
      setResult(out[0] ?? null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page">
      <div className="atmosphere" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />

      <header className="top">
        <a className="gh" href="https://github.com/wassgha/weightlift">
          GitHub
        </a>
      </header>

      <main className="stage">
        <p className="brand">weightlift</p>
        <h1 className="headline">Load models. Watch the bytes land.</h1>
        <p className="lede">
          In-browser ML registry with download progress — try DistilBERT
          sentiment below.
        </p>

        <div className="actions">
          {!model.isReady ? (
            <button
              type="button"
              className="cta"
              onClick={onLoad}
              disabled={busy}
            >
              {busy ? "Loading…" : "Load model"}
            </button>
          ) : (
            <button
              type="button"
              className="cta"
              onClick={onClassify}
              disabled={running || !text.trim()}
            >
              {running ? "Running…" : "Classify"}
            </button>
          )}
          {model.isReady && (
            <button
              type="button"
              className="ghost"
              onClick={() => void model.unload()}
            >
              Unload
            </button>
          )}
        </div>

        <section
          className={`meter ${busy ? "is-live" : ""} ${model.isReady ? "is-ready" : ""}`}
          aria-live="polite"
        >
          <div className="meter-row">
            <span className="meter-status">{statusLabel}</span>
            <span className="meter-pct">
              {busy
                ? percent != null
                  ? `${percent}%`
                  : "…"
                : model.isReady
                  ? "100%"
                  : "0%"}
            </span>
          </div>
          <div
            className="track"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent ?? undefined}
            aria-label="Model download progress"
          >
            <div
              className={`fill ${model.indeterminate && busy ? "indeterminate" : ""}`}
              style={{
                width: model.isReady
                  ? "100%"
                  : percent != null
                    ? `${percent}%`
                    : busy
                      ? "30%"
                      : "0%",
              }}
            />
          </div>
          <div className="meter-meta">
            <span>
              {model.loadedBytes > 0
                ? `${formatBytes(model.loadedBytes)}${
                    model.totalBytes != null
                      ? ` / ${formatBytes(model.totalBytes)}`
                      : ""
                  }`
                : "—"}
            </span>
            <span className="mono">{MODEL_ID}</span>
          </div>
        </section>

        {model.hasError && (
          <p className="error">{model.error?.message ?? "Load failed"}</p>
        )}

        {model.isReady && (
          <section className="try">
            <label className="field">
              <span>Try a sentence</span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
              />
            </label>
            {result && (
              <p className="verdict">
                <span className="verdict-label">{result.label}</span>
                <span className="verdict-score">
                  {(result.score * 100).toFixed(1)}%
                </span>
              </p>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
