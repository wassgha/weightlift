import { useMemo, useState, type ReactNode } from "react";
import { Highlight, themes, type PrismTheme } from "prism-react-renderer";
import type { ModelDefinition } from "weightlift";
import { ModelManager } from "weightlift";
import { useModel, useModelManager } from "weightlift/react";

const codeTheme: PrismTheme = {
  ...themes.nightOwl,
  plain: {
    ...themes.nightOwl.plain,
    backgroundColor: "transparent",
  },
};

type Scored = { label: string; score: number };
type SentimentPipe = (
  input: string,
  options?: { top_k?: number | null }
) => Promise<Scored[] | Scored[][]>;
type ZeroShotImagePipe = (
  image: string,
  labels: string[]
) => Promise<Scored[]>;
type FillMaskPipe = (input: string) => Promise<Scored[] | Scored[][]>;

type ExampleId = "sentiment" | "vision" | "fillmask" | "custom";

interface Example {
  id: ExampleId;
  title: string;
  blurb: string;
  size: string;
  code: string;
}

const SAMPLE_IMAGES = [
  {
    id: "tiger",
    label: "tiger",
    src: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg",
  },
  {
    id: "cats",
    label: "cats",
    src: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg",
  },
  {
    id: "beach",
    label: "beach",
    src: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/beach.png",
  },
] as const;

const EXAMPLES: Example[] = [
  {
    id: "vision",
    title: "Zero-shot vision",
    blurb: "CLIP ranks your own labels against an image.",
    size: "~150 MB",
    code: `import { pipeline } from "@huggingface/transformers";
import { ModelManager } from "weightlift";
import { transformersModel } from "weightlift/transformers";

const models = new ModelManager({
  models: {
    vision: transformersModel({
      pipeline,
      task: "zero-shot-image-classification",
      modelId: "Xenova/clip-vit-base-patch32",
    }),
  },
});

const clip = await models.load("vision");
await clip(imageUrl, ["tiger", "lion", "house cat"]);`,
  },
  {
    id: "sentiment",
    title: "Sentiment analysis",
    blurb: "DistilBERT SST-2 — is this sentence positive or negative?",
    size: "~67 MB",
    code: `import { pipeline } from "@huggingface/transformers";
import { ModelManager } from "weightlift";
import { transformersModel } from "weightlift/transformers";

const models = new ModelManager({
  models: {
    sentiment: transformersModel({
      pipeline,
      task: "sentiment-analysis",
      modelId:
        "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    }),
  },
});

const clf = await models.load("sentiment");
const out = await clf("I love transformers!", { top_k: null });`,
  },
  {
    id: "fillmask",
    title: "Fill the blank",
    blurb: "BERT guesses the [MASK] token.",
    size: "~110 MB",
    code: `import { pipeline } from "@huggingface/transformers";
import { ModelManager } from "weightlift";
import { transformersModel } from "weightlift/transformers";

const models = new ModelManager({
  models: {
    fillmask: transformersModel({
      pipeline,
      task: "fill-mask",
      modelId: "Xenova/bert-base-uncased",
    }),
  },
});

const unmask = await models.load("fillmask");
await unmask("The browser can run [MASK] models.");`,
  },
  {
    id: "custom",
    title: "Custom ModelDefinition",
    blurb: "Any runtime — map progress events yourself.",
    size: "demo",
    code: `import { ModelManager } from "weightlift";

const models = new ModelManager({
  models: {
    custom: {
      load: async ({ progress }) => {
        progress.dispatch({ type: "start" });
        // …fetch / init, dispatch progress events…
        progress.dispatch({ type: "ready" });
        return myModel;
      },
    },
  },
});

await models.load("custom");`,
  },
];

const ASCII = `██╗    ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗██╗     ██╗███████╗████████╗
██║    ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██║     ██║██╔════╝╚══██╔══╝
██║ █╗ ██║█████╗  ██║██║  ███╗███████║   ██║   ██║     ██║█████╗     ██║
██║███╗██║██╔══╝  ██║██║   ██║██╔══██║   ██║   ██║     ██║██╔══╝     ██║
╚███╔███╔╝███████╗██║╚██████╔╝██║  ██║   ██║   ███████╗██║██║        ██║
 ╚══╝╚══╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝        ╚═╝`;

const INSTALL = "npm install weightlift";

const MODEL_IDS: Record<Exclude<ExampleId, "custom">, string> = {
  sentiment: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
  vision: "Xenova/clip-vit-base-patch32",
  fillmask: "Xenova/bert-base-uncased",
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Flatten nested pipeline outputs and pick the highest score. */
function topScored(raw: unknown): Scored | null {
  const rows = flattenScored(raw);
  if (!rows.length) return null;
  return rows.reduce((best, row) => (row.score > best.score ? row : best));
}

function flattenScored(raw: unknown): Scored[] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  if (Array.isArray(raw[0])) {
    return flattenScored(raw[0]);
  }
  return raw.filter(
    (row): row is Scored =>
      !!row &&
      typeof row === "object" &&
      typeof (row as Scored).label === "string" &&
      typeof (row as Scored).score === "number"
  );
}

function RankedList({ rows }: { rows: Scored[] }) {
  const sorted = [...rows].sort((a, b) => b.score - a.score).slice(0, 5);
  return (
    <ul className="ranked">
      {sorted.map((row) => (
        <li key={row.label}>
          <span>{row.label}</span>
          <span className="muted">{(row.score * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

function customDefinition(): ModelDefinition<{ ok: true }> {
  return {
    isCached: () => false,
    load: async ({ progress }) => {
      const files = ["config.json", "tokenizer.json", "model.onnx"];
      const totals = [4_096, 28_000, 420_000];
      progress.dispatch({ type: "start" });
      for (let i = 0; i < files.length; i++) {
        const file = files[i]!;
        const total = totals[i]!;
        progress.dispatch({ type: "initiate", file });
        for (let loaded = 0; loaded <= total; loaded += Math.ceil(total / 12)) {
          progress.dispatch({
            type: "progress",
            file,
            loaded: Math.min(loaded, total),
            total,
          });
          await sleep(40);
        }
        progress.dispatch({ type: "done", file });
      }
      progress.dispatch({ type: "ready" });
      return { ok: true };
    },
  };
}

let runtimePromise: Promise<{
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformersModel: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pipeline: any;
}> | null = null;

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = Promise.all([
      import("@huggingface/transformers"),
      import("weightlift/transformers"),
    ]).then(([tf, wl]) => {
      tf.env.allowLocalModels = false;
      tf.env.useBrowserCache = true;
      if (tf.env.backends.onnx.wasm) {
        tf.env.backends.onnx.wasm.proxy = false;
      }
      return {
        pipeline: tf.pipeline,
        transformersModel: wl.transformersModel,
      };
    });
  }
  return runtimePromise;
}

async function definitionFor(id: ExampleId): Promise<ModelDefinition<any>> {
  if (id === "custom") return customDefinition();

  const { pipeline, transformersModel } = await getRuntime();
  const modelId = MODEL_IDS[id];

  if (id === "sentiment") {
    return transformersModel({
      pipeline,
      task: "sentiment-analysis",
      modelId,
    });
  }
  if (id === "vision") {
    return transformersModel({
      pipeline,
      task: "zero-shot-image-classification",
      modelId,
    });
  }
  return transformersModel({
    pipeline,
    task: "fill-mask",
    modelId,
  });
}

function GitHubIcon() {
  return (
    <svg
      className="gh-icon"
      viewBox="0 0 16 16"
      width="14"
      height="14"
      aria-hidden="true"
    >
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"
      />
    </svg>
  );
}

function CopyButton({ value, label = "copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* ignore */
    }
  }

  return (
    <button type="button" className="copy-btn" onClick={copy}>
      {copied ? "copied" : label}
    </button>
  );
}

function CopyCommand({ value }: { value: string }) {
  return (
    <div className="cmd">
      <span className="cmd-prompt">$</span>
      <code>{value}</code>
      <CopyButton value={value} />
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <div className="code">
      <div className="code-bar">
        <span>typescript</span>
        <CopyButton value={code} />
      </div>
      <Highlight theme={codeTheme} code={code.trimEnd()} language="typescript">
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={`code-pre ${className}`} style={style}>
            <code>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  <span className="line-no">{i + 1}</span>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  );
}

function modelRowLabel(
  status: string,
  fromCache: boolean | null
): { text: string; tone: string } {
  if (status === "loading") {
    return {
      text: fromCache ? "loading · cache" : "loading · download",
      tone: "loading",
    };
  }
  if (status === "ready") {
    return {
      text: fromCache ? "ready · cached" : "ready",
      tone: "ready",
    };
  }
  if (status === "error") {
    return { text: "error", tone: "error" };
  }
  return { text: "idle", tone: "idle" };
}

function PlaygroundStatus({ manager }: { manager: ModelManager }) {
  const snapshot = useModelManager(manager);

  function scrollToExample(id: string) {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="status-board" aria-live="polite">
      <div className="status-board-head">
        <span>Model</span>
        <span>State</span>
        <span>Progress</span>
        <span></span>
      </div>
      <ul className="status-board-list">
        {EXAMPLES.map((example) => {
          const rec = snapshot.models[example.id];
          const status = rec?.status ?? "idle";
          const fromCache = rec?.fromCache ?? null;
          const { text, tone } = modelRowLabel(status, fromCache);
          const percent =
            rec?.percent != null ? Math.round(rec.percent * 100) : null;
          const progress =
            status === "ready"
              ? "100%"
              : status === "loading"
                ? percent != null
                  ? `${percent}%`
                  : "…"
                : status === "error"
                  ? "—"
                  : example.size;

          return (
            <li key={example.id} className={`status-row tone-${tone}`}>
              <a className="status-id" href={`#${example.id}`}>
                {example.id}
              </a>
              <span className="status-state">{text}</span>
              <span className="status-progress">
                <span className="status-track" aria-hidden="true">
                  <span
                    className="status-fill"
                    style={{
                      width:
                        status === "ready"
                          ? "100%"
                          : percent != null
                            ? `${percent}%`
                            : status === "loading"
                              ? "20%"
                              : "0%",
                    }}
                  />
                </span>
                <span className="status-pct">{progress}</span>
              </span>
              <button
                type="button"
                className="status-try"
                onClick={() => scrollToExample(example.id)}
              >
                try
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ExampleSection({
  example,
  manager,
  codeFirst,
}: {
  example: Example;
  manager: ModelManager;
  codeFirst: boolean;
}) {
  const model = useModel(manager, example.id);
  const [booting, setBooting] = useState(false);
  const [text, setText] = useState(
    example.id === "sentiment"
      ? "I love loading models in the browser!"
      : example.id === "fillmask"
        ? "The browser can run [MASK] models."
        : ""
  );
  const [labels, setLabels] = useState("tiger, lion, house cat, dog");
  const [imageSrc, setImageSrc] = useState<string>(SAMPLE_IMAGES[0].src);
  const [output, setOutput] = useState<ReactNode>(null);
  const [running, setRunning] = useState(false);

  const busy = model.isLoading || booting;
  const percent =
    model.percent != null ? Math.round(model.percent * 100) : null;

  const status = booting
    ? "preparing runtime"
    : model.isLoading
      ? model.fromCache
        ? "loading from cache"
        : "downloading"
      : model.isReady
        ? "ready"
        : model.hasError
          ? "error"
          : "idle";

  async function load() {
    setOutput(null);
    setBooting(true);
    try {
      if (!manager.has(example.id)) {
        manager.define(example.id, await definitionFor(example.id));
      }
      setBooting(false);
      await model.load();
    } catch {
      setBooting(false);
    }
  }

  async function onPickFile(file: File | undefined) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageSrc(url);
  }

  async function run() {
    setRunning(true);
    setOutput(null);
    try {
      if (example.id === "sentiment") {
        const fn = model.value as SentimentPipe;
        const raw = await fn(text, { top_k: null });
        const rows = flattenScored(raw);
        const hit = topScored(rows);
        if (!hit) throw new Error("No classification returned");
        setOutput(
          <div className="result-stack">
            <p className="out">
              <strong>{hit.label}</strong>{" "}
              <span className="muted">{(hit.score * 100).toFixed(1)}%</span>
            </p>
            <RankedList rows={rows} />
          </div>
        );
      } else if (example.id === "vision") {
        const fn = model.value as ZeroShotImagePipe;
        const labelList = labels
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        const raw = await fn(imageSrc, labelList);
        const rows = flattenScored(raw);
        setOutput(<RankedList rows={rows} />);
      } else if (example.id === "fillmask") {
        const fn = model.value as FillMaskPipe;
        const raw = await fn(text);
        const rows = flattenScored(raw).map((row) => ({
          ...row,
          label: row.label.replace(/^##/, ""),
        }));
        setOutput(<RankedList rows={rows} />);
      } else {
        setOutput(
          <p className="out">
            <strong>ready</strong>
            <span className="muted"> · custom loader finished</span>
          </p>
        );
      }
    } catch (err) {
      setOutput(
        <p className="out err">
          {err instanceof Error ? err.message : "run failed"}
        </p>
      );
    } finally {
      setRunning(false);
    }
  }

  const demo = (
    <div className="demo-pane">
      <div className="example-head">
        <div>
          <h2>{example.title}</h2>
          <p className="blurb">{example.blurb}</p>
        </div>
        <div className="example-actions">
          {model.isReady ? (
            <button
              type="button"
              className="primary"
              onClick={() => void model.unload()}
            >
              unload
            </button>
          ) : (
            <button
              type="button"
              className="primary"
              disabled={busy}
              onClick={() => void load()}
            >
              {busy ? "loading…" : "load"}
            </button>
          )}
        </div>
      </div>

      <div className="meter" aria-live="polite">
        <div className="meter-row">
          <span className={`status status-${model.status}`}>{status}</span>
          <span className="meter-pct">
            {busy
              ? percent != null
                ? `${percent}%`
                : "…"
              : model.isReady
                ? "100%"
                : example.size}
          </span>
        </div>
        <div
          className="track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent ?? undefined}
        >
          <div
            className={`fill ${model.indeterminate && busy ? "indeterminate" : ""}`}
            style={{
              width: model.isReady
                ? "100%"
                : percent != null
                  ? `${percent}%`
                  : busy
                    ? "24%"
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
          {model.hasError && (
            <span className="err">
              {model.error?.message ?? "load failed"}
            </span>
          )}
        </div>
      </div>

      <div className="try">
        {example.id === "sentiment" && (
          <label className="field">
            <span>sentence</span>
            <textarea
              value={text}
              rows={2}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
        )}

        {example.id === "fillmask" && (
          <label className="field">
            <span>sentence with [MASK]</span>
            <textarea
              value={text}
              rows={2}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
        )}

        {example.id === "vision" && (
          <>
            <div className="field">
              <span>image</span>
              <div className="thumbs">
                {SAMPLE_IMAGES.map((img) => (
                  <button
                    key={img.id}
                    type="button"
                    className={`thumb ${imageSrc === img.src ? "is-active" : ""}`}
                    onClick={() => setImageSrc(img.src)}
                    title={img.label}
                  >
                    <img src={img.src} alt={img.label} />
                  </button>
                ))}
              </div>
              <label className="upload">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => void onPickFile(e.target.files?.[0])}
                />
                upload your own
              </label>
              <div className="preview">
                <img src={imageSrc} alt="Selected" />
              </div>
            </div>
            <label className="field">
              <span>candidate labels</span>
              <input
                value={labels}
                onChange={(e) => setLabels(e.target.value)}
              />
            </label>
          </>
        )}

        {example.id === "custom" && (
          <p className="hint">Synthetic progress — no model weights required.</p>
        )}

        <div className="try-row">
          <button
            type="button"
            className="ghost"
            disabled={!model.isReady || running}
            onClick={() => void run()}
          >
            {running
              ? "running…"
              : !model.isReady
                ? "load model to run"
                : example.id === "custom"
                  ? "confirm"
                  : "run"}
          </button>
        </div>
        {output}
      </div>
    </div>
  );

  const code = (
    <div className="code-pane">
      <CodeBlock code={example.code} />
    </div>
  );

  return (
    <section
      className={`example ${codeFirst ? "code-first" : "demo-first"}`}
      id={example.id}
    >
      {codeFirst ? (
        <>
          {code}
          {demo}
        </>
      ) : (
        <>
          {demo}
          {code}
        </>
      )}
    </section>
  );
}

export function App() {
  const manager = useMemo(() => new ModelManager(), []);

  return (
    <div className="page">
      <header className="nav">
        <div className="nav-brand">
          <span className="mark" aria-hidden="true" />
          weightlift
        </div>
        <a
          className="gh-link"
          href="https://github.com/wassgha/weightlift"
          target="_blank"
          rel="noreferrer"
        >
          <GitHubIcon />
          GitHub
        </a>
      </header>

      <main>
        <section className="hero">
          <pre className="ascii" aria-label="weightlift">
            {ASCII}
          </pre>
          <p className="eyebrow">IN-BROWSER AI MODEL MANAGER</p>
          <div className="hero-grid">
            <p className="lede">
              Load, cache and manage client-side AI models with live progress tracking.
            </p>
            <div className="cta-block">
              <div className="label">TRY IT NOW</div>
              <CopyCommand value={INSTALL} />
            </div>
          </div>
        </section>

        <section className="playground-head" id="playground">
          <p className="eyebrow">PLAYGROUND</p>
          <h2 className="playground-title">Load models. Watch the bytes land.</h2>
          <p className="lede">
            Manage multiple models with one <code>ModelManager</code> — support for transformers.js
            and custom model loaders via <code>ModelDefinition</code>.
          </p>
          <PlaygroundStatus manager={manager} />
        </section>

        {EXAMPLES.map((example, i) => (
          <ExampleSection
            key={example.id}
            example={example}
            manager={manager}
            codeFirst={i % 2 === 1}
          />
        ))}
      </main>

      <footer className="footer">
        <span>MIT</span>
        <a href="https://github.com/wassgha/weightlift">wassgha/weightlift</a>
      </footer>
    </div>
  );
}
