import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ModelManager } from "../src/manager.js";
import { Weightlift } from "../src/store.js";
import {
  fallbackDevicePolicy,
  transformersEvent,
  transformersModel,
  transformersProgress,
  type TransformersDevice,
  type TransformersDevicePolicy,
} from "../src/transformers.js";

function isolatedPolicy(
  initial: TransformersDevice = "wasm"
): TransformersDevicePolicy {
  let forceWasm = initial === "wasm";
  return {
    get forceWasm() {
      return forceWasm;
    },
    preferWasm() {
      forceWasm = true;
    },
    async pickDevice() {
      return forceWasm ? "wasm" : "webgpu";
    },
  };
}

describe("transformersProgress", () => {
  it("maps progress_total (percent → bytes)", () => {
    assert.deepEqual(
      transformersEvent({
        status: "progress_total",
        progress: 25,
        total: 400,
      }),
      { type: "progress_total", loaded: 100, total: 400 }
    );
  });

  it("feeds a Weightlift store via transformersProgress()", () => {
    const wl = new Weightlift();
    wl.start();
    const cb = transformersProgress(wl);
    cb({ status: "progress_total", progress: 50, total: 200 });
    assert.equal(wl.getSnapshot().percent, 0.5);
  });
});

describe("transformersModel", () => {
  it("loads via injected pipeline with progress + dtype", async () => {
    const calls: Array<{ device: string; dtype: unknown }> = [];
    const def = transformersModel<{ ok: true }>({
      pipeline: async (_task, _id, opts) => {
        calls.push({
          device: opts.device as string,
          dtype: opts.dtype,
        });
        const cb = opts.progress_callback as (p: {
          status: string;
          progress: number;
          total: number;
        }) => void;
        cb({ status: "progress_total", progress: 100, total: 10 });
        return { ok: true };
      },
      task: "automatic-speech-recognition",
      modelId: "onnx-community/whisper-base",
      dtype: { webgpu: "fp32", wasm: "q4" },
      devicePolicy: isolatedPolicy("wasm"),
    });

    const models = new ModelManager({ models: { w: def } });
    const value = await models.load<{ ok: true }>("w");
    assert.deepEqual(value, { ok: true });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].device, "wasm");
    assert.equal(calls[0].dtype, "q4");
    assert.equal(models.status("w").percent, 1);
  });

  it("defaults to fallbackDevicePolicy when none is passed", async () => {
    assert.equal(typeof fallbackDevicePolicy.pickDevice, "function");
    const def = transformersModel({
      pipeline: async (_task, _id, opts) => {
        assert.ok(opts.device === "webgpu" || opts.device === "wasm");
        return "ok";
      },
      task: "feature-extraction",
      modelId: "x",
      // no devicePolicy — uses shared fallbackDevicePolicy
    });
    const models = new ModelManager({ models: { x: def } });
    assert.equal(await models.load("x"), "ok");
  });

  it("retries on WASM when WebGPU pipeline() throws", async () => {
    const devices: TransformersDevice[] = [];
    const seen: string[] = [];
    const policy = isolatedPolicy("webgpu");

    const def = transformersModel({
      pipeline: async (_task, _id, opts) => {
        seen.push(opts.device as string);
        if (opts.device === "webgpu") throw new Error("gpu down");
        return "session";
      },
      task: "feature-extraction",
      modelId: "Xenova/clip",
      dtype: { webgpu: "fp16", wasm: "fp32" },
      devicePolicy: policy,
      onDevice: (d) => devices.push(d),
    });

    const models = new ModelManager({ models: { clip: def } });
    assert.equal(await models.load("clip"), "session");
    assert.deepEqual(seen, ["webgpu", "wasm"]);
    assert.deepEqual(devices, ["webgpu", "wasm"]);
    assert.equal(policy.forceWasm, true);
  });

  it("does not retry when already on WASM", async () => {
    let calls = 0;
    const def = transformersModel({
      pipeline: async () => {
        calls += 1;
        throw new Error("boom");
      },
      task: "feature-extraction",
      modelId: "x",
      devicePolicy: isolatedPolicy("wasm"),
    });
    const models = new ModelManager({ models: { x: def } });
    await assert.rejects(() => models.load("x"), /boom/);
    assert.equal(calls, 1);
  });
});

describe("isTransformersModelCached", () => {
  const modelId = "Xenova/demo-model";

  function mockCaches(
    entries: Array<{ url: string; status: number; length?: number }>
  ) {
    const map = new Map(
      entries.map((e) => {
        const headers = new Headers();
        if (e.length != null) headers.set("content-length", String(e.length));
        return [
          e.url,
          new Response(null, { status: e.status, headers }),
        ] as const;
      })
    );
    const keys = [...map.keys()].map((url) => new Request(url));
    return {
      open: async () => ({
        keys: async () => keys,
        match: async (req: Request) => map.get(req.url) ?? undefined,
      }),
    };
  }

  it("ignores redirect stubs without weight bytes", async () => {
    const { isTransformersModelCached } = await import(
      "../src/transformers.js"
    );
    const previous = globalThis.caches;
    // @ts-expect-error test mock
    globalThis.caches = mockCaches([
      {
        url: `https://huggingface.co/${modelId}/resolve/main/onnx/model.onnx`,
        status: 302,
      },
    ]);
    try {
      assert.equal(await isTransformersModelCached(modelId), false);
    } finally {
      globalThis.caches = previous;
    }
  });

  it("returns true when enough ONNX bytes are cached", async () => {
    const { isTransformersModelCached } = await import(
      "../src/transformers.js"
    );
    const previous = globalThis.caches;
    // @ts-expect-error test mock
    globalThis.caches = mockCaches([
      {
        url: `https://huggingface.co/${modelId}/resolve/main/onnx/model.onnx`,
        status: 200,
        length: 5_000_000,
      },
    ]);
    try {
      assert.equal(await isTransformersModelCached(modelId), true);
    } finally {
      globalThis.caches = previous;
    }
  });
});
