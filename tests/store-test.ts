import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  Weightlift,
  createModelLoader,
  createModelRegistry,
} from "../src/store.js";

describe("Weightlift", () => {
  it("starts idle and becomes loading on start()", () => {
    const wl = new Weightlift();
    assert.equal(wl.status, "idle");
    assert.equal(wl.isIdle, true);
    wl.start();
    assert.equal(wl.isLoading, true);
  });

  it("aggregates parallel per-file progress into one percent", () => {
    const wl = new Weightlift();
    wl.start();
    wl.dispatch({ type: "progress", file: "a.onnx", loaded: 50, total: 100 });
    wl.dispatch({ type: "progress", file: "b.onnx", loaded: 0, total: 100 });
    const s = wl.getSnapshot();
    assert.equal(s.loadedBytes, 50);
    assert.equal(s.totalBytes, 200);
    assert.equal(s.percent, 0.25);
    assert.equal(s.indeterminate, false);
  });

  it("marks indeterminate when a file has no total", () => {
    const wl = new Weightlift();
    wl.start();
    wl.dispatch({ type: "progress", file: "a.onnx", loaded: 50 });
    const s = wl.getSnapshot();
    assert.equal(s.indeterminate, true);
    assert.equal(s.percent, null);
  });

  it("prefers progress_total over per-file events", () => {
    const wl = new Weightlift();
    wl.start();
    wl.dispatch({ type: "progress_total", loaded: 10, total: 100 });
    wl.dispatch({ type: "progress", file: "a.onnx", loaded: 100, total: 100 });
    const s = wl.getSnapshot();
    assert.equal(s.percent, 0.1);
    assert.equal(s.loadedBytes, 10);
  });

  it("rescales best when total grows", () => {
    const wl = new Weightlift();
    wl.start();
    wl.dispatch({ type: "progress_total", loaded: 50, total: 100 });
    assert.equal(wl.getSnapshot().percent, 0.5);
    wl.dispatch({ type: "progress_total", loaded: 50, total: 200 });
    assert.ok(wl.getSnapshot().percent !== null);
    assert.ok((wl.getSnapshot().percent as number) <= 0.5);
    assert.ok(Math.abs((wl.getSnapshot().percent as number) - 0.25) < 1e-9);
  });

  it("notifies subscribers and unsubscribes cleanly", () => {
    const wl = new Weightlift();
    const seen: number[] = [];
    const unsub = wl.subscribe((s) => {
      if (s.percent != null) seen.push(s.percent);
    });
    wl.dispatch({ type: "progress_total", loaded: 1, total: 10 });
    unsub();
    wl.dispatch({ type: "progress_total", loaded: 5, total: 10 });
    assert.deepEqual(seen, [0.1]);
  });

  it("ready / fail / reset lifecycle", () => {
    const wl = new Weightlift();
    wl.start();
    wl.ready();
    assert.equal(wl.isReady, true);
    assert.equal(wl.getSnapshot().percent, 1);

    wl.fail(new Error("boom"));
    assert.equal(wl.hasError, true);
    assert.equal(wl.getSnapshot().error?.message, "boom");

    wl.reset();
    assert.equal(wl.isIdle, true);
    assert.equal(wl.getSnapshot().error, null);
  });
});

describe("createModelLoader", () => {
  it("dedupes concurrent load() calls", async () => {
    let calls = 0;
    const loader = createModelLoader(async (wl) => {
      calls += 1;
      wl.dispatch({ type: "progress_total", loaded: 1, total: 1 });
      return "model";
    });
    const [a, b] = await Promise.all([loader.load(), loader.load()]);
    assert.equal(a, "model");
    assert.equal(b, "model");
    assert.equal(calls, 1);
    assert.equal(loader.weightlift.isReady, true);
  });

  it("retries after failure when invalidated", async () => {
    let calls = 0;
    const loader = createModelLoader(async () => {
      calls += 1;
      if (calls === 1) throw new Error("nope");
      return "ok";
    });
    await assert.rejects(() => loader.load());
    assert.equal(loader.weightlift.hasError, true);
    const value = await loader.load();
    assert.equal(value, "ok");
    assert.equal(calls, 2);
  });
});

describe("createModelRegistry", () => {
  it("keeps separate loaders per key", async () => {
    const registry = createModelRegistry<string>();
    const a = registry.get("base", async () => "base");
    const b = registry.get("small", async () => "small");
    assert.notEqual(a, b);
    assert.equal(await a.load(), "base");
    assert.equal(await b.load(), "small");
  });
});
