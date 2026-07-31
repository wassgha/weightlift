import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ModelManager } from "../src/manager.js";
import type { Weightlift } from "../src/store.js";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("ModelManager", () => {
  it("define + load returns the model and marks ready", async () => {
    const models = new ModelManager();
    models.define("clip", {
      load: async () => ({ name: "clip" }),
    });
    const value = await models.load<{ name: string }>("clip");
    assert.deepEqual(value, { name: "clip" });
    assert.equal(models.isReady("clip"), true);
    assert.deepEqual(models.get("clip"), { name: "clip" });
    assert.equal(models.status("clip").status, "ready");
  });

  it("dedupes concurrent load() calls", async () => {
    let calls = 0;
    const gate = deferred<void>();
    const models = new ModelManager();
    models.define("m", {
      load: async () => {
        calls += 1;
        await gate.promise;
        return "ok";
      },
    });
    const p1 = models.load("m");
    const p2 = models.load("m");
    assert.equal(models.isLoading("m"), true);
    gate.resolve();
    assert.equal(await p1, "ok");
    assert.equal(await p2, "ok");
    assert.equal(calls, 1);
  });

  it("load(id, definition) registers lazily", async () => {
    const models = new ModelManager();
    const value = await models.load("lazy", {
      load: async () => 42,
    });
    assert.equal(value, 42);
    assert.equal(models.has("lazy"), true);
  });

  it("exposes fromCache without owning UI copy", async () => {
    const models = new ModelManager();
    await models.load("w", {
      isCached: async () => true,
      load: async () => "x",
    });
    assert.equal(models.status("w").fromCache, true);

    const models2 = new ModelManager();
    await models2.load("w", {
      isCached: async () => false,
      load: async () => "x",
    });
    assert.equal(models2.status("w").fromCache, false);
  });

  it("exposes progress through the load context", async () => {
    const models = new ModelManager();
    let progress: Weightlift | undefined;
    const gate = deferred<void>();
    const done = models.load("p", {
      load: async (ctx) => {
        progress = ctx.progress;
        ctx.progress.dispatch({
          type: "progress_total",
          loaded: 25,
          total: 100,
        });
        await gate.promise;
        return "model";
      },
    });
    await Promise.resolve();
    await Promise.resolve();
    assert.ok(progress);
    assert.equal(models.status("p").percent, 0.25);
    assert.equal(models.getSnapshot().loading.includes("p"), true);
    gate.resolve();
    await done;
    assert.equal(models.getSnapshot().ready.includes("p"), true);
  });

  it("unload clears the instance so load runs again", async () => {
    let calls = 0;
    const models = new ModelManager();
    models.define("x", {
      load: async () => {
        calls += 1;
        return calls;
      },
    });
    assert.equal(await models.load("x"), 1);
    await models.unload("x");
    assert.equal(models.get("x"), undefined);
    assert.equal(models.isReady("x"), false);
    assert.equal(await models.load("x"), 2);
  });

  it("calls dispose on unload", async () => {
    let disposed: string | null = null;
    const models = new ModelManager();
    models.define("gpu", {
      load: async () => "session",
      dispose: async (v) => {
        disposed = v;
      },
    });
    await models.load("gpu");
    await models.unload("gpu");
    assert.equal(disposed, "session");
  });

  it("retries after failure", async () => {
    let calls = 0;
    const models = new ModelManager();
    models.define("flaky", {
      load: async () => {
        calls += 1;
        if (calls === 1) throw new Error("nope");
        return "ok";
      },
    });
    await assert.rejects(() => models.load("flaky"));
    assert.equal(models.status("flaky").status, "error");
    assert.equal(await models.load("flaky"), "ok");
    assert.equal(calls, 2);
  });

  it("preload loads many ids", async () => {
    const models = new ModelManager();
    models.define("a", { load: async () => "a" });
    models.define("b", { load: async () => "b" });
    await models.preload(["a", "b"]);
    assert.deepEqual(models.getSnapshot().ready.sort(), ["a", "b"]);
  });

  it("rejects load of unknown id", async () => {
    const models = new ModelManager();
    await assert.rejects(() => models.load("missing"), /unknown model/);
  });
});
