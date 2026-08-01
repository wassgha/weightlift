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
  it("registers models in the constructor and loads by id", async () => {
    const models = new ModelManager({
      models: {
        clip: {
          load: async () => ({ name: "clip" }),
        },
      },
    });
    assert.equal(models.has("clip"), true);
    assert.deepEqual(models.ids(), ["clip"]);
    const value = await models.load<{ name: string }>("clip");
    assert.deepEqual(value, { name: "clip" });
    assert.equal(models.isReady("clip"), true);
    assert.deepEqual(models.get("clip"), { name: "clip" });
    assert.equal(models.status("clip").status, "ready");
  });

  it("define adds models after init", async () => {
    const models = new ModelManager();
    models.define("clip", {
      load: async () => ({ name: "clip" }),
    });
    const value = await models.load<{ name: string }>("clip");
    assert.deepEqual(value, { name: "clip" });
  });

  it("dedupes concurrent load() calls", async () => {
    let calls = 0;
    const gate = deferred<void>();
    const models = new ModelManager({
      models: {
        m: {
          load: async () => {
            calls += 1;
            await gate.promise;
            return "ok";
          },
        },
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

  it("remove drops the definition; unload keeps it", async () => {
    const models = new ModelManager({
      models: {
        x: {
          load: async () => "v",
        },
      },
    });
    await models.load("x");
    await models.unload("x");
    assert.equal(models.has("x"), true);
    assert.equal(models.isReady("x"), false);
    assert.equal(await models.load("x"), "v");

    await models.remove("x");
    assert.equal(models.has("x"), false);
    await assert.rejects(() => models.load("x"), /unknown model/);
  });

  it("exposes fromCache without owning UI copy", async () => {
    const models = new ModelManager({
      models: {
        w: {
          isCached: async () => true,
          load: async () => "x",
        },
      },
    });
    await models.load("w");
    assert.equal(models.status("w").fromCache, true);

    const models2 = new ModelManager({
      models: {
        w: {
          isCached: async () => false,
          load: async () => "x",
        },
      },
    });
    await models2.load("w");
    assert.equal(models2.status("w").fromCache, false);
  });

  it("exposes progress through the load context", async () => {
    const models = new ModelManager({
      models: {
        p: {
          load: async (ctx) => {
            ctx.progress.dispatch({
              type: "progress_total",
              loaded: 25,
              total: 100,
            });
            return "model";
          },
        },
      },
    });
    let progress: Weightlift | undefined;
    const gate = deferred<void>();
    models.define("p2", {
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
    const done = models.load("p2");
    await Promise.resolve();
    await Promise.resolve();
    assert.ok(progress);
    assert.equal(models.status("p2").percent, 0.25);
    assert.equal(models.getSnapshot().loading.includes("p2"), true);
    gate.resolve();
    await done;
    assert.equal(models.getSnapshot().ready.includes("p2"), true);
  });

  it("unload clears the instance so load runs again", async () => {
    let calls = 0;
    const models = new ModelManager({
      models: {
        x: {
          load: async () => {
            calls += 1;
            return calls;
          },
        },
      },
    });
    assert.equal(await models.load("x"), 1);
    await models.unload("x");
    assert.equal(models.get("x"), undefined);
    assert.equal(models.isReady("x"), false);
    assert.equal(await models.load("x"), 2);
  });

  it("unload during an in-flight load keeps the manager idle", async () => {
    const gate = deferred<void>();
    let disposed: string | null = null;
    const models = new ModelManager({
      models: {
        x: {
          load: async ({ progress }) => {
            progress.dispatch({
              type: "progress_total",
              loaded: 50,
              total: 100,
            });
            await gate.promise;
            progress.dispatch({
              type: "progress_total",
              loaded: 100,
              total: 100,
            });
            return "model";
          },
          dispose: async (value) => {
            disposed = value;
          },
        },
      },
    });

    const pending = models.load("x");
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(models.status("x").status, "loading");

    await models.unload("x");
    assert.equal(models.status("x").status, "idle");
    assert.equal(models.get("x"), undefined);

    gate.resolve();
    await assert.rejects(pending, /unloaded while loading/);
    assert.equal(disposed, "model");
    assert.equal(models.status("x").status, "idle");
    assert.equal(models.get("x"), undefined);
    assert.equal(models.isReady("x"), false);
  });

  it("unload during load does not poison a subsequent load", async () => {
    const gate1 = deferred<void>();
    const gate2 = deferred<void>();
    let phase: "first" | "second" = "first";
    const models = new ModelManager({
      models: {
        x: {
          load: async () => {
            if (phase === "first") {
              await gate1.promise;
              return "first";
            }
            await gate2.promise;
            return "second";
          },
        },
      },
    });

    const first = models.load("x");
    await Promise.resolve();
    await models.unload("x");
    phase = "second";
    const second = models.load("x");
    gate1.resolve();
    await assert.rejects(first, /unloaded while loading/);
    gate2.resolve();
    assert.equal(await second, "second");
    assert.equal(models.get("x"), "second");
    assert.equal(models.status("x").status, "ready");
  });

  it("unloadAll drops every cached instance", async () => {
    const models = new ModelManager({
      models: {
        a: { load: async () => "a" },
        b: { load: async () => "b" },
      },
    });
    await models.preload(["a", "b"]);
    await models.unloadAll();
    assert.equal(models.isReady("a"), false);
    assert.equal(models.isReady("b"), false);
    assert.equal(models.has("a"), true);
    assert.equal(models.has("b"), true);
  });

  it("calls dispose on unload", async () => {
    let disposed: string | null = null;
    const models = new ModelManager({
      models: {
        gpu: {
          load: async () => "session",
          dispose: async (v) => {
            disposed = v;
          },
        },
      },
    });
    await models.load("gpu");
    await models.unload("gpu");
    assert.equal(disposed, "session");
  });

  it("retries after failure", async () => {
    let calls = 0;
    const models = new ModelManager({
      models: {
        flaky: {
          load: async () => {
            calls += 1;
            if (calls === 1) throw new Error("nope");
            return "ok";
          },
        },
      },
    });
    await assert.rejects(() => models.load("flaky"));
    assert.equal(models.status("flaky").status, "error");
    assert.equal(await models.load("flaky"), "ok");
    assert.equal(calls, 2);
  });

  it("preload loads many ids", async () => {
    const models = new ModelManager({
      models: {
        a: { load: async () => "a" },
        b: { load: async () => "b" },
      },
    });
    await models.preload(["a", "b"]);
    assert.deepEqual(models.getSnapshot().ready.sort(), ["a", "b"]);
  });

  it("rejects load of unknown id", async () => {
    const models = new ModelManager();
    await assert.rejects(() => models.load("missing"), /unknown model/);
  });
});
