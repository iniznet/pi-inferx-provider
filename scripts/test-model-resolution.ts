#!/usr/bin/env node
/**
 * Behavioural checks for the InferX provider's model pipeline.
 *
 * Runs against the REAL exported helpers in index.ts (no re-implementation) plus the
 * catalog parser in scripts/update-models.js, and covers the failure modes this provider
 * actually has:
 *
 *   - the catalog parser must take the id from the href (InferX slugs are case-sensitive and
 *     the row's data-search attribute is lowercased) and must read the BILLED price out of a
 *     cell that also shows the struck-through list price
 *   - pi throws `invalid contextWindow` for contextWindow <= 0, so every shipped and every
 *     sanitised model must clear that bar or the whole provider fails to register
 *   - unmapped thinking levels make pi send its own level names to a vLLM parser that
 *     rejects them, so reasoning models need a complete thinkingLevelMap
 *   - merge order (base -> deprecated -> patch -> custom -> sanitize) and the 14-day
 *     delisting window must hold at the TTL boundary
 *   - the session_start hot swap must re-register with live data merged over the embedded
 *     catalog, and must not touch the user's real ~/.pi
 *
 * Cache/config IO is isolated to a temp agent dir; no network is used.
 */

import fs from "fs";
import os from "os";
import path from "path";

const tmpHome = path.join(os.tmpdir(), `pi-inferx-test-${Date.now()}`);
fs.mkdirSync(tmpHome, { recursive: true });
process.env.PI_CODING_AGENT_DIR = tmpHome;
delete process.env.INFERX_BASE_URL;

const root = path.resolve(import.meta.dirname, "..");
const modelsData = JSON.parse(fs.readFileSync(path.join(root, "models.json"), "utf8"));
const patchData = JSON.parse(fs.readFileSync(path.join(root, "patch.json"), "utf8"));

const index = await import("../index.ts");
type JsonModel = import("../index.ts").JsonModel;
const { parseCatalog } = await import("./update-models.js");
const {
  applyPatch,
  buildModels,
  sanitizeModels,
  mergeWithEmbedded,
  selectActiveDeprecated,
  withDeprecated,
  transformApiModel,
  resolveBaseUrl,
  DEPRECATED_MODEL_TTL_MS,
} = index;

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, message: string) {
  if (condition) {
    passed++;
    console.log(`  \u2713 ${message}`);
  } else {
    failures.push(message);
    console.error(`  \u2717 ${message}`);
  }
}
function eq(actual: unknown, expected: unknown, message: string) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) console.error(`     expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  check(ok, message);
}
function section(title: string) {
  console.log(`\n\u2500\u2500 ${title}`);
}

// ─── A. Catalog parser ────────────────────────────────────────────────────────

section("A. public catalog parser");

const FIXTURE = `
<tbody id="endpoint-table-body">
  <tr data-search="qwen3.8-27b-fp8 qwen " data-tags="" data-use-cases="" data-provider="qwen">
    <td style="text-align: left;"><a class="endpoint-slug-link" href="/catalog/endpoints/Qwen3.8-27B-FP8?tenant=public">Qwen3.8-27B-FP8</a></td>
    <td style="text-align: center;">262000</td>
    <td class="endpoint-discount-cell" style="text-align: center;">96.00% off</td>
    <td class="endpoint-price-cell" style="text-align: center;"><del>$0.45</del> $0.018</td>
    <td class="endpoint-price-cell" style="text-align: center;"><del>$3.2</del> $0.128</td>
    <td class="endpoint-price-cell" style="text-align: center;"><del>$0</del> $0</td>
  </tr>
  <tr data-search="deepseek-v4-flash-0731 " data-tags="" data-use-cases="" data-provider="">
    <td><a class="endpoint-slug-link" href="/catalog/endpoints/deepseek-v4-flash-0731?tenant=public">deepseek-v4-flash-0731</a></td>
    <td>\u2014</td>
    <td class="endpoint-discount-cell">80.00% off</td>
    <td class="endpoint-price-cell">\u2014</td>
    <td class="endpoint-price-cell">\u2014</td>
    <td class="endpoint-price-cell">\u2014</td>
  </tr>
  <tr><th>not a row</th></tr>
</tbody>`;

const parsed = parseCatalog(FIXTURE);
eq(parsed.length, 2, "ignores rows without an endpoint link");
eq(parsed[0]?.id, "Qwen3.8-27B-FP8", "id comes from the href and keeps its case (data-search is lowercased)");
eq(parsed[0]?.contextWindow, 262000, "published context length is parsed");
eq(parsed[0]?.cost, { input: 0.018, output: 0.128, cacheRead: 0 }, "billed price is the last number in a cell that also shows list price");
eq(parsed[1]?.contextWindow, 0, "an unpublished context length reads as 0 rather than NaN");
eq(parsed[1]?.cost, { input: 0, output: 0, cacheRead: 0 }, "anonymous em-dash price cells read as 0 (curated data is preserved instead)");
eq(parsed[0]?.provider, "qwen", "upstream owner is read from data-provider");
eq(parseCatalog("<html><body>no table</body></html>"), [], "a page without the table yields no models");

// ─── B. Shipped catalog integrity ─────────────────────────────────────────────

section("B. shipped catalog integrity (models.json + patch.json)");

const EXPECTED_IDS = [
  "Agents-A1",
  "Qwen3-Coder-Next-FP8",
  "Qwen3.6-35B-A3B-FP8",
  "Qwen3.8-27B-FP8",
  "deepseek-v4-flash-0731",
  "deepseek-v4.1-flash",
  "gemma-4-31B-it-fp8",
  "glm-5.3-flash",
  "gpt-oss-20b",
];
// Delisted endpoints still inside the 14-day grace window stay registered. Read the
// shipped graveyard dynamically so this gate survives its natural eviction instead of
// hardcoding ids that are guaranteed to disappear.
const shippedDeprecated = JSON.parse(
  fs.readFileSync(path.join(root, "deprecated-models.json"), "utf8"),
) as Record<string, JsonModel & { deprecatedAt?: string }>;
const shippedGraceIds = selectActiveDeprecated(shippedDeprecated).map((m: JsonModel) => m.id);
const EXPECTED_REGISTERED_IDS = [...EXPECTED_IDS, ...shippedGraceIds];
const PI_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const VLLM_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
// Per-model accepted effort set, probed live against the InferX gateway (each vLLM
// reasoning parser owns a subset; e.g. qwen38-flash-next rejects "high" but accepts
// "xhigh", and Qwen3.8-27B-FP8 accepts only "low"/"medium"). These are the load-bearing
// vocabulary bounds that the global set alone cannot express.
const VERIFIED_EFFORT: Record<string, string[]> = {
  "deepseek-v4-flash-0731": VLLM_EFFORTS,
  "deepseek-v4-flash-vision-exp": VLLM_EFFORTS,
  "gemma-4-31B-it-fp8": ["none", "low", "medium", "high"],
  "glm-5.3-flash": VLLM_EFFORTS,
  "Qwen3.6-35B-A3B-FP8": ["none", "low", "medium", "high"],
  "Qwen3.8-27B-FP8": ["low", "medium"],
  "qwen38-flash-next": ["none", "low", "medium", "xhigh"],
};
// Which pi levels each reasoning model actually exposes. pi shows a level only when its
// thinkingLevelMap entry is non-null, so levels that fold onto an already-present endpoint
// effort are hidden (null) to keep the selector at the endpoint's true granularity.
const EXPECTED_EXPOSED: Record<string, string[]> = {
  "deepseek-v4-flash-0731": PI_LEVELS,
  "deepseek-v4-flash-vision-exp": PI_LEVELS,
  "gemma-4-31B-it-fp8": ["off", "low", "medium", "high"],
  "glm-5.3-flash": PI_LEVELS,
  "Qwen3.6-35B-A3B-FP8": ["off", "low", "medium", "high"],
  "Qwen3.8-27B-FP8": ["low", "medium"],
  "qwen38-flash-next": ["off", "low", "medium", "xhigh"],
};

const built = buildModels(modelsData, [], patchData);
const byId = new Map(built.map((m: any) => [m.id, m]));

eq(built.map((m: any) => m.id).sort(), [...EXPECTED_REGISTERED_IDS].sort(), "every published endpoint is registered with its exact slug");
check(built.every((m: any) => m.contextWindow > 0), "no model can trip pi's `invalid contextWindow` guard");
check(built.every((m: any) => m.maxTokens > 0 && m.maxTokens <= m.contextWindow), "maxTokens is positive and fits inside the window");
check(built.every((m: any) => [m.cost.input, m.cost.output, m.cost.cacheRead, m.cost.cacheWrite].every((c: number) => Number.isFinite(c) && c >= 0)), "all cost fields are finite and non-negative");
check(EXPECTED_IDS.every((id) => patchData[id]), "every endpoint has a curated patch entry (no capability left at the raw default)");
check(
  built.every((m: any) => m.compat?.maxTokensField === "max_tokens" && m.compat.supportsStore === false && m.compat.supportsDeveloperRole === false),
  "every model pins the vLLM request surface (max_tokens, no store, system role)",
);
check(
  built.filter((m: any) => m.reasoning).every((m: any) => PI_LEVELS.every((level) => level in (m.thinkingLevelMap || {}))),
  "reasoning models map every pi thinking level (an unmapped level would send pi's name to vLLM)",
);
check(
  built.filter((m: any) => m.reasoning).every((m: any) => Object.values(m.thinkingLevelMap).every((v: any) => v === null || VLLM_EFFORTS.includes(v))),
  "thinking level values stay inside the vLLM effort vocabulary",
);
check(
  built.filter((m: any) => m.reasoning).every((m: any) => {
    const exposed = Object.entries(m.thinkingLevelMap)
      .filter(([, v]) => v !== null)
      .map(([k]) => k)
      .sort();
    const expected = [...(EXPECTED_EXPOSED[m.id] || [])].sort();
    const allowed = VERIFIED_EFFORT[m.id] || [];
    return (
      JSON.stringify(exposed) === JSON.stringify(expected) &&
      Object.values(m.thinkingLevelMap).every((v: any) => v === null || allowed.includes(v))
    );
  }),
  "each reasoning model exposes exactly the pi levels its endpoint truly offers (null hides the rest) and every exposed value is accepted",
);
check(
  built.filter((m: any) => !m.reasoning).every((m: any) => !m.thinkingLevelMap && !m.compat?.thinkingFormat && !m.compat?.supportsReasoningEffort),
  "non-reasoning models carry no thinking controls",
);
eq(
  built.filter((m: any) => m.input.includes("image")).map((m: any) => m.id).sort(),
  ["Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B-FP8", "deepseek-v4.1-flash", "gemma-4-31B-it-fp8", "glm-5.3-flash", ...shippedGraceIds.filter((id: string) => patchData[id]?.input?.includes("image"))].sort(),
  "vision is declared exactly where InferX evidence or a majority of sibling E2E catalogs points",
);
eq(
  built.filter((m: any) => m.reasoning).map((m: any) => m.id).sort(),
  ["Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B-FP8", "deepseek-v4-flash-0731", "gemma-4-31B-it-fp8", "glm-5.3-flash", ...shippedGraceIds.filter((id: string) => patchData[id]?.reasoning)].sort(),
  "reasoning is declared exactly where a vLLM reasoning parser (or the same upstream weights) is evidenced",
);
eq(byId.get("gemma-4-31B-it-fp8")?.maxTokens, 16384, "gemma keeps the sibling-verified 16K completion cap instead of the generic ceiling");
eq(byId.get("deepseek-v4-flash-0731")?.contextWindow, 1000000, "the endpoint InferX publishes no length for carries the curated upstream window");

// ─── C. Merge semantics ───────────────────────────────────────────────────────

section("C. merge semantics");

const base: JsonModel = {
  id: "x", name: "X", reasoning: false, input: ["text"], cost: { input: 1, output: 2, cacheRead: 3, cacheWrite: 0 },
  contextWindow: 1000, maxTokens: 100,
};
const frozen = JSON.parse(JSON.stringify(base));
const patched = applyPatch(base, { reasoning: true, cost: { input: 0.5 }, compat: { thinkingFormat: "openai" }, thinkingLevelMap: { off: "none" } });
eq(JSON.stringify(base), JSON.stringify(frozen), "applyPatch does not mutate the input model");
eq([patched.reasoning, patched.cost.input, patched.cost.output], [true, 0.5, 2], "patch deep-merges cost and replaces scalars");
check(patched.compat?.thinkingFormat === "openai" && !!patched.thinkingLevelMap, "thinking controls survive on a reasoning model");
const stripped = applyPatch({ ...base, compat: { thinkingFormat: "openai", supportsReasoningEffort: true, maxTokensField: "max_tokens" }, thinkingLevelMap: { off: "none" } }, { reasoning: false });
check(!stripped.compat?.thinkingFormat && !stripped.compat?.supportsReasoningEffort && !stripped.thinkingLevelMap, "turning reasoning off strips every thinking control");
check(stripped.compat?.maxTokensField === "max_tokens", "the request-shape pins survive that strip");

const custom: JsonModel = { ...base, id: "x", name: "Custom X", contextWindow: 2000 };
eq(buildModels([base], [custom], { x: { name: "Patched" } })[0].name, "Patched", "custom replaces base and still receives its patch");
eq(buildModels([base], [], { ghost: { name: "Ghost" } }).map((m: any) => m.id).sort(), ["x", ...shippedGraceIds].sort(), "a patch entry for an unknown id adds no model (only the shipped grace entries ride along)");
eq(buildModels([base], [], {}).map((m: any) => m.id).sort(), ["x", ...shippedGraceIds].sort(), "buildModels is stable with an empty patch");

const live: JsonModel = { ...base, name: "Live", contextWindow: 4000, maxTokens: 4000, cost: { input: 9, output: 0, cacheRead: 0, cacheWrite: 0 } };
const merged = mergeWithEmbedded([live, { ...base, id: "new", name: "New", contextWindow: 10, maxTokens: 10 }], [base, { ...base, id: "only-embedded", name: "Only" }]);
const mergedX = merged.find((m: any) => m.id === "x");
eq([mergedX.name, mergedX.cost.input, mergedX.cost.output, mergedX.contextWindow], ["X", 9, 2, 4000], "live data refreshes reported fields while curated fields and silent costs win");
eq(merged.map((m: any) => m.id), ["x", "new", "only-embedded"], "live-only models are added and embedded-only models are kept");

const sanitized = sanitizeModels([
  { ...base, contextWindow: 0, maxTokens: 0 },
  { ...base, id: "over", contextWindow: 262144, maxTokens: 999999 },
  { ...base, id: "tiny", contextWindow: 1000, maxTokens: 999999 },
  base,
]);
eq([sanitized[0].contextWindow, sanitized[0].maxTokens], [131072, 32768], "an unpublished length falls back to the safe default instead of breaking registration");
eq(sanitized[1].maxTokens, 32768, "a completion cap larger than the window is pulled back inside it");
eq(sanitized[2].maxTokens, 1000, "a window smaller than the default cap keeps the cap inside that window");
check(sanitized[3] === base, "already-valid models are returned untouched (no needless copies)");

const day = 24 * 60 * 60 * 1000;
const stamp = (ageMs: number) => new Date(Date.now() - ageMs).toISOString();
const graveyard: Record<string, JsonModel & { deprecatedAt?: string }> = {
  fresh: { ...base, id: "fresh", deprecatedAt: stamp(day) },
  boundary: { ...base, id: "boundary", deprecatedAt: stamp(DEPRECATED_MODEL_TTL_MS) },
  expired: { ...base, id: "expired", deprecatedAt: stamp(DEPRECATED_MODEL_TTL_MS + 1) },
  broken: { ...base, id: "broken", deprecatedAt: "not-a-date" },
  junk: { deprecatedAt: stamp(day) } as any,
};
eq(selectActiveDeprecated(graveyard).map((m: any) => m.id), ["fresh", "boundary"], "the 14-day grace window keeps day-1 and the exact boundary, evicts past-boundary and unparseable stamps");
check(!("deprecatedAt" in (selectActiveDeprecated(graveyard)[0] as any)), "deprecation metadata never reaches the provider registration");
eq(selectActiveDeprecated({}, Date.now()).length, 0, "an empty graveyard yields nothing");
const now = Date.now();
const freshGraveyard: Record<string, JsonModel & { deprecatedAt?: string }> = {
  fresh: { ...base, id: "fresh", deprecatedAt: new Date(now - day).toISOString() },
  gone: { ...base, id: "gone", deprecatedAt: new Date(now - 40 * day).toISOString() },
};
eq(withDeprecated([base], now, freshGraveyard).map((m: any) => m.id), ["x", "fresh"], "a delisted endpoint inside the grace window keeps working");
eq(withDeprecated([base, { ...base, id: "fresh" }], now, freshGraveyard).map((m: any) => m.id), ["x", "fresh"], "a live listing is never duplicated by the graveyard");
eq(buildModels([base], [], {}).length, 1 + shippedGraceIds.length, "the shipped graveyard contributes exactly its active grace entries");

// ─── D. Live /models transforms ───────────────────────────────────────────────

section("D. live /models transforms");

const openaiEntry = transformApiModel({ id: "MixedCase-Endpoint", object: "model", owned_by: "inferx" });
eq([openaiEntry?.id, openaiEntry?.name, openaiEntry?.reasoning, openaiEntry?.contextWindow], ["MixedCase-Endpoint", "MixedCase-Endpoint", false, 0], "a bare OpenAI model record keeps its case and defaults capability conservatively");
const vllmEntry = transformApiModel({ id: "y", name: "InferX: Y", context_length: 262144, max_completion_tokens: 131072, reasoning: true, pricing: { prompt: "0.044", completion: "0.088", cache_prompt: "0.0014" } });
eq([vllmEntry?.name, vllmEntry?.contextWindow, vllmEntry?.maxTokens, vllmEntry?.reasoning], ["Y", 262144, 131072, true], "vLLM-style extras (length, cap, reasoning flag) are read when present");
eq(vllmEntry?.cost, { input: 0.044, output: 0.088, cacheRead: 0.0014, cacheWrite: 0 }, "string prices are parsed without rounding sub-cent cache rates away");
eq(transformApiModel({ name: "no id" }), null, "a record without an id is dropped");
eq(transformApiModel(null), null, "null is dropped");
eq(transformApiModel("string"), null, "a non-object is dropped");

// ─── E. Provider registration + hot swap ──────────────────────────────────────

section("E. provider registration");

eq(resolveBaseUrl(), "https://model.inferx.net/endpoints/v1", "the shared token-billed surface is the default base URL");
process.env.INFERX_BASE_URL = "  https://model.inferx.net/funccall/acme/endpoints/glm-5.3-flash/v1  ";
eq(resolveBaseUrl(), "https://model.inferx.net/funccall/acme/endpoints/glm-5.3-flash/v1", "INFERX_BASE_URL selects a dedicated per-tenant endpoint (and is trimmed)");
process.env.INFERX_BASE_URL = "   ";
eq(resolveBaseUrl(), "https://model.inferx.net/endpoints/v1", "a blank override falls back to the shared surface");
delete process.env.INFERX_BASE_URL;

const registered: Array<{ id: string; config: any }> = [];
const handlers: Record<string, Array<(event: any, ctx: any) => any>> = {};
const fakePi = {
  registerProvider: (id: string, config: any) => registered.push({ id, config }),
  on: (event: string, handler: any) => {
    handlers[event] = handlers[event] || [];
    handlers[event].push(handler);
  },
};

const extension = index.default;
extension(fakePi as any);

eq(registered.length, 1, "registration happens once, synchronously, at load");
const first = registered[0];
eq([first.id, first.config.api, first.config.apiKey, first.config.name], ["inferx", "openai-completions", "$INFERX_API_KEY", "InferX"], "provider identity matches the public integration docs");
eq(first.config.baseUrl, "https://model.inferx.net/endpoints/v1", "the shared surface is registered");
eq(first.config.models.length, EXPECTED_REGISTERED_IDS.length, "all endpoints are available before any network call");
check(first.config.models.every((m: any) => m.contextWindow > 0 && m.maxTokens > 0), "the registered catalog clears pi's validation");
check(typeof handlers.session_start?.[0] === "function" && typeof handlers.session_shutdown?.[0] === "function", "session_start and session_shutdown handlers are wired");

// Background refresh must hot-swap the catalog and must never touch the real ~/.pi.
const originalFetch = globalThis.fetch;
globalThis.fetch = (async () => ({
  ok: true,
  json: async () => ({ data: [{ id: "glm-5.3-flash", context_length: 262144, pricing: { prompt: "0.02" } }, { id: "brand-new-endpoint" }] }),
})) as any;
const ctx = {
  modelRegistry: { getApiKeyForProvider: async (id: string) => (id === "inferx" ? "ix_test_key" : undefined) },
};

await handlers.session_start[0]({}, ctx);
for (let i = 0; i < 50 && registered.length < 2; i++) await new Promise((r) => setTimeout(r, 20));

eq(registered.length, 2, "session_start re-registers once the live catalog lands");
const swapped = registered[1].config.models;
eq(swapped.length, EXPECTED_REGISTERED_IDS.length + 1, "a new endpoint from the live listing is added without dropping the curated ones");
const glm = swapped.find((m: any) => m.id === "glm-5.3-flash");
eq([glm.cost.input, glm.name, glm.reasoning], [0.02, "GLM 5.3 Flash", true], "live pricing is adopted while curated identity and capability survive");
check(fs.existsSync(path.join(tmpHome, "cache", "inferx-models.json")), "the refreshed catalog is cached under the isolated agent dir");

// A keyless session must leave the stale catalog alone (InferX answers 401 without a key).
const before = registered.length;
globalThis.fetch = (async () => ({ ok: false, status: 401, json: async () => ({}) })) as any;
await handlers.session_start[0]({}, { modelRegistry: { getApiKeyForProvider: async () => undefined } });
await new Promise((r) => setTimeout(r, 120));
eq(registered.length, before, "no key means no re-registration: the embedded catalog stays authoritative");

globalThis.fetch = originalFetch;
await handlers.session_shutdown[0]({}, ctx);

// ─── Result ───────────────────────────────────────────────────────────────────

fs.rmSync(tmpHome, { recursive: true, force: true });
console.log(`\n${failures.length === 0 ? "\u2713" : "\u2717"} ${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  console.error("\nFailures:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
