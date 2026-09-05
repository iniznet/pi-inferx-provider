/**
 * InferX Provider Extension
 *
 * Registers InferX (model.inferx.net) as a custom provider using the openai-completions API.
 * Base URL: https://model.inferx.net/endpoints/v1
 *
 * InferX publishes serverless, token-billed inference endpoints, each one a stock vLLM
 * server (the public catalog pages embed the exact launch command in their "Model Spec").
 * Model ids are the case-sensitive endpoint slugs (`Agents-A1`, `Qwen3.8-27B-FP8`,
 * `gemma-4-31B-it-fp8`), so they must never be lowercased.
 *
 * Key API differences from OpenAI-native endpoints:
 *   - Uses `max_tokens` (the published sample request body), not `max_completion_tokens`
 *   - `store` is not part of the vLLM request surface
 *   - Chat templates take the `system` role; `developer` is not universally mapped
 *   - Reasoning is enabled per endpoint by `--reasoning-parser <family>` and is controlled
 *     through `reasoning_effort`, whose accepted vocabulary is parser-owned
 *     (`none|low|medium|high`, plus `max` on some builds) — pi's own level names
 *     `minimal`/`xhigh` are not accepted, hence the explicit thinkingLevelMap entries
 *   - `/endpoints/v1/models` is tenant-key-gated: it answers 401 without an `ix_` key, so
 *     the catalog cannot be refreshed anonymously (unlike the sibling providers this is
 *     modelled on). Without a key the embedded models.json is the catalog of record.
 *   - Endpoints scale to zero (`min_replica: 0` in the published Model Spec policy), so the
 *     first request after idle pays a cold-start TTFT; there is no key-authenticated usage
 *     or credits endpoint to surface (verified 404), so this extension shows no balance.
 *
 * Model resolution strategy: Stale-While-Revalidate
 *   1. Serve stale immediately: disk cache -> embedded models.json (zero-latency)
 *   2. Revalidate in background: live /models -> merge with embedded -> cache -> hot-swap
 *   3. patch.json + custom-models.json applied on top of whichever source won
 *
 * Merge order: [live|cache|embedded] -> + grace-period deprecated -> apply patch.json
 *              -> merge custom-models.json -> sanitize
 *
 * Usage:
 *   # Option 1: Store in auth.json (recommended)
 *   # Add to ~/.pi/agent/auth.json:
 *   #   "inferx": { "type": "api_key", "key": "ix_your-api-key" }
 *
 *   # Option 2: Set as environment variable (the name InferX publishes in its own docs)
 *   export INFERX_API_KEY=ix_your-api-key
 *
 *   # Optional: point at a dedicated GPU endpoint instead of the shared surface
 *   export INFERX_BASE_URL=https://model.inferx.net/funccall/<tenant>/endpoints/<model>/v1
 *
 *   # Run pi with the extension
 *   pi -e /path/to/pi-inferx-provider
 *
 * @see https://model.inferx.net/catalog/endpoints
 */

import { getAgentDir, type ExtensionAPI, type ModelRegistry } from "@earendil-works/pi-coding-agent";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };
import deprecatedData from "./deprecated-models.json" with { type: "json" };
import fs from "fs";
import path from "path";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface JsonModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: {
    supportsDeveloperRole?: boolean;
    supportsStore?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    thinkingFormat?: "openai" | "zai" | "qwen" | "qwen-chat-template";
    supportsReasoningEffort?: boolean;
  };
}

export interface PatchEntry {
  name?: string;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  contextWindow?: number;
  maxTokens?: number;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: Record<string, unknown>;
}

export type PatchData = Record<string, PatchEntry>;

// ─── Patch Application ────────────────────────────────────────────────────────

export function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
  const result: JsonModel = { ...model };

  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = { ...patch.thinkingLevelMap };

  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }

  // Thinking controls are meaningless on a non-reasoning endpoint and pi would still send
  // reasoning_effort if the flags survived a patch that turned reasoning off.
  if (!result.reasoning) {
    if (result.compat?.thinkingFormat) delete result.compat.thinkingFormat;
    if (result.compat?.supportsReasoningEffort) delete result.compat.supportsReasoningEffort;
    delete result.thinkingLevelMap;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }

  return result;
}

// ─── Sanitisation ─────────────────────────────────────────────────────────────

// pi's provider composer throws `invalid contextWindow` for any model whose contextWindow
// is <= 0 (dist/core/provider-composer.js). InferX publishes no context length for some
// endpoints and a live /models entry may carry none at all, so a floor is mandatory here:
// without it a single incomplete upstream record would break registration of the whole
// provider and the user would lose every InferX model.
const FALLBACK_CONTEXT_WINDOW = 131072;
const FALLBACK_MAX_TOKENS = 32768;

export function sanitizeModels(models: JsonModel[]): JsonModel[] {
  return models.map((model) => {
    const contextWindow = model.contextWindow > 0 ? model.contextWindow : FALLBACK_CONTEXT_WINDOW;
    // maxTokens must stay inside the window or pi can request a completion the endpoint
    // cannot serve (vLLM rejects max_tokens larger than the remaining context).
    const maxTokens =
      model.maxTokens > 0 && model.maxTokens <= contextWindow
        ? model.maxTokens
        : Math.min(contextWindow, FALLBACK_MAX_TOKENS);
    if (contextWindow === model.contextWindow && maxTokens === model.maxTokens) return model;
    return { ...model, contextWindow, maxTokens };
  });
}

/** Full pipeline: base models -> grace-period deprecated -> patch -> custom -> sanitize */
export function buildModels(base: JsonModel[], custom: JsonModel[], patch: PatchData): JsonModel[] {
  const modelMap = new Map<string, JsonModel>();

  // Seed with the base list plus grace-period deprecated models so patch.json entries
  // apply to deprecated models exactly as while the model was live (withDeprecated keeps
  // live data on id conflicts).
  for (const model of withDeprecated(base)) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patch)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of custom) {
    const existing = modelMap.get(model.id);
    const patchEntry = patch[model.id];
    if (existing && patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else if (existing) {
      modelMap.set(model.id, model);
    } else if (patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else {
      modelMap.set(model.id, model);
    }
  }

  return sanitizeModels(Array.from(modelMap.values()));
}

// ─── Stale-While-Revalidate Model Sync ────────────────────────────────────────

export const PROVIDER_ID = "inferx";
export const PROVIDER_NAME = "InferX";
const SHARED_BASE_URL = "https://model.inferx.net/endpoints/v1";

/**
 * Dedicated per-tenant endpoints are served under
 * `https://model.inferx.net/funccall/<tenant>/endpoints/<model>/v1`, which pi cannot infer
 * (the tenant is account-specific), so the shared surface is the default and the
 * environment overrides it. pi interpolates `$ENV` references in `apiKey` and headers only
 * — `baseUrl` is used verbatim — hence reading process.env here.
 */
export function resolveBaseUrl(): string {
  const configured = process.env.INFERX_BASE_URL?.trim();
  return configured && configured.length > 0 ? configured : SHARED_BASE_URL;
}

const LIVE_FETCH_TIMEOUT_MS = 8000;
const CACHE_DIR = path.join(getAgentDir(), "cache");
const CACHE_PATH = path.join(CACHE_DIR, `${PROVIDER_ID}-models.json`);

/**
 * Transform a model from the InferX /models response.
 *
 * The route is tenant-key-gated, so its payload could not be captured without a credential.
 * It is an OpenAI-compatible `/models` listing, therefore the reader accepts the OpenAI
 * shape (`{ data: [{ id, owned_by }] }`) plus the vLLM/SGLang extensions this ecosystem
 * publishes alongside it (`context_length`, `max_completion_tokens`, `pricing.*`). Every
 * capability field defaults to the conservative value and is corrected by patch.json —
 * exactly the split the sibling providers use.
 */
export function transformApiModel(apiModel: unknown): JsonModel | null {
  if (!apiModel || typeof apiModel !== "object") return null;
  const raw = apiModel as Record<string, any>;
  const id = typeof raw.id === "string" ? raw.id : typeof raw.model === "string" ? raw.model : null;
  if (!id) return null;

  const numberOr = (...values: unknown[]): number => {
    for (const value of values) {
      const n = typeof value === "string" ? Number.parseFloat(value) : (value as number);
      if (typeof n === "number" && Number.isFinite(n) && n > 0) return n;
    }
    return 0;
  };

  // InferX quotes prices per million tokens; keep six decimals so sub-cent cache rates
  // (e.g. 0.0042) survive instead of rounding to zero.
  const priceOr = (...values: unknown[]): number => {
    const n = numberOr(...values);
    return n > 0 ? Math.round(n * 1e6) / 1e6 : 0;
  };

  const pricing = (raw.pricing ?? raw.price ?? raw.cost ?? {}) as Record<string, unknown>;
  const contextWindow = numberOr(raw.context_length, raw.context_window, raw.max_model_len, raw.context);
  const maxTokens = numberOr(raw.max_completion_tokens, raw.max_tokens);

  return {
    id,
    name: typeof raw.name === "string" && raw.name
      ? raw.name.replace(/^[^:]+:\s*/, "")
      : id,
    reasoning: raw.reasoning === true || raw.custom_reasoning === true || raw.reasoning_effort === true,
    input: ["text"],
    cost: {
      input: priceOr(pricing.prompt, pricing.input, raw.prompt_price),
      output: priceOr(pricing.completion, pricing.output, raw.completion_price),
      cacheRead: priceOr(pricing.cache_prompt, pricing.cacheRead, raw.cache_price),
      cacheWrite: priceOr(pricing.cache_completion, pricing.cacheWrite),
    },
    contextWindow,
    maxTokens: maxTokens || contextWindow,
  };
}

export async function fetchLiveModels(apiKey: string | undefined, baseUrl: string, signal?: AbortSignal): Promise<JsonModel[] | null> {
  if (!apiKey) return null;
  try {
    const timeout = AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS);
    const response = await fetch(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
    });
    if (!response.ok) return null;
    const data = await response.json();
    const apiModels: unknown = Array.isArray(data) ? data : (data as { data?: unknown }).data;
    if (!Array.isArray(apiModels) || apiModels.length === 0) return null;
    const models = apiModels.map(transformApiModel).filter((m): m is JsonModel => m !== null);
    return models.length > 0 ? models : null;
  } catch {
    // A cold/offline/permission-blocked catalog must never degrade the running provider.
    return null;
  }
}

export function loadCachedModels(): JsonModel[] | null {
  try {
    const data = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

export function cacheModels(models: JsonModel[]): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2) + "\n");
  } catch {
    // Cache write failure is non-fatal
  }
}

export function mergeWithEmbedded(liveModels: JsonModel[], embeddedModels: JsonModel[]): JsonModel[] {
  const embeddedMap = new Map(embeddedModels.map((m) => [m.id, m]));
  const seen = new Set<string>();
  const result: JsonModel[] = [];
  for (const liveModel of liveModels) {
    const embedded = embeddedMap.get(liveModel.id);
    seen.add(liveModel.id);
    if (embedded) {
      // Self-heal: the live listing is authoritative for the fields it reports. Prefer the
      // live cost when non-zero and fall back to the curated one when the API is silent, so
      // curation (reasoning/input/compat/name) is never clobbered by a sparse record.
      result.push({
        ...liveModel,
        ...embedded,
        cost: {
          input: liveModel.cost.input || embedded.cost.input,
          output: liveModel.cost.output || embedded.cost.output,
          cacheRead: liveModel.cost.cacheRead || embedded.cost.cacheRead,
          cacheWrite: liveModel.cost.cacheWrite || embedded.cost.cacheWrite,
        },
        contextWindow: liveModel.contextWindow || embedded.contextWindow,
      });
    } else {
      result.push(liveModel);
    }
  }
  // Append any embedded models that the live API did not return
  for (const em of embeddedModels) {
    if (!seen.has(em.id)) result.push(em);
  }
  return result;
}

// Grace period for delisted models. When the provider API stops listing a model,
// update-models.js moves its last-known definition into deprecated-models.json (stamped
// with deprecatedAt) instead of dropping it. For 14 days the model keeps working here so
// in-flight sessions and saved model settings do not break; afterwards it is evicted.
export const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

// Grace-period deprecated models with deprecation metadata stripped. Split from the
// embedded file read so the window arithmetic is testable without rewriting models.json.
export function selectActiveDeprecated(
  entries: Record<string, JsonModel & { deprecatedAt?: string }>,
  now: number = Date.now(),
): JsonModel[] {
  const result: JsonModel[] = [];
  for (const entry of Object.values(entries)) {
    if (!entry?.id) continue;
    const removedAt = Date.parse(entry.deprecatedAt ?? "");
    if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
    const model = { ...entry } as JsonModel & { deprecatedAt?: string };
    delete model.deprecatedAt;
    result.push(model);
  }
  return result;
}

export function activeDeprecatedModels(now: number = Date.now()): JsonModel[] {
  return selectActiveDeprecated(deprecatedData as Record<string, JsonModel & { deprecatedAt?: string }>, now);
}

// Append grace-period deprecated models the list does not already have (live data wins).
// `entries` is injectable so the append/dedup branches are testable without editing the
// shipped graveyard.
export function withDeprecated(
  models: JsonModel[],
  now: number = Date.now(),
  entries: Record<string, JsonModel & { deprecatedAt?: string }> = deprecatedData as Record<string, JsonModel & { deprecatedAt?: string }>,
): JsonModel[] {
  const seen = new Set(models.map((m) => m.id));
  const extras = selectActiveDeprecated(entries, now).filter((m) => !seen.has(m.id));
  return extras.length > 0 ? [...models, ...extras] : models;
}

export function loadStaleModels(embeddedModels: JsonModel[]): JsonModel[] {
  const cached = loadCachedModels();
  if (!cached || cached.length === 0) return embeddedModels;

  // Merge embedded models that are missing from cache (newly added models)
  const cachedMap = new Map(cached.map((m) => [m.id, m]));
  for (const em of embeddedModels) {
    if (!cachedMap.has(em.id)) cached.push(em);
  }
  return cached;
}

export async function revalidateModels(
  apiKey: string | undefined,
  baseUrl: string,
  embeddedModels: JsonModel[],
  signal?: AbortSignal,
): Promise<JsonModel[] | null> {
  if (!apiKey) return null;
  const liveModels = await fetchLiveModels(apiKey, baseUrl, signal);
  if (!liveModels || liveModels.length === 0) return null;
  const merged = mergeWithEmbedded(liveModels, embeddedModels);
  cacheModels(merged);
  return merged;
}

// ─── API Key Resolution (via ModelRegistry) ───────────────────────────────────

let cachedApiKey: string | undefined;
let revalidateAbort: AbortController | null = null;

export async function resolveApiKey(modelRegistry: ModelRegistry): Promise<string | undefined> {
  cachedApiKey = (await modelRegistry.getApiKeyForProvider(PROVIDER_ID)) ?? undefined;
  return cachedApiKey;
}

// ─── Extension Entry Point ────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchData;
  const baseUrl = resolveBaseUrl();

  const providerConfig = (models: JsonModel[]) => ({
    name: PROVIDER_NAME,
    baseUrl,
    apiKey: "$INFERX_API_KEY",
    api: "openai-completions",
    models,
  });

  const staleBase = loadStaleModels(embeddedModels);
  pi.registerProvider(PROVIDER_ID, providerConfig(buildModels(staleBase, customModels, patches)));

  pi.on("session_start", async (_event, ctx) => {
    revalidateAbort?.abort();
    revalidateAbort = new AbortController();
    const signal = revalidateAbort.signal;

    // Non-blocking: the stale catalog is already registered, so a slow or key-gated
    // /models round trip never delays the first turn.
    void resolveApiKey(ctx.modelRegistry).then(async (apiKey) => {
      const freshBase = await revalidateModels(apiKey, baseUrl, embeddedModels, signal);
      if (freshBase && !signal.aborted) {
        pi.registerProvider(PROVIDER_ID, providerConfig(buildModels(freshBase, customModels, patches)));
      }
    });
  });

  pi.on("session_shutdown", () => {
    revalidateAbort?.abort();
  });
}
