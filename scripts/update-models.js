#!/usr/bin/env node
/**
 * Update InferX models from the provider
 *
 * Fetches the published endpoint catalog and updates:
 * - models.json: base model definitions (no patches baked in)
 * - deprecated-models.json: graveyard for delisted endpoints (14-day grace)
 * - README.md: model table with patch.json overrides applied
 *
 * Why two sources (the sibling providers only need one):
 *
 *   1. GET {BASE_URL}/models  — the authoritative OpenAI-compatible listing, but InferX
 *      gates it on a tenant key (401 "no tenant context" without one), so it only works
 *      when a credential resolves.
 *   2. GET /catalog/endpoints — the public, server-rendered endpoint table. It carries the
 *      case-sensitive slug (the API model name), the published context length and the
 *      upstream owner, but renders price cells as an em dash to anonymous visitors.
 *
 * The two are merged (ids = live union catalog) so the sync works with or without a key.
 *
 * Pricing: InferX bills a per-tenant promotional discount on top of list price and exposes
 * neither over a keyless route, so cost is *preserved curated data* (see transformModel):
 * whatever models.json already says for a known id survives a sync unless a source actually
 * reports a price. Curated capability (reasoning / vision / thinking levels / compat) lives
 * in patch.json and is never written here.
 *
 * API key: optional. Resolved from INFERX_API_KEY, then from the `inferx` credential in
 * ~/.pi/agent/auth.json (literal value or a `$VAR` reference; shell-command values are
 * deliberately not executed by this script).
 */

import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROVIDER_ID = 'inferx';
const SHARED_BASE_URL = process.env.INFERX_BASE_URL?.trim() || 'https://model.inferx.net/endpoints/v1';
const MODELS_API_URL = `${SHARED_BASE_URL}/models`;
const CATALOG_URL = 'https://model.inferx.net/catalog/endpoints';
const FETCH_TIMEOUT_MS = 15000;

const MODELS_JSON_PATH = path.join(__dirname, '..', 'models.json');
const PATCH_JSON_PATH = path.join(__dirname, '..', 'patch.json');
const CUSTOM_MODELS_JSON_PATH = path.join(__dirname, '..', 'custom-models.json');
const DEPRECATED_JSON_PATH = path.join(__dirname, '..', 'deprecated-models.json');
const README_PATH = path.join(__dirname, '..', 'README.md');

// pi rejects contextWindow <= 0 and a completion cap larger than the window, so the base
// catalog is written with the same clamp the runtime applies (index.ts sanitizeModels).
const DEFAULT_MAX_TOKENS_CEILING = 131072;

/** Humanise an endpoint slug for a model that has no curated display name yet. */
function humanize(id) {
  return id
    .replace(/[-_.]+/g, ' ')
    .replace(/\b([a-z])/g, (m, c) => c.toUpperCase())
    .trim();
}

/**
 * Parse a catalog price cell.
 *
 * Anonymous visitors see an em dash. A tenant-rendered cell shows list price struck through
 * next to the billed price, so the LAST quoted number is the one actually charged — which is
 * what pi should cost-estimate against.
 */
function parsePriceCell(text) {
  const prices = (text.match(/\$\s*([0-9]*\.?[0-9]+)/g) || []).map((s) => Number.parseFloat(s.replace(/[^0-9.]/g, '')));
  const last = prices[prices.length - 1];
  return Number.isFinite(last) ? Math.round(last * 1e6) / 1e6 : 0;
}

/**
 * Parse the public /catalog/endpoints table.
 *
 * Row shape (verified against a saved copy of the live page):
 *   <tr data-search="<lowercased slug> <provider> …" data-tags="…" data-use-cases="…" data-provider="…">
 *     <td><a class="endpoint-slug-link" href="/catalog/endpoints/<SLUG>?tenant=public"><name></a></td>
 *     <td>context length | —</td>
 *     <td class="endpoint-discount-cell">…</td>
 *     <td class="endpoint-price-cell">input / 1M</td>
 *     <td class="endpoint-price-cell">output / 1M</td>
 *     <td class="endpoint-price-cell">cached / 1M</td>
 *   </tr>
 *
 * The id MUST come from the href: data-search is lowercased and would corrupt the five
 * mixed-case slugs (Agents-A1, Qwen3.8-27B-FP8, …) that InferX treats case-sensitively.
 */
export function parseCatalog(html) {
  const body = html.split('id="endpoint-table-body"')[1];
  if (!body) return [];
  const rows = body.split('</tbody>')[0];
  const models = [];
  for (const row of rows.matchAll(/<tr([\s\S]*?)<\/tr>/g)) {
    const block = row[1];
    const href = block.match(/href="\/catalog\/endpoints\/([^?"]+)/);
    if (!href) continue;
    const id = decodeURIComponent(href[1]);
    const attr = (name) => {
      const m = block.match(new RegExp(name + '="([^"]*)"'));
      return m ? m[1] : '';
    };
    const cells = [...block.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((c) =>
      c[1].replace(/<[^>]+>/g, ' ').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim(),
    );
    if (cells.length === 0) continue;
    const context = Number.parseInt(cells[1] || '', 10);
    models.push({
      id,
      name: cells[0] || id,
      provider: attr('data-provider'),
      contextWindow: Number.isFinite(context) && context > 0 ? context : 0,
      cost: {
        input: parsePriceCell(cells[3] || ''),
        output: parsePriceCell(cells[4] || ''),
        cacheRead: parsePriceCell(cells[5] || ''),
      },
    });
  }
  return models;
}

async function fetchText(url, headers = {}) {
  const response = await fetch(url, { headers, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.text();
}

async function fetchCatalogModels() {
  try {
    const models = parseCatalog(await fetchText(CATALOG_URL));
    if (models.length === 0) throw new Error('catalog table contained no rows');
    console.log(`\u2713 Fetched ${models.length} endpoints from the public catalog`);
    return models;
  } catch (error) {
    console.warn(`\u26a0 Public catalog unavailable (${error.message}); falling back to the key-gated API alone`);
    return [];
  }
}

/** OpenAI-style listing transformed to base records; capability flags are unreliable. */
function parseLiveModels(payload) {
  const list = Array.isArray(payload) ? payload : payload?.data;
  if (!Array.isArray(list)) return [];
  const numberOr = (v) => {
    const n = typeof v === 'string' ? Number.parseFloat(v) : v;
    return typeof n === 'number' && Number.isFinite(n) && n > 0 ? n : 0;
  };
  const models = [];
  for (const entry of list) {
    const id = typeof entry?.id === 'string' ? entry.id : typeof entry?.model === 'string' ? entry.model : null;
    if (!id) continue;
    const pricing = entry.pricing || entry.price || entry.cost || {};
    const contextWindow = numberOr(entry.context_length) || numberOr(entry.context_window) || numberOr(entry.max_model_len);
    models.push({
      id,
      name: typeof entry.name === 'string' && entry.name ? entry.name.replace(/^[^:]+:\s*/, '') : id,
      provider: typeof entry.owned_by === 'string' ? entry.owned_by : '',
      contextWindow,
      maxTokens: numberOr(entry.max_completion_tokens) || numberOr(entry.max_tokens),
      cost: {
        input: numberOr(pricing.prompt) || numberOr(pricing.input),
        output: numberOr(pricing.completion) || numberOr(pricing.output),
        cacheRead: numberOr(pricing.cache_prompt) || numberOr(pricing.cacheRead),
      },
    });
  }
  return models;
}

async function fetchLiveModels(apiKey) {
  if (!apiKey) {
    console.log('- Skipping /models (no API key; the route is tenant-gated and answers 401 anonymously)');
    return [];
  }
  try {
    const payload = JSON.parse(await fetchText(MODELS_API_URL, { Authorization: `Bearer ${apiKey}` }));
    const models = parseLiveModels(payload);
    console.log(`\u2713 Fetched ${models.length} models from ${MODELS_API_URL}`);
    return models;
  } catch (error) {
    console.warn(`\u26a0 Live /models failed (${error.message}); using the public catalog only`);
    return [];
  }
}

/**
 * Resolve the API key without executing config-file shell commands.
 * Supports a literal `key` or a `$VAR` / `${VAR}` reference into the environment.
 */
function resolveApiKey() {
  if (process.env.INFERX_API_KEY) return process.env.INFERX_API_KEY;
  const agentDir = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), '.pi', 'agent');
  try {
    const auth = JSON.parse(fs.readFileSync(path.join(agentDir, 'auth.json'), 'utf8'));
    const entry = auth?.[PROVIDER_ID];
    const key = typeof entry === 'string' ? entry : entry?.key;
    if (typeof key !== 'string' || key.length === 0) return undefined;
    if (key.startsWith('!')) {
      console.warn('\u26a0 auth.json stores the inferx key as a shell command; export INFERX_API_KEY instead');
      return undefined;
    }
    const ref = key.match(/^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/);
    if (ref) return process.env[ref[1]] || undefined;
    return key;
  } catch {
    return undefined;
  }
}

/**
 * Apply patch.json overrides on top of transformed models (README generation only).
 * Deep merges compat/cost, replaces everything else — mirrors applyPatch() in index.ts.
 */
function applyPatch(model, patch) {
  const result = { ...model };
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
  if (patch.compat) result.compat = { ...(result.compat || {}), ...patch.compat };
  if (!result.reasoning && result.compat?.thinkingFormat) delete result.compat.thinkingFormat;
  if (!result.reasoning && result.compat?.supportsReasoningEffort) delete result.compat.supportsReasoningEffort;
  if (!result.reasoning && result.thinkingLevelMap) delete result.thinkingLevelMap;
  if (result.compat && Object.keys(result.compat).length === 0) delete result.compat;
  return result;
}

function buildModels(baseModels, customModels, patchData) {
  const modelMap = new Map();
  for (const model of withDeprecatedForReadme(baseModels)) modelMap.set(model.id, model);
  for (const [id, patchEntry] of Object.entries(patchData)) {
    const existing = modelMap.get(id);
    if (existing) modelMap.set(id, applyPatch(existing, patchEntry));
  }
  for (const model of customModels) {
    const patchEntry = patchData[model.id];
    modelMap.set(model.id, patchEntry ? applyPatch(model, patchEntry) : model);
  }
  const seen = new Set();
  return Array.from(modelMap.values()).filter((model) => {
    if (seen.has(model.id)) return false;
    seen.add(model.id);
    return true;
  });
}

/**
 * Fold the two sources into base records, preserving curated data for known ids.
 *
 * A source only overwrites a field when it actually reports one, so prices and the context
 * window of endpoints InferX leaves unpublished (deepseek-v4-flash-0731 publishes no length)
 * survive every sync instead of decaying to zero.
 */
function transformModel(source, existing) {
  const model = existing ? { ...existing } : {
    id: source.id,
    name: humanize(source.id),
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 0,
    maxTokens: 0,
    _meta: {},
  };

  if (source.contextWindow > 0) model.contextWindow = source.contextWindow;
  if (source.maxTokens > 0) model.maxTokens = source.maxTokens;
  if (source.cost?.input > 0) model.cost = { ...model.cost, input: source.cost.input };
  if (source.cost?.output > 0) model.cost = { ...model.cost, output: source.cost.output };
  if (source.cost?.cacheRead > 0) model.cost = { ...model.cost, cacheRead: source.cost.cacheRead };
  if (model.contextWindow > 0 && !(model.maxTokens > 0)) {
    model.maxTokens = Math.min(model.contextWindow, DEFAULT_MAX_TOKENS_CEILING);
  }
  model._meta = {
    isFree: (model.cost?.input || 0) === 0 && (model.cost?.output || 0) === 0,
    provider: source.provider || existing?._meta?.provider || undefined,
  };
  return model;
}

function loadJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return fallback;
  }
}

function formatCost(cost, isFree) {
  if (isFree) return '**Free**';
  if (cost === 0 || cost === null || cost === undefined) return '\u2014';
  // InferX bills sub-cent rates (0.0014, 0.024), so cent rounding would print $0.00 for a
  // price the user actually pays. Two decimals above a dollar, exact-but-trimmed below.
  return '$' + (cost >= 1 ? cost.toFixed(2) : cost.toFixed(6).replace(/0+$/, '').replace(/\.$/, ''));
}

function formatContextWindow(n) {
  if (!n) return '\u2014';
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

function generateReadmeTable(models) {
  const lines = [
    '| Model | Context | Vision | Reasoning | Input $/M | Cache Read $/M | Output $/M |',
    '|-------|---------|--------|-----------|-----------|-----------------|------------|',
  ];
  for (const model of models) {
    lines.push([
      `| ${model.name}`,
      formatContextWindow(model.contextWindow),
      model.input.includes('image') ? '\u2705' : '\u274c',
      model.reasoning ? '\u2705' : '\u274c',
      formatCost(model.cost.input, model._meta?.isFree),
      formatCost(model.cost.cacheRead, model._meta?.isFree),
      formatCost(model.cost.output, model._meta?.isFree) + ' |',
    ].join(' | '));
  }
  return lines.join('\n');
}

function updateReadme(models) {
  let readme = fs.readFileSync(README_PATH, 'utf8');
  const newTable = generateReadmeTable(models);
  const tableRegex = /(## Available Models\n\n)\| Model \| Context \| Vision \| Reasoning \| Input \$\/M \| Cache Read \$\/M \| Output \$\/M \|\n\|[-| ]+\|(\n\|[^\n]+\|)*\n*/;
  if (!tableRegex.test(readme)) {
    console.warn('\u26a0 Could not find the model table in the "## Available Models" section');
    return;
  }
  readme = readme.replace(tableRegex, (match, header) => `${header}${newTable}\n\n`);
  fs.writeFileSync(README_PATH, readme);
  console.log('\u2713 Updated README.md');
}

function cleanModelForJson(model) {
  const { _meta, ...cleanModel } = model;
  return cleanModel;
}

// Grace period for delisted models: an endpoint InferX unpublishes moves to
// deprecated-models.json (stamped deprecatedAt) instead of disappearing; the runtime
// appends those back for 14 days so saved model settings keep working, then evicts them.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

function updateDeprecatedModels(newModels) {
  const oldModels = loadJson(MODELS_JSON_PATH, []);
  const deprecated = loadJson(DEPRECATED_JSON_PATH, {});
  const currentIds = new Set(newModels.map((m) => m.id));
  const now = new Date().toISOString();
  const added = [], resurrected = [], evicted = [];

  for (const old of Array.isArray(oldModels) ? oldModels : []) {
    if (old && old.id && !currentIds.has(old.id) && !deprecated[old.id]) {
      deprecated[old.id] = { ...old, deprecatedAt: now };
      added.push(old.id);
    }
  }
  for (const [id, entry] of Object.entries(deprecated)) {
    if (currentIds.has(id)) {
      delete deprecated[id];
      resurrected.push(id);
      continue;
    }
    const removedAt = Date.parse(entry && entry.deprecatedAt ? entry.deprecatedAt : '');
    if (Number.isNaN(removedAt) || Date.now() - removedAt > DEPRECATED_MODEL_TTL_MS) {
      delete deprecated[id];
      evicted.push(id);
    }
  }

  if (added.length || resurrected.length || evicted.length) {
    fs.writeFileSync(DEPRECATED_JSON_PATH, JSON.stringify(deprecated, null, 2) + '\n');
    console.log('Updated deprecated-models.json ' + JSON.stringify({ added, resurrected, evicted }));
  }
}

function withDeprecatedForReadme(models) {
  const deprecated = loadJson(DEPRECATED_JSON_PATH, {});
  const now = Date.now();
  const seen = new Set(models.map((m) => m.id));
  const extras = [];
  for (const entry of Object.values(deprecated)) {
    if (!entry || !entry.id || seen.has(entry.id)) continue;
    const removedAt = Date.parse(entry.deprecatedAt || '');
    if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
    const model = { ...entry };
    delete model.deprecatedAt;
    extras.push(model);
  }
  return extras.length ? [...models, ...extras] : models;
}

async function main() {
  console.log(`Syncing InferX endpoints (${SHARED_BASE_URL})...`);
  const apiKey = resolveApiKey();

  const [live, catalog] = await Promise.all([fetchLiveModels(apiKey), fetchCatalogModels()]);
  if (live.length === 0 && catalog.length === 0) {
    console.error('Error: neither the live /models listing nor the public catalog returned models');
    process.exit(1);
  }

  const existing = loadJson(MODELS_JSON_PATH, []);
  const existingMap = {};
  for (const model of Array.isArray(existing) ? existing : []) existingMap[model.id] = model;

  // Catalog first: the live listing is authoritative for existence, the catalog for the
  // published context length, so folding catalog over live keeps both.
  const byId = new Map();
  for (const source of live) byId.set(source.id, source);
  for (const source of catalog) {
    byId.set(source.id, { ...byId.get(source.id), ...source, reportedReasoning: byId.get(source.id)?.reportedReasoning ?? false });
  }

  const transformed = [...byId.values()].map((source) => transformModel(source, existingMap[source.id]));
  // Surface the upstream owners the catalog reported: they are the signal that an endpoint
  // was re-pointed at different weights, which is when patch.json curation needs re-checking.
  const owners = new Set(transformed.map((m) => m._meta?.provider).filter(Boolean));
  transformed.sort((a, b) => {
    const aFree = !!a._meta?.isFree;
    const bFree = !!b._meta?.isFree;
    if (aFree && !bFree) return 1;
    if (!aFree && bFree) return -1;
    return a.id.localeCompare(b.id);
  });

  const cleanModels = transformed.map(cleanModelForJson);
  // Must run BEFORE models.json is overwritten: it diffs against the previous file.
  updateDeprecatedModels(cleanModels);
  fs.writeFileSync(MODELS_JSON_PATH, JSON.stringify(cleanModels, null, 2) + '\n');
  console.log(`\u2713 Updated models.json (${cleanModels.length} base models)`);

  const patch = loadJson(PATCH_JSON_PATH, {});
  const custom = loadJson(CUSTOM_MODELS_JSON_PATH, []);
  const readmeModels = buildModels(transformed, Array.isArray(custom) ? custom : [], patch);
  readmeModels.sort((a, b) => a.name.localeCompare(b.name));
  updateReadme(readmeModels);

  console.log('\n--- Summary ---');
  console.log(`Total models: ${readmeModels.length}`);
  console.log(`Reasoning models (patched): ${readmeModels.filter((m) => m.reasoning).length}`);
  console.log(`Vision models: ${readmeModels.filter((m) => m.input.includes('image')).length}`);
  console.log(`Free (fully discounted) models: ${readmeModels.filter((m) => m._meta?.isFree).length}`);
  console.log(`Upstream owners reported by the catalog: ${owners.size ? [...owners].join(', ') : 'none'}`);

  const oldIds = new Set(Array.isArray(existing) ? existing.map((m) => m.id) : []);
  const added = [...byId.keys()].filter((id) => !oldIds.has(id));
  const removed = [...oldIds].filter((id) => !byId.has(id));
  if (added.length) console.log(`\nNew endpoints: ${added.join(', ')}`);
  if (removed.length) console.log(`\nDelisted endpoints (now in the grace period): ${removed.join(', ')}`);
}

// Only run when executed directly, so tests can import the parser helpers.
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((error) => {
    console.error('Error:', error.message);
    process.exit(1);
  });
}
