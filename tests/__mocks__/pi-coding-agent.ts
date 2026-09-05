/**
 * Runtime stub for the @earendil-works/pi-coding-agent peer dependency, used only by
 * `scripts/test-model-resolution.ts` (redirected through tests/loader.mjs).
 *
 * index.ts calls getAgentDir() at module scope to build CACHE_DIR, and the published
 * package barrel pulls in optional server dependencies that plain `node` cannot resolve,
 * so the test suite redirects the bare specifier here. pi itself always supplies the real
 * module when the extension is loaded, and `npm run check` type-checks against the real
 * package installed from package.json - never against this file.
 */

import os from "os";
import path from "path";

/** Mirrors the real getAgentDir(): respects PI_CODING_AGENT_DIR, defaults to ~/.pi/agent. */
export function getAgentDir(): string {
  return process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
}

export interface ModelRegistry {
  getApiKeyForProvider(provider: string): Promise<string | undefined>;
}

export interface ExtensionContext {
  modelRegistry: ModelRegistry;
  [key: string]: any;
}

export interface ExtensionAPI {
  registerProvider(name: string, provider: any): void;
  on(event: string, handler: (event: any, ctx: ExtensionContext) => any): void;
}
