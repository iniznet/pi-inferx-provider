const SPECIFIER = "@earendil-works/pi-coding-agent";
const STUB = new URL("./__mocks__/pi-coding-agent.ts", import.meta.url);

// Both hook APIs require shortCircuit on a handled specifier: it is what stops Node from
// re-resolving the bare specifier against node_modules.
function redirect() {
  return { url: STUB.href, shortCircuit: true };
}

/** Worker loader hook: redirect only the peer dependency, never anything else. */
export async function resolve(specifier, context, nextResolve) {
  if (specifier === SPECIFIER) {
    return redirect();
  }
  return nextResolve(specifier, context);
}

/** In-thread hook used when node:module#registerHooks is available. */
export const hooks = {
  resolve(specifier, context, nextResolve) {
    return specifier === SPECIFIER ? redirect() : nextResolve(specifier, context);
  },
};
