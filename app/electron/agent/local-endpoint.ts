// Local inference endpoints (VST-015): a local provider whose server is not listening must
// refuse the run BEFORE the first model call with a named, actionable error — not die
// mid-run with a generic "fetch failed". This is the availability half of treating
// providers as resources; the security half (private-IP/DNS checks for CLOUD providers)
// lives in security.ts assertProviderEndpointResolved, which deliberately skips local kinds.
//
// Semantics: "unavailable" means NOTHING is listening (connection refused / timeout / DNS
// failure). Any HTTP response — even a 404 — proves a server is up, so the run proceeds and
// protocol-level errors surface through the normal request path with their own messages.

import type { Provider } from './types'
import { createLocalProviderFetch } from './secure-fetch'
import { discardResponseBody } from './pinned-fetch'

export const LOCAL_ENDPOINT_PROBE_TIMEOUT_MS = 1500

// CLI-backed engines have no HTTP endpoint to probe — the same transport set
// assertProviderEndpointResolved (security.ts) exempts from its DNS check.
const CLI_TRANSPORTS = new Set(['codex-cli', 'claude-cli', 'opencode', 'kilo', 'pi-cli'])

export async function assertLocalEndpointAvailable(provider: Provider, fetchImpl: typeof fetch = fetch): Promise<void> {
  if (provider.kind !== 'local') return
  if (CLI_TRANSPORTS.has(provider.transport ?? '')) return
  // The built-in engine's base URL is a placeholder until the run starts the engine on a fresh
  // port; streamChat's engine route serves the model and names its own failures.
  if (provider.source === 'tensorfold') return
  const base = String(provider.baseUrl ?? '').trim()
  if (!base) return // no endpoint configured — the missing-baseUrl error belongs to settings validation
  try {
    const response = await createLocalProviderFetch(base, fetchImpl)(base, { method: 'GET', redirect: 'manual', signal: AbortSignal.timeout(LOCAL_ENDPOINT_PROBE_TIMEOUT_MS) })
    await discardResponseBody(response)
  } catch (err) {
    const cause = (err as { cause?: { code?: string; message?: string } })?.cause
    const reason = cause?.code ?? cause?.message ?? (err instanceof Error && err.name === 'TimeoutError' ? 'no response within 1.5s' : (err as Error)?.message) ?? 'connection failed'
    throw new Error(`Local endpoint ${base} is unavailable (${reason}). Start the local server (e.g. \`ollama serve\` or LM Studio) or fix the URL in Settings → Providers.`)
  }
}
