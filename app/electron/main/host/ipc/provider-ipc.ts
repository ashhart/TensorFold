import { discoverLocalProvider } from '../../../agent/local-providers'
import { providerWithAccount } from '../../../agent/provider-accounts'
import { assertLocalProviderDiscoveryUrl, auditPreview } from '../../../agent/security'
import { modelReadiness } from '../../../agent/cli-auth'
import { checkCliAuth, cliBinaryNames, startCliLogin, cancelCliLogin } from '../../../agent/cli-auth-runtime'
import type { Provider, ProviderDiscoveryKind, VoiceTranscriptionInput } from '../../../agent/types'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { isEngineProvider } from '../../../agent/engine/provider'
import { engineService } from '../engine-runtime'
import { listProviderModels } from '../provider-runtime'
import { discoverLocalTts, synthesizeLocalTts } from '../voice-runtime'

export function registerProviderIpc() {

handle('provider:models', async (_e, provider: Provider) => {
  requireSecurityRole('operator', 'fetch provider models')
  return listProviderModels(provider)
})

handle('provider:discoverLocal', async (_e, payload: ProviderDiscoveryKind | { kind: ProviderDiscoveryKind; baseUrl?: string; apiKey?: string }) => {
  requireSecurityRole('operator', 'discover local providers')
  const kind = typeof payload === 'string' ? payload : payload.kind
  const baseUrl = typeof payload === 'string' ? undefined : payload.baseUrl
  const apiKey = typeof payload === 'string' ? '' : (payload.apiKey ?? '')
  if (!['ollama', 'lmstudio', 'omlx'].includes(kind)) throw new Error('Unsupported local provider discovery kind.')
  // A renderer-supplied baseUrl must point at loopback/LAN — never a public host that would
  // receive the apiKey as a Bearer token (SSRF + credential exfil).
  if (baseUrl) await assertLocalProviderDiscoveryUrl(baseUrl)
  const discovered = await discoverLocalProvider(kind, baseUrl, apiKey)
  await auditEvent('provider.discover_local', {
    kind,
    found: discovered.found,
    baseUrl: auditPreview(discovered.baseUrl),
    models: discovered.models.length,
    hasApiKey: !!discovered.apiKey,
  })
  return discovered
})

handle('provider:discoverOmlx', async () => {
  const discovered = await discoverLocalProvider('omlx')
  await auditEvent('provider.discover_omlx', {
    found: discovered.found,
    baseUrl: auditPreview(discovered.baseUrl),
    models: discovered.models.length,
    hasApiKey: !!discovered.apiKey,
  })
  return discovered
})

// ---- Voice (main-process provider network boundary) ----
// Every provider HTTP call uses the host's protected settings. The renderer supplies only
// text or bounded recorded audio — never a URL, model, provider, or API key.
// Cloud speech would send replies (which can quote confidential work) and the user's own voice
// to an external service, so only the system voice and a local speech server are used.
const CLOUD_VOICE_OFF = 'Cloud voice is off: TensorFold keeps confidential work on this Mac.'
handle('voice:synthesize', async (_e, _payload: { text: string }) => {
  return { ok: false, fallback: true, error: `${CLOUD_VOICE_OFF} Using the system voice.` }
})

handle('voice:localSynthesize', async (_e, payload: { text: string }) => {
  return synthesizeLocalTts(String(payload?.text ?? ''))
})

handle('voice:transcribe', async (_e, _payload: VoiceTranscriptionInput) => {
  return { ok: false, error: CLOUD_VOICE_OFF }
})

handle('voice:discoverLocalTts', async (_e, payload?: { baseUrl?: string }) => {
  if (payload?.baseUrl) await assertLocalProviderDiscoveryUrl(payload.baseUrl)
  return discoverLocalTts(payload?.baseUrl)
})

// Point-of-use guard: is there a usable model for this provider? CLI providers
// resolve auth by spawning `<cli> <status>`; HTTP providers check for a key here
// (where the real, unmasked secret lives).
handle('provider:readiness', async (_e, input: { providerId?: string | null; accountId?: string | null }) => {
  requireSecurityRole('operator', 'check model readiness')
  const providerId = input?.providerId ?? store.settings.activeProviderId
  const base = store.settings.providers.find((p) => p.id === providerId)
  if (!base) return modelReadiness({ provider: null })
  const accountId = input?.accountId !== undefined ? input.accountId : (store.settings.activeProviderAccountId ?? base.defaultAccountId ?? null)
  const provider = providerWithAccount(base, accountId)
  if (isEngineProvider(provider)) {
    const model = providerId === store.settings.activeProviderId ? store.settings.activeModel ?? null : null
    return engineService()?.readiness(model) ?? { ready: false, reason: 'needs-endpoint', message: 'The built-in engine is not available in this process.' }
  }
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
    const auth = await checkCliAuth(provider)
    return { ...modelReadiness({ provider, cliBinaryPresent: auth.binaryPresent, cliLoggedIn: auth.loggedIn }), detail: auth.detail }
  }
  return modelReadiness({ provider, hasApiKey: !!provider.apiKey?.trim() })
})

// Authorize a CLI provider in-app via its OAuth flow (`codex login` /
// `claude auth login`). The CLI opens the browser; this resolves once sign-in
// completes and re-confirms status.
handle('cliAuth:login', async (_e, input: { providerId: string; accountId?: string | null }) => {
  requireSecurityRole('operator', 'authorize CLI provider')
  const base = store.settings.providers.find((p) => p.id === input?.providerId)
  if (!base) throw new Error('Unknown provider.')
  const provider = providerWithAccount(base, input?.accountId ?? base.defaultAccountId ?? null)
  if (provider.transport !== 'codex-cli' && provider.transport !== 'claude-cli') throw new Error('This provider does not use CLI sign-in.')
  const result = await startCliLogin(provider)
  await auditEvent('provider.cli_login', { providerId: base.id, transport: provider.transport, ok: result.ok })
  return result
})

handle('cliAuth:cancelLogin', async (_e, providerId: string) => {
  requireSecurityRole('operator', 'cancel CLI sign-in')
  return { cancelled: cancelCliLogin(providerId) }
})

// Check whether a CLI transport is already logged in, without needing a saved
// provider. Used by onboarding to detect an existing Codex/Claude CLI login so
// the user can connect in one click instead of signing in again.
handle('cliAuth:checkTransport', async (_e, input: { transport: 'codex-cli' | 'claude-cli'; cliPath?: string }) => {
  // Gated + path-validated: checkCliAuth spawn()s the resolved binary with the app's full env.
  requireSecurityRole('operator', 'check CLI auth status')
  // A renderer-supplied cliPath must name the transport's own CLI binary — never an
  // arbitrary executable a compromised renderer could point at.
  const cliPath = input.cliPath?.trim()
  if (cliPath) {
    const names = cliBinaryNames(input.transport)
    const base = cliPath.split(/[\\/]/).pop() ?? ''
    if (!names.includes(base)) throw new Error(`Custom CLI path must point to the ${names[0]} binary.`)
  }
  const provider: Provider = {
    id: 'probe', name: 'probe', transport: input.transport,
    baseUrl: '', apiKey: '', models: [], kind: 'cloud',
    cliPath,
  }
  return checkCliAuth(provider)
})

}
