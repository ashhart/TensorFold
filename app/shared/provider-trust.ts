// Which providers may see local work. Pure so the renderer (badges, notes) and the main
// process (the outbound guard, plan handoff) share one answer.
//
// Trusted local: the built-in TensorFold engine, and kind:'local' OpenAI-compatible
// providers whose endpoint is loopback (oMLX, Ollama, LM Studio on this Mac). Everything
// else is external and runs planning only: cloud HTTP providers, LAN endpoints, and every
// delegated CLI or server engine (Codex, Claude, Cursor, Gemini, Pi, OpenCode, Kilo),
// whatever model they are pointed at.

export type ProviderTrust = 'trusted-local' | 'external'

export interface ProviderTrustInput {
  id?: string
  name?: string
  source?: string
  kind?: string
  transport?: string
  baseUrl?: string
}

export const BUILT_IN_PROVIDER_ID = 'tensorfold'

export function isLoopbackUrl(raw: string | null | undefined): boolean {
  let url: URL
  try { url = new URL(String(raw ?? '').trim()) } catch { return false }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false
  if (url.username || url.password) return false
  const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host === '::1') return true
  if (/^127(?:\.\d{1,3}){3}$/.test(host)) return true
  // IPv4-mapped loopback, in the dotted or the hex form WHATWG URL normalizes to.
  if (/^::ffff:127(?:\.\d{1,3}){3}$/.test(host)) return true
  const mapped = /^::ffff:([0-9a-f]{1,4}):[0-9a-f]{1,4}$/.exec(host)
  return !!mapped && parseInt(mapped[1], 16) >> 8 === 127
}

export function isBuiltInProvider(provider: ProviderTrustInput | null | undefined): boolean {
  return provider?.id === BUILT_IN_PROVIDER_ID && provider?.source === BUILT_IN_PROVIDER_ID
}

/** Classify a provider; `baseUrl` is the endpoint a call will actually use, when it differs. */
export function classifyProvider(provider: ProviderTrustInput | null | undefined, baseUrl?: string | null): ProviderTrust {
  const transport = provider?.transport || 'chat-completions'
  if (transport !== 'chat-completions') return 'external'
  const urls = [provider?.baseUrl, baseUrl].map((url) => String(url ?? '').trim()).filter(Boolean)
  const allLoopback = urls.every(isLoopbackUrl)
  // The built-in engine's live URL is assigned per launch, so an empty one is still local.
  if (isBuiltInProvider(provider)) return allLoopback ? 'trusted-local' : 'external'
  if (!urls.length) return 'external'
  if (!provider) return allLoopback ? 'trusted-local' : 'external'
  return provider.kind === 'local' && allLoopback ? 'trusted-local' : 'external'
}

export function isExternalProvider(provider: ProviderTrustInput | null | undefined, baseUrl?: string | null): boolean {
  return classifyProvider(provider, baseUrl) === 'external'
}

/** The engine a planning reply hands off to: the built-in one, else any trusted local provider. */
export function pickPlanRunner<P extends ProviderTrustInput>(providers: readonly P[] | null | undefined): P | null {
  const list = providers ?? []
  const builtIn = list.find((provider) => isBuiltInProvider(provider) && classifyProvider(provider) === 'trusted-local')
  return builtIn ?? list.find((provider) => classifyProvider(provider) === 'trusted-local') ?? null
}
