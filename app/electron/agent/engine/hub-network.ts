import { isIP } from 'node:net'
import { createPinnedFetch, resolvePinnedHost, type PinnedHostResolver } from '../pinned-fetch'
import { isAlwaysForbiddenConnectorIp, isBenchmarkIp, isGloballyRoutableIp } from '../security-addresses'
import { hubError, hubFailureCause } from '../../../shared/hub-failure'

export const HF_ORIGIN = 'https://huggingface.co'

// Prefix paths are refused so the app's builders and the native pull child share one exact origin.
export function resolveHubEndpoint(value?: string): string {
  const raw = value?.trim() || HF_ORIGIN
  let url: URL
  try { url = new URL(raw) } catch { throw hubError('policy', 'HF_ENDPOINT must be a named HTTPS Hub origin.') }
  const host = url.hostname.replace(/^\[|\]$/g, '')
  const normalizedHost = host.toLowerCase().replace(/\.$/, '')
  const reservedHost = normalizedHost === 'localhost' || normalizedHost.endsWith('.localhost') || normalizedHost === 'metadata.google.internal'
  const namedHost = /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.?$/i.test(host)
  if (!/^https:\/\/[^/?#]+\/?$/i.test(raw) || /[\u0000-\u0020\u007f]/.test(raw) ||
      url.protocol !== 'https:' || url.username || url.password || raw.includes('@') ||
      raw.includes('\\') || url.pathname !== '/' || !namedHost || host.length > 253 ||
      isIP(host) || reservedHost) {
    throw hubError('policy', 'HF_ENDPOINT must be a named HTTPS origin without credentials, a path, query or fragment.')
  }
  return url.origin
}

export interface HubNetworkOptions { lookup?: PinnedHostResolver; fetchImpl?: typeof fetch }

export function createHubFetch(endpoint = HF_ORIGIN, options: HubNetworkOptions = {}): typeof fetch {
  const origin = resolveHubEndpoint(endpoint)
  const hostname = new URL(origin).hostname
  return (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const request = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
    if (request.origin !== origin || request.protocol !== 'https:' || request.username || request.password) {
      throw hubError('policy', 'The Hub request must use its configured HTTPS origin without URL credentials.')
    }
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined))
    if (headers.has('authorization') || headers.has('cookie')) throw hubError('policy', 'Hub model checks do not accept account tokens or cookies.')
    if (init?.redirect === 'follow') throw hubError('policy', 'Hub redirects must be checked and pinned one hop at a time.')
    try {
      const target = await resolvePinnedHost(request, (address) => {
        if (address.includes('%') || isAlwaysForbiddenConnectorIp(address) || (!isGloballyRoutableIp(address) && !isBenchmarkIp(address))) {
          throw hubError('policy', `The Hub host ${hostname} resolved to a refused address (${address}). Add this host to the proxy's fake-ip-filter or configure a valid HF_ENDPOINT, then retry.`)
        }
      }, { lookup: options.lookup })
      return await createPinnedFetch(target, options.fetchImpl ?? fetch)(input, { ...init, headers, credentials: 'omit' })
    } catch (cause) {
      const failure = hubFailureCause(cause)
      if (failure) throw hubError(failure.category, failure.message, cause)
      throw hubError('transport', `Could not reach the Hub host ${hostname}. Check the connection or proxy settings, then retry.`, cause)
    }
  }) as typeof fetch
}
