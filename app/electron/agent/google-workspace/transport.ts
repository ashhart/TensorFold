import { createConnectorFetch } from '../secure-fetch'
import { readBoundedResponseText, isJsonResponseMime } from '../bounded-response'
import type { PinnedHostResolver } from '../pinned-fetch'
import { GOOGLE_REVOKE_URL, GOOGLE_TOKEN_URL } from './client-config'

export interface GoogleTransportDeps { fetchImpl?: typeof fetch; lookup?: PinnedHostResolver; timeoutMs?: number }
export class GoogleApiError extends Error {
  constructor(readonly status: number, readonly category: 'authorization' | 'permission' | 'rate-limit' | 'service' | 'request') {
    super(status === 401 ? 'Google sign-in expired. Reconnect this account.' : status === 403 ? 'Google denied this operation. Check the granted permissions and enabled API.' : status === 429 ? 'Google is rate limiting this connection. Try again later.' : status >= 500 ? 'Google could not complete this request.' : 'Google rejected this request.')
  }
}
export class GoogleTransport {
  constructor(private readonly deps: GoogleTransportDeps = {}) {}
  async request(url: string, init: RequestInit, signal?: AbortSignal): Promise<any> {
    const target = new URL(url)
    const valid = target.origin === 'https://gmail.googleapis.com' && target.pathname.startsWith('/gmail/v1/users/me/')
      || target.origin === 'https://www.googleapis.com' && (target.pathname.startsWith('/calendar/v3/') || target.pathname === '/oauth2/v3/userinfo')
      || url === GOOGLE_TOKEN_URL || url === GOOGLE_REVOKE_URL
    if (!valid || target.username || target.password || target.hash || target.port) throw new Error('Google endpoint is not allowed.')
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.deps.timeoutMs ?? 20_000)
    const abort = () => controller.abort()
    if (signal?.aborted) controller.abort()
    else signal?.addEventListener('abort', abort, { once: true })
    try {
      const fetcher = createConnectorFetch(target.origin, 'Google Workspace', { lookup: this.deps.lookup }, this.deps.fetchImpl ?? fetch)
      const response = await fetcher(url, { ...init, redirect: 'error', signal: controller.signal })
      if (!response.ok) {
        await response.body?.cancel().catch(() => {})
        throw new GoogleApiError(response.status, response.status === 401 ? 'authorization' : response.status === 403 ? 'permission' : response.status === 429 ? 'rate-limit' : response.status >= 500 ? 'service' : 'request')
      }
      if (response.status === 204 || url === GOOGLE_REVOKE_URL) { await response.body?.cancel().catch(() => {}); return {} }
      const text = await readBoundedResponseText(response, 'Google Workspace', 2 * 1024 * 1024, isJsonResponseMime)
      try {
        const value = JSON.parse(text)
        if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('shape')
        return value
      } catch { throw new Error('Google returned invalid JSON.') }
    } catch (error) {
      if (error instanceof GoogleApiError || error instanceof Error && (/^Google (endpoint|returned)/.test(error.message) || /^Google Workspace (response exceeds|returned a non-JSON)/.test(error.message))) throw error
      throw new Error(controller.signal.aborted ? 'Google request was cancelled or timed out.' : 'Could not reach Google. Check your connection.')
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
  }
  json(url: string, token: string, body?: unknown, method = 'GET', signal?: AbortSignal, headers: Record<string, string> = {}): Promise<any> {
    if (!token || token.length > 16_384 || /[\r\n\0]/.test(token)) throw new Error('Google credential is invalid. Reconnect this account.')
    return this.request(url, { method, headers: { Authorization: `Bearer ${token}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) }, signal)
  }
  form(url: string, body: URLSearchParams, signal?: AbortSignal): Promise<any> {
    return this.request(url, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: body.toString() }, signal)
  }
}
