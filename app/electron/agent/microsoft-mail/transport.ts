import { fetchConnectorJson, fetchConnectorResponse, type ConnectorHttpDeps } from '../connector-http'

export const LOGIN = 'https://login.microsoftonline.com'
export const GRAPH = 'https://graph.microsoft.com'

export class MicrosoftMailTransport {
  constructor(private deps: ConnectorHttpDeps) { }
  async json(origin: string, route: string, init: RequestInit = {}) {
    try {
      return await fetchConnectorJson(origin, 'Microsoft mail', new URL(route, origin), { ...init, signal: AbortSignal.timeout(20_000) },
        { fetchImpl: this.deps.fetchImpl, lookup: this.deps.lookup, retry: { maxAttempts: 1 } })
    } catch { throw new Error('Microsoft mail could not be reached securely. Check your connection and try again.') }
  }
  async send(route: string, body: unknown, headers: Record<string, string>, assertCurrent: () => void): Promise<Response> {
    assertCurrent()
    const guardedFetch: typeof fetch = (input, init) => { assertCurrent(); return (this.deps.fetchImpl ?? fetch)(input, init) }
    try {
      return await fetchConnectorResponse(GRAPH, 'Microsoft mail', new URL(route, GRAPH), {
        method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(20_000),
      }, { fetchImpl: guardedFetch, lookup: this.deps.lookup, retry: { maxAttempts: 1 } })
    } catch { throw new Error('Microsoft mail delivery could not be confirmed. Check Sent Items before creating another message.') }
  }
}
