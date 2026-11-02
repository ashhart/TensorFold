import { pinnedFetch, discardResponseBody, type PinnedHostResolver } from '../../pinned-fetch'
import { resolveUpdateHost } from '../../update-network'
import { CATALOG_LIMIT, RUNTIME_FEED } from './catalog'

export async function fetchRuntimeCatalog(signal: AbortSignal, deps: { fetchImpl?: typeof fetch; lookup?: PinnedHostResolver } = {}): Promise<string> {
  const target = await resolveUpdateHost(RUNTIME_FEED, deps.lookup)
  const response = await pinnedFetch(target, { signal, redirect: 'error', headers: { Accept: 'application/json' } }, deps.fetchImpl ?? fetch)
  if (!response.ok || !/^application\/(?:[\w.+-]*\+)?json(?:;|$)/i.test(response.headers.get('content-type') ?? '')) {
    await discardResponseBody(response)
    throw new Error(`Runtime catalog unavailable, HTTP ${response.status}.`)
  }
  const length = response.headers.get('content-length')
  if (length && (!/^\d+$/.test(length) || Number(length) > CATALOG_LIMIT)) { await discardResponseBody(response); throw new Error('Runtime catalog is too large.') }
  if (!response.body) throw new Error('Runtime catalog is empty.')
  const reader = response.body.getReader()
  const chunks: Buffer[] = []
  let bytes = 0
  try {
    for (;;) {
      signal.throwIfAborted()
      const item = await reader.read()
      if (item.done) break
      bytes += item.value.byteLength
      if (bytes > CATALOG_LIMIT) throw new Error('Runtime catalog is too large.')
      chunks.push(Buffer.from(item.value))
    }
    return Buffer.concat(chunks).toString('utf8')
  } finally { await reader.cancel().catch(() => {}) }
}
