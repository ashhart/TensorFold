import type { Provider } from './types'
export type ManagedComputeFetch = (provider: Provider, url: string, init: RequestInit, assertCurrent?: () => void) => Promise<Response>
let transport: ManagedComputeFetch | null = null
/** The main host installs this after its protected deployment manager is registered. */
export function configureManagedComputeTransport(value: ManagedComputeFetch | null): void { transport = value }
export async function fetchManagedCompute(provider: Provider, url: string, init: RequestInit, assertCurrent?: () => void): Promise<Response> {
  if (!transport) throw new Error('Reconnect and verify the managed compute endpoint in the desktop app before sending work.')
  return transport(provider, url, init, assertCurrent)
}

/** Streaming inference keeps its workspace reservation until physical consumption or cancellation. */
export function holdManagedComputeResponse(response: Response, release: () => void, maxBytes = 32 * 1024 * 1024, assertCurrent?: () => void): Response {
  const reader = response.body?.getReader()
  if (!reader) { release(); return response }
  let finished = false, bytes = 0
  const finish = () => { if (!finished) { finished = true; release() } }
  let output: ReadableStreamDefaultController<Uint8Array> | null = null
  const body = new ReadableStream<Uint8Array>({
    start(controller) { output = controller },
    async pull(controller) {
      try {
        assertCurrent?.()
        const value = await reader.read()
        assertCurrent?.()
        if (value.done) { finish(); controller.close(); return }
        bytes += value.value.byteLength
        if (bytes > maxBytes) { await reader.cancel(); throw new Error('The managed compute response exceeded its size limit.') }
        controller.enqueue(value.value)
      } catch (error) { await reader.cancel().catch(() => {}); finish(); controller.error(error) }
    },
    async cancel(reason) { try { await reader.cancel(reason) } finally { finish() } },
  })
  void reader.closed.catch(error => { if (!finished) { finish(); try { output?.error(error) } catch { /* Already cancelled. */ } } })
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers })
}
