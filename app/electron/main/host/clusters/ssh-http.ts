import http from 'node:http'
import { Readable } from 'node:stream'
import type { ClusterForwardSession } from './ssh-forward'

/** HTTP is carried inside the captured verified SSH channel, never a reusable local port. */
export function createSshHttpRequest(session: ClusterForwardSession, port: number, assertCurrent: () => void | Promise<void>) {
  let closed = false
  const unsubscribe = session.onClosed(() => { closed = true })
  const current = async () => { if (closed) throw new Error('The verified SSH session closed.'); await assertCurrent(); if (closed) throw new Error('The verified SSH session closed.') }
  const request = async (rawUrl: string, init: RequestInit): Promise<Response> => {
    const url = new URL(rawUrl)
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !['GET', 'POST'].includes(init.method ?? 'GET') || init.body !== undefined && typeof init.body !== 'string') throw new Error('Unsupported app-owned SSH HTTP request.')
    if (typeof init.body === 'string' && Buffer.byteLength(init.body) > 1024 * 1024) throw new Error('The SSH request body exceeds its size limit.')
    await current()
    const channel = await session.forwardLoopback(port, init.signal ?? undefined, 15_000)
    try { await current() } catch (error) { channel.destroy(); throw error }
    return new Promise<Response>((resolve, reject) => {
      const agent = new http.Agent({ keepAlive: false, maxSockets: 1 })
      agent.createConnection = () => channel as any
      let settled = false
      const abort = () => { channel.destroy(); agent.destroy() }
      const req = http.request({ host: '127.0.0.1', port, path: url.pathname + url.search, method: init.method ?? 'GET', headers: Object.fromEntries(new Headers(init.headers)), agent, signal: init.signal ?? undefined }, res => {
        const headers = new Headers()
        for (let index = 0; index < res.rawHeaders.length; index += 2) headers.append(res.rawHeaders[index], res.rawHeaders[index + 1])
        res.once('close', () => { init.signal?.removeEventListener('abort', abort); agent.destroy() })
        try {
          const body = [204, 205, 304].includes(res.statusCode ?? 0) ? null : Readable.toWeb(res) as ReadableStream<Uint8Array>
          if (!body) res.resume()
          const response = new Response(body, { status: res.statusCode ?? 502, headers }); settled = true; resolve(response)
        } catch { abort(); reject(new Error('The verified SSH response headers were malformed.')) }
      })
      init.signal?.addEventListener('abort', abort, { once: true })
      req.once('error', () => { init.signal?.removeEventListener('abort', abort); abort(); if (!settled) reject(new Error('The verified SSH HTTP request did not settle.')) })
      if (init.signal?.aborted || closed) { req.destroy(); abort(); reject(new Error('The verified SSH request was cancelled before headers.')); return }
      req.end(init.body)
    })
  }
  return { request, close: unsubscribe }
}
