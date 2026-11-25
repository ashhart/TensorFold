import type { IncomingMessage, ServerResponse } from 'node:http'

const MAX_BODY_BYTES = 20 * 1024 * 1024
export const PAIRING_BODY_BYTES = 64 * 1024
const REMOTE_WEB_CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
const jsonBodies = new WeakMap<IncomingMessage, Promise<Record<string, unknown>>>()


async function parseJson(req: IncomingMessage, maxBodyBytes: number): Promise<Record<string, unknown>> {
  const declared = Number(req.headers['content-length'])
  if (Number.isFinite(declared) && declared > maxBodyBytes) throw Object.assign(new Error('Payload too large.'), { code: 413 })
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    const cleanup = () => {
      req.off('data', onData)
      req.off('end', onEnd)
      req.off('error', onError)
    }
    const onData = (chunk: unknown) => {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array)
      size += buf.length
      if (size > maxBodyBytes) {
        chunks.length = 0
        cleanup()
        // Async-iterator cancellation destroys the socket before the handler can write 413.
        // Discard the remaining request bytes instead, leaving the response socket intact.
        req.resume()
        reject(Object.assign(new Error('Payload too large.'), { code: 413 }))
        return
      }
      chunks.push(buf)
    }
    const onEnd = () => {
      cleanup()
      if (!chunks.length) { resolve({}); return }
      try {
        const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'))
        resolve(parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {})
      } catch {
        reject(Object.assign(new Error('Invalid JSON.'), { code: 400 }))
      }
    }
    const onError = (err: unknown) => { cleanup(); reject(err) }
    req.on('data', onData)
    req.on('end', onEnd)
    req.on('error', onError)
  })
}

export function readJson(req: IncomingMessage, maxBodyBytes = MAX_BODY_BYTES): Promise<Record<string, unknown>> {
  let body = jsonBodies.get(req)
  if (!body) {
    body = parseJson(req, maxBodyBytes)
    jsonBodies.set(req, body)
  }
  return body
}

export function writeJson(res: ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, {
    'content-type': 'application/json',
    'cache-control': 'no-store',
  })
  res.end(JSON.stringify(body))
}

export async function writeArchiveMutation<T>(res: ServerResponse, operation: Promise<T | null>): Promise<void> {
  const record = await operation
  if (!record) {
    writeJson(res, 404, { error: 'not_found' })
    return
  }
  writeJson(res, 202, record)
}

export function writeRemoteStatic(res: ServerResponse, body: Buffer, contentType: string, cacheControl: string): void {
  res.writeHead(200, {
    'content-type': contentType,
    'cache-control': cacheControl,
    'content-security-policy': REMOTE_WEB_CSP,
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
    'referrer-policy': 'no-referrer',
    'cross-origin-resource-policy': 'same-origin',
  })
  res.end(body)
}
