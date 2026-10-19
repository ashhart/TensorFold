import type { IncomingMessage, ServerResponse } from 'node:http'

const MAX_BODY_BYTES = 20 * 1024 * 1024
const REMOTE_WEB_CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"

export async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
    size += buf.length
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error('Payload too large.'), { code: 413 })
    chunks.push(buf)
  }
  if (!chunks.length) return {}
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    throw Object.assign(new Error('Invalid JSON.'), { code: 400 })
  }
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
