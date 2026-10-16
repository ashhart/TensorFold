import { discardResponseBody } from './pinned-fetch'

export const DEFAULT_JSON_RESPONSE_MAX_BYTES = 2 * 1024 * 1024
export const DEFAULT_ERROR_RESPONSE_MAX_BYTES = 64 * 1024

type ResponseWithBody = Pick<Response, 'headers' | 'body'>

export function isJsonResponseMime(value: string): boolean {
  return /^(?:application\/json|application\/[a-z0-9.+-]+\+json)\b/i.test(value.trim())
}

function byteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

/**
 * Consume a response body through a byte-counted stream. This is intentionally shared by
 * providers, connector clients, registries and local service adapters: every peer is treated as
 * untrusted even when it normally runs on loopback.
 */
export async function readBoundedResponseText(
  response: Pick<Response, 'headers' | 'body' | 'text'>,
  label: string,
  maxBytes: number,
  acceptsMime?: (mime: string) => boolean,
): Promise<string> {
  // A few unit tests use deliberately tiny fetch doubles. Keep those usable, but still apply the
  // post-read byte limit. Real fetch Response objects always take the streaming path below.
  if (!response.headers || !response.body || typeof response.body.getReader !== 'function') {
    const text = await response.text()
    if (byteLength(text) > maxBytes) throw new Error(`${label} response exceeds the ${maxBytes}-byte limit.`)
    return text
  }

  const mime = response.headers.get('content-type') ?? ''
  if (acceptsMime && !acceptsMime(mime)) {
    await discardResponseBody(response as Response)
    throw new Error(`${label} returned a non-JSON or otherwise unexpected content type.`)
  }
  const declared = response.headers.get('content-length')
  if (declared != null) {
    const bytes = Number(declared)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > maxBytes) {
      await discardResponseBody(response as Response)
      throw new Error(`${label} response exceeds the ${maxBytes}-byte limit.`)
    }
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let bytes = 0
  let text = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.byteLength
      if (bytes > maxBytes) {
        await reader.cancel().catch(() => {})
        throw new Error(`${label} response exceeds the ${maxBytes}-byte limit.`)
      }
      text += decoder.decode(value, { stream: true })
    }
    text += decoder.decode()
    return text
  } finally {
    reader.releaseLock()
  }
}

/** Read at most maxBytes and cancel the remainder, for deliberately truncated previews. */
export async function readResponseTextPrefix(
  response: Pick<Response, 'body' | 'text'>,
  maxBytes: number,
): Promise<string> {
  if (!response.body || typeof response.body.getReader !== 'function') {
    return (await response.text()).slice(0, maxBytes)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let remaining = maxBytes
  let text = ''
  let complete = false
  try {
    while (remaining > 0) {
      const { done, value } = await reader.read()
      if (done) { complete = true; break }
      const take = Math.min(remaining, value.byteLength)
      text += decoder.decode(value.subarray(0, take), { stream: true })
      remaining -= take
      if (take < value.byteLength || remaining === 0) break
    }
    text += decoder.decode()
    return text
  } finally {
    if (!complete) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

/** Read and parse a MIME-checked, byte-bounded JSON response. */
export async function readBoundedJsonResponse(
  response: Pick<Response, 'headers' | 'body' | 'text' | 'json'>,
  label: string,
  maxBytes = DEFAULT_JSON_RESPONSE_MAX_BYTES,
): Promise<any> {
  // Preserve compatibility with json()-only test doubles. Production responses are streamed and
  // bounded by readBoundedResponseText.
  if (!response.headers || !response.body || typeof response.body.getReader !== 'function') {
    if (typeof response.text === 'function') {
      const text = await readBoundedResponseText(response, label, maxBytes)
      return JSON.parse(text)
    }
    return response.json()
  }
  const text = await readBoundedResponseText(response, label, maxBytes, isJsonResponseMime)
  return JSON.parse(text)
}
