import { discardResponseBody } from './pinned-fetch'
import type { RemoteControlEvent } from './types'

export const REMOTE_SSE_FRAME_MAX_CHARS = 512 * 1024
export const REMOTE_SSE_CONNECT_TIMEOUT_MS = 15_000
export const REMOTE_SSE_IDLE_TIMEOUT_MS = 45_000

function streamAbortError(): Error {
  const error = new Error('Orchestrator event stream was aborted.')
  error.name = 'AbortError'
  return error
}

/**
 * Bound the phase before an SSE response's headers arrive. The read-side idle timeout
 * cannot help while fetch itself is still connecting, so keep a separate hard deadline
 * and bridge caller cancellation into the request controller used by the native fetch.
 */
export function fetchRemoteControlSse(
  fetchImpl: typeof fetch,
  input: Parameters<typeof fetch>[0],
  init: RequestInit = {},
  connectTimeoutMs = REMOTE_SSE_CONNECT_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<Response> {
  const bounded = Math.max(1, connectTimeoutMs)
  const requestController = new AbortController()
  const callerSignal = signal ?? init.signal ?? undefined
  const pending = Promise.resolve().then(() => fetchImpl(input, { ...init, signal: requestController.signal }))

  if (callerSignal?.aborted) {
    requestController.abort()
    void pending.catch(() => {})
    return Promise.reject(streamAbortError())
  }

  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      callerSignal?.removeEventListener('abort', onAbort)
      fn()
    }
    const onAbort = () => {
      finish(() => reject(streamAbortError()))
      requestController.abort()
    }
    const timer = setTimeout(() => {
      finish(() => reject(new Error(
        `Orchestrator event stream did not connect within ${bounded}ms; reconnecting.`,
      )))
      requestController.abort()
    }, bounded)
    callerSignal?.addEventListener('abort', onAbort, { once: true })
    pending.then(
      (response) => finish(() => resolve(response)),
      (error) => finish(() => reject(error)),
    )
  })
}

/** Bound one stream read so a half-open TCP connection cannot wait forever. */
export function readRemoteSseChunk(
  reader: Pick<ReadableStreamDefaultReader<Uint8Array>, 'read'>,
  idleTimeoutMs = REMOTE_SSE_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  const bounded = Math.max(1, idleTimeoutMs)
  if (signal?.aborted) return Promise.reject(streamAbortError())
  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      fn()
    }
    const onAbort = () => finish(() => reject(streamAbortError()))
    const timer = setTimeout(() => finish(() => reject(new Error(
      `Orchestrator event stream was idle for ${bounded}ms; reconnecting.`,
    ))), bounded)
    signal?.addEventListener('abort', onAbort, { once: true })
    reader.read().then(
      (result) => finish(() => resolve(result)),
      (error) => finish(() => reject(error)),
    )
  })
}

/** MIME-check and incrementally parse the remote-control SSE stream with a hard frame cap. */
export async function readRemoteControlSse(
  response: Pick<Response, 'headers' | 'body'>,
  onEvent: (event: RemoteControlEvent) => void,
  maxFrameChars = REMOTE_SSE_FRAME_MAX_CHARS,
  idleTimeoutMs = REMOTE_SSE_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<void> {
  const mime = String(response.headers.get('content-type') ?? '').toLowerCase()
  if (!mime.startsWith('text/event-stream')) {
    await discardResponseBody(response as Response)
    throw new Error('Orchestrator event stream returned an invalid content type.')
  }
  if (!response.body) throw new Error('Orchestrator event stream was empty.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let complete = false
  try {
    while (true) {
      const { done, value } = await readRemoteSseChunk(reader, idleTimeoutMs, signal)
      if (done) { complete = true; break }
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
      if (buffer.length > maxFrameChars && !buffer.includes('\n\n')) {
        throw new Error('Orchestrator event stream frame exceeded the safety limit.')
      }
      let split = buffer.indexOf('\n\n')
      while (split >= 0) {
        const raw = buffer.slice(0, split)
        buffer = buffer.slice(split + 2)
        if (raw.length > maxFrameChars) throw new Error('Orchestrator event stream frame exceeded the safety limit.')
        const data = raw.split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n')
        if (data) onEvent(JSON.parse(data) as RemoteControlEvent)
        split = buffer.indexOf('\n\n')
      }
      if (buffer.length > maxFrameChars) throw new Error('Orchestrator event stream frame exceeded the safety limit.')
    }
  } finally {
    if (!complete) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
