import {
  HEAD_TIMEOUT_MS,
  IDLE_TIMEOUT_MS,
  MAX_INFLIGHT,
  MAX_REQUEST_BYTES,
  MAX_STREAM_BYTES,
  type RelayFrame,
} from './protocol'

type HeadFrame = Extract<RelayFrame, { t: 'head' }>
type ErrFrame = Extract<RelayFrame, { t: 'err' }>

interface Pending {
  rid: string
  head: HeadFrame | null
  chunks: Uint8Array[]
  bytes: number
  writer: WritableStreamDefaultWriter<Uint8Array> | null
  waiter: ((frame: RelayFrame) => void) | null
  headTimer: ReturnType<typeof setTimeout> | null
  idleTimer: ReturnType<typeof setTimeout> | null
  done: boolean
}

const reply = (status: number, message: string): Response =>
  Response.json({ error: { message } }, { status })

const sha256 = async (value: string): Promise<string> => {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

const toBase64 = (bytes: Uint8Array): string => {
  let binary = ''
  const step = 0x8000
  for (let i = 0; i < bytes.length; i += step) binary += String.fromCharCode(...bytes.subarray(i, i + step))
  return btoa(binary)
}

/** Reads at most `limit` bytes; null means the request declared or delivered more than that. */
async function readBoundedBody(request: Request, limit: number): Promise<Uint8Array | null> {
  const declared = Number(request.headers.get('content-length') ?? '0')
  if (Number.isFinite(declared) && declared > limit) return null
  if (!request.body) return new Uint8Array()
  const reader = request.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    total += value.byteLength
    if (total > limit) {
      await reader.cancel().catch(() => {})
      return null
    }
    chunks.push(value)
  }
  const body = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return body
}

const fromBase64 = (value: string): Uint8Array => {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/**
 * One object per card id. It holds the donor's outbound WebSocket and the
 * renter requests waiting on it, so the donor never accepts inbound traffic.
 *
 * The relay is deliberately not the security boundary: it forwards the
 * renter's `authorization` header untouched and the donor's proxy still checks
 * the card token, the window and the caps. A compromised relay can deny
 * service or read prompts, but cannot mint a valid token.
 */
export class DonorRelay implements DurableObject {
  private socket: WebSocket | null = null
  private pending = new Map<string, Pending>()

  constructor(private readonly state: DurableObjectState) {}

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url)
    if (request.headers.get('upgrade')?.toLowerCase() === 'websocket') return this.attach(request)
    if (url.pathname === '/__relay/status') {
      return Response.json({ attached: this.socket?.readyState === 1, inflight: this.pending.size })
    }
    return this.proxy(request)
  }

  /**
   * The donor key travels as the second WebSocket subprotocol: Node and browser
   * clients cannot set headers, and a query parameter would end up in logs.
   */
  private async attach(request: Request): Promise<Response> {
    const offered = (request.headers.get('sec-websocket-protocol') ?? '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean)
    const key = offered[1] ?? ''
    if (!/^[A-Za-z0-9_-]{16,128}$/.test(key)) return reply(400, 'A donor key is required as the second subprotocol.')

    const digest = await sha256(key)
    const stored = await this.state.storage.get<string>('donorKeyHash')
    if (stored && stored !== digest) return reply(403, 'That card id is already claimed by another donor key.')
    if (!stored) await this.state.storage.put('donorKeyHash', digest)
    if (this.socket && this.socket.readyState === 1) return reply(409, 'A donor is already attached for this card.')

    const pair = new WebSocketPair()
    const server = pair[1]
    server.accept()
    this.socket = server
    const drop = () => this.dropSocket(server, 'The donor disconnected.')
    server.addEventListener('message', (event) => this.onFrame(String((event as MessageEvent).data)))
    server.addEventListener('close', drop)
    server.addEventListener('error', drop)

    return new Response(null, {
      status: 101,
      webSocket: pair[0],
      headers: { 'Sec-WebSocket-Protocol': offered[0] ?? 'tf-donation-v1' },
    })
  }

  private async proxy(request: Request): Promise<Response> {
    const raw = request.method === 'GET' || request.method === 'HEAD' ? new Uint8Array() : await readBoundedBody(request, MAX_REQUEST_BYTES)
    if (raw === null) return reply(413, 'The request body is larger than the relay forwards.')

    const socket = this.socket
    if (!socket || socket.readyState !== 1) return reply(503, 'No donor is attached to this card right now.')
    if (this.pending.size >= MAX_INFLIGHT) return reply(429, 'Too many requests are open for this card.')

    const headers: Record<string, string> = {}
    for (const name of ['authorization', 'content-type', 'accept']) {
      const value = request.headers.get(name)
      if (value) headers[name] = value
    }

    const url = new URL(request.url)
    const rid = crypto.randomUUID()
    const pending: Pending = {
      rid,
      head: null,
      chunks: [],
      bytes: 0,
      writer: null,
      waiter: null,
      headTimer: null,
      idleTimer: null,
      done: false,
    }
    this.pending.set(rid, pending)

    socket.send(JSON.stringify({ t: 'req', rid, method: request.method, path: url.pathname.replace(/^\/[^/]+/, ''), headers, body: toBase64(raw) } satisfies RelayFrame))

    const head = await this.waitForHead(pending)
    if (!head || head.t === 'err') {
      const failure = head && head.t === 'err' ? head : null
      this.finish(rid)
      return reply(failure?.status ?? 504, failure?.message ?? 'The donor did not answer in time.')
    }
    if (head.t !== 'head') {
      this.finish(rid)
      return reply(502, 'The donor answered with a frame the relay did not expect.')
    }

    const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>()
    pending.writer = writable.getWriter()
    for (const chunk of pending.chunks.splice(0)) void pending.writer.write(chunk).catch(() => this.finish(rid))
    this.armIdle(pending)

    return new Response(readable, { status: head.status, headers: head.headers })
  }

  private waitForHead(pending: Pending): Promise<RelayFrame | null> {
    return new Promise((resolve) => {
      pending.waiter = resolve
      pending.headTimer = setTimeout(() => {
        if (pending.done) return
        pending.done = true
        this.finish(pending.rid)
        resolve({ t: 'err', rid: pending.rid, status: 504, message: 'The donor did not answer in time.' })
      }, HEAD_TIMEOUT_MS)
    })
  }

  private armIdle(pending: Pending): void {
    if (pending.idleTimer) clearTimeout(pending.idleTimer)
    pending.idleTimer = setTimeout(() => {
      void pending.writer?.abort(new Error('The relay stopped receiving data from the donor.')).catch(() => {})
      this.finish(pending.rid)
    }, IDLE_TIMEOUT_MS)
  }

  private onFrame(raw: string): void {
    let frame: RelayFrame
    try {
      frame = JSON.parse(raw) as RelayFrame
    } catch {
      return
    }
    if (frame.t === 'ping') {
      this.socket?.send(JSON.stringify({ t: 'pong' } satisfies RelayFrame))
      return
    }
    if (!('rid' in frame)) return
    const pending = this.pending.get(frame.rid)
    if (!pending) return

    if (frame.t === 'head') {
      pending.head = frame
      if (pending.headTimer) clearTimeout(pending.headTimer)
      pending.headTimer = null
      if (pending.waiter) {
        const waiter = pending.waiter
        pending.waiter = null
        pending.done = false
        waiter(frame)
      }
      return
    }

    if (frame.t === 'chunk') {
      const bytes = fromBase64(frame.data)
      pending.bytes += bytes.byteLength
      if (pending.bytes > MAX_STREAM_BYTES) {
        void pending.writer?.abort(new Error('The relay stopped forwarding this response: it grew past its limit.')).catch(() => {})
        this.finish(pending.rid)
        return
      }
      this.armIdle(pending)
      if (pending.writer) void pending.writer.write(bytes).catch(() => this.finish(pending.rid))
      else pending.chunks.push(bytes)
      return
    }

    if (frame.t === 'end') {
      void pending.writer?.close().catch(() => {})
      this.finish(pending.rid)
      return
    }

    if (frame.t === 'err') {
      const failure = frame as ErrFrame
      if (pending.waiter) {
        const waiter = pending.waiter
        pending.waiter = null
        waiter(failure)
      } else {
        void pending.writer?.abort(new Error(failure.message)).catch(() => {})
      }
      this.finish(pending.rid)
    }
  }

  private finish(rid: string): void {
    const pending = this.pending.get(rid)
    if (!pending) return
    if (pending.headTimer) clearTimeout(pending.headTimer)
    if (pending.idleTimer) clearTimeout(pending.idleTimer)
    pending.waiter = null
    this.pending.delete(rid)
  }

  private dropSocket(socket: WebSocket, reason: string): void {
    if (this.socket !== socket) return
    this.socket = null
    for (const pending of [...this.pending.values()]) {
      if (pending.waiter) {
        const waiter = pending.waiter
        pending.waiter = null
        waiter({ t: 'err', rid: pending.rid, status: 502, message: reason })
      }
      void pending.writer?.abort(new Error(reason)).catch(() => {})
      this.finish(pending.rid)
    }
  }
}
