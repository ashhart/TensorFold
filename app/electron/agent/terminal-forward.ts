// Pure protocol for forwarding terminal (PTY) sessions over the orchestrator remote-control link.
// The mini (orchestrator/host) OWNS the PTY; the client (MBP) drives + observes it. These are the
// request validators (client→server actions) and the SSE event builders (server→client output),
// plus a coalescer so high-rate PTY output is batched into few SSE frames instead of firehosing
// the shared /v1/events stream (which also carries thread/room/mission control events).
//
// Kept pure/dependency-free so both the mini's remote-control routes and the client's dispatcher
// can share the exact shapes, and so it's unit-testable without a server or a real PTY.

// ---- client → server action bodies ------------------------------------------

export interface TerminalStartInput { projectId: string; cliId: string; command?: string; label?: string }
export function parseTerminalStart(body: unknown): TerminalStartInput | { error: string } {
  if (!body || typeof body !== 'object') return { error: 'Expected a JSON object.' }
  const b = body as Record<string, unknown>
  const projectId = typeof b.projectId === 'string' ? b.projectId.trim() : ''
  const cliId = typeof b.cliId === 'string' ? b.cliId.trim() : ''
  if (!projectId) return { error: 'projectId is required.' }
  if (!cliId) return { error: 'cliId is required.' }
  const out: TerminalStartInput = { projectId, cliId }
  if (typeof b.command === 'string' && b.command) out.command = b.command
  if (typeof b.label === 'string' && b.label) out.label = b.label
  return out
}

export interface TerminalWriteInput { data: string }
export function parseTerminalWrite(body: unknown): TerminalWriteInput | { error: string } {
  if (!body || typeof body !== 'object') return { error: 'Expected a JSON object.' }
  const data = (body as Record<string, unknown>).data
  if (typeof data !== 'string') return { error: 'data (string) is required.' }
  return { data }
}

export interface TerminalResizeInput { cols: number; rows: number }
export function parseTerminalResize(body: unknown): TerminalResizeInput | { error: string } {
  if (!body || typeof body !== 'object') return { error: 'Expected a JSON object.' }
  const b = body as Record<string, unknown>
  const cols = Number(b.cols)
  const rows = Number(b.rows)
  if (!Number.isFinite(cols) || !Number.isFinite(rows)) return { error: 'cols and rows must be numbers.' }
  const c = Math.floor(cols)
  const r = Math.floor(rows)
  if (c <= 0 || r <= 0) return { error: 'cols and rows must be positive.' }
  return { cols: c, rows: r }
}

export interface TerminalRenameInput { title: string }
export function parseTerminalRename(body: unknown): TerminalRenameInput | { error: string } {
  if (!body || typeof body !== 'object') return { error: 'Expected a JSON object.' }
  const b = body as Record<string, unknown>
  const title = typeof b.title === 'string' ? b.title.trim() : ''
  if (!title) return { error: 'title is required.' }
  return { title }
}

// ---- server → client events (ride RemoteControlEvent kind:'terminal') -------

export type TerminalEvent =
  | { type: 'data'; sessionId: string; chunk: string /* base64 of the raw PTY bytes */ }
  | { type: 'status'; sessionId: string; status: string }
  | { type: 'sessions'; projectId: string; sessions: unknown[] }
  | { type: 'exit'; sessionId: string }

// base64 is the safe framing: PTY output is split arbitrarily (mid-multibyte, mid-ANSI), and raw
// utf8 in JSON can carry lone surrogates that corrupt on the wire — encoding the bytes avoids it.
export function terminalDataEvent(sessionId: string, raw: string | Uint8Array): TerminalEvent {
  const buf = typeof raw === 'string' ? Buffer.from(raw, 'utf8') : Buffer.from(raw)
  return { type: 'data', sessionId, chunk: buf.toString('base64') }
}

export function decodeTerminalChunk(chunk: string): string {
  return Buffer.from(String(chunk || ''), 'base64').toString('utf8')
}

export function terminalStatusEvent(sessionId: string, status: string): TerminalEvent {
  return { type: 'status', sessionId, status }
}
export function terminalSessionsEvent(projectId: string, sessions: unknown[]): TerminalEvent {
  return { type: 'sessions', projectId, sessions }
}
export function terminalExitEvent(sessionId: string): TerminalEvent {
  return { type: 'exit', sessionId }
}

export function stampRemoteTerminalSession<T extends object>(session: T, orchestratorHostId: string): T & {
  hostLocation: 'orchestrator'
  managedBy: 'local'
  orchestratorHostId: string
} {
  return { ...session, hostLocation: 'orchestrator', managedBy: 'local', orchestratorHostId }
}

export function clientVisibleRemoteSessions(sessions: unknown[], orchestratorHostId: string): unknown[] {
  return sessions.map((session) => (
    session && typeof session === 'object'
      ? stampRemoteTerminalSession(session as Record<string, unknown>, orchestratorHostId)
      : session
  ))
}

// CLIENT side: map a forwarded terminal event onto the LOCAL renderer channel + payload, so a
// remote (mini-hosted) terminal renders through the exact same useXterm listeners as a local one —
// the renderer never knows the difference. Data chunks arrive base64-encoded on the wire; decode
// them back to the raw string the renderer's `terminal:data` handler expects. Returns null for an
// unrecognized event.
export function remoteTerminalEventToRenderer(ev: TerminalEvent, orchestratorHostId: string): { channel: string; payload: unknown } | null {
  switch (ev.type) {
    case 'data': return { channel: 'terminal:data', payload: { sessionId: ev.sessionId, chunk: decodeTerminalChunk(ev.chunk), orchestratorHostId } }
    case 'status': return { channel: 'terminal:status', payload: { sessionId: ev.sessionId, status: ev.status, orchestratorHostId } }
    case 'sessions': return {
      channel: 'terminal:sessions',
      payload: { projectId: ev.projectId, sessions: clientVisibleRemoteSessions(ev.sessions, orchestratorHostId), orchestratorHostId },
    }
    case 'exit': return { channel: 'terminal:status', payload: { sessionId: ev.sessionId, status: 'done', orchestratorHostId } }
    default: return null
  }
}

// ---- output coalescer -------------------------------------------------------
// Accumulates PTY output per session and hands back ONE combined (base64) chunk per session on
// drain(), so a chatty terminal produces a handful of SSE frames per tick instead of thousands.
// Timer-free (pure): the terminal runtime wraps it with a real interval + a size trip-wire.

export interface OutputCoalescer {
  // Buffer raw output for a session. Returns true when total buffered bytes have crossed maxBytes
  // (the caller should drain + publish now rather than wait for the next tick).
  push(sessionId: string, data: string): boolean
  // Combined per-session chunks (base64), in first-seen session order, then clears the buffer.
  drain(): Array<{ sessionId: string; chunk: string }>
  pending(): number
}

export function makeOutputCoalescer(opts: { maxBytes?: number } = {}): OutputCoalescer {
  const maxBytes = opts.maxBytes && opts.maxBytes > 0 ? opts.maxBytes : 256 * 1024
  const order: string[] = []
  const bufs = new Map<string, string>()
  let bytes = 0
  return {
    push(sessionId, data) {
      if (!bufs.has(sessionId)) { bufs.set(sessionId, ''); order.push(sessionId) }
      bufs.set(sessionId, bufs.get(sessionId)! + data)
      bytes += Buffer.byteLength(data, 'utf8')
      return bytes >= maxBytes
    },
    drain() {
      const out = order.map((sessionId) => ({
        sessionId,
        chunk: Buffer.from(bufs.get(sessionId) ?? '', 'utf8').toString('base64'),
      }))
      order.length = 0
      bufs.clear()
      bytes = 0
      return out
    },
    pending() { return bytes },
  }
}
