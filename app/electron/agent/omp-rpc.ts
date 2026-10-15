// OMP RPC transport (VST-050): speak the headless embedding protocol of the installed
// `omp` coding agent (@oh-my-pi/pi-coding-agent). Wire facts taken from the INSTALLED
// package source (src/modes/rpc/*) plus a live rpc-mode probe on this machine — not from
// memory:
// - `omp --mode=rpc` reads newline-delimited JSON commands on stdin and writes
//   newline-delimited JSON frames on stdout: `ready`, id-correlated `response` frames,
//   streamed AgentSessionEvent objects, and `rpc_chunk` frames that split one large
//   payload across several lines ({chunkId,index,count,data} — concat data in index
//   order, then JSON.parse).
// - Framing is STRICT LF-delimited JSONL (same family as the pi engine, VST-033):
//   stdout is split manually on \n only — readline also splits on U+2028/U+2029, which
//   are valid inside JSON strings.
// - stdin EOF lets the process exit cleanly; abort falls back to the managed
//   terminateChildProcess lifecycle.
// - Commands larger than maxFrameBytes must be length-prefixed; we never send frames
//   that big (prompts carry file URIs, not file bodies) and refuse instead.

import { spawn, type ChildProcess } from 'node:child_process'
import { resolveCliBinary } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'

export const OMP_SILENCE_TIMEOUT_MS = 180_000
export const OMP_REQUEST_TIMEOUT_MS = 30_000

export interface OmpReadyFrame {
  protocolVersion: number
  supportedProtocolVersions: number[]
  maxFrameBytes: number
  maxReassembledFrameBytes: number
}

export type OmpSessionEvent = Record<string, unknown> & { type: string }

export interface OmpResponseFrame {
  id?: string
  command: string
  success: boolean
  data?: unknown
  error?: string | { message?: string } | Record<string, unknown>
}

interface PendingChunk {
  count: number
  parts: Map<number, string>
}

/** Reassembles rpc_chunk payloads into parsed frames. Pure; returns null while incomplete. */
export function reassembleRpcChunks(chunks: Array<{ chunkId: string; index: number; count: number; data: string }>): { complete: Array<{ chunkId: string; frame: unknown }>; pending: number } {
  const byId = new Map<string, PendingChunk>()
  for (const chunk of chunks) {
    const entry = byId.get(chunk.chunkId) ?? { count: chunk.count, parts: new Map<number, string>() }
    entry.parts.set(chunk.index, chunk.data)
    byId.set(chunk.chunkId, entry)
  }
  const complete: Array<{ chunkId: string; frame: unknown }> = []
  let pending = 0
  for (const [chunkId, entry] of byId) {
    if (entry.parts.size < entry.count) { pending++; continue }
    const joined = Array.from({ length: entry.count }, (_, index) => entry.parts.get(index) ?? '').join('')
    try {
      complete.push({ chunkId, frame: JSON.parse(joined) })
    } catch {
      // A corrupt reassembly is dropped, not fatal: the correlated request times out.
    }
  }
  return { complete, pending }
}

export interface OmpRpcClientOptions {
  binary?: string
  args?: string[]
  cwd: string
  env?: Record<string, string>
  signal?: AbortSignal
  silenceTimeoutMs?: number
  /** Test seam: inject a pre-spawned child. */
  child?: ChildProcess
}

export class OmpRpcClient {
  readonly child: ChildProcess
  readonly ready: Promise<OmpReadyFrame>
  /** Resolves when the process exits — lets callers stop waiting for events that will never come. */
  readonly exited: Promise<void>
  private readonly pending = new Map<string, { resolve: (value: OmpResponseFrame) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
  private readonly eventListeners = new Set<(event: OmpSessionEvent) => void>()
  private readonly chunkBuffers = new Map<string, PendingChunk>()
  private buffer = ''
  private closed = false
  private silenceTimer: ReturnType<typeof setTimeout> | null = null
  private readonly silenceTimeoutMs: number
  private nextId = 1
  private settleExited: () => void = () => {}
  private readonly stdin: NonNullable<ChildProcess['stdin']>
  private readonly stdout: NonNullable<ChildProcess['stdout']>

  constructor(private readonly options: OmpRpcClientOptions) {
    this.silenceTimeoutMs = options.silenceTimeoutMs ?? OMP_SILENCE_TIMEOUT_MS
    this.child = options.child ?? spawn(options.binary ?? 'omp', ['--mode', 'rpc', ...(options.args ?? [])], {
      cwd: options.cwd,
      env: cliSpawnEnv(options.env),
      detached: detachedProcessGroup(),
    })
    if (!this.child.stdin || !this.child.stdout) throw new Error('omp rpc child is missing stdio pipes.')
    this.exited = new Promise<void>((resolve) => { this.settleExited = resolve })
    let settleReady: ((frame: OmpReadyFrame) => void) | undefined
    let failReady: ((error: Error) => void) | undefined
    this.ready = new Promise<OmpReadyFrame>((resolve, reject) => {
      settleReady = resolve
      failReady = reject
    })
    const readyListener = (event: OmpSessionEvent) => {
      if (event.type !== 'ready' || settleReady === undefined) return
      const resolve = settleReady
      settleReady = undefined
      failReady = undefined
      resolve({
        protocolVersion: Number(event.protocolVersion ?? 1),
        supportedProtocolVersions: Array.isArray(event.supportedProtocolVersions) ? event.supportedProtocolVersions.map(Number) : [1],
        maxFrameBytes: Number(event.maxFrameBytes ?? 1_048_576),
        maxReassembledFrameBytes: Number(event.maxReassembledFrameBytes ?? 8_388_608),
      })
    }
    // These stay attached for the child's whole life (an unhandled 'error' event on a
    // ChildProcess throws); they only act before ready settles.
    const onSpawnError = (error: Error) => { failReady?.(error); settleReady = undefined; failReady = undefined }
    const onEarlyExit = (code: number | null) => { onSpawnError(new Error(`omp rpc process exited before ready (code ${code ?? 'unknown'}).`)) }
    this.eventListeners.add(readyListener)
    this.child.once('error', onSpawnError)
    this.child.once('exit', onEarlyExit)
    this.stdin = this.child.stdin
    this.stdout = this.child.stdout
    this.stdout.setEncoding('utf8')
    this.stdout.on('data', (chunk: string) => this.onStdout(chunk))
    this.child.stderr?.setEncoding('utf8')
    this.child.once('exit', () => this.finish())
    options.signal?.addEventListener('abort', () => this.dispose(), { once: true })
    this.armSilenceTimer()
  }

  onEvent(listener: (event: OmpSessionEvent) => void): () => void {
    this.eventListeners.add(listener)
    return () => this.eventListeners.delete(listener)
  }

  send(command: Record<string, unknown>): void {
    if (this.closed) throw new Error('omp rpc client is closed.')
    const line = `${JSON.stringify(command)}\n`
    this.stdin.write(line)
    this.armSilenceTimer()
  }

  request<T = OmpResponseFrame>(command: Record<string, unknown>, timeoutMs = OMP_REQUEST_TIMEOUT_MS): Promise<T> {
    const id = `vontra-${this.nextId++}`
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`omp rpc command "${String(command.type ?? '?')}" timed out after ${timeoutMs}ms.`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (frame) => { resolve(frame as unknown as T) },
        reject,
        timer,
      })
      try {
        this.send({ id, ...command })
      } catch (error) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  dispose(): void {
    if (this.closed) return
    this.closed = true
    if (this.silenceTimer) clearTimeout(this.silenceTimer)
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer)
      entry.reject(new Error('omp rpc client closed.'))
    }
    this.pending.clear()
    // stdin EOF is the clean shutdown path; force-kill only if the child lingers.
    try { this.stdin.end() } catch { /* already closed */ }
    const child = this.child
    setTimeout(() => { terminateChildProcess(child) }, 1_500).unref?.()
  }

  private finish(): void {
    this.settleExited()
    if (this.closed) return
    this.closed = true
    if (this.silenceTimer) clearTimeout(this.silenceTimer)
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer)
      entry.reject(new Error('omp rpc process exited.'))
    }
    this.pending.clear()
  }

  private armSilenceTimer(): void {
    if (this.silenceTimer) clearTimeout(this.silenceTimer)
    this.silenceTimer = setTimeout(() => this.dispose(), this.silenceTimeoutMs)
  }

  private onStdout(chunk: string): void {
    // STRICT LF splitting: never split on U+2028/U+2029 (valid inside JSON strings).
    this.buffer += chunk
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (line) this.handleLine(line)
      newline = this.buffer.indexOf('\n')
    }
    this.armSilenceTimer()
  }

  private handleLine(line: string): void {
    let frame: unknown
    try {
      frame = JSON.parse(line)
    } catch {
      return // malformed lines must not desync the stream
    }
    if (!frame || typeof frame !== 'object') return
    const record = frame as Record<string, unknown>
    if (record.type === 'rpc_chunk') {
      this.handleChunk(record as unknown as { chunkId: string; index: number; count: number; data: string })
      return
    }
    if (record.type === 'response') {
      const response = frame as unknown as OmpResponseFrame
      if (response.id) {
        const entry = this.pending.get(response.id)
        if (entry) {
          this.pending.delete(response.id)
          clearTimeout(entry.timer)
          entry.resolve(response)
        }
      }
      return
    }
    if (record.type === 'prompt_result' || record.type === 'ready') {
      // prompt_result is a streaming-side frame (agent_invoked echo); `ready` is consumed
      // by the ready promise's listener through the normal event path.
    }
    for (const listener of [...this.eventListeners]) {
      try { listener(record as OmpSessionEvent) } catch { /* listener isolation */ }
    }
  }

  private handleChunk(chunk: { chunkId: string; index: number; count: number; data: string }): void {
    const entry = this.chunkBuffers.get(chunk.chunkId) ?? { count: chunk.count, parts: new Map<number, string>() }
    entry.parts.set(chunk.index, chunk.data)
    this.chunkBuffers.set(chunk.chunkId, entry)
    if (entry.parts.size < entry.count) return
    this.chunkBuffers.delete(chunk.chunkId)
    const joined = Array.from({ length: entry.count }, (_, index) => entry.parts.get(index) ?? '').join('')
    this.handleLine(joined)
  }
}

/** Locates the omp binary: explicit path, then known bin dirs, then bare name. */
export function discoverOmpBinary(explicit?: string | null): string {
  return resolveCliBinary(explicit ?? undefined, ['omp'])
}
