import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { terminateChildProcess } from '../child-process-lifecycle'

// Client for the native tensorfold-computer helper: newline-delimited JSON over stdio to one long-lived
// process, started on first use and again after it exits. Safety checks live in the helper itself.

export const COMPUTER_HELPER_BINARY = 'tensorfold-computer'
export const COMPUTER_HELPER_TIMEOUT_MS = 15_000
const OPEN_APP_TIMEOUT_MS = 30_000
const STOP_TIMEOUT_MS = 3_000
const MAX_LINE_CHARS = 16 * 1024 * 1024
const CHILD_ENV_KEYS = ['PATH', 'HOME', 'TMPDIR'] as const

export type ComputerHelperErrorCode =
  | 'bad_json' | 'bad_request' | 'unknown_cmd' | 'no_accessibility' | 'own_app' | 'bad_point' | 'stale_element'
  | 'secure_field' | 'bad_key' | 'not_found' | 'no_app' | 'launch_failed' | 'internal'
  | 'timeout' | 'exited' | 'spawn_failed' | 'missing_helper' | 'protocol'

export class ComputerHelperError extends Error {
  readonly code: ComputerHelperErrorCode | (string & {})

  constructor(code: ComputerHelperErrorCode | (string & {}), message: string) {
    super(message)
    this.name = 'ComputerHelperError'
    this.code = code
  }
}

export interface ComputerPoint { x: number; y: number }
export interface ComputerFrame { x: number; y: number; width: number; height: number }
export interface ComputerApp { name: string; bundleId: string | null; pid: number }
export interface ComputerRunningApp extends ComputerApp { active: boolean }
export interface ComputerWindow { title: string; frame: ComputerFrame }
export interface ComputerPermissions { accessibility: boolean; screenRecording: boolean }
export interface ComputerApps { apps: ComputerRunningApp[] }
export interface ComputerFrontmost { app: ComputerApp; window: ComputerWindow | null }
// Global points, top-left origin; scale is backing pixels per point.
export interface ComputerScreen { x: number; y: number; width: number; height: number; scale: number }

export interface ComputerElement {
  id: number
  role: string
  subrole?: string
  title?: string
  value?: string
  description?: string
  placeholder?: string
  frame: ComputerFrame
  enabled: boolean
  focused: boolean
  secure: boolean
}

export interface ComputerObservation {
  app: ComputerApp
  window: ComputerWindow | null
  screen: ComputerScreen
  elements: ComputerElement[]
  // The element limit, the visit cap or the time budget cut the walk short.
  truncated: boolean
}

// What every input action returns: the frontmost app and window once the action settled.
export interface ComputerActionResult {
  app: ComputerApp | null
  window: ComputerWindow | null
  via?: 'press' | 'mouse'
  typed?: number
}

export interface ComputerPermissionRequest { accessibility?: boolean; screenRecording?: boolean }
export interface ComputerObserveOptions { maxElements?: number; pid?: number }
// Either x and y (global points) or element (an id from the latest observe); the helper refuses both or neither.
export interface ComputerClickInput { x?: number; y?: number; element?: number; button?: 'left' | 'right'; count?: 1 | 2 }
// dy > 0 scrolls toward later content and dx > 0 to the right, in lines; a point or element moves the pointer there first.
export interface ComputerScrollInput { x?: number; y?: number; element?: number; dx?: number; dy?: number }
export interface ComputerDragInput { from: ComputerPoint; to: ComputerPoint }
// Exactly one of name or bundleId.
export interface ComputerOpenAppInput { name?: string; bundleId?: string }

export interface ComputerHelperOptions {
  binaryPath: string
  ownBundleId: string
  timeoutMs?: number
  // Source of PATH, HOME and TMPDIR for the child; nothing else is passed on.
  env?: NodeJS.ProcessEnv
  onLog?: (line: string) => void
}

export function computerHelperPath(resourcesPath: string, devRoot: string): string {
  const packaged = path.join(resourcesPath, 'computer-use', COMPUTER_HELPER_BINARY)
  return existsSync(packaged) ? packaged : path.join(devRoot, 'build', 'computer-use', COMPUTER_HELPER_BINARY)
}

export function computerHelperEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const out: NodeJS.ProcessEnv = {}
  for (const key of CHILD_ENV_KEYS) {
    const value = env[key]
    if (typeof value === 'string' && value) out[key] = value
  }
  return out
}

// The helper paces typing at roughly a millisecond a character, plus a check before every line.
export function typeTimeoutMs(text: string, baseMs = COMPUTER_HELPER_TIMEOUT_MS): number {
  return baseMs + text.length * 5
}

interface Pending {
  proc: ChildProcessWithoutNullStreams
  cmd: string
  timer: NodeJS.Timeout
  resolve: (body: Record<string, unknown>) => void
  reject: (error: Error) => void
}

export class ComputerHelper {
  private proc: ChildProcessWithoutNullStreams | null = null
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private readonly timeoutMs: number

  constructor(private readonly options: ComputerHelperOptions) {
    this.timeoutMs = options.timeoutMs ?? COMPUTER_HELPER_TIMEOUT_MS
  }

  get running(): boolean {
    return this.proc !== null
  }

  get pid(): number | null {
    return this.proc?.pid ?? null
  }

  request<T extends object>(cmd: string, params: Record<string, unknown> = {}, timeoutMs = this.timeoutMs): Promise<T> {
    let proc: ChildProcessWithoutNullStreams
    try {
      proc = this.ensureProcess()
    } catch (error) {
      return Promise.reject(error)
    }
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return
        reject(new ComputerHelperError('timeout', `The computer helper did not answer ${cmd} within ${timeoutMs} ms.`))
        // The helper runs one request at a time, so a stuck one would hold up every later call.
        this.discard(proc)
      }, timeoutMs)
      this.pending.set(id, { proc, cmd, timer, resolve: (body) => resolve(body as T), reject })
      proc.stdin.write(`${JSON.stringify({ ...params, id, cmd })}\n`)
    })
  }

  permissions(): Promise<ComputerPermissions> {
    return this.request('permissions')
  }

  requestPermissions(which: ComputerPermissionRequest = { accessibility: true, screenRecording: true }): Promise<ComputerPermissions> {
    return this.request('request_permissions', { ...which })
  }

  apps(): Promise<ComputerApps> {
    return this.request('apps')
  }

  frontmost(): Promise<ComputerFrontmost> {
    return this.request('frontmost')
  }

  observe(options: ComputerObserveOptions = {}): Promise<ComputerObservation> {
    return this.request('observe', { ...options })
  }

  click(input: ComputerClickInput): Promise<ComputerActionResult> {
    return this.request('click', { ...input })
  }

  move(point: ComputerPoint): Promise<ComputerActionResult> {
    return this.request('move', { x: point.x, y: point.y })
  }

  drag(input: ComputerDragInput): Promise<ComputerActionResult> {
    return this.request('drag', { from: { x: input.from.x, y: input.from.y }, to: { x: input.to.x, y: input.to.y } })
  }

  scroll(input: ComputerScrollInput): Promise<ComputerActionResult> {
    return this.request('scroll', { ...input })
  }

  type(input: { text: string }): Promise<ComputerActionResult> {
    return this.request('type', { text: input.text }, typeTimeoutMs(input.text, this.timeoutMs))
  }

  key(input: { keys: string }): Promise<ComputerActionResult> {
    return this.request('key', { keys: input.keys })
  }

  openApp(input: ComputerOpenAppInput): Promise<ComputerActionResult> {
    return this.request('open_app', { ...input }, Math.max(this.timeoutMs, OPEN_APP_TIMEOUT_MS))
  }

  // Asks the helper to exit and waits for it; the next call starts a fresh one.
  async stop(): Promise<void> {
    const proc = this.proc
    if (!proc) return
    const closed = new Promise<void>((resolve) => proc.once('close', () => resolve()))
    const reply = this.request('stop', {}, STOP_TIMEOUT_MS).catch(() => undefined)
    if (this.proc === proc) this.proc = null
    await reply
    const timer = setTimeout(() => terminateChildProcess(proc), STOP_TIMEOUT_MS)
    await closed
    clearTimeout(timer)
  }

  // Synchronous teardown for app quit: no goodbye, pending calls fail with exited.
  dispose(): void {
    const proc = this.proc
    if (proc) this.discard(proc)
  }

  private discard(proc: ChildProcessWithoutNullStreams): void {
    if (this.proc === proc) this.proc = null
    terminateChildProcess(proc)
  }

  private ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.proc) return this.proc
    const binary = this.options.binaryPath
    if (!existsSync(binary)) throw new ComputerHelperError('missing_helper', `The computer helper is not at ${binary}.`)
    const proc = spawn(binary, ['--own-bundle-id', this.options.ownBundleId], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: computerHelperEnv(this.options.env ?? process.env),
    })
    this.proc = proc
    let stdout = ''
    let stderr = ''
    proc.stdout.setEncoding('utf8')
    proc.stdout.on('data', (chunk: string) => {
      stdout += chunk
      let newline = stdout.indexOf('\n')
      while (newline >= 0) {
        this.onLine(proc, stdout.slice(0, newline))
        stdout = stdout.slice(newline + 1)
        newline = stdout.indexOf('\n')
      }
      if (stdout.length > MAX_LINE_CHARS) {
        this.options.onLog?.('computer helper wrote an oversized line; restarting it')
        stdout = ''
        this.discard(proc)
      }
    })
    proc.stderr.setEncoding('utf8')
    proc.stderr.on('data', (chunk: string) => {
      stderr = (stderr + chunk).slice(-65_536)
      const lines = stderr.split('\n')
      stderr = lines.pop() ?? ''
      for (const line of lines) if (line.trim()) this.options.onLog?.(line)
    })
    // EPIPE after the helper died; the close handler fails whatever was waiting.
    proc.stdin.on('error', () => {})
    proc.on('error', (error) => this.onClose(proc, new ComputerHelperError('spawn_failed', `The computer helper could not start: ${error.message}`)))
    proc.on('close', (code, signal) => this.onClose(proc, new ComputerHelperError('exited', `The computer helper exited (${signal ?? `code ${code}`}).`)))
    return proc
  }

  private onLine(proc: ChildProcessWithoutNullStreams, line: string): void {
    if (!line.trim()) return
    let message: any
    try {
      message = JSON.parse(line)
    } catch {
      this.options.onLog?.(`computer helper wrote a non-JSON line: ${line.slice(0, 200)}`)
      return
    }
    if (!message || typeof message !== 'object' || typeof message.id !== 'number') {
      if (message?.ok === false) this.options.onLog?.(`computer helper rejected a request: ${String(message.code)} ${String(message.error ?? '')}`)
      return
    }
    const entry = this.pending.get(message.id)
    if (!entry || entry.proc !== proc) return
    this.pending.delete(message.id)
    clearTimeout(entry.timer)
    if (message.ok === true) {
      const { id: _id, ok: _ok, ...body } = message
      entry.resolve(body)
      return
    }
    const code = typeof message.code === 'string' && message.code ? message.code : 'protocol'
    const text = typeof message.error === 'string' && message.error ? message.error : `The computer helper refused ${entry.cmd}.`
    entry.reject(new ComputerHelperError(code, text))
  }

  private onClose(proc: ChildProcessWithoutNullStreams, error: ComputerHelperError): void {
    if (this.proc === proc) this.proc = null
    for (const [id, entry] of this.pending) {
      if (entry.proc !== proc) continue
      this.pending.delete(id)
      clearTimeout(entry.timer)
      entry.reject(error)
    }
  }
}
