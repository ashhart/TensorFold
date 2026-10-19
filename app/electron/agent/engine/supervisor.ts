import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import http from 'node:http'
import net from 'node:net'
import type { EngineState, EngineModelGenerationSettings } from '../types'
import type { ModelInfo } from './cli'
import { assertRepoId, type EngineBinary } from './paths'
import { killGroup } from './pull'
import { loweredEngineMemoryBudget } from './memory-budget'
import { engineLoadLimitOverride } from '../../../shared/engine-advanced-env'
import { CORTHEON_RELEASE_POLICY, cortheonEnabledForRelease } from '../../../shared/release-feature-policy'
import { modelLaunchOptions } from './launch-options'
import { assertEngineModelSamplingSupported, isGreedyOnlyEngineModel } from '../../../shared/model-profiles'

// Native admission accounts for loaded buffers and streams; the weights-only floor cannot reserve retained states.

// One engine process at a time: `serve MODEL` on a free loopback port with a fresh API key per launch,
// ready once /v1/models answers. Switching models stops the old process first; a model whose memory
// floor is above the limit (70% of RAM by default) is refused before anything starts.

export interface SupervisorDeps {
  binary: () => EngineBinary | null
  env: () => NodeJS.ProcessEnv
  info: (repo: string) => Promise<ModelInfo>
  resolveModelArgument?: (repo: string) => Promise<string>
  drafterFor: (repo: string) => string | null
  // Bytes of a cached drafter checkpoint, or null when it is not in the cache.
  drafterBytes: (repo: string) => Promise<number | null>
  totalMemoryBytes: number
  memoryBudgetBytes?: number
  memoryFraction?: number
  readyTimeoutMs?: number
  pollMs?: number
  stopGraceMs?: number
  onChange?: () => void
  onPid?: (pid: number | null) => void
  // True when the next start should pass --cortheon.
  cortheon?: () => boolean
  learning?: (repo: string) => { enabled: boolean; graphFile: string } | null
  modelSettings?: (repo: string) => EngineModelGenerationSettings | null
}

export interface LiveEngine {
  model: string
  baseUrl: string
  apiKey: string
  greedyOnly?: boolean
}

export interface EngineRequestLease extends LiveEngine {
  release(): void
}

export interface SupervisorStatus {
  state: EngineState
  model: string | null
  drafter: string | null
  port: number | null
  startedAt: number | null
  readyAt: number | null
  message: string | null
  logTail: string[]
  memoryLimitBytes: number
  contextTokens: number | null
  // The running (or starting) server was given --cortheon.
  cortheonActive: boolean
  learningActive: boolean
}

const LOG_LINES = 200
const GIB = 1024 ** 3

export function formatGiB(bytes: number): string {
  return `${(bytes / GIB).toFixed(1)} GB`
}

export async function freeLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0, exclusive: true }, () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      server.close(() => (port ? resolve(port) : reject(new Error('No free loopback port.'))))
    })
  })
}

// One JSON request to the engine with the launch key; null on any transport, status or parse failure.
function engineJson(port: number, apiKey: string, method: 'GET' | 'POST', path: string, body?: unknown, timeoutMs = 2000): Promise<any | null> {
  return new Promise((resolve) => {
    const payload = body === undefined ? undefined : JSON.stringify(body)
    const headers: Record<string, string> = { Authorization: `Bearer ${apiKey}`, ...(payload ? { 'Content-Type': 'application/json' } : {}) }
    const req = http.request({ host: '127.0.0.1', port, path, method, headers, timeout: timeoutMs }, (res) => {
      let text = ''
      res.setEncoding('utf8')
      res.on('data', (chunk) => { if (text.length < 65_536) text += chunk })
      res.on('end', () => {
        if (res.statusCode !== 200) return resolve(null)
        try { resolve(JSON.parse(text)) } catch { resolve(null) }
      })
    })
    req.on('timeout', () => req.destroy())
    req.on('error', () => resolve(null))
    req.end(payload)
  })
}

// GET /v1/models; the model list once the server answers, else null.
export async function probeModels(port: number, apiKey: string, timeoutMs = 2000): Promise<{ data: any[] } | null> {
  const doc = await engineJson(port, apiKey, 'GET', '/v1/models', undefined, timeoutMs)
  return Array.isArray(doc?.data) ? doc : null
}

function positiveInt(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : null
}

// The served prompt-plus-reply limit: /v1/models reports max_model_len when the engine has one;
// otherwise /tokenize's max_model_len (null there means no limit is set).
export async function readContextWindow(port: number, apiKey: string, model: string, models: { data: any[] } | null): Promise<number | null> {
  const entry = models?.data.find((m) => m?.id === model) ?? models?.data[0]
  const listed = positiveInt(entry?.max_model_len) ?? positiveInt(entry?.context_length)
  if (listed) return listed
  const tokenized = await engineJson(port, apiKey, 'POST', '/tokenize', { prompt: 'hi' })
  return positiveInt(tokenized?.max_model_len)
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export class EngineSupervisor {
  private state: EngineState = 'stopped'
  private model: string | null = null
  private drafter: string | null = null
  private port: number | null = null
  private apiKey: string | null = null
  private child: ChildProcess | null = null
  private startedAt: number | null = null
  private readyAt: number | null = null
  private message: string | null = null
  private contextTokens: number | null = null
  private log: string[] = []
  private starting: { model: string; promise: Promise<LiveEngine> } | null = null
  private queue: Promise<unknown> = Promise.resolve()
  private generation = 0
  private stopping = false
  private activeRequests = 0
  private exclusiveRequest = false
  private sequenceOwner: symbol | null = null
  private sequenceLabel = 'model comparison'
  private activeSettings: EngineModelGenerationSettings | null = null
  private restoreSettings: { model: string; settings: EngineModelGenerationSettings | null } | null = null
  private cancellation = 0
  private requestModel: string | null = null
  private cortheon = false
  private learning = false
  private greedyOnly = false

  constructor(private readonly deps: SupervisorDeps) {}

  get activeRequestCount(): number { return this.activeRequests }

  get memoryLimitBytes(): number {
    const base = Math.floor(this.memoryBudgetBytes * (this.deps.memoryFraction ?? 0.7))
    // A user-set engine load-limit variable (Settings → Engine advanced) is the engine's own
    // ceiling for its family; the floor check must not veto what the engine was told to allow.
    const override = engineLoadLimitOverride(this.deps.env())
    return override ? Math.max(base, override.bytes) : base
  }

  get memoryBudgetBytes(): number {
    return loweredEngineMemoryBudget(this.deps.totalMemoryBytes, this.deps.memoryBudgetBytes)
  }

  status(): SupervisorStatus {
    return {
      state: this.state,
      model: this.model,
      drafter: this.drafter,
      port: this.port,
      startedAt: this.startedAt,
      readyAt: this.readyAt,
      message: this.message,
      logTail: this.log.slice(-12),
      memoryLimitBytes: this.memoryLimitBytes,
      contextTokens: this.contextTokens,
      cortheonActive: this.cortheon && (this.state === 'starting' || this.state === 'ready'),
      learningActive: this.learning && (this.state === 'starting' || this.state === 'ready'),
    }
  }

  // The running endpoint when it is ready (and serving `model`, if given).
  live(model?: string): LiveEngine | null {
    if (this.state !== 'ready' || !this.model || !this.port || !this.apiKey) return null
    if (model && model !== this.model) return null
    return { model: this.model, baseUrl: `http://127.0.0.1:${this.port}/v1`, apiKey: this.apiKey, ...(this.greedyOnly ? { greedyOnly: true } : {}) }
  }

  serve(model: string): Promise<LiveEngine> {
    try { assertRepoId(model) } catch (error) { return Promise.reject(error) }
    try { this.assertSequenceOwner() } catch (error) { return Promise.reject(error) }
    const ready = this.live(model)
    if (ready) return Promise.resolve(ready)
    if (this.starting?.model === model) return this.starting.promise
    const next = this.queue.then(() => {
      this.assertRequestIdle(model)
      const current = this.live(model)
      if (current) return current
      if (this.starting?.model === model) return this.starting.promise
      return this.switchWithRecovery(model)
    })
    this.queue = next.catch(() => {})
    return next
  }

  // Start the serving model again so a changed serve setting (such as --cortheon) takes effect.
  // Refused while a reply is running; a stopped or failed engine has nothing to restart.
  restart(): Promise<LiveEngine | null> {
    const model = this.model
    if (!model || this.state === 'stopped' || this.state === 'failed') return Promise.resolve(null)
    try { this.assertRequestIdle() } catch (error) { return Promise.reject(error) }
    const next = this.queue.then(() => { this.assertRequestIdle(); return this.switchTo(model) })
    this.queue = next.catch(() => {})
    return next
  }

  // Both the persistent choice and the running process change under the same request guard.
  // No preference is changed when a reply or lesson holds a lease.
  reconfigure(apply: () => Promise<void>, rollback: () => Promise<void>): Promise<void> {
    const next = this.queue.then(async () => {
      this.assertRequestIdle()
      const model = this.model
      const running = model != null && (this.state === 'ready' || this.state === 'starting')
      try {
        await apply()
        if (running) await this.switchTo(model!)
      } catch (error) {
        await rollback()
        if (running && !this.live(model!)) await this.switchTo(model!).catch(() => {})
        throw error
      }
    })
    this.queue = next.catch(() => {})
    return next
  }

  // Reserve the serving model before another queued load can replace its process. Same-model
  // requests share it; another model gets a defined refusal rather than killing an active turn.
  acquire(model: string, options: { start: boolean; signal?: AbortSignal; exclusive?: boolean; owner?: symbol }): Promise<EngineRequestLease> {
    const next = this.queue.then(async () => {
      if (options.signal?.aborted) throw options.signal.reason ?? new Error('Aborted.')
      if (this.exclusiveRequest || options.exclusive && this.activeRequests > 0) throw new Error('Finish or stop active local requests before measuring model speed.')
      this.assertRequestIdle(model, options.owner)
      let live = this.live(model)
      if (!live && !options.start) throw new Error(`The built-in engine is not serving ${model}.`)
      live ??= await this.switchWithRecovery(assertRepoId(model), options.signal)
      if (options.signal?.aborted) throw options.signal.reason ?? new Error('Aborted.')
      this.activeRequests++
      if (options.exclusive) this.exclusiveRequest = true
      this.requestModel = model
      let released = false
      return { ...live, release: () => {
        if (released) return
        released = true
        this.activeRequests--
        if (options.exclusive) this.exclusiveRequest = false
        if (!this.activeRequests) this.requestModel = null
      } }
    })
    this.queue = next.catch(() => {})
    return next
  }

  private assertSequenceOwner(owner?: symbol): void {
    if (this.sequenceOwner && owner !== this.sequenceOwner) throw new Error(`Finish or cancel the ${this.sequenceLabel} before changing or using the local engine.`)
  }

  // Keep admission closed between two measured requests and through restoring the old runtime.
  // The owner is a main-process symbol, never a renderer or model-supplied identifier.
  reserveSequence(label = 'model comparison'): Promise<{ owner: symbol; previousModel: string | null; pause(): Promise<void>; restore(signal?: AbortSignal): Promise<void>; release(): void }> {
    const next = this.queue.then(() => {
      this.assertRequestIdle()
      if (this.state === 'starting') throw new Error('Wait for the current model to finish loading before comparing models.')
      const owner = Symbol('model-comparison')
      const previousModel = this.live()?.model ?? null
      const previousSettings = structuredClone(this.activeSettings)
      this.sequenceOwner = owner
      this.sequenceLabel = label
      let released = false
      return { owner, previousModel,
        pause: async () => {
          const pausing = this.queue.then(async () => { this.assertSequenceOwner(owner); if (this.activeRequests) throw new Error('The owned local engine still has active requests.'); await this.stopNow('stopped', null) })
          this.queue = pausing.catch(() => {})
          await pausing
        },
        restore: async (signal?: AbortSignal) => {
          const restoring = this.queue.then(async () => {
            this.assertSequenceOwner(owner)
            if (signal?.aborted) throw new Error('Local model restoration was cancelled while TensorFold closed.')
            if (this.activeRequests) throw new Error('The comparison request has not stopped yet.')
            if (previousModel) {
              this.restoreSettings = { model: previousModel, settings: previousSettings }
              try { if (!this.live(previousModel)) await this.switchTo(previousModel, signal) } finally { this.restoreSettings = null }
            }
            else await this.stopNow('stopped', null)
          })
          this.queue = restoring.catch(() => {})
          await restoring
        },
        release: () => { if (!released) { released = true; if (this.sequenceOwner === owner) this.sequenceOwner = null } },
      }
    })
    this.queue = next.catch(() => {})
    return next
  }

  private assertRequestIdle(model?: string, owner?: symbol): void {
    this.assertSequenceOwner(owner)
    if (this.activeRequests && (model == null || model !== this.requestModel)) {
      throw Object.assign(new Error('Finish or stop active TensorFold requests before switching or unloading the built-in engine.'), { code: 'ENGINE_MODEL_BUSY' })
    }
  }

  stop(): Promise<void> {
    try { this.assertRequestIdle() } catch (error) { return Promise.reject(error) }
    this.cancellation++
    // A load in progress holds the queue until it settles, so cancel it before queueing the stop.
    if (this.state === 'starting') {
      this.generation++
      killGroup(this.child, 'SIGTERM')
    }
    const next = this.queue.then(() => { this.assertRequestIdle(); return this.stopNow('stopped', null) })
    this.queue = next.catch(() => {})
    return next
  }

  // Synchronous quit path: SIGTERM now (or SIGKILL with force), nothing awaited.
  killNow(signal: NodeJS.Signals = 'SIGKILL'): void {
    this.cancellation++
    this.generation++
    this.stopping = true
    killGroup(this.child, signal)
  }

  private changed(): void {
    try { this.deps.onChange?.() } catch { /* listeners never break the supervisor */ }
  }

  private append(text: string): void {
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trimEnd()
      if (!line.trim()) continue
      this.log.push(line)
      if (this.log.length > LOG_LINES) this.log.shift()
    }
  }

  private async switchTo(model: string, signal?: AbortSignal): Promise<LiveEngine> {
    if (this.child) await this.stopNow('stopped', null)
    if (signal?.aborted) throw signal.reason ?? new Error('Engine start was cancelled.')
    const promise = this.start(model, signal)
    this.starting = { model, promise }
    try {
      return await promise
    } finally {
      if (this.starting?.promise === promise) this.starting = null
    }
  }

  private async switchWithRecovery(model: string, signal?: AbortSignal): Promise<LiveEngine> {
    const previous = this.live()?.model
    const cancellation = this.cancellation
    try { return await this.switchTo(model, signal) }
    catch (error) {
      if (!previous || previous === model || cancellation !== this.cancellation || signal?.aborted) throw error
      try { await this.switchTo(previous) }
      catch (restoreError) { throw new Error(`${(error as Error).message} Restoring the previous model also failed: ${(restoreError as Error).message}`, { cause: error }) }
      throw new Error(`${(error as Error).message} The previous model has been restored.`, { cause: error })
    }
  }

  // A profile for an idle different model can be saved without restarting the current model.
  configureModel(model: string, apply: () => Promise<void>, rollback: () => Promise<void>): Promise<void> {
    const next = this.queue.then(async () => {
      this.assertSequenceOwner()
      const running = this.model === model && (this.state === 'ready' || this.state === 'starting')
      if (running) this.assertRequestIdle()
      try { await apply(); if (running) await this.switchTo(model) }
      catch (error) {
        await rollback()
        if (running && !this.live(model)) {
          try { await this.switchTo(model) }
          catch (restoreError) { throw new Error(`${(error as Error).message} Restoring the previous profile also failed: ${(restoreError as Error).message}`, { cause: error }) }
        }
        throw error
      }
    })
    this.queue = next.catch(() => {})
    return next
  }

  // Keep cache deletion ordered with launches; its callback performs no engine requests.
  mutateCache<T>(run: () => Promise<T>): Promise<T> {
    const next = this.queue.then(() => { this.assertSequenceOwner(); return run() })
    this.queue = next.catch(() => {})
    return next
  }

  private fail(message: string): never {
    this.state = 'failed'
    this.message = message
    this.changed()
    throw new Error(message)
  }

  private async start(model: string, signal?: AbortSignal): Promise<LiveEngine> {
    const generation = ++this.generation
    this.state = 'starting'
    this.model = model
    this.drafter = null
    this.port = null
    this.apiKey = null
    this.startedAt = Date.now()
    this.readyAt = null
    this.contextTokens = null
    this.message = 'Checking memory'
    this.log = []
    this.changed()

    const cancel = () => {
      if (generation === this.generation && this.state === 'starting') {
        if (this.sequenceOwner) { this.generation++; killGroup(this.child, 'SIGTERM') }
        else void this.stop().catch(() => {})
      }
    }
    signal?.addEventListener('abort', cancel, { once: true })
    if (signal?.aborted) cancel()
    try {

    const binary = this.deps.binary()
    if (!binary) this.fail('The TensorFold engine is not bundled with this build.')
    let info: ModelInfo
    try {
      info = await this.deps.info(model)
    } catch (error) {
      this.fail((error as Error).message)
    }
    const settings = this.restoreSettings?.model === model ? this.restoreSettings.settings : this.deps.modelSettings?.(model)
    this.activeSettings = settings ? structuredClone(settings) : null
    this.greedyOnly = isGreedyOnlyEngineModel(info)
    if (settings) assertEngineModelSamplingSupported(info, settings)
    const profileArgs = modelLaunchOptions(settings, info.contextTokens)
    const drafter = settings?.speculativeDecoding === 'off' ? null : this.deps.drafterFor(model)
    let floor = info.memoryFloorBytes
    if (drafter) {
      const drafterBytes = await this.deps.drafterBytes(drafter)
      if (drafterBytes == null) this.fail(`${model} needs its drafter ${drafter}. Download the model again to fetch it.`)
      floor += drafterBytes
    }
    if (floor > this.memoryLimitBytes) {
      const raisedBase = Math.floor(this.memoryBudgetBytes * (this.deps.memoryFraction ?? 0.7))
      const override = engineLoadLimitOverride(this.deps.env())
      const basis = override && override.bytes > raisedBase
        ? `${override.key}=${String(override.gb)} GB, already above the 70% default`
        : this.memoryBudgetBytes < this.deps.totalMemoryBytes ? `70% of the lowered ${formatGiB(this.memoryBudgetBytes)} qualification budget` : `70% of this Mac's ${formatGiB(this.deps.totalMemoryBytes)}`
      this.fail(`${model} needs at least ${formatGiB(floor)} of memory, more than the ${formatGiB(this.memoryLimitBytes)} limit (${basis}).`)
    }
    if (generation !== this.generation) this.fail('Engine start was cancelled.')

    let argument = model
    let drafterArgument = drafter
    try {
      argument = await this.deps.resolveModelArgument?.(model) ?? model
      if (drafter) drafterArgument = await this.deps.resolveModelArgument?.(drafter) ?? drafter
    } catch (error) { this.fail((error as Error).message) }
    if (generation !== this.generation) this.fail('Engine start was cancelled.')

    const port = await freeLoopbackPort()
    if (generation !== this.generation) this.fail('Engine start was cancelled.')
    const apiKey = randomBytes(32).toString('hex')
    this.cortheon = cortheonEnabledForRelease(this.deps.cortheon?.())
    const learning = this.deps.learning?.(model)
    this.learning = learning?.enabled === true
    const args = ['serve', argument, '--host', '127.0.0.1', '--port', String(port), '--name', model, ...profileArgs, ...(drafterArgument ? ['--drafter', drafterArgument] : []), ...(this.cortheon ? ['--cortheon'] : []), ...(learning ? ['--slide-graph', learning.graphFile, ...(learning.enabled ? ['--slide'] : [])] : [])]
    // Independently reject a stale injected binary prefix as well as the normal serve choice.
    if (!CORTHEON_RELEASE_POLICY.enabled && [...binary.args, ...args].includes('--cortheon')) this.fail(CORTHEON_RELEASE_POLICY.reason)
    const env = { ...this.deps.env(), TENSORFOLD_API_KEY: apiKey, TENSORFOLD_NO_UPDATE_CHECK: '1' }
    const child = spawn(binary.command, [...binary.args, ...args], { env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
    this.child = child
    this.drafter = drafter
    this.port = port
    this.apiKey = apiKey
    this.message = `Loading ${model}`
    this.deps.onPid?.(child.pid ?? null)
    this.changed()
    let exited: string | null = null
    child.stdout?.on('data', (chunk: Buffer) => this.append(chunk.toString('utf8')))
    child.stderr?.on('data', (chunk: Buffer) => this.append(chunk.toString('utf8')))
    child.once('error', (error) => { exited = `Could not start the engine: ${error.message}` })
    child.once('exit', (code, signal) => {
      exited = exited ?? `The engine exited (${signal ?? `code ${code}`})${this.log.length ? `: ${this.log.at(-1)}` : '.'}`
      if (this.child !== child) return
      this.child = null
      this.deps.onPid?.(null)
      if (this.state === 'ready') {
        this.state = this.stopping ? 'stopped' : 'failed'
        this.message = this.stopping ? null : exited
        if (this.stopping) this.model = null
        this.port = null
        this.apiKey = null
        this.changed()
      }
    })

    const deadline = Date.now() + (this.deps.readyTimeoutMs ?? 15 * 60_000)
    let models: { data: any[] } | null = null
    while (!(models = await probeModels(port, apiKey))) {
      if (generation !== this.generation) {
        killGroup(child, 'SIGTERM')
        this.fail('Engine start was cancelled.')
      }
      if (exited) this.fail(exited)
      if (Date.now() > deadline) {
        killGroup(child, 'SIGKILL')
        this.fail(`${model} did not finish loading in time.`)
      }
      await delay(this.deps.pollMs ?? 500)
    }
    if (exited || generation !== this.generation) this.fail(exited ?? 'Engine start was cancelled.')
    this.contextTokens = await readContextWindow(port, apiKey, model, models)
    if (generation !== this.generation) this.fail('Engine start was cancelled.')
    this.state = 'ready'
    this.readyAt = Date.now()
    this.message = null
    this.changed()
    return { model, baseUrl: `http://127.0.0.1:${port}/v1`, apiKey, ...(this.greedyOnly ? { greedyOnly: true } : {}) }
    } finally { signal?.removeEventListener('abort', cancel) }
  }

  private async stopNow(state: EngineState, message: string | null): Promise<void> {
    const child = this.child
    if (child) {
      this.stopping = true
      const exited = new Promise<void>((resolve) => {
        if (child.exitCode !== null || child.signalCode !== null) return resolve()
        child.once('exit', () => resolve())
      })
      killGroup(child, 'SIGTERM')
      const hardKill = setTimeout(() => killGroup(child, 'SIGKILL'), this.deps.stopGraceMs ?? 10_000)
      await exited
      clearTimeout(hardKill)
      if (this.child === child) this.child = null
      this.stopping = false
      this.deps.onPid?.(null)
    }
    this.state = state
    this.model = null
    this.drafter = null
    this.port = null
    this.apiKey = null
    this.startedAt = null
    this.readyAt = null
    this.contextTokens = null
    this.message = message
    this.changed()
  }
}
