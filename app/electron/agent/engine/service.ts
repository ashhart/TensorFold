import { execFile } from 'node:child_process'
import { promises as fs, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { EngineCapabilities, EngineConnection, EngineEvent, EngineInstalledModel, EnginePerformance, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness } from '../types'
import { FAMILY_TITLES, SUGGESTED_MODELS, drafterFor } from './catalog'
import { parseCapabilities, parseInfoOutput, parseModelsOutput, parseVersion, runEngineCli, type ModelInfo } from './cli'
import { createHubFetch, preflightRepo, searchHub } from './hub'
import { assertRepoId, freeDiskBytes, hasCachedSnapshot, hubCacheDir, isRevision, repoCacheDir, repoCachedBytes, type EngineBinary } from './paths'
import { setEnginePerformanceRecorder, setEngineServeHook } from './provider'
import { createTurnStats, finalizeTurnStats, readEngineTimings, recordModelCall, type ModelCallSample } from '../turn-stats'
import { recordEngineContext } from './context'
import { startPull, type PullHandle } from './pull'
import { EngineSupervisor } from './supervisor'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'

// Glue for the main process: one supervisor, the pull jobs, cached CLI answers, and the events the
// renderer mirrors. Electron-free; main-app passes the binary path, app home and an emitter.

export interface EngineServiceOptions {
  binaryPath: string
  appHome: string
  env?: NodeJS.ProcessEnv
  // Exact environment for engine processes; tests use it to reach the fake engine's knobs.
  childEnv?: NodeJS.ProcessEnv
  binary?: EngineBinary
  fetch?: typeof fetch
  totalMemoryBytes?: number
  memoryBudgetBytes?: number
  readyTimeoutMs?: number
  pollMs?: number
  emit: (event: EngineEvent) => void
  onInstalled?: (repos: string[]) => void
}

const CHILD_ENV_KEYS = ['HOME', 'PATH', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'USER', 'LOGNAME', 'SHELL', 'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE']

// The engine gets ordinary OS values and the Hugging Face cache settings, never the app's provider keys.
export function engineChildEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const key of CHILD_ENV_KEYS) if (typeof base[key] === 'string') env[key] = base[key]
  env.HOME = env.HOME || os.homedir()
  env.PATH = env.PATH || '/usr/bin:/bin:/usr/sbin:/sbin'
  return env
}

function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM' }
}

export class EngineService {
  readonly supervisor: EngineSupervisor
  private readonly env: NodeJS.ProcessEnv
  private readonly totalMemoryBytes: number
  private version: string | null = null
  private versionChecked = false
  private capabilities: EngineCapabilities | null = null
  private performance: EnginePerformance | null = null
  private installedCache: EngineInstalledModel[] | null = null
  private readonly pulls = new Map<string, { handle: PullHandle; last: PullProgress }>()
  private readonly finished = new Map<string, PullProgress>()
  private readonly preflights = new Map<string, ModelPreflight>()
  private readonly lastPullEmit = new Map<string, number>()
  private hubFetch: typeof fetch | null
  private readonly revisions: ModelRevisions
  private readonly preferences: EnginePreferenceStore
  // Whether the bundled engine lists --cortheon among its serve flags; null until asked.
  private cortheonSupported: boolean | null = null

  constructor(private readonly options: EngineServiceOptions) {
    this.env = options.childEnv ?? engineChildEnv(options.env ?? process.env)
    this.totalMemoryBytes = options.totalMemoryBytes ?? os.totalmem()
    this.hubFetch = options.fetch ?? null
    this.revisions = new ModelRevisions(options.appHome, () => this.hub)
    this.preferences = new EnginePreferenceStore(options.appHome)
    this.supervisor = new EngineSupervisor({
      binary: () => this.binary(),
      env: () => this.env,
      info: (repo) => this.info(repo),
      resolveModelArgument: (repo) => this.modelArgument(repo),
      drafterFor,
      drafterBytes: (repo) => this.cachedModelBytes(repo),
      totalMemoryBytes: this.totalMemoryBytes,
      memoryBudgetBytes: options.memoryBudgetBytes,
      readyTimeoutMs: options.readyTimeoutMs,
      pollMs: options.pollMs,
      onChange: () => this.emitStatus(),
      onPid: (pid) => { void this.recordPid(pid) },
      cortheon: () => this.cortheonSupported === true && this.preferences.get().cortheon,
    })
  }

  get hub(): string {
    return hubCacheDir(this.env)
  }

  binary(): EngineBinary | null {
    if (this.options.binary) return this.options.binary
    return existsSync(this.options.binaryPath) ? { command: this.options.binaryPath, args: [] } : null
  }

  private requireBinary(): EngineBinary {
    const binary = this.binary()
    if (!binary) throw new Error('The TensorFold engine is not bundled with this build.')
    return binary
  }

  // Wires streamChat's engine hook to this service and clears an engine a crashed run left behind.
  install(): void {
    setEnginePerformanceRecorder((model, sample) => this.recordPerformance(model, sample))
    setEngineServeHook((model, options) => this.supervisor.acquire(model, options))
    void this.reapStaleEngine()
  }

  async status(): Promise<EngineStatus> {
    if (!this.versionChecked && this.binary()) {
      this.versionChecked = true
      try { this.version = parseVersion((await runEngineCli(this.requireBinary(), ['--version'], this.env, 10_000)).stdout) } catch { this.version = null }
    }
    if (this.cortheonSupported == null && this.binary()) {
      try { this.cortheonSupported = (await this.capabilitiesDoc()).flags.includes('--cortheon') } catch { this.cortheonSupported = false }
    }
    return this.statusNow()
  }

  // Saves the Cortheon choice and restarts a loaded model so its server picks the flag up.
  async setCortheon(enabled: boolean): Promise<EngineStatus> {
    await this.status()
    if (enabled && !this.cortheonSupported) throw new Error("This build's TensorFold engine does not offer Cortheon yet.")
    await this.preferences.set({ cortheon: enabled })
    this.emitStatus()
    const running = this.supervisor.status()
    if ((running.state === 'ready' || running.state === 'starting') && running.cortheonActive !== enabled) await this.supervisor.restart()
    return this.statusNow()
  }

  private statusNow(): EngineStatus {
    const { cortheonActive, ...s } = this.supervisor.status()
    const binary = this.binary()
    return {
      ...s, cortheon: { supported: this.cortheonSupported === true, enabled: this.preferences.get().cortheon, active: cortheonActive }, available: !!binary, version: this.version, binaryPath: binary ? this.options.binaryPath : null, totalMemoryBytes: this.totalMemoryBytes, memoryBudgetBytes: this.supervisor.memoryBudgetBytes,
      hardware: { platform: os.platform(), architecture: os.arch(), chip: os.cpus()[0]?.model?.trim() || null },
    }
  }

  private emitStatus(): void {
    const status = this.statusNow()
    if (status.state !== 'ready' || status.model !== this.performance?.model) this.clearPerformance()
    if (status.state === 'ready') recordEngineContext(status.model, status.contextTokens)
    this.options.emit({ kind: 'status', status })
  }

  async snapshot(): Promise<EngineSnapshot> {
    const [status, installed] = await Promise.all([this.status(), this.installed()])
    return { status, performance: this.performance, installed, pulls: this.pullList(), suggestions: SUGGESTED_MODELS, hubCache: this.hub }
  }

  // For an explicit client-connection request, without starting a model or emitting credentials.
  async connection(): Promise<EngineConnection | null> {
    return this.supervisor.live()
  }

  private clearPerformance(): void {
    if (!this.performance) return
    this.performance = null
    this.options.emit({ kind: 'performance', performance: null })
  }

  private recordPerformance(model: string, sample: ModelCallSample): void {
    const status = this.supervisor.status()
    // Reject old requests even when the same model was unloaded and launched again.
    if (!this.supervisor.live(model) || !Number.isFinite(sample.requestAt) || status.readyAt == null || sample.requestAt < status.readyAt) return
    const raw = sample.engine
    const engine = raw ? readEngineTimings({ tokens_per_second: raw.decodeTokensPerSecond, seconds: raw.decodeSeconds, prefill_seconds: raw.prefillSeconds, time_to_first_token: raw.firstTokenSeconds }) : null
    const stats = createTurnStats()
    recordModelCall(stats, { ...sample, engine })
    const latest = finalizeTurnStats(stats, { model, providerId: 'tensorfold' })
    const metric = (value: number | undefined): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
    this.performance = {
      model, measuredAt: Date.now(),
      decodeTokensPerSecond: metric(engine && engine.decodeSeconds > 0 ? engine.decodeTokensPerSecond : latest.decodeTokensPerSecond),
      prefillTokensPerSecond: metric(latest.prefillTokensPerSecond),
      firstTokenMs: metric(latest.firstTokenMs),
      completionTokens: metric(latest.completionTokens),
    }
    this.options.emit({ kind: 'performance', performance: this.performance })
  }

  async info(repo: string): Promise<ModelInfo> {
    const result = await runEngineCli(this.requireBinary(), ['info', await this.modelArgument(repo)], this.env)
    if (result.code !== 0) throw new Error((result.stdout || result.stderr).trim().split(/\r?\n/)[0] || `tensorfold-native info ${repo} failed.`)
    return parseInfoOutput(result.stdout)
  }

  private async modelArgument(repo: string): Promise<string> {
    assertRepoId(repo)
    if (this.pulls.has(repo) || [...this.pulls.keys()].some((model) => drafterFor(model) === repo)) throw new Error('Wait for the model download to finish before loading it.')
    return await this.revisions.selected(repo) ?? repo
  }

  private async cachedModelBytes(repo: string): Promise<number | null> {
    try {
      const selected = await this.revisions.selected(repo)
      if (selected) return this.revisions.snapshotBytes(selected)
      return await hasCachedSnapshot(this.hub, repo) ? repoCachedBytes(this.hub, repo) : null
    } catch { return null }
  }

  async capabilitiesDoc(): Promise<EngineCapabilities> {
    if (this.capabilities) return this.capabilities
    const result = await runEngineCli(this.requireBinary(), ['capabilities', '--json'], this.env)
    if (result.code !== 0) throw new Error(result.stderr.trim() || 'tensorfold-native capabilities --json failed.')
    this.capabilities = parseCapabilities(result.stdout)
    return this.capabilities
  }

  async installed(refresh = false): Promise<EngineInstalledModel[]> {
    if (this.installedCache && !refresh) return this.installedCache
    if (!this.binary()) return []
    const listed = await runEngineCli(this.requireBinary(), ['models'], this.env)
    const selected = await this.revisions.read()
    const models = new Map(parseModelsOutput(listed.stdout).map((model) => [model.repo, model]))
    for (const repo of Object.keys(selected)) if (!models.has(repo)) models.set(repo, { repo, family: '', modelType: '' })
    const out: EngineInstalledModel[] = []
    for (const model of models.values()) {
      const drafter = drafterFor(model.repo)
      const drafterBytes = drafter ? await this.cachedModelBytes(drafter) : null
      const entry: EngineInstalledModel = { ...model, format: null, contextTokens: null, weightsBytes: null, memoryFloorBytes: null, fitsMemory: null, drafter, drafterInstalled: drafter ? drafterBytes != null : null }
      try {
        const info = await this.info(model.repo)
        entry.family = info.family
        entry.modelType = info.modelType
        entry.format = info.format
        entry.contextTokens = info.contextTokens
        entry.weightsBytes = info.weightsBytes
        entry.memoryFloorBytes = info.memoryFloorBytes + (drafterBytes ?? 0)
        entry.fitsMemory = entry.memoryFloorBytes <= this.supervisor.memoryLimitBytes
      } catch {
        if (selected[model.repo]) continue // Never present a different legacy ref as the selected model.
        /* Legacy listed but unreadable: keep the row without sizes. */
      }
      out.push(entry)
    }
    this.installedCache = out
    this.options.emit({ kind: 'installed', installed: out })
    this.options.onInstalled?.(out.map((m) => m.repo))
    return out
  }

  async search(query: string): Promise<HubModelSummary[]> {
    return searchHub(query, this.hubDeps())
  }

  private hubDeps() {
    this.hubFetch ??= createHubFetch()
    return { fetch: this.hubFetch, offline: /^(1|true|yes|on)$/i.test(this.env.HF_HUB_OFFLINE ?? '') }
  }

  async preflight(repo: string, revision: string | null = null): Promise<ModelPreflight> {
    assertRepoId(repo)
    if (revision != null && !isRevision(revision)) throw new Error('Expected a branch, tag or commit for the revision.')
    // Offline: fail before asking the engine for its families or touching the network.
    const hub = this.hubDeps()
    if (hub.offline) throw new Error('Hugging Face is offline in this session (HF_HUB_OFFLINE is set).')
    const [caps, installed] = await Promise.all([this.capabilitiesDoc(), this.installed()])
    const result = await preflightRepo(repo, revision, {
      ...hub,
      families: caps.families,
      familyTitles: FAMILY_TITLES,
      totalMemoryBytes: this.totalMemoryBytes,
      memoryLimitBytes: this.supervisor.memoryLimitBytes,
      memoryBudgetBytes: this.supervisor.memoryBudgetBytes,
      installed: new Set(installed.map((m) => m.repo)),
      drafterFor,
      cachedBytes: (r) => repoCachedBytes(this.hub, r),
      freeDiskBytes: () => freeDiskBytes(this.hub),
    })
    this.preflights.set(JSON.stringify([repo, revision]), result)
    return result
  }

  pullList(): PullProgress[] {
    const running = [...this.pulls.values()].map((p) => p.last)
    return [...running, ...[...this.finished.values()].filter((p) => !this.pulls.has(p.repo))]
  }

  async pull(repo: string, revision: string | null = null): Promise<PullProgress> {
    if (this.pulls.has(assertRepoId(repo))) return this.pulls.get(repo)!.last
    await this.revisions.read() // Preserve invalid registry bytes and refuse before starting a child.
    this.assertPullIdle(repo)
    const pre = this.preflights.get(JSON.stringify([repo, revision])) ?? await this.preflight(repo, revision)
    this.assertPullIdle(repo)
    if (this.pulls.has(repo)) return this.pulls.get(repo)!.last
    if (!['ok', 'installed'].includes(pre.verdict)) throw new Error(pre.reasons[0] ?? `${repo} cannot be downloaded for the built-in engine.`)
    if (!pre.resolvedRevision || (pre.drafter && !pre.drafter.resolvedRevision)) throw new Error('Retry the model checks to resolve an immutable checkpoint before downloading.')
    const steps = [{ repo, revision: pre.resolvedRevision }, ...(pre.drafter ? [{ repo: pre.drafter.repo, revision: pre.drafter.resolvedRevision! }] : [])]
    this.finished.delete(repo)
    const handle = startPull({
      binary: this.requireBinary(),
      env: this.env,
      hub: this.hub,
      steps,
      totalBytes: pre.totalBytes,
      pollMs: this.options.pollMs ? Math.max(50, this.options.pollMs) : 1000,
      onUpdate: (progress) => this.onPullUpdate(progress),
    })
    this.pulls.set(repo, { handle, last: handle.progress() })
    this.options.emit({ kind: 'pull', pull: handle.progress() })
    void handle.done.then(async (final) => {
      if (final.state === 'done') {
        try { await this.revisions.commit(steps) } catch (error) {
          final = { ...final, state: 'failed', phase: 'Failed', message: (error as Error).message }
        }
      }
      this.pulls.delete(repo)
      this.finished.set(repo, final)
      this.invalidatePreflight(repo)
      this.installedCache = null
      this.options.emit({ kind: 'pull', pull: final })
      if (final.state === 'done') await this.installed(true).catch(() => {})
    })
    return handle.progress()
  }

  private onPullUpdate(progress: PullProgress): void {
    if (progress.state !== 'running') return // Completion is emitted only after the selection is durable.
    const job = this.pulls.get(progress.repo)
    if (job) job.last = progress
    const now = Date.now()
    if (progress.state === 'running' && now - (this.lastPullEmit.get(progress.repo) ?? 0) < 250) return
    this.lastPullEmit.set(progress.repo, now)
    this.options.emit({ kind: 'pull', pull: progress })
  }

  private assertPullIdle(repo: string): void {
    const status = this.supervisor.status()
    if ((status.model === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error('Unload the model before downloading another checkpoint for it.')
  }

  private invalidatePreflight(repo: string): void {
    for (const [key, value] of this.preflights) if (value.repo === repo) this.preflights.delete(key)
  }

  cancelPull(repo: string): boolean {
    const job = this.pulls.get(assertRepoId(repo))
    if (!job) return false
    job.handle.cancel()
    return true
  }

  async load(repo: string): Promise<EngineStatus> {
    await this.supervisor.serve(assertRepoId(repo))
    return this.statusNow()
  }

  async unload(): Promise<EngineStatus> {
    await this.supervisor.stop()
    this.clearPerformance()
    return this.statusNow()
  }

  // Removes the repo's folder from the shared cache, and its drafter when no other installed model uses it.
  async remove(repo: string): Promise<EngineInstalledModel[]> {
    assertRepoId(repo)
    await this.revisions.read()
    const status = this.supervisor.status()
    if (status.model === repo && status.state !== 'stopped' && status.state !== 'failed') throw new Error(`Unload ${repo} before deleting it.`)
    if (this.pulls.has(repo)) throw new Error(`Cancel the download of ${repo} before deleting it.`)
    const installed = await this.installed()
    const drafter = drafterFor(repo)
    const targets = [repo]
    if (drafter && !installed.some((m) => m.repo !== repo && m.drafter === drafter)) targets.push(drafter)
    for (const target of targets) await this.removeRepoFolder(target)
    await this.revisions.remove(targets)
    this.invalidatePreflight(repo)
    this.finished.delete(repo)
    return this.installed(true)
  }

  private async removeRepoFolder(repo: string): Promise<void> {
    const hub = await fs.realpath(this.hub).catch(() => null)
    if (!hub) return
    const folder = repoCacheDir(hub, repo)
    let info
    try { info = await fs.lstat(folder) } catch { return }
    if (info.isSymbolicLink() || !info.isDirectory()) throw new Error(`${folder} is not a cache folder; refusing to delete it.`)
    if (path.dirname(folder) !== hub || !path.basename(folder).startsWith('models--')) throw new Error('Refusing to delete outside the Hugging Face cache.')
    await fs.rm(folder, { recursive: true, force: true })
  }

  async readiness(model: string | null): Promise<Readiness> {
    if (!this.binary()) return { ready: false, reason: 'needs-endpoint', message: 'The TensorFold engine is missing from this build.' }
    const installed = await this.installed().catch(() => [])
    if (!installed.length) return { ready: false, reason: 'needs-model', message: 'Download a model for the built-in TensorFold engine in Models.' }
    const entry = model ? installed.find((m) => m.repo === model) : installed[0]
    if (!entry) return { ready: false, reason: 'needs-model', message: `${model} is not downloaded. Open Models to get it.` }
    if (entry.fitsMemory === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs more memory than this Mac allows. Pick a smaller model in Models.` }
    if (entry.drafter && entry.drafterInstalled === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Download the model again in Models.` }
    return { ready: true, reason: 'ok', message: '' }
  }

  // Quit path, without waiting: SIGTERM at before-quit, SIGKILL (force) when the process exits.
  shutdown(force = true): void {
    for (const { handle } of this.pulls.values()) handle.cancel()
    this.supervisor.killNow(force ? 'SIGKILL' : 'SIGTERM')
    if (force) void this.recordPid(null)
  }

  private pidFile(): string {
    return path.join(this.options.appHome, 'engine', 'serve.json')
  }

  private async recordPid(pid: number | null): Promise<void> {
    try {
      if (pid == null) { await fs.rm(this.pidFile(), { force: true }); return }
      await fs.mkdir(path.dirname(this.pidFile()), { recursive: true, mode: 0o700 })
      await fs.writeFile(this.pidFile(), JSON.stringify({ pid, binary: this.options.binaryPath, owner: process.pid }), { mode: 0o600 })
    } catch { /* bookkeeping only */ }
  }

  // An engine left by a crashed app keeps its model in memory. Stop it only when the app that started
  // it is gone and the recorded pid still runs this exact binary, so a reused pid is never touched.
  private async reapStaleEngine(): Promise<void> {
    try {
      const record = JSON.parse(await fs.readFile(this.pidFile(), 'utf8'))
      const pid = Number(record?.pid)
      const owner = Number(record?.owner)
      if (!Number.isInteger(pid) || pid <= 1 || record?.binary !== this.options.binaryPath) return
      if (Number.isInteger(owner) && owner > 1 && owner !== process.pid && processAlive(owner)) return
      const command = await new Promise<string>((resolve) => execFile('/bin/ps', ['-o', 'command=', '-p', String(pid)], (_e, stdout) => resolve(String(stdout ?? '').trim())))
      if (command.startsWith(`${this.options.binaryPath} serve `)) {
        try { process.kill(-pid, 'SIGTERM') } catch { try { process.kill(pid, 'SIGTERM') } catch { /* gone */ } }
      }
    } catch { /* no record */ }
    await this.recordPid(null)
  }
}
