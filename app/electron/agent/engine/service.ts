import { execFile } from 'node:child_process'
import { promises as fs, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { EngineCapabilities, EngineConnection, EngineEvent, EngineInstalledModel, EngineLearningSnapshot, EngineLibraryScanOptions, EngineModelLibrarySnapshot, EnginePerformance, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness } from '../types'
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
import { LearningCopies, LEARNING_MODEL } from './learning-copy'
import { EngineLearning } from './learning'
import { isLocalModelId, ModelLibrary } from './library'

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
  private readonly library: ModelLibrary
  private readonly localRepos = new Map<string, string>()
  // Whether the bundled engine lists --cortheon among its serve flags; null until asked.
  private cortheonSupported: boolean | null = null
  private learningSupported = false
  private readonly learningCopies: LearningCopies
  private readonly learning: EngineLearning
  private featureChanges: Promise<unknown> = Promise.resolve()
  private quitting = false
  private featurePending = 0
  private learningPurge: Promise<void> | null = null
  private purgingLearning = false

  constructor(private readonly options: EngineServiceOptions) {
    this.env = options.childEnv ?? engineChildEnv(options.env ?? process.env)
    this.totalMemoryBytes = options.totalMemoryBytes ?? os.totalmem()
    this.hubFetch = options.fetch ?? null
    this.revisions = new ModelRevisions(options.appHome, () => this.hub)
    this.preferences = new EnginePreferenceStore(options.appHome)
    this.library = new ModelLibrary({ appHome: options.appHome, env: this.env, capabilities: () => this.capabilitiesDoc(), memoryLimitBytes: () => this.supervisor.memoryLimitBytes })
    this.learningCopies = new LearningCopies(options.appHome)
    this.supervisor = new EngineSupervisor({
      binary: () => this.binary(),
      env: () => this.env,
      info: (repo) => this.info(repo),
      resolveModelArgument: (repo) => this.modelArgument(repo),
      drafterFor: (repo) => drafterFor(this.localRepos.get(repo) ?? repo),
      drafterBytes: (repo) => this.cachedModelBytes(repo),
      totalMemoryBytes: this.totalMemoryBytes,
      memoryBudgetBytes: options.memoryBudgetBytes,
      readyTimeoutMs: options.readyTimeoutMs,
      pollMs: options.pollMs,
      onChange: () => this.emitStatus(),
      onPid: (pid) => { void this.recordPid(pid) },
      cortheon: () => this.cortheonSupported === true && this.preferences.get().cortheon,
      learning: (repo) => {
        const prefs = this.preferences.get()
        return this.learningSupported && repo === prefs.learningModel ? { enabled: prefs.learningEnabled, graphFile: this.learningCopies.graphFile(repo) } : null
      },
    })
    this.learning = new EngineLearning({ appHome: options.appHome, supervisor: this.supervisor, copies: this.learningCopies,
      settings: () => ({ supported: this.learningSupported, enabled: this.preferences.get().learningEnabled, model: this.preferences.get().learningModel }),
      emit: (snapshot) => {
        if (snapshot.job && ['done', 'cancelled', 'interrupted'].includes(snapshot.job.state)) this.installedCache = null
        options.emit({ kind: 'learning', snapshot })
      }, pollMs: options.pollMs,
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
    setEngineServeHook(async (model, options) => { this.assertNotQuitting(); await this.status(); return this.supervisor.acquire(model, options) })
    void this.reapStaleEngine()
  }

  async status(): Promise<EngineStatus> {
    if (!this.versionChecked && this.binary()) {
      this.versionChecked = true
      try { this.version = parseVersion((await runEngineCli(this.requireBinary(), ['--version'], this.env, 10_000)).stdout) } catch { this.version = null }
    }
    if (this.cortheonSupported == null && this.binary()) {
      try {
        const flags = (await this.capabilitiesDoc()).flags
        this.cortheonSupported = flags.includes('--cortheon')
        this.learningSupported = flags.includes('--slide') && flags.includes('--slide-graph') && os.platform() === 'darwin' && os.arch() === 'arm64'
      } catch { this.cortheonSupported = false }
    }
    return this.statusNow()
  }

  // Saves the Cortheon choice and restarts a loaded model so its server picks the flag up.
  setCortheon(enabled: boolean): Promise<EngineStatus> {
    return this.changeFeature(() => this.setCortheonNow(enabled))
  }

  private changeFeature<T>(run: () => Promise<T>): Promise<T> {
    this.featurePending++
    const next = this.featureChanges.then(() => { this.assertNotQuitting(); return run() }).finally(() => { this.featurePending-- })
    this.featureChanges = next.catch(() => {})
    return next
  }

  private async setCortheonNow(enabled: boolean): Promise<EngineStatus> {
    await this.status()
    if (enabled && !this.cortheonSupported) throw new Error("This build's TensorFold engine does not offer Cortheon yet.")
    if (this.learning.busy) throw new Error('Wait for Living Weights to finish before changing Cortheon.')
    if (this.preferences.get().cortheon === enabled) return this.statusNow()
    const prior = this.preferences.get()
    await this.supervisor.reconfigure(async () => { await this.preferences.set({ cortheon: enabled }) }, async () => { await this.preferences.set(prior) })
    this.emitStatus()
    return this.statusNow()
  }

  async learningSnapshot(): Promise<EngineLearningSnapshot> {
    await this.status()
    return this.learning.snapshot()
  }

  setLearning(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    return this.changeFeature(() => this.setLearningNow(enabled, model))
  }

  private async setLearningNow(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    await this.status()
    if (enabled && !this.learningSupported) throw new Error('Living Weights is unavailable in this build or on this hardware.')
    if (this.learning.busy) throw new Error('Wait for the current lesson or model preparation to finish.')
    const prior = this.preferences.get()
    const repo = enabled ? assertRepoId(model ?? this.supervisor.status().model ?? prior.learningModel ?? '') : prior.learningModel
    if (enabled && repo !== LEARNING_MODEL) throw new Error('Living Weights currently supports only Nemotron 3.5 Lightning MLX 4-bit.')
    if (prior.learningEnabled === enabled && prior.learningModel === repo) return this.learning.snapshot()
    const running = this.supervisor.status()
    if (enabled && running.model && running.model !== repo && ['ready', 'starting'].includes(running.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
    try {
      await this.supervisor.reconfigure(async () => {
        if (enabled) {
          const current = this.supervisor.status()
          if (current.model && current.model !== repo && ['ready', 'starting'].includes(current.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
          if ((await this.info(repo!)).memoryFloorBytes > this.supervisor.memoryLimitBytes) throw new Error('This Mac does not have enough model memory for Living Weights on Nemotron. Choose a smaller model for ordinary chat.')
          if (!await this.learningCopies.existing(repo!)) {
            let source = await this.revisions.selected(repo!)
            if (!source) {
              const cache = repoCacheDir(await fs.realpath(this.hub), repo!)
              const revision = (await fs.readFile(path.join(cache, 'refs', 'main'), 'utf8')).trim()
              source = await this.revisions.snapshot(repo!, revision)
            }
            this.learning.preparation(0, 0, 'Preparing an independent private model copy; the original download stays untouched.')
            await this.learningCopies.prepare(repo!, source, repoCacheDir(this.hub, repo!), (copied, total) => this.learning.preparation(copied, total))
          }
        }
        await this.preferences.set({ learningEnabled: enabled, learningModel: repo })
      }, async () => { await this.preferences.set(prior) })
    } catch (error) {
      this.learning.preparation(null, 0, (error as Error).message)
      throw error
    }
    this.learning.preparation(null)
    this.installedCache = null
    this.emitStatus()
    const snapshot = await this.learning.snapshot()
    this.options.emit({ kind: 'learning', snapshot })
    return snapshot
  }

  async teach(text: string): Promise<EngineLearningSnapshot> { this.assertNotQuitting(); await this.status(); this.assertNotQuitting(); return this.learning.teach(text) }
  async cancelTeach(): Promise<EngineLearningSnapshot> { return this.learning.cancel() }

  isLearningBusy(): boolean { return this.learning.busy || this.featurePending > 0 }

  private assertNotQuitting(): void {
    if (this.quitting) throw new Error('TensorFold is waiting for Living Weights to save before quitting.')
    if (this.purgingLearning) throw new Error('TensorFold is pausing local models while learned data is erased.')
  }

  async prepareToQuit(): Promise<void> {
    this.quitting = true
    await this.featureChanges
    await this.learning.settle()
  }

  beginLearningPurge(): Promise<void> {
    // Close admission synchronously, before the caller acquires/awaits its global purge lease.
    this.purgingLearning = true
    this.learning.setPaused(true)
    return this.learningPurge ??= (async () => {
      await this.featureChanges
      await this.learning.settle()
    })()
  }

  async purgeLearningData(): Promise<{ removedPaths: string[]; files: number }> {
    if (!this.purgingLearning) throw new Error('Close local model admission before erasing learned data.')
    await this.learningPurge
    await this.learning.settle()
    await this.supervisor.stop() // Active chats must quiesce too; never force-kill a write or reply.
    const copies = await this.learningCopies.purge()
    const journal = await this.learning.reset()
    await this.preferences.set({ learningEnabled: false, learningModel: null })
    this.installedCache = null
    this.emitStatus()
    this.learning.changed()
    return { removedPaths: [...copies.removedPaths, ...journal.removedPaths], files: copies.files + journal.files }
  }

  endLearningPurge(): void {
    this.purgingLearning = false
    this.learning.setPaused(false)
    this.learningPurge = null
  }

  private statusNow(): EngineStatus {
    const { cortheonActive, learningActive, ...s } = this.supervisor.status()
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
    // Enabling learning restarts the server and rotates the previously shared key. Keep the new
    // learner key inside the main process so external requests cannot bypass its save/quit guards.
    if (this.preferences.get().learningEnabled || this.supervisor.status().learningActive) throw new Error('Local API sharing is paused while Living Weights is enabled. Turn teaching off before copying a connection for another app.')
    return this.supervisor.live()
  }

  hasExternalResearchEnabled(): boolean { return this.preferences.get().cortheon || this.supervisor.status().cortheonActive }

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
    if (isLocalModelId(repo)) {
      const local = await this.library.resolve(repo)
      if (local.repo) this.localRepos.set(repo, local.repo)
      return local.path
    }
    if (this.pulls.has(repo) || [...this.pulls.keys()].some((model) => drafterFor(model) === repo)) throw new Error('Wait for the model download to finish before loading it.')
    if (repo === this.preferences.get().learningModel) {
      const owned = await this.learningCopies.existing(repo)
      if (!owned) throw new Error('The private Living Weights model is missing; preserve its profile before repairing it.')
      return owned
    }
    const selected = await this.revisions.selected(repo)
    if (selected) return selected
    if (!await hasCachedSnapshot(this.hub, repo)) {
      const local = await this.localDrafter(repo)
      if (local) return local.path
    }
    return repo
  }

  private async cachedModelBytes(repo: string): Promise<number | null> {
    try {
      const selected = await this.revisions.selected(repo)
      if (selected) return this.revisions.snapshotBytes(selected)
      if (await hasCachedSnapshot(this.hub, repo)) return repoCachedBytes(this.hub, repo)
      return (await this.localDrafter(repo))?.weightsBytes ?? null
    } catch { return null }
  }

  private async localDrafter(repo: string) {
    // Resolve only a known companion identity, never an arbitrary repo against a nearby folder.
    if (!SUGGESTED_MODELS.some((model) => drafterFor(model.repo) === repo)) return null
    const entries = (await this.library.snapshot()).models.filter((model) => model.repo === repo && model.drafterOnly && model.complete)
    if (entries.length !== 1) return null
    return this.library.resolve(entries[0].id, true)
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
      const entry: EngineInstalledModel = { ...model, source: selected[model.repo] ? 'download' : 'cache', format: null, contextTokens: null, weightsBytes: null, memoryFloorBytes: null, fitsMemory: null, drafter, drafterInstalled: drafter ? drafterBytes != null : null }
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
    for (const local of (await this.library.snapshot()).models.filter((model) => model.loadable)) {
      try {
        const checked = await this.library.resolve(local.id)
        const drafter = checked.repo ? drafterFor(checked.repo) : null
        const drafterBytes = drafter ? await this.cachedModelBytes(drafter) : null
        const memoryFloorBytes = checked.memoryFloorBytes == null ? null : checked.memoryFloorBytes + (drafterBytes ?? 0)
        if (checked.repo) this.localRepos.set(local.id, checked.repo)
        out.push({ repo: local.id, source: 'library', title: local.title, localPath: checked.path, family: FAMILY_TITLES[checked.modelType!] ?? checked.modelType!, modelType: checked.modelType!, format: checked.format, contextTokens: null, weightsBytes: checked.weightsBytes, memoryFloorBytes, fitsMemory: memoryFloorBytes == null ? null : memoryFloorBytes <= this.supervisor.memoryLimitBytes, drafter, drafterInstalled: drafter ? drafterBytes != null : null })
      } catch { /* Moved or changed checkpoints are unavailable until scanned again. */ }
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
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place. Scan their folder instead of downloading them.')
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
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place and cannot be downloaded by their library ID.')
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
    this.assertNotQuitting()
    await this.status() // Restore managed flags before starting a model after an app restart.
    if (isLocalModelId(repo)) await this.library.resolve(repo)
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
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints are read-only. Remove their search folder to unlink them; their files stay untouched.')
    const selected = await this.revisions.read()
    if (!selected[repo]) throw new Error('This checkpoint belongs to the shared cache, rather than an app-managed download. TensorFold will not delete it.')
    const status = this.supervisor.status()
    if (status.model === repo && status.state !== 'stopped' && status.state !== 'failed') throw new Error(`Unload ${repo} before deleting it.`)
    if (this.pulls.has(repo)) throw new Error(`Cancel the download of ${repo} before deleting it.`)
    const installed = await this.installed()
    const drafter = drafterFor(repo)
    const targets = [repo]
    if (drafter && selected[drafter] && !installed.some((m) => m.repo !== repo && m.drafter === drafter)) targets.push(drafter)
    for (const target of targets) await this.removeRepoFolder(target, selected[target])
    await this.revisions.remove(targets)
    this.invalidatePreflight(repo)
    this.finished.delete(repo)
    return this.installed(true)
  }

  private async removeRepoFolder(repo: string, revision: string): Promise<void> {
    const hub = await fs.realpath(this.hub).catch(() => null)
    if (!hub) return
    const folder = repoCacheDir(hub, repo)
    let info
    try { info = await fs.lstat(folder) } catch { return }
    if (info.isSymbolicLink() || !info.isDirectory()) throw new Error(`${folder} is not a cache folder; refusing to delete it.`)
    if (path.dirname(folder) !== hub || !path.basename(folder).startsWith('models--')) throw new Error('Refusing to delete outside the Hugging Face cache.')
    const snapshots = path.join(folder, 'snapshots')
    if (await fs.realpath(snapshots).catch(() => null) !== snapshots) throw new Error('The model snapshots folder was redirected; refusing to delete it.')
    const entries = await fs.readdir(snapshots)
    if (entries.some((name) => name !== revision)) {
      // Other HF clients may keep different revisions here. Preserve those revisions, shared
      // blobs and refs; only unlink our exact selected checkpoint, without following its links.
      const selected = path.join(snapshots, revision)
      const stat = await fs.lstat(selected).catch(() => null)
      if (stat?.isSymbolicLink() || stat && !stat.isDirectory()) throw new Error('The selected checkpoint was redirected; refusing to delete it.')
      if (stat) await fs.rm(selected, { recursive: true })
      return
    }
    await fs.rm(folder, { recursive: true, force: true })
  }

  async readiness(model: string | null): Promise<Readiness> {
    if (!this.binary()) return { ready: false, reason: 'needs-endpoint', message: 'The TensorFold engine is missing from this build.' }
    const installed = await this.installed().catch(() => [])
    if (!installed.length) return { ready: false, reason: 'needs-model', message: 'Download a model for the built-in TensorFold engine in Models.' }
    const entry = model ? installed.find((m) => m.repo === model) : installed[0]
    if (!entry) return { ready: false, reason: 'needs-model', message: `${model} is not downloaded. Open Models to get it.` }
    if (entry.source === 'library') {
      try { await this.library.resolve(entry.repo) } catch (error) { return { ready: false, reason: 'needs-model', message: (error as Error).message } }
      if (entry.drafter && await this.cachedModelBytes(entry.drafter) == null) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Scan the companion checkpoint again in Models.` }
    }
    if (entry.fitsMemory === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs more memory than this Mac allows. Pick a smaller model in Models.` }
    if (entry.drafter && entry.drafterInstalled === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Download the model again in Models.` }
    return { ready: true, reason: 'ok', message: '' }
  }

  async librarySnapshot(): Promise<EngineModelLibrarySnapshot> {
    const snapshot = await this.library.snapshot()
    for (const model of snapshot.models) {
      const drafter = model.repo ? drafterFor(model.repo) : null
      if (!drafter || !model.loadable) continue
      const bytes = await this.cachedModelBytes(drafter)
      if (bytes == null) { model.loadable = false; model.reason = `This checkpoint needs its drafter ${drafter}. Download the full model in Models or scan the companion checkpoint too.` }
      else if (model.memoryFloorBytes != null) {
        model.memoryFloorBytes += bytes
        model.fitsMemory = model.memoryFloorBytes <= this.supervisor.memoryLimitBytes
        if (!model.fitsMemory) { model.loadable = false; model.reason = 'This model and its drafter need more model memory than the current hardware allows.' }
      }
    }
    return snapshot
  }

  private async libraryChanged(_snapshot: EngineModelLibrarySnapshot): Promise<EngineModelLibrarySnapshot> {
    this.installedCache = null
    await this.installed(true)
    return this.librarySnapshot()
  }

  async scanLibrary(options: EngineLibraryScanOptions = {}): Promise<EngineModelLibrarySnapshot> {
    this.assertNotQuitting()
    return this.libraryChanged(await this.library.scan(options))
  }

  async addLibraryFolder(directory: string): Promise<EngineModelLibrarySnapshot> {
    this.assertNotQuitting()
    return this.libraryChanged(await this.library.addFolder(directory))
  }

  async removeLibraryFolder(id: string): Promise<EngineModelLibrarySnapshot> {
    this.assertNotQuitting()
    // Unlinking a search folder never stops a loaded API or modifies its checkpoint files.
    return this.libraryChanged(await this.library.removeFolder(id))
  }

  async cancelLibraryScan(): Promise<EngineModelLibrarySnapshot> { return this.library.cancel() }

  // Quit path, without waiting: SIGTERM at before-quit, SIGKILL (force) when the process exits.
  shutdown(force = true): void {
    if (!force && this.learning.busy) {
      void this.prepareToQuit().then(() => this.shutdown(false)).catch(() => {})
      return
    }
    this.learning.shutdown()
    void this.library.cancel().catch(() => {})
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
