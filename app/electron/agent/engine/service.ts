import { execFile } from 'node:child_process'
import { promises as fs, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import type { EngineCapabilities, EngineConnection, EngineEvent, EngineInstalledModel, EngineLearningSnapshot, EngineLibraryScanOptions, EngineModelLibrarySnapshot, EnginePerformance, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness, EngineModelGenerationSettings, EngineModelPreviewResult, EngineModelComparisonInput, EngineModelComparisonSnapshot } from '../types'
import { FAMILY_TITLES, SUGGESTED_MODELS, drafterFor } from './catalog'
import { parseCapabilities, parseInfoOutput, parseModelsOutput, parseVersion, runEngineCli, type ModelInfo } from './cli'
import { ModelHub } from './model-hub'
import { engineLaunchEnv, pinnedEngineEnv } from './engine-env'
import { normalizeEngineEnvOverrides, type EngineEnvOverrides } from '../../../shared/engine-advanced-env'
export { engineChildEnv } from './engine-env'
import { assertRepoId, hasCachedSnapshot, hubCacheDir, repoCacheDir, repoCachedBytes, type EngineBinary } from './paths'
import { setEnginePerformanceRecorder, setEngineServeHook } from './provider'
import { createTurnStats, finalizeTurnStats, readEngineTimings, recordModelCall, type ModelCallSample } from '../turn-stats'
import { recordEngineContext } from './context'
import { PullQueue, type PullHandle } from './pull'
import { EngineSupervisor } from './supervisor'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'
import { LearningCopies, LEARNING_MODEL } from './learning-copy'
import { EngineLearning } from './learning'
import { isLocalModelId, ModelLibrary } from './library'
import { assertCortheonReleaseAllowed, CORTHEON_RELEASE_POLICY, cortheonEnabledForRelease } from '../../../shared/release-feature-policy'
import { measureModelPreview } from './model-preview'
import { ModelComparisons } from './model-comparison'
import { modelLaunchOptions } from './launch-options'
import type { ModelBenchmarkInput, ModelBenchmarkReceipt, ModelBenchmarkSnapshot } from '../../../shared/model-benchmark'
import { ModelBenchmarks } from './benchmark/controller'
import { createModelBenchmarkStore, benchmarkWorkspace, readBenchmarkReceiptFile, type ModelBenchmarkStore } from './benchmark/storage'
import { startBenchmarkProcess } from './benchmark/process'
import { assertBundledBenchmarkVersion } from './benchmark/version'

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
  assertComparisonAllowed?: () => void
  modelSettings?: (repo: string) => EngineModelGenerationSettings | null
  benchmarkStore?: ModelBenchmarkStore
}

function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM' }
}

export class EngineService {
  readonly supervisor: EngineSupervisor
  private readonly launch: { env: NodeJS.ProcessEnv; endpoint: string }
  // Advanced user variables applied over the launch environment; null until preferences load.
  private envOverridesApplied: EngineEnvOverrides | null = null
  private readonly totalMemoryBytes: number
  private version: string | null = null
  private versionChecked = false
  private capabilities: EngineCapabilities | null = null
  private performance: EnginePerformance | null = null
  private installedCache: EngineInstalledModel[] | null = null
  private readonly pulls = new Map<string, { handle: PullHandle; last: PullProgress }>()
  private readonly finished = new Map<string, PullProgress>()
  private readonly pullQueue = new PullQueue()
  private deletingModel = false
  private preview: { model: string; controller: AbortController; profile: EngineModelGenerationSettings | null; promise: Promise<EngineModelPreviewResult> } | null = null
  private readonly comparisons: ModelComparisons
  private readonly benchmarks: ModelBenchmarks
  private benchmarkAdmission: (() => void) | null = null
  private sharedEndpointKey: string | null = null
  private comparisonAdmission: (() => void) | null = null
  private comparisonRequest: { owner: symbol; model: string; profile: EngineModelGenerationSettings | null } | null = null
  private readonly lastPullEmit = new Map<string, number>()
  private readonly models: ModelHub
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

  // A supplied childEnv stays live for the fake engine's knobs; the authorized Hub origin stays pinned either way.
  // Saved advanced variables layer on top; the running server keeps its launch-time values.
  private get env(): NodeJS.ProcessEnv {
    const base = this.options.childEnv ? pinnedEngineEnv(this.options.childEnv, this.launch.endpoint) : this.launch.env
    return this.envOverridesApplied ? { ...base, ...this.envOverridesApplied } : base
  }

  constructor(private readonly options: EngineServiceOptions) {
    this.launch = engineLaunchEnv(options)
    this.totalMemoryBytes = options.totalMemoryBytes ?? os.totalmem()
    // Preferences load first so a saved HF_HUB_OFFLINE or cache location governs every dependent.
    this.preferences = new EnginePreferenceStore(options.appHome)
    this.envOverridesApplied = this.preferences.get().envOverrides
    this.models = new ModelHub({
      endpoint: this.launch.endpoint, fetch: options.fetch, env: () => this.env, totalMemoryBytes: this.totalMemoryBytes,
      capabilities: () => this.capabilitiesDoc(), installed: () => this.installed(), cacheDir: () => this.hub,
      memory: () => ({ limit: this.supervisor.memoryLimitBytes, budget: this.supervisor.memoryBudgetBytes, limitFor: (modelType) => this.supervisor.memoryPolicyFor(modelType) }),
    })
    this.revisions = new ModelRevisions(options.appHome, () => this.hub)
    this.library = new ModelLibrary({ appHome: options.appHome, env: () => this.env, capabilities: () => this.capabilitiesDoc(), memoryLimitBytes: () => this.supervisor.memoryLimitBytes })
    this.learningCopies = new LearningCopies(options.appHome)
    this.supervisor = new EngineSupervisor({
      binary: () => this.binary(),
      env: () => this.env,
      info: (repo) => this.info(repo),
      resolveModelArgument: (repo) => this.modelArgument(repo),
      drafterFor: (repo) => this.requiredDrafter(repo),
      drafterBytes: (repo) => this.cachedModelBytes(repo),
      modelSettings: (repo) => this.options.modelSettings?.(repo) ?? null,
      totalMemoryBytes: this.totalMemoryBytes,
      memoryBudgetBytes: options.memoryBudgetBytes,
      readyTimeoutMs: options.readyTimeoutMs,
      pollMs: options.pollMs,
      onChange: () => this.emitStatus(),
      onPid: (pid) => { void this.recordPid(pid) },
      cortheon: () => this.cortheonSupported === true && cortheonEnabledForRelease(this.preferences.get().cortheon),
      learning: (repo) => {
        const prefs = this.preferences.get()
        return this.learningSupported && repo === prefs.learningModel ? { enabled: prefs.learningEnabled, graphFile: this.learningCopies.graphFile(repo) } : null
      },
    })
    this.comparisons = new ModelComparisons({
      assertAllowed: () => this.assertComparisonAllowed(),
      reserve: () => this.supervisor.reserveSequence(),
      validateModels: async (models) => {
        const installed = await this.installed(true)
        for (const model of models) {
          const entry = installed.find((item) => item.repo === model)
          if (!entry) throw new Error('Choose models that are installed in your local library.')
          if (entry.fitsMemory !== true) throw new Error(`${model} has not been confirmed to fit this Mac's model memory budget.`)
          // info() checks the actual admitted checkpoint and registered engine family.
          await this.info(model)
        }
      },
      measure: async (model, prompt, signal, owner) => {
        const request = { owner, model, profile: null as EngineModelGenerationSettings | null }
        this.comparisonRequest = request
        try { return await measureModelPreview(model, signal, () => request.profile, undefined, { prompt, owner, assertAllowed: () => this.assertComparisonAllowed() }) }
        finally { if (this.comparisonRequest === request) this.comparisonRequest = null }
      },
    })
    this.learning = new EngineLearning({ appHome: options.appHome, supervisor: this.supervisor, copies: this.learningCopies,
      settings: () => ({ supported: this.learningSupported, enabled: this.preferences.get().learningEnabled, model: this.preferences.get().learningModel }),
      emit: (snapshot) => {
        if (snapshot.job && ['done', 'cancelled', 'interrupted'].includes(snapshot.job.state)) this.installedCache = null
        options.emit({ kind: 'learning', snapshot })
      }, pollMs: options.pollMs,
    })
    this.benchmarks = new ModelBenchmarks({
      assertAllowed: () => this.assertBenchmarkAllowed(),
      validateModel: async (model) => {
        const installed = await this.installed(true)
        const entry = installed.find(item => item.repo === model)
        if (!entry || entry.fitsMemory !== true || entry.source === 'library' || isLocalModelId(model) || !(await hasCachedSnapshot(this.hub, model))) throw new Error('Choose a cached installed checkpoint confirmed to fit this Mac.')
        const revision = (await fs.readFile(path.join(repoCacheDir(this.hub, model), 'refs', 'main'), 'utf8')).trim()
        const snapshot = await this.revisions.snapshot(model, revision)
        const argument = await this.modelArgument(model)
        if (argument !== model && await fs.realpath(argument) !== snapshot) throw new Error('The selected checkpoint differs from the cached public model.')
        const status = await this.status()
        const engineVersion = assertBundledBenchmarkVersion(status.version)
        await this.info(model)
        return { revision, configSha256: createHash('sha256').update(await fs.readFile(path.join(snapshot, 'config.json'))).digest('hex'), engineVersion }
      },
      reserve: () => this.supervisor.reserveSequence('local benchmark'),
      workspace: () => benchmarkWorkspace(options.appHome),
      process: (input, workspace, progress, signal) => startBenchmarkProcess({ binary: this.requireBinary(), input, ...workspace, workingDirectory: workspace.directory, env: this.env, progress, signal }),
      readReceipt: (file, input) => readBenchmarkReceiptFile(file, input),
      store: options.benchmarkStore ?? createModelBenchmarkStore(options.appHome),
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
    setEngineServeHook(async (model, options) => {
      this.assertModelAdmission()
      const comparison = this.comparisonRequest
      if (this.comparisons.busy && (!comparison || comparison.owner !== options.owner || comparison.model !== model)) throw new Error('Finish or cancel the model comparison before using the local engine.')
      if (options.owner && (!this.comparisons.busy || comparison?.owner !== options.owner || comparison.model !== model)) throw new Error('This comparison no longer owns the local engine.')
      if (options.owner) this.assertComparisonAllowed()
      await this.status(); this.assertModelAdmission()
      if (options.owner) this.assertComparisonAllowed()
      const lease = await this.supervisor.acquire(model, options)
      try {
        const saved = this.options.modelSettings?.(model) ?? null
        const generation = options.owner && comparison ? { ...(saved ?? { contextTokens: this.supervisor.status().contextTokens ?? 4096, topP: 1, topK: 0, minP: 0, speculativeDecoding: 'off' as const }), temperature: 0, maxOutputTokens: 256 } : saved
        if (options.owner && comparison) comparison.profile = generation
        if (options.exclusive && this.preview?.model === model) this.preview.profile = generation
        return { ...lease, generation: generation ?? undefined }
      } catch (error) { lease.release(); throw error }
    })
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

  // Refuse paused features before probes, mutations or queued engine work.
  // Disabling remains available to clear a previously active engine safely.
  setCortheon(enabled: boolean): Promise<EngineStatus> {
    try { assertCortheonReleaseAllowed(enabled) } catch (error) { return Promise.reject(error) }
    return this.changeFeature(() => this.setCortheonNow(enabled))
  }

  // Advanced engine environment: validated, persisted in engine preferences, and handed to
  // every engine process at its next launch; a running server keeps its launch-time values.
  envOverrides(): EngineEnvOverrides {
    return this.preferences.get().envOverrides
  }

  async setEnvOverrides(next: EngineEnvOverrides): Promise<EngineEnvOverrides> {
    const normalized = normalizeEngineEnvOverrides(next)
    const prior = this.preferences.get().envOverrides
    // A cache-location move must never race an operation bound to the current root.
    const cacheMoved = normalized.HF_HOME !== prior.HF_HOME || normalized.HF_HUB_CACHE !== prior.HF_HUB_CACHE
    if (cacheMoved && (this.pulls.size || this.pullQueue.busy || this.deletingModel)) throw new Error('Finish model downloads and cache cleanup before changing the model cache location.')
    await this.preferences.set({ envOverrides: normalized })
    this.envOverridesApplied = normalized
    // Verdicts cached at the old limit are stale the moment the limit moves; re-answer them all.
    this.models.clearPreflights()
    await this.library.refreshEligibility((modelType) => this.supervisor.memoryLimitFor(modelType))
    this.installedCache = null
    this.emitStatus()
    await this.installed(true)
    return normalized
  }

  private changeFeature<T>(run: () => Promise<T>): Promise<T> {
    if (this.benchmarks?.busy) return Promise.reject(new Error('Finish or cancel the local benchmark before changing engine settings.'))
    if (this.comparisons.busy) return Promise.reject(new Error('Finish or cancel the model comparison before changing local engine settings.'))
    this.featurePending++
    const next = this.featureChanges.then(() => { this.assertNotQuitting(); return run() }).finally(() => { this.featurePending-- })
    this.featureChanges = next.catch(() => {})
    return next
  }

  private async setCortheonNow(enabled: boolean): Promise<EngineStatus> {
    assertCortheonReleaseAllowed(enabled)
    await this.status()
    if (enabled && !this.cortheonSupported) throw new Error("This build's TensorFold engine does not offer Cortheon yet.")
    if (this.learning.busy) throw new Error('Wait for Living Weights to finish before changing Cortheon.')
    if (this.preferences.get().cortheon === enabled && (enabled || !this.supervisor.status().cortheonActive)) {
      // An explicit Off also clears an older on-disk true without restarting the model.
      if (!enabled) await this.preferences.set({ cortheon: false })
      return this.statusNow()
    }
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
          const learningInfo = await this.info(repo!)
          if (learningInfo.memoryFloorBytes > this.supervisor.memoryLimitFor(learningInfo.modelType)) throw new Error('This Mac does not have enough model memory for Living Weights on Nemotron. Choose a smaller model for ordinary chat.')
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

  async teach(text: string): Promise<EngineLearningSnapshot> { if (this.benchmarks.busy) throw new Error('Finish or cancel the local benchmark before teaching.'); if (this.comparisons.busy) throw new Error('Finish or cancel the model comparison before teaching.'); this.assertNotQuitting(); await this.status(); this.assertNotQuitting(); return this.learning.teach(text) }
  async cancelTeach(): Promise<EngineLearningSnapshot> { return this.learning.cancel() }

  isLearningBusy(): boolean { return this.learning.busy || this.featurePending > 0 || this.preview != null || this.comparisons.busy || this.benchmarks.busy }

  applyModelSettings(model: string, apply: () => Promise<void>, rollback: () => Promise<void>): Promise<void> {
    this.assertModelAdmission()
    assertRepoId(model)
    if (this.learning.busy || this.preview) return Promise.reject(new Error('Finish the current lesson or model preview before applying model settings.'))
    return this.changeFeature(async () => {
      try {
        await this.supervisor.configureModel(model, async () => {
          await apply()
          const settings = this.options.modelSettings?.(model)
          if (settings) modelLaunchOptions(settings, (await this.info(model)).contextTokens)
        }, rollback)
      } finally { this.installedCache = null; this.emitStatus() }
    })
  }

  tryModel(model: string): Promise<EngineModelPreviewResult> {
    this.assertModelAdmission()
    assertRepoId(model)
    if (this.isLearningBusy()) return Promise.reject(new Error('Finish the current lesson, settings change or preview before measuring another model.'))
    if (this.hasExternalResearchEnabled()) return Promise.reject(new Error('Turn external research off before running a local model preview.'))
    const controller = new AbortController()
    const job = { model, controller, profile: null as EngineModelGenerationSettings | null, promise: Promise.resolve(null as unknown as EngineModelPreviewResult) }
    this.preview = job
    const timeout = setTimeout(() => controller.abort(new Error('The model preview timed out. Try a smaller model or context window.')), 180_000)
    job.promise = measureModelPreview(model, controller.signal, () => job.profile).finally(() => { clearTimeout(timeout); if (this.preview === job) this.preview = null })
    return job.promise
  }

  assertLocalWorkAdmission(): void {
    this.assertModelAdmission()
    if (this.comparisons.busy) throw new Error('Finish or cancel the model comparison before starting local model work.')
  }
  setComparisonAdmission(assertAllowed: (() => void) | null): void { this.comparisonAdmission = assertAllowed }
  compareModels(input: EngineModelComparisonInput): Promise<EngineModelComparisonSnapshot> { return this.comparisons.start(input) }
  comparisonStatus(id: string): EngineModelComparisonSnapshot | null { return this.comparisons.snapshot(id) }
  cancelComparison(id: string): boolean { return this.comparisons.cancel(id) }
  setBenchmarkAdmission(assertAllowed: (() => void) | null): void { this.benchmarkAdmission = assertAllowed }
  startBenchmark(input: ModelBenchmarkInput): Promise<ModelBenchmarkSnapshot> { return this.benchmarks.start(input) }
  benchmarkStatus(id?: string): Promise<ModelBenchmarkSnapshot | null> { return this.benchmarks.snapshot(id) }
  benchmarkHistory(): Promise<ModelBenchmarkSnapshot[]> { return this.benchmarks.history() }
  cancelBenchmark(id: string): boolean { return this.benchmarks.cancel(id) }
  benchmarkReceiptForExport(id: string): Promise<ModelBenchmarkReceipt> { return this.benchmarks.receiptForExport(id) }
  settleBenchmarks(id?: string): Promise<void> { return this.benchmarks.settle(id) }
  private assertBenchmarkAllowed(): void {
    this.assertNotQuitting()
    this.benchmarkAdmission?.()
    if (this.deletingModel || this.pullQueue.busy || this.learning.busy || this.featurePending || this.preview || this.comparisons.busy || this.supervisor.status().learningActive || this.preferences.get().learningEnabled) throw new Error('Finish local work, downloads and model settings changes, and turn Living Weights off before benchmarking.')
    if (this.hasExternalResearchEnabled()) throw new Error('Turn external research off before benchmarking local models.')
    if (this.sharedEndpointKey && this.supervisor.live()?.apiKey === this.sharedEndpointKey) throw new Error('Unload the shared API session before benchmarking; external clients may still be using it.')
  }
  private assertComparisonAllowed(): void {
    this.assertModelAdmission()
    this.options.assertComparisonAllowed?.()
    this.comparisonAdmission?.()
    if (this.learning.busy || this.featurePending || this.preview || this.supervisor.status().learningActive || this.preferences.get().learningEnabled) throw new Error('Finish local work and turn Living Weights off before comparing models.')
    if (this.hasExternalResearchEnabled()) throw new Error('Turn external research off before comparing local models.')
    if (this.sharedEndpointKey && this.supervisor.live()?.apiKey === this.sharedEndpointKey) throw new Error('Unload the model to close its shared API session before comparing models; external clients may still be using it.')
  }

  cancelTryModel(): boolean {
    if (!this.preview) return false
    this.preview.controller.abort(new Error('Model preview cancelled.'))
    return true
  }

  private assertNotQuitting(): void {
    if (this.quitting) throw new Error('TensorFold is waiting for Living Weights to save before quitting.')
    if (this.purgingLearning) throw new Error('TensorFold is pausing local models while learned data is erased.')
  }

  private assertModelAdmission(): void {
    this.assertNotQuitting()
    if (this.benchmarks?.busy) throw new Error('Finish or cancel the local benchmark before changing or using the local engine.')
    if (this.deletingModel) throw new Error('Wait for model cache cleanup before loading or using a model.')
  }

  async prepareToQuit(): Promise<void> {
    this.quitting = true
    this.benchmarks.dispose(false)
    await this.benchmarks.settle()
    const comparisonId = this.comparisons.currentId
    if (comparisonId) this.cancelComparison(comparisonId)
    await this.comparisons.settle()
    this.comparisons.clear()
    const preview = this.preview
    this.cancelTryModel()
    await preview?.promise.catch(() => {})
    await this.featureChanges
    await this.learning.settle()
  }

  beginLearningPurge(): Promise<void> {
    // Close admission synchronously, before the caller acquires/awaits its global purge lease.
    this.purgingLearning = true
    const benchmark = this.benchmarks.currentId
    if (benchmark) this.benchmarks.cancel(benchmark)
    const preview = this.preview
    this.cancelTryModel() // Close the owned preview request before waiting on global purge leases.
    this.options.emit({ kind: 'comparison-cleared' })
    this.learning.setPaused(true)
    const comparisonId = this.comparisons.currentId
    if (comparisonId) this.cancelComparison(comparisonId)
    return this.learningPurge ??= (async () => {
      await preview?.promise.catch(() => {})
      await this.comparisons.settle()
      await this.benchmarks.settle()
      this.comparisons.clear()
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
      ...s, cortheon: { supported: this.cortheonSupported === true, enabled: this.preferences.get().cortheon, active: cortheonActive, disabledReason: CORTHEON_RELEASE_POLICY.enabled ? undefined : CORTHEON_RELEASE_POLICY.reason }, available: !!binary, version: this.version, binaryPath: binary ? this.options.binaryPath : null, totalMemoryBytes: this.totalMemoryBytes, memoryBudgetBytes: this.supervisor.memoryBudgetBytes,
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
    if (this.comparisons.busy || this.benchmarks.busy) throw new Error('API sharing is paused during local model comparisons and benchmarks.')
    // Enabling learning restarts the server and rotates the previously shared key. Keep the new
    // learner key inside the main process so external requests cannot bypass its save/quit guards.
    if (this.preferences.get().learningEnabled || this.supervisor.status().learningActive) throw new Error('Local API sharing is paused while Living Weights is enabled. Turn teaching off before copying a connection for another app.')
    const live = this.supervisor.live()
    if (live) this.sharedEndpointKey = live.apiKey
    return live
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
      if (local.repo && (this.pullQueue.hasRepo(local.repo) || this.pullQueue.hasRepo(this.requiredDrafter(repo, local.repo) ?? ''))) throw new Error('Wait for the model download to finish before loading it.')
      if (local.repo) this.localRepos.set(repo, local.repo)
      return local.path
    }
    if (this.pullQueue.hasRepo(repo)) throw new Error('Wait for the model download to finish before loading it.')
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

  private requiredDrafter(model: string, canonical = this.localRepos.get(model) ?? model): string | null {
    return this.options.modelSettings?.(model)?.speculativeDecoding === 'off' ? null : drafterFor(canonical)
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
      const drafter = this.requiredDrafter(model.repo)
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
        entry.fitsMemory = entry.memoryFloorBytes <= this.supervisor.memoryLimitFor(info.modelType)
      } catch {
        if (selected[model.repo]) continue // Never present a different legacy ref as the selected model.
        /* Legacy listed but unreadable: keep the row without sizes. */
      }
      out.push(entry)
    }
    for (const local of (await this.library.snapshot()).models.filter((model) => model.loadable)) {
      try {
        const checked = await this.library.resolve(local.id)
        const drafter = checked.repo ? this.requiredDrafter(local.id, checked.repo) : null
        const drafterBytes = drafter ? await this.cachedModelBytes(drafter) : null
        const memoryFloorBytes = checked.memoryFloorBytes == null ? null : checked.memoryFloorBytes + (drafterBytes ?? 0)
        if (checked.repo) this.localRepos.set(local.id, checked.repo)
        out.push({ repo: local.id, source: 'library', title: local.title, localPath: checked.path, family: FAMILY_TITLES[checked.modelType!] ?? checked.modelType!, modelType: checked.modelType!, format: checked.format, contextTokens: null, weightsBytes: checked.weightsBytes, memoryFloorBytes, fitsMemory: memoryFloorBytes == null ? null : memoryFloorBytes <= this.supervisor.memoryLimitFor(checked.modelType), drafter, drafterInstalled: drafter ? drafterBytes != null : null })
      } catch { /* Moved or changed checkpoints are unavailable until scanned again. */ }
    }
    this.installedCache = out
    this.options.emit({ kind: 'installed', installed: out })
    this.options.onInstalled?.(out.map((m) => m.repo))
    return out
  }

  search(query: string): Promise<HubModelSummary[]> { return this.models.search(query) }

  preflight(repo: string, revision: string | null = null): Promise<ModelPreflight> { return this.models.preflight(repo, revision) }

  pullList(): PullProgress[] {
    const running = [...this.pulls.values()].map((p) => p.last)
    return [...running, ...[...this.finished.values()].filter((p) => !this.pulls.has(p.repo))]
  }

  async pull(repo: string, revision: string | null = null): Promise<PullProgress> {
    this.assertModelAdmission()
    if (this.deletingModel) throw new Error('Wait for model cache cleanup before starting a download.')
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place and cannot be downloaded by their library ID.')
    if (this.pulls.has(assertRepoId(repo))) return this.pulls.get(repo)!.last
    await this.revisions.read() // Preserve invalid registry bytes and refuse before starting a child.
    this.assertPullIdle(repo)
    const pre = this.models.cached(JSON.stringify([repo, revision])) ?? await this.preflight(repo, revision)
    this.assertModelAdmission()
    if (this.deletingModel) throw new Error('Wait for model cache cleanup before starting a download.')
    this.assertPullIdle(repo)
    if (this.pulls.has(repo)) return this.pulls.get(repo)!.last
    if (!['ok', 'installed'].includes(pre.verdict)) throw new Error(pre.reasons[0] ?? `${repo} cannot be downloaded for the built-in engine.`)
    if (!pre.resolvedRevision || (pre.drafter && !pre.drafter.resolvedRevision)) throw new Error('Retry the model checks to resolve an immutable checkpoint before downloading.')
    const steps = [{ repo, revision: pre.resolvedRevision }, ...(pre.drafter ? [{ repo: pre.drafter.repo, revision: pre.drafter.resolvedRevision! }] : [])]
    for (const step of steps) this.assertPullIdle(step.repo)
    this.finished.delete(repo)
    const startedHub = this.hub
    const handle = this.pullQueue.enqueue({
      binary: this.requireBinary(),
      env: this.env,
      hub: this.hub,
      steps,
      totalBytes: pre.totalBytes,
      pollMs: this.options.pollMs ? Math.max(50, this.options.pollMs) : 1000,
      onUpdate: (progress) => this.onPullUpdate(progress),
    }, async (final) => { await this.revisions.commit(steps, startedHub); return final })
    this.pulls.set(repo, { handle, last: handle.progress() })
    this.options.emit({ kind: 'pull', pull: handle.progress() })
    void handle.done.then(async (final) => {
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
    if (!['queued', 'running', 'verifying'].includes(progress.state)) return // Completion follows durable selection.
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
    this.models.invalidate(repo)
  }

  cancelPull(repo: string): boolean {
    const job = this.pulls.get(assertRepoId(repo))
    if (!job) return false
    job.handle.cancel()
    return true
  }

  async load(repo: string): Promise<EngineStatus> {
    this.assertModelAdmission()
    await this.status() // Restore managed flags before starting a model after an app restart.
    if (isLocalModelId(repo)) await this.library.resolve(repo)
    this.assertModelAdmission()
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
    this.assertModelAdmission()
    if (this.deletingModel || this.pullQueue.busy) throw new Error('Finish or cancel model downloads before cleaning the shared cache.')
    this.deletingModel = true
    try { return await this.supervisor.mutateCache(() => this.removeModelNow(repo)) } finally { this.deletingModel = false }
  }

  private async removeModelNow(repo: string): Promise<EngineInstalledModel[]> {
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints are read-only. Remove their search folder to unlink them; their files stay untouched.')
    const selected = await this.revisions.read()
    if (!selected[repo]) throw new Error('This checkpoint belongs to the shared cache, rather than an app-managed download. TensorFold will not delete it.')
    const status = this.supervisor.status()
    if ((status.model === repo || this.localRepos.get(status.model ?? '') === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error(`Unload ${repo} before deleting it.`)
    if (this.pulls.has(repo)) throw new Error(`Cancel the download of ${repo} before deleting it.`)
    const installed = await this.installed()
    const drafter = drafterFor(repo)
    const targets = [repo]
    if (drafter && selected[drafter] && !installed.some((m) => m.repo !== repo && drafterFor(this.localRepos.get(m.repo) ?? m.repo) === drafter)) targets.push(drafter)
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
      const drafter = model.repo ? this.requiredDrafter(model.id, model.repo) : null
      if (!drafter || !model.loadable) continue
      const bytes = await this.cachedModelBytes(drafter)
      if (bytes == null) { model.loadable = false; model.reason = `This checkpoint needs its drafter ${drafter}. Download the full model in Models or scan the companion checkpoint too.` }
      else if (model.memoryFloorBytes != null) {
        model.memoryFloorBytes += bytes
        model.fitsMemory = model.memoryFloorBytes <= this.supervisor.memoryLimitFor(model.modelType)
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
    this.assertModelAdmission()
    return this.libraryChanged(await this.library.scan(options))
  }

  async addLibraryFolder(directory: string): Promise<EngineModelLibrarySnapshot> {
    this.assertModelAdmission()
    return this.libraryChanged(await this.library.addFolder(directory))
  }

  async removeLibraryFolder(id: string): Promise<EngineModelLibrarySnapshot> {
    this.assertModelAdmission()
    // Unlinking a search folder never stops a loaded API or modifies its checkpoint files.
    return this.libraryChanged(await this.library.removeFolder(id))
  }

  async cancelLibraryScan(): Promise<EngineModelLibrarySnapshot> { return this.library.cancel() }

  // Quit path, without waiting: SIGTERM at before-quit, SIGKILL (force) when the process exits.
  shutdown(force = true): void {
    if (!force && (this.learning.busy || this.benchmarks.busy)) {
      void this.prepareToQuit().then(() => this.shutdown(false)).catch(() => {})
      return
    }
    this.learning.shutdown()
    this.benchmarks.dispose()
    this.cancelTryModel()
    this.pullQueue.cancelAll()
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
