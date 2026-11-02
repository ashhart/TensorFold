import { EngineFeatureControls } from './feature-controls'
import { EngineInventory } from './inventory'
import { EngineDownloads } from './downloads'
import { EngineProcessRecord } from './process-record'
import { EnginePerformanceRecorder } from './performance-recorder'
import { promises as fs, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import type { EngineCapabilities, EngineConnection, EngineEvent, EngineInstalledModel, EngineLearningSnapshot, EngineLibraryScanOptions, EngineModelLibrarySnapshot, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness, EngineModelGenerationSettings, EngineModelPreviewResult, EngineModelComparisonInput, EngineModelComparisonSnapshot } from '../types'
import { SUGGESTED_MODELS, drafterFor } from './catalog'
import { parseCapabilities, parseVersion, runEngineCli, type ModelInfo } from './cli'
import { ModelHub } from './model-hub'
import { engineLaunchEnv, pinnedEngineEnv } from './engine-env'
import type { EngineEnvOverrides } from '../../../shared/engine-advanced-env'
export { engineChildEnv } from './engine-env'
import { assertRepoId, hasCachedSnapshot, hubCacheDir, repoCacheDir, type EngineBinary } from './paths'
import { setEnginePerformanceRecorder, setEngineServeHook } from './provider'
import { recordEngineContext } from './context'
import { EngineSupervisor } from './supervisor'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'
import { LearningCopies } from './learning-copy'
import { EngineLearning } from './learning'
import { isLocalModelId, ModelLibrary } from './library'
import { CORTHEON_RELEASE_POLICY, cortheonEnabledForRelease } from '../../../shared/release-feature-policy'
import { measureModelPreview } from './model-preview'
import { ModelComparisons } from './model-comparison'
import { modelLaunchOptions } from './launch-options'
import type { ModelBenchmarkInput, ModelBenchmarkReceipt, ModelBenchmarkSnapshot } from '../../../shared/model-benchmark'
import { ModelBenchmarks } from './benchmark/controller'
import { createModelBenchmarkStore, benchmarkWorkspace, readBenchmarkReceiptFile, type ModelBenchmarkStore } from './benchmark/storage'
import { startBenchmarkProcess } from './benchmark/process'
import { assertCompatibleBenchmarkVersion } from './benchmark/version'

export interface EngineServiceOptions {
  binaryPath: string
  appHome: string
  env?: NodeJS.ProcessEnv
  // Exact environment for engine processes; tests use it to reach the fake engine's knobs.
  childEnv?: NodeJS.ProcessEnv
  binary?: EngineBinary
  runtimeBinary?: () => string | null
  runtimeReady?: () => Promise<void>
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

export class EngineService {
  readonly supervisor: EngineSupervisor
  private readonly features: EngineFeatureControls
  private readonly inventory: EngineInventory
  private readonly downloads: EngineDownloads
  private readonly processRecord: EngineProcessRecord
  private readonly metrics: EnginePerformanceRecorder
  private readonly launch: { env: NodeJS.ProcessEnv; endpoint: string }
  private readonly totalMemoryBytes: number
  private runtimeSwitching = false
  private runtimeEpoch = 0
  private version: string | null = null
  private versionChecked = false
  private capabilities: EngineCapabilities | null = null
  private deletingModel = false
  private preview: { model: string; controller: AbortController; profile: EngineModelGenerationSettings | null; promise: Promise<EngineModelPreviewResult> } | null = null
  private readonly comparisons: ModelComparisons
  private readonly benchmarks: ModelBenchmarks
  private benchmarkAdmission: (() => void) | null = null
  private sharedEndpointKey: string | null = null
  private comparisonAdmission: (() => void) | null = null
  private comparisonRequest: { owner: symbol; model: string; profile: EngineModelGenerationSettings | null } | null = null
  private readonly models: ModelHub
  private readonly revisions: ModelRevisions
  private readonly preferences: EnginePreferenceStore
  private readonly library: ModelLibrary
  // Whether the bundled engine lists --cortheon among its serve flags; null until asked.
  private cortheonSupported: boolean | null = null
  private learningSupported = false
  private readonly learningCopies: LearningCopies
  private readonly learning: EngineLearning
  private quitting = false
  private learningPurge: Promise<void> | null = null
  private purgingLearning = false

  private get env(): NodeJS.ProcessEnv {
    const base = this.options.childEnv ? pinnedEngineEnv(this.options.childEnv, this.launch.endpoint) : this.launch.env
    return this.features.appliedEnvOverrides ? { ...base, ...this.features.appliedEnvOverrides } : base
  }

  constructor(private readonly options: EngineServiceOptions) {
    this.launch = engineLaunchEnv(options)
    this.totalMemoryBytes = options.totalMemoryBytes ?? os.totalmem()
    // Preferences load first so a saved HF_HUB_OFFLINE or cache location governs every dependent.
    this.preferences = new EnginePreferenceStore(options.appHome)
    this.features = new EngineFeatureControls({
      preferences: this.preferences, learningCopies: () => this.learningCopies, revisions: () => this.revisions,
      library: () => this.library, supervisor: () => this.supervisor, learning: () => this.learning, hub: () => this.hub,
      benchmarkBusy: () => this.benchmarks?.busy ?? false, comparisonBusy: () => this.comparisons.busy,
      cacheBusy: () => this.downloads.busy || this.deletingModel, assertAllowed: () => this.assertNotQuitting(),
      cortheonSupported: () => this.cortheonSupported, learningSupported: () => this.learningSupported,
      capabilitiesDoc: () => this.capabilitiesDoc(), status: () => this.status(), statusNow: () => this.statusNow(), info: repo => this.info(repo),
      clearPreflights: () => this.models.clearPreflights(), invalidateInstalled: () => this.inventory.invalidate(),
      refreshInstalled: () => this.installed(true), emitStatus: () => this.emitStatus(), emit: options.emit,
    })
    this.models = new ModelHub({
      endpoint: this.launch.endpoint, fetch: options.fetch, env: () => this.env, totalMemoryBytes: this.totalMemoryBytes,
      capabilities: () => this.capabilitiesDoc(), installed: () => this.installed(), cacheDir: () => this.hub,
      memory: () => ({ limit: this.supervisor.memoryLimitBytes, budget: this.supervisor.memoryBudgetBytes, limitFor: (modelType) => this.supervisor.memoryPolicyFor(modelType) }),
    })
    this.revisions = new ModelRevisions(options.appHome, () => this.hub)
    this.library = new ModelLibrary({ appHome: options.appHome, env: () => this.env, capabilities: () => this.capabilitiesDoc(), memoryLimitBytes: () => this.supervisor.memoryLimitBytes })
    this.learningCopies = new LearningCopies(options.appHome)
    this.processRecord = new EngineProcessRecord(options)
    this.metrics = new EnginePerformanceRecorder({ supervisor: () => this.supervisor, emit: options.emit })
    this.inventory = new EngineInventory({
      binary: () => this.binary(), requireBinary: () => this.requireBinary(), env: () => this.env, hub: () => this.hub,
      runtimeReady: options.runtimeReady, runtimeEpoch: () => this.runtimeEpoch, modelSettings: options.modelSettings,
      library: this.library, revisions: this.revisions, preferences: this.preferences, learningCopies: this.learningCopies,
      hasPull: repo => this.downloads.hasRepo(repo), memoryLimitFor: type => this.supervisor.memoryLimitFor(type),
      emit: options.emit, onInstalled: options.onInstalled,
    })
    this.downloads = new EngineDownloads({
      binary: () => this.requireBinary(), env: () => this.env, hub: () => this.hub, pollMs: options.pollMs,
      assertAllowed: () => this.assertModelAdmission(), deletingModel: () => this.deletingModel,
      cachedPreflight: key => this.models.cached(key), preflight: (repo, revision) => this.preflight(repo, revision),
      revisions: this.revisions, status: () => this.supervisor.status(), invalidate: repo => this.models.invalidate(repo),
      invalidateInstalled: () => this.inventory.invalidate(), refreshInstalled: () => this.installed(true), emit: options.emit,
    })
    this.supervisor = new EngineSupervisor({
      binary: () => this.binary(),
      env: () => this.env,
      info: (repo) => this.info(repo),
      resolveModelArgument: (repo) => this.inventory.modelArgument(repo),
      drafterFor: (repo) => this.inventory.requiredDrafter(repo),
      drafterBytes: (repo) => this.inventory.cachedModelBytes(repo),
      modelSettings: (repo) => this.options.modelSettings?.(repo) ?? null,
      totalMemoryBytes: this.totalMemoryBytes,
      memoryBudgetBytes: options.memoryBudgetBytes,
      readyTimeoutMs: options.readyTimeoutMs,
      pollMs: options.pollMs,
      onChange: () => this.emitStatus(),
      onPid: (pid) => { void this.processRecord.record(pid) },
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
    this.learning = new EngineLearning({
      appHome: options.appHome, supervisor: this.supervisor, copies: this.learningCopies,
      settings: () => ({ supported: this.learningSupported, enabled: this.preferences.get().learningEnabled, model: this.preferences.get().learningModel }),
      emit: (snapshot) => {
        if (snapshot.job && ['done', 'cancelled', 'interrupted'].includes(snapshot.job.state)) this.inventory.invalidate()
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
        const argument = await this.inventory.modelArgument(model)
        if (argument !== model && await fs.realpath(argument) !== snapshot) throw new Error('The selected checkpoint differs from the cached public model.')
        const status = await this.status()
        const engineVersion = assertCompatibleBenchmarkVersion(status.version)
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
    const command = this.options.runtimeBinary?.() ?? this.options.binaryPath
    return existsSync(command) ? { command, args: [] } : null
  }

  private requireBinary(): EngineBinary {
    const binary = this.binary()
    if (!binary) throw new Error('The TensorFold engine is not bundled with this build.')
    return binary
  }

  // Wires streamChat's engine hook to this service and clears an engine a crashed run left behind.
  install(): void {
    setEnginePerformanceRecorder((model, sample) => this.metrics.record(model, sample))
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
    void this.processRecord.reap()
  }

  async reconfigureRuntime(apply: () => Promise<void>, rollback: () => Promise<void>): Promise<void> {
    this.assertModelAdmission()
    if (this.isLearningBusy() || this.library.scanning || this.downloads.busy || this.deletingModel ||
      this.preferences.get().learningEnabled || this.sharedEndpointKey === this.supervisor.live()?.apiKey) {
      throw new Error('Finish local work and downloads, turn Living Weights off and unload any shared API session before switching runtimes.')
    }
    this.runtimeSwitching = true
    const invalidate = () => {
      this.runtimeEpoch++; this.version = null; this.versionChecked = false; this.capabilities = null
      this.cortheonSupported = null; this.learningSupported = false; this.inventory.invalidate()
      this.metrics.clear(); this.models.clearPreflights()
    }
    try {
      await this.supervisor.reconfigure(async () => { await apply(); invalidate(); await this.library.scan({}) },
        async () => { await rollback(); invalidate(); await this.library.scan({}) })
    } finally { this.runtimeSwitching = false; await this.status(); this.emitStatus() }
  }

  async status(): Promise<EngineStatus> {
    await this.options.runtimeReady?.()
    const epoch = this.runtimeEpoch
    if (!this.versionChecked && this.binary()) {
      this.versionChecked = true
      try {
        const version = parseVersion((await runEngineCli(this.requireBinary(), ['--version'], this.env, 10_000)).stdout)
        if (epoch === this.runtimeEpoch) this.version = version
      } catch { if (epoch === this.runtimeEpoch) this.version = null }
    }
    if (this.cortheonSupported == null && this.binary()) {
      try {
        const flags = (await this.capabilitiesDoc()).flags
        if (epoch !== this.runtimeEpoch) return this.status()
        this.cortheonSupported = flags.includes('--cortheon')
        this.learningSupported = flags.includes('--slide') && flags.includes('--slide-graph') && os.platform() === 'darwin' && os.arch() === 'arm64'
      } catch { this.cortheonSupported = false }
    }
    if (epoch !== this.runtimeEpoch) return this.status()
    return this.statusNow()
  }

  async learningSnapshot(): Promise<EngineLearningSnapshot> {
    await this.status()
    return this.learning.snapshot()
  }

  async teach(text: string): Promise<EngineLearningSnapshot> { if (this.benchmarks.busy) throw new Error('Finish or cancel the local benchmark before teaching.'); if (this.comparisons.busy) throw new Error('Finish or cancel the model comparison before teaching.'); this.assertNotQuitting(); await this.status(); this.assertNotQuitting(); return this.learning.teach(text) }
  async cancelTeach(): Promise<EngineLearningSnapshot> { return this.learning.cancel() }

  isLearningBusy(): boolean { return this.learning.busy || this.features.pending > 0 || this.preview != null || this.comparisons.busy || this.benchmarks.busy }

  applyModelSettings(model: string, apply: () => Promise<void>, rollback: () => Promise<void>): Promise<void> {
    this.assertModelAdmission()
    assertRepoId(model)
    if (this.learning.busy || this.preview) return Promise.reject(new Error('Finish the current lesson or model preview before applying model settings.'))
    return this.features.change(async () => {
      try {
        await this.supervisor.configureModel(model, async () => {
          await apply()
          const settings = this.options.modelSettings?.(model)
          if (settings) modelLaunchOptions(settings, (await this.info(model)).contextTokens)
        }, rollback)
      } finally { this.inventory.invalidate(); this.emitStatus() }
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
    if (this.deletingModel || this.downloads.queueBusy || this.learning.busy || this.features.pending || this.preview || this.comparisons.busy || this.supervisor.status().learningActive || this.preferences.get().learningEnabled) throw new Error('Finish local work, downloads and model settings changes, and turn Living Weights off before benchmarking.')
    if (this.hasExternalResearchEnabled()) throw new Error('Turn external research off before benchmarking local models.')
    if (this.sharedEndpointKey && this.supervisor.live()?.apiKey === this.sharedEndpointKey) throw new Error('Unload the shared API session before benchmarking; external clients may still be using it.')
  }
  private assertComparisonAllowed(): void {
    this.assertModelAdmission()
    this.options.assertComparisonAllowed?.()
    this.comparisonAdmission?.()
    if (this.learning.busy || this.features.pending || this.preview || this.supervisor.status().learningActive || this.preferences.get().learningEnabled) throw new Error('Finish local work and turn Living Weights off before comparing models.')
    if (this.hasExternalResearchEnabled()) throw new Error('Turn external research off before comparing local models.')
    if (this.sharedEndpointKey && this.supervisor.live()?.apiKey === this.sharedEndpointKey) throw new Error('Unload the model to close its shared API session before comparing models; external clients may still be using it.')
  }

  cancelTryModel(): boolean {
    if (!this.preview) return false
    this.preview.controller.abort(new Error('Model preview cancelled.'))
    return true
  }

  private assertNotQuitting(): void {
    if (this.runtimeSwitching) throw new Error('Wait for the engine runtime switch to finish.')
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
    await preview?.promise.catch(() => { })
    await this.features.settle()
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
      await preview?.promise.catch(() => { })
      await this.comparisons.settle()
      await this.benchmarks.settle()
      this.comparisons.clear()
      await this.features.settle()
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
    this.inventory.invalidate()
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
      ...s, cortheon: { supported: this.cortheonSupported === true, enabled: this.preferences.get().cortheon, active: cortheonActive, disabledReason: CORTHEON_RELEASE_POLICY.enabled ? undefined : CORTHEON_RELEASE_POLICY.reason }, available: !!binary, version: this.version, binaryPath: binary?.command ?? null, totalMemoryBytes: this.totalMemoryBytes, memoryBudgetBytes: this.supervisor.memoryBudgetBytes,
      hardware: { platform: os.platform(), architecture: os.arch(), chip: os.cpus()[0]?.model?.trim() || null },
    }
  }

  private emitStatus(): void {
    const status = this.statusNow()
    if (status.state !== 'ready' || status.model !== this.metrics.current?.model) this.metrics.clear()
    if (status.state === 'ready') recordEngineContext(status.model, status.contextTokens)
    this.options.emit({ kind: 'status', status })
  }

  async snapshot(): Promise<EngineSnapshot> {
    const [status, installed] = await Promise.all([this.status(), this.installed()])
    return { status, performance: this.metrics.current, installed, pulls: this.pullList(), suggestions: SUGGESTED_MODELS, hubCache: this.hub }
  }

  // For an explicit client-connection request, without starting a model or emitting credentials.
  async connection(): Promise<EngineConnection | null> {
    if (this.comparisons.busy || this.benchmarks.busy) throw new Error('API sharing is paused during local model comparisons and benchmarks.')

    if (this.preferences.get().learningEnabled || this.supervisor.status().learningActive) throw new Error('Local API sharing is paused while Living Weights is enabled. Turn teaching off before copying a connection for another app.')
    const live = this.supervisor.live()
    if (live) this.sharedEndpointKey = live.apiKey
    return live
  }

  hasExternalResearchEnabled(): boolean { return this.preferences.get().cortheon || this.supervisor.status().cortheonActive }

  async capabilitiesDoc(): Promise<EngineCapabilities> {
    await this.options.runtimeReady?.()
    const epoch = this.runtimeEpoch
    if (this.capabilities) return this.capabilities
    const result = await runEngineCli(this.requireBinary(), ['capabilities', '--json'], this.env)
    if (result.code !== 0) throw new Error(result.stderr.trim() || 'tensorfold-native capabilities --json failed.')
    if (epoch !== this.runtimeEpoch) return this.capabilitiesDoc()
    this.capabilities = parseCapabilities(result.stdout)
    return this.capabilities
  }

  search(query: string): Promise<HubModelSummary[]> { return this.models.search(query) }

  preflight(repo: string, revision: string | null = null): Promise<ModelPreflight> { return this.models.preflight(repo, revision) }

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
    this.metrics.clear()
    return this.statusNow()
  }

  // Removes the repo's folder from the shared cache, and its drafter when no other installed model uses it.
  async remove(repo: string): Promise<EngineInstalledModel[]> {
    assertRepoId(repo)
    this.assertModelAdmission()
    if (this.deletingModel || this.downloads.queueBusy) throw new Error('Finish or cancel model downloads before cleaning the shared cache.')
    this.deletingModel = true
    try { return await this.supervisor.mutateCache(() => this.removeModelNow(repo)) } finally { this.deletingModel = false }
  }

  private async removeModelNow(repo: string): Promise<EngineInstalledModel[]> {
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints are read-only. Remove their search folder to unlink them; their files stay untouched.')
    const selected = await this.revisions.read()
    if (!selected[repo]) throw new Error('This checkpoint belongs to the shared cache, rather than an app-managed download. TensorFold will not delete it.')
    const status = this.supervisor.status()
    if ((status.model === repo || this.inventory.resolvedRepo(status.model ?? '') === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error(`Unload ${repo} before deleting it.`)
    if (this.downloads.hasJob(repo)) throw new Error(`Cancel the download of ${repo} before deleting it.`)
    const installed = await this.installed()
    const drafter = drafterFor(repo)
    const targets = [repo]
    if (drafter && selected[drafter] && !installed.some((m) => m.repo !== repo && drafterFor(this.inventory.resolvedRepo(m.repo) ?? m.repo) === drafter)) targets.push(drafter)
    for (const target of targets) await this.removeRepoFolder(target, selected[target])
    await this.revisions.remove(targets)
    this.models.invalidate(repo)
    this.downloads.forget(repo)
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

      const selected = path.join(snapshots, revision)
      const stat = await fs.lstat(selected).catch(() => null)
      if (stat?.isSymbolicLink() || stat && !stat.isDirectory()) throw new Error('The selected checkpoint was redirected; refusing to delete it.')
      if (stat) await fs.rm(selected, { recursive: true })
      return
    }
    await fs.rm(folder, { recursive: true, force: true })
  }

  private async libraryChanged(_snapshot: EngineModelLibrarySnapshot): Promise<EngineModelLibrarySnapshot> {
    this.inventory.invalidate()
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
      void this.prepareToQuit().then(() => this.shutdown(false)).catch(() => { })
      return
    }
    this.learning.shutdown()
    this.benchmarks.dispose()
    this.cancelTryModel()
    this.downloads.cancelAll()
    void this.library.cancel().catch(() => { })
    this.supervisor.killNow(force ? 'SIGKILL' : 'SIGTERM')
    if (force) void this.processRecord.record(null)
  }

  info(repo: string): Promise<ModelInfo> { return this.inventory.info(repo) }
  installed(refresh = false): Promise<EngineInstalledModel[]> { return this.inventory.installed(refresh) }
  readiness(model: string | null): Promise<Readiness> { return this.inventory.readiness(model) }
  librarySnapshot(): Promise<EngineModelLibrarySnapshot> { return this.inventory.librarySnapshot() }
  pullList(): PullProgress[] { return this.downloads.pullList() }
  pull(repo: string, revision: string | null = null): Promise<PullProgress> { return this.downloads.pull(repo, revision) }
  cancelPull(repo: string): boolean { return this.downloads.cancelPull(repo) }

  setCortheon(enabled: boolean): Promise<EngineStatus> { return this.features.setCortheon(enabled) }
  envOverrides(): EngineEnvOverrides { return this.features.envOverrides() }
  setEnvOverrides(next: EngineEnvOverrides): Promise<EngineEnvOverrides> { return this.features.setEnvOverrides(next) }
  declaredEngineEnv(): Promise<string[] | null> { return this.features.declaredEngineEnv() }
  setLearning(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> { return this.features.setLearning(enabled, model) }
}
