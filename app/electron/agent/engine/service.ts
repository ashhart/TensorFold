import { existsSync } from 'node:fs'
import os from 'node:os'
import type { EngineCapabilities, EngineConnection, EngineControlVectorSnapshot, EngineEvent, EngineInstalledModel, EngineLearningSnapshot, EngineLibraryScanOptions, EngineModelLibrarySnapshot, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness, EngineModelGenerationSettings, EngineModelPreviewResult, EngineModelComparisonInput, EngineModelComparisonSnapshot } from '../types'
import { SUGGESTED_MODELS } from './catalog'
import type { ModelInfo } from './cli'
import { ModelHub } from './model-hub'
import { engineLaunchEnv, pinnedEngineEnv } from './engine-env'
import { assertDeclaredEngineEnv, normalizeEngineEnvOverrides, type EngineEnvOverrides } from '../../../shared/engine-advanced-env'
export { engineChildEnv } from './engine-env'
import { assertRepoId, hubCacheDir, type EngineBinary } from './paths'
import { setEnginePerformanceRecorder, setEngineServeHook } from './provider'
import { recordEngineContext } from './context'
import { EngineSupervisor } from './supervisor'
import { ControlVectorSetting } from './control-vector'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'
import { LearningCopies } from './learning-copy'
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
import { checkBenchmarkModel } from './benchmark/model-check'
import { InstalledModels } from './installed-models'
import { ModelDownloads } from './model-downloads'
import { EngineProcessRecord } from './engine-process-record'
import { EnginePerformanceTracker } from './performance-tracker'
import { configureLearning } from './learning-setup'
import { EngineFeatures } from './engine-features'

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

export class EngineService {
  readonly supervisor: EngineSupervisor
  private readonly launch: { env: NodeJS.ProcessEnv; endpoint: string }
  // Advanced user variables applied over the launch environment; null until preferences load.
  private envOverridesApplied: EngineEnvOverrides | null = null
  private readonly totalMemoryBytes: number
  private readonly features: EngineFeatures
  private readonly performance: EnginePerformanceTracker
  private readonly downloads: ModelDownloads
  private readonly installedModels: InstalledModels
  private readonly processRecord: EngineProcessRecord
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
  // The control vector the operator chose, and what the engine in use can do with it.
  private controlVector!: ControlVectorSetting
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
    this.features = new EngineFeatures({ binary: () => this.binary(), requireBinary: () => this.requireBinary(), env: () => this.env })
    this.models = new ModelHub({
      endpoint: this.launch.endpoint, fetch: options.fetch, env: () => this.env, totalMemoryBytes: this.totalMemoryBytes,
      capabilities: () => this.capabilitiesDoc(), installed: () => this.installed(), cacheDir: () => this.hub,
      memory: () => ({ limit: this.supervisor.memoryLimitBytes, budget: this.supervisor.memoryBudgetBytes, limitFor: (modelType) => this.supervisor.memoryPolicyFor(modelType) }),
    })
    this.revisions = new ModelRevisions(options.appHome, () => this.hub)
    this.library = new ModelLibrary({ appHome: options.appHome, env: () => this.env, capabilities: () => this.capabilitiesDoc(), memoryLimitBytes: () => this.supervisor.memoryLimitBytes })
    this.learningCopies = new LearningCopies(options.appHome)
    this.processRecord = new EngineProcessRecord(options.appHome, options.binaryPath)
    this.installedModels = new InstalledModels({
      binary: () => this.binary(), requireBinary: () => this.requireBinary(), env: () => this.env, hub: () => this.hub,
      library: this.library, revisions: this.revisions, preferences: this.preferences, learningCopies: this.learningCopies,
      modelSettings: (repo) => options.modelSettings?.(repo) ?? null,
      memoryLimitFor: (modelType) => this.supervisor.memoryLimitFor(modelType),
      isPulling: (repo) => this.downloads.hasRepo(repo),
      emit: (event) => options.emit(event),
      onInstalled: (repos) => options.onInstalled?.(repos),
    })
    this.supervisor = new EngineSupervisor({
      binary: () => this.binary(),
      env: () => this.env,
      info: (repo) => this.info(repo),
      resolveModelArgument: (repo) => this.installedModels.argument(repo),
      drafterFor: (repo) => this.installedModels.requiredDrafter(repo),
      drafterBytes: (repo) => this.installedModels.cachedBytes(repo),
      modelSettings: (repo) => this.options.modelSettings?.(repo) ?? null,
      totalMemoryBytes: this.totalMemoryBytes,
      memoryBudgetBytes: options.memoryBudgetBytes,
      readyTimeoutMs: options.readyTimeoutMs,
      pollMs: options.pollMs,
      onChange: () => this.emitStatus(),
      onPid: (pid) => { void this.processRecord.record(pid) },
      cortheon: () => this.features.cortheon === true && cortheonEnabledForRelease(this.preferences.get().cortheon),
      controlVector: () => {
        const prefs = this.preferences.get().controlVector
        // The engine takes the vector only when it offers --glp and the operator has a file.
        if (this.features.glp !== true || !prefs.enabled || !prefs.file) return null
        return { file: prefs.file, alpha: prefs.alpha }
      },
      learning: (repo) => {
        const prefs = this.preferences.get()
        return this.features.learning && repo === prefs.learningModel ? { enabled: prefs.learningEnabled, graphFile: this.learningCopies.graphFile(repo) } : null
      },
    })
    this.controlVector = new ControlVectorSetting({ preferences: this.preferences, supervisor: this.supervisor, supported: () => this.features.glp === true })
    this.performance = new EnginePerformanceTracker({ supervisor: this.supervisor, emit: (event) => options.emit(event) })
    this.downloads = new ModelDownloads({
      models: this.models, revisions: this.revisions, installed: this.installedModels, supervisor: this.supervisor,
      requireBinary: () => this.requireBinary(), env: () => this.env, hub: () => this.hub, pollMs: options.pollMs,
      emit: (event) => options.emit(event), assertAdmission: () => this.assertModelAdmission(),
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
      settings: () => ({ supported: this.features.learning, enabled: this.preferences.get().learningEnabled, model: this.preferences.get().learningModel }),
      emit: (snapshot) => {
        if (snapshot.job && ['done', 'cancelled', 'interrupted'].includes(snapshot.job.state)) this.installedModels.invalidate()
        options.emit({ kind: 'learning', snapshot })
      }, pollMs: options.pollMs,
    })
    this.benchmarks = new ModelBenchmarks({
      assertAllowed: () => this.assertBenchmarkAllowed(),
      validateModel: (model) => checkBenchmarkModel({ installed: this.installedModels, revisions: this.revisions, hub: () => this.hub, status: () => this.status() }, model),
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
    setEnginePerformanceRecorder((model, sample) => this.performance.record(model, sample))
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
    void this.processRecord.reapStale()
  }

  async status(): Promise<EngineStatus> {
    await this.features.probe()
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
    // The engine's own capabilities doc is the authority on what this build honours; a name it
    // does not list is refused rather than stored and ignored. Null when the binary could not be
    // asked, and then the deny list in the shared validator is the only boundary.
    assertDeclaredEngineEnv(normalized, await this.declaredEngineEnv())
    const prior = this.preferences.get().envOverrides
    // A cache-location move must never race an operation bound to the current root.
    const cacheMoved = normalized.HF_HOME !== prior.HF_HOME || normalized.HF_HUB_CACHE !== prior.HF_HUB_CACHE
    if (cacheMoved && (this.downloads.activeCount || this.downloads.queueBusy || this.downloads.deleting)) throw new Error('Finish model downloads and cache cleanup before changing the model cache location.')
    await this.preferences.set({ envOverrides: normalized })
    this.envOverridesApplied = normalized
    // Verdicts cached at the old limit are stale the moment the limit moves; re-answer them all.
    this.models.clearPreflights()
    await this.library.refreshEligibility((modelType) => this.supervisor.memoryLimitFor(modelType))
    this.installedModels.invalidate()
    this.emitStatus()
    await this.installed(true)
    return normalized
  }

  /** The environment this engine build declares it honours, or null when it could not be asked. */
  declaredEngineEnv(): Promise<string[] | null> { return this.features.declaredEnv() }

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
    if (enabled && !this.features.cortheon) throw new Error("This build's TensorFold engine does not offer Cortheon yet.")
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

  // The control vector the operator chose, and what the engine in use can do with it.
  async controlVectorSnapshot(): Promise<EngineControlVectorSnapshot> {
    await this.status()
    return this.controlVector.snapshot()
  }

  // The file the operator picked, from the OS picker: it is remembered, not applied.
  async chooseControlVector(file: string): Promise<EngineControlVectorSnapshot> {
    await this.status()
    return this.controlVector.choose(file)
  }

  // Whether the vector is in force, and at what strength.
  async setControlVector(input: { enabled?: unknown; alpha?: unknown }): Promise<EngineControlVectorSnapshot> {
    await this.status()
    return this.changeFeature(() => this.controlVector.set(input))
  }

  // Forget the file: the engine stops taking a vector at its next start.
  async clearControlVector(): Promise<EngineControlVectorSnapshot> {
    await this.status()
    return this.changeFeature(() => this.controlVector.clear())
  }
  setLearning(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    return this.changeFeature(() => this.setLearningNow(enabled, model))
  }

  private async setLearningNow(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    await this.status()
    return configureLearning({
      learning: this.learning, supervisor: this.supervisor, preferences: this.preferences, learningCopies: this.learningCopies,
      revisions: this.revisions, installed: this.installedModels, hub: () => this.hub, supported: () => this.features.learning,
      emit: (event) => this.options.emit(event), emitStatus: () => this.emitStatus(),
    }, enabled, model)
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
      } finally { this.installedModels.invalidate(); this.emitStatus() }
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
    if (this.downloads.deleting || this.downloads.queueBusy || this.learning.busy || this.featurePending || this.preview || this.comparisons.busy || this.supervisor.status().learningActive || this.preferences.get().learningEnabled) throw new Error('Finish local work, downloads and model settings changes, and turn Living Weights off before benchmarking.')
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
    if (this.downloads.deleting) throw new Error('Wait for model cache cleanup before loading or using a model.')
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
    this.installedModels.invalidate()
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
    const { cortheonActive, controlVectorActive, learningActive, ...s } = this.supervisor.status()
    const binary = this.binary()
    return {
      ...s, cortheon: { supported: this.features.cortheon === true, enabled: this.preferences.get().cortheon, active: cortheonActive, disabledReason: CORTHEON_RELEASE_POLICY.enabled ? undefined : CORTHEON_RELEASE_POLICY.reason }, controlVector: { supported: this.features.glp === true, enabled: this.preferences.get().controlVector.enabled, file: this.preferences.get().controlVector.file, alpha: this.preferences.get().controlVector.alpha, active: controlVectorActive }, available: !!binary, version: this.features.version, binaryPath: binary ? this.options.binaryPath : null, totalMemoryBytes: this.totalMemoryBytes, memoryBudgetBytes: this.supervisor.memoryBudgetBytes,
      hardware: { platform: os.platform(), architecture: os.arch(), chip: os.cpus()[0]?.model?.trim() || null },
    }
  }

  private emitStatus(): void {
    const status = this.statusNow()
    if (status.state !== 'ready' || status.model !== this.performance.current?.model) this.performance.clear()
    if (status.state === 'ready') recordEngineContext(status.model, status.contextTokens)
    this.options.emit({ kind: 'status', status })
  }

  async snapshot(): Promise<EngineSnapshot> {
    const [status, installed] = await Promise.all([this.status(), this.installed()])
    return { status, performance: this.performance.current, installed, pulls: this.pullList(), suggestions: SUGGESTED_MODELS, hubCache: this.hub }
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

  info(repo: string): Promise<ModelInfo> { return this.installedModels.info(repo) }

  capabilitiesDoc(): Promise<EngineCapabilities> { return this.features.capabilities() }

  installed(refresh = false): Promise<EngineInstalledModel[]> { return this.installedModels.list(refresh) }

  search(query: string): Promise<HubModelSummary[]> { return this.models.search(query) }

  preflight(repo: string, revision: string | null = null): Promise<ModelPreflight> { return this.models.preflight(repo, revision) }

  pullList(): PullProgress[] { return this.downloads.list() }

  pull(repo: string, revision: string | null = null): Promise<PullProgress> { return this.downloads.pull(repo, revision) }

  cancelPull(repo: string): boolean { return this.downloads.cancel(repo) }

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
    this.performance.clear()
    return this.statusNow()
  }

  // Removes the repo's folder from the shared cache, and its drafter when no other installed model uses it.
  remove(repo: string): Promise<EngineInstalledModel[]> { return this.downloads.remove(repo) }

  readiness(model: string | null): Promise<Readiness> { return this.installedModels.readiness(model) }

  librarySnapshot(): Promise<EngineModelLibrarySnapshot> { return this.installedModels.librarySnapshot() }

  async scanLibrary(options: EngineLibraryScanOptions = {}): Promise<EngineModelLibrarySnapshot> {
    this.assertModelAdmission()
    await this.library.scan(options)
    return this.installedModels.libraryChanged()
  }

  async addLibraryFolder(directory: string): Promise<EngineModelLibrarySnapshot> {
    this.assertModelAdmission()
    await this.library.addFolder(directory)
    return this.installedModels.libraryChanged()
  }

  async removeLibraryFolder(id: string): Promise<EngineModelLibrarySnapshot> {
    this.assertModelAdmission()
    // Unlinking a search folder never stops a loaded API or modifies its checkpoint files.
    await this.library.removeFolder(id)
    return this.installedModels.libraryChanged()
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
    this.downloads.cancelQueued()
    void this.library.cancel().catch(() => {})
    this.downloads.cancelActive()
    this.supervisor.killNow(force ? 'SIGKILL' : 'SIGTERM')
    if (force) void this.processRecord.record(null)
  }
}
