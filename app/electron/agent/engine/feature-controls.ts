import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EngineCapabilities, EngineEvent, EngineInstalledModel, EngineLearningSnapshot, EngineStatus } from '../types'
import { type ModelInfo } from './cli'
import { assertDeclaredEngineEnv, normalizeEngineEnvOverrides, type EngineEnvOverrides } from '../../../shared/engine-advanced-env'
import { assertRepoId, repoCacheDir } from './paths'
import { EngineSupervisor } from './supervisor'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'
import { LearningCopies, LEARNING_MODEL } from './learning-copy'
import { EngineLearning } from './learning'
import { ModelLibrary } from './library'
import { assertCortheonReleaseAllowed } from '../../../shared/release-feature-policy'

interface EngineFeatureOptions {
  preferences: EnginePreferenceStore
  learningCopies: () => LearningCopies
  revisions: () => ModelRevisions
  library: () => ModelLibrary
  supervisor: () => EngineSupervisor
  learning: () => EngineLearning
  hub: () => string
  benchmarkBusy: () => boolean
  comparisonBusy: () => boolean
  cacheBusy: () => boolean
  assertAllowed: () => void
  cortheonSupported: () => boolean | null
  learningSupported: () => boolean
  capabilitiesDoc: () => Promise<EngineCapabilities>
  status: () => Promise<EngineStatus>
  statusNow: () => EngineStatus
  info: (repo: string) => Promise<ModelInfo>
  clearPreflights: () => void
  invalidateInstalled: () => void
  refreshInstalled: () => Promise<EngineInstalledModel[]>
  emitStatus: () => void
  emit: (event: EngineEvent) => void
}

export class EngineFeatureControls {
  pending = 0
  private changes: Promise<unknown> = Promise.resolve()
  appliedEnvOverrides: EngineEnvOverrides | null
  constructor(private options: EngineFeatureOptions) { this.appliedEnvOverrides = options.preferences.get().envOverrides }
  settle(): Promise<unknown> { return this.changes }
  setCortheon(enabled: boolean): Promise<EngineStatus> {
    try { assertCortheonReleaseAllowed(enabled) } catch (error) { return Promise.reject(error) }
    return this.change(() => this.setCortheonNow(enabled))
  }

  envOverrides(): EngineEnvOverrides {
    return this.options.preferences.get().envOverrides
  }

  async setEnvOverrides(next: EngineEnvOverrides): Promise<EngineEnvOverrides> {
    const normalized = normalizeEngineEnvOverrides(next)

    assertDeclaredEngineEnv(normalized, await this.declaredEngineEnv())
    const prior = this.options.preferences.get().envOverrides
    // A cache-location move must never race an operation bound to the current root.
    const cacheMoved = normalized.HF_HOME !== prior.HF_HOME || normalized.HF_HUB_CACHE !== prior.HF_HUB_CACHE
    if (cacheMoved && (this.options.cacheBusy())) throw new Error('Finish model downloads and cache cleanup before changing the model cache location.')
    await this.options.preferences.set({ envOverrides: normalized })
    this.appliedEnvOverrides = normalized
    // Verdicts cached at the old limit are stale the moment the limit moves; re-answer them all.
    this.options.clearPreflights()
    await this.options.library().refreshEligibility((modelType) => this.options.supervisor().memoryLimitFor(modelType))
    this.options.invalidateInstalled()
    this.options.emitStatus()
    await this.options.refreshInstalled()
    return normalized
  }

  async declaredEngineEnv(): Promise<string[] | null> {
    try {
      const caps = await this.options.capabilitiesDoc()
      return Array.isArray(caps.env) ? caps.env : null
    } catch {
      return null
    }
  }

  change<T>(run: () => Promise<T>): Promise<T> {
    if (this.options.benchmarkBusy()) return Promise.reject(new Error('Finish or cancel the local benchmark before changing engine settings.'))
    if (this.options.comparisonBusy()) return Promise.reject(new Error('Finish or cancel the model comparison before changing local engine settings.'))
    this.pending++
    const next = this.changes.then(() => { this.options.assertAllowed(); return run() }).finally(() => { this.pending-- })
    this.changes = next.catch(() => { })
    return next
  }

  private async setCortheonNow(enabled: boolean): Promise<EngineStatus> {
    assertCortheonReleaseAllowed(enabled)
    await this.options.status()
    if (enabled && !this.options.cortheonSupported()) throw new Error("This build's TensorFold engine does not offer Cortheon yet.")
    if (this.options.learning().busy) throw new Error('Wait for Living Weights to finish before changing Cortheon.')
    if (this.options.preferences.get().cortheon === enabled && (enabled || !this.options.supervisor().status().cortheonActive)) {
      // An explicit Off also clears an older on-disk true without restarting the model.
      if (!enabled) await this.options.preferences.set({ cortheon: false })
      return this.options.statusNow()
    }
    const prior = this.options.preferences.get()
    await this.options.supervisor().reconfigure(async () => { await this.options.preferences.set({ cortheon: enabled }) }, async () => { await this.options.preferences.set(prior) })
    this.options.emitStatus()
    return this.options.statusNow()
  }

  setLearning(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    return this.change(() => this.setLearningNow(enabled, model))
  }

  private async setLearningNow(enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
    await this.options.status()
    if (enabled && !this.options.learningSupported()) throw new Error('Living Weights is unavailable in this build or on this hardware.')
    if (this.options.learning().busy) throw new Error('Wait for the current lesson or model preparation to finish.')
    const prior = this.options.preferences.get()
    const repo = enabled ? assertRepoId(model ?? this.options.supervisor().status().model ?? prior.learningModel ?? '') : prior.learningModel
    if (enabled && repo !== LEARNING_MODEL) throw new Error('Living Weights currently supports only Nemotron 3.5 Lightning MLX 4-bit.')
    if (prior.learningEnabled === enabled && prior.learningModel === repo) return this.options.learning().snapshot()
    const running = this.options.supervisor().status()
    if (enabled && running.model && running.model !== repo && ['ready', 'starting'].includes(running.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
    try {
      await this.options.supervisor().reconfigure(async () => {
        if (enabled) {
          const current = this.options.supervisor().status()
          if (current.model && current.model !== repo && ['ready', 'starting'].includes(current.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
          const learningInfo = await this.options.info(repo!)
          if (learningInfo.memoryFloorBytes > this.options.supervisor().memoryLimitFor(learningInfo.modelType)) throw new Error('This Mac does not have enough model memory for Living Weights on Nemotron. Choose a smaller model for ordinary chat.')
          if (!await this.options.learningCopies().existing(repo!)) {
            let source = await this.options.revisions().selected(repo!)
            if (!source) {
              const cache = repoCacheDir(await fs.realpath(this.options.hub()), repo!)
              const revision = (await fs.readFile(path.join(cache, 'refs', 'main'), 'utf8')).trim()
              source = await this.options.revisions().snapshot(repo!, revision)
            }
            this.options.learning().preparation(0, 0, 'Preparing an independent private model copy; the original download stays untouched.')
            await this.options.learningCopies().prepare(repo!, source, repoCacheDir(this.options.hub(), repo!), (copied, total) => this.options.learning().preparation(copied, total))
          }
        }
        await this.options.preferences.set({ learningEnabled: enabled, learningModel: repo })
      }, async () => { await this.options.preferences.set(prior) })
    } catch (error) {
      this.options.learning().preparation(null, 0, (error as Error).message)
      throw error
    }
    this.options.learning().preparation(null)
    this.options.invalidateInstalled()
    this.options.emitStatus()
    const snapshot = await this.options.learning().snapshot()
    this.options.emit({ kind: 'learning', snapshot })
    return snapshot
  }
}
