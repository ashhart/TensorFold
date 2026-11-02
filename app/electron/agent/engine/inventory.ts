import type { EngineEvent, EngineInstalledModel, EngineModelLibrarySnapshot, Readiness, EngineModelGenerationSettings } from '../types'
import { FAMILY_TITLES, SUGGESTED_MODELS, drafterFor } from './catalog'
import { parseInfoOutput, parseModelsOutput, runEngineCli, type ModelInfo } from './cli'
import { assertRepoId, hasCachedSnapshot, repoCachedBytes, type EngineBinary } from './paths'
import { ModelRevisions } from './model-revisions'
import { EnginePreferenceStore } from './preferences'
import { LearningCopies } from './learning-copy'
import { isLocalModelId, ModelLibrary } from './library'

interface EngineInventoryOptions {
  binary: () => EngineBinary | null
  requireBinary: () => EngineBinary
  env: () => NodeJS.ProcessEnv
  hub: () => string
  runtimeReady?: () => Promise<void>
  runtimeEpoch: () => number
  modelSettings?: (repo: string) => EngineModelGenerationSettings | null
  library: ModelLibrary
  revisions: ModelRevisions
  preferences: EnginePreferenceStore
  learningCopies: LearningCopies
  hasPull: (repo: string) => boolean
  memoryLimitFor: (modelType?: string | null) => number
  emit: (event: EngineEvent) => void
  onInstalled?: (repos: string[]) => void
}

export class EngineInventory {
  private cache: EngineInstalledModel[] | null = null
  private readonly canonical = new Map<string, string>()
  constructor(private options: EngineInventoryOptions) { }
  invalidate(): void { this.cache = null }
  resolvedRepo(repo: string): string | undefined { return this.canonical.get(repo) }
  async info(repo: string): Promise<ModelInfo> {
    const result = await runEngineCli(this.options.requireBinary(), ['info', await this.modelArgument(repo)], this.options.env())
    if (result.code !== 0) throw new Error((result.stdout || result.stderr).trim().split(/\r?\n/)[0] || `tensorfold-native info ${repo} failed.`)
    return parseInfoOutput(result.stdout)
  }

  async modelArgument(repo: string): Promise<string> {
    assertRepoId(repo)
    if (isLocalModelId(repo)) {
      const local = await this.options.library.resolve(repo)
      if (local.repo && (this.options.hasPull(local.repo) || this.options.hasPull(this.requiredDrafter(repo, local.repo) ?? ''))) throw new Error('Wait for the model download to finish before loading it.')
      if (local.repo) this.canonical.set(repo, local.repo)
      return local.path
    }
    if (this.options.hasPull(repo)) throw new Error('Wait for the model download to finish before loading it.')
    if (repo === this.options.preferences.get().learningModel) {
      const owned = await this.options.learningCopies.existing(repo)
      if (!owned) throw new Error('The private Living Weights model is missing; preserve its profile before repairing it.')
      return owned
    }
    const selected = await this.options.revisions.selected(repo)
    if (selected) return selected
    if (!await hasCachedSnapshot(this.options.hub(), repo)) {
      const local = await this.localDrafter(repo)
      if (local) return local.path
    }
    return repo
  }

  async cachedModelBytes(repo: string): Promise<number | null> {
    try {
      const selected = await this.options.revisions.selected(repo)
      if (selected) return this.options.revisions.snapshotBytes(selected)
      if (await hasCachedSnapshot(this.options.hub(), repo)) return repoCachedBytes(this.options.hub(), repo)
      return (await this.localDrafter(repo))?.weightsBytes ?? null
    } catch { return null }
  }

  requiredDrafter(model: string, canonical = this.canonical.get(model) ?? model): string | null {
    return this.options.modelSettings?.(model)?.speculativeDecoding === 'off' ? null : drafterFor(canonical)
  }

  private async localDrafter(repo: string) {
    // Resolve only a known companion identity, never an arbitrary repo against a nearby folder.
    if (!SUGGESTED_MODELS.some((model) => drafterFor(model.repo) === repo)) return null
    const entries = (await this.options.library.snapshot()).models.filter((model) => model.repo === repo && model.drafterOnly && model.complete)
    if (entries.length !== 1) return null
    return this.options.library.resolve(entries[0].id, true)
  }

  async installed(refresh = false): Promise<EngineInstalledModel[]> {
    await this.options.runtimeReady?.()
    const epoch = this.options.runtimeEpoch()
    if (this.cache && !refresh) return this.cache
    if (!this.options.binary()) return []
    const listed = await runEngineCli(this.options.requireBinary(), ['models'], this.options.env())
    const selected = await this.options.revisions.read()
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
        entry.fitsMemory = entry.memoryFloorBytes <= this.options.memoryLimitFor(info.modelType)
      } catch {
        if (selected[model.repo]) continue
      }
      out.push(entry)
    }
    for (const local of (await this.options.library.snapshot()).models.filter((model) => model.loadable)) {
      try {
        const checked = await this.options.library.resolve(local.id)
        const drafter = checked.repo ? this.requiredDrafter(local.id, checked.repo) : null
        const drafterBytes = drafter ? await this.cachedModelBytes(drafter) : null
        const memoryFloorBytes = checked.memoryFloorBytes == null ? null : checked.memoryFloorBytes + (drafterBytes ?? 0)
        if (checked.repo) this.canonical.set(local.id, checked.repo)
        out.push({ repo: local.id, source: 'library', title: local.title, localPath: checked.path, family: FAMILY_TITLES[checked.modelType!] ?? checked.modelType!, modelType: checked.modelType!, format: checked.format, contextTokens: null, weightsBytes: checked.weightsBytes, memoryFloorBytes, fitsMemory: memoryFloorBytes == null ? null : memoryFloorBytes <= this.options.memoryLimitFor(checked.modelType), drafter, drafterInstalled: drafter ? drafterBytes != null : null })
      } catch { }
    }
    if (epoch !== this.options.runtimeEpoch()) return this.installed(true)
    this.cache = out
    this.options.emit({ kind: 'installed', installed: out })
    this.options.onInstalled?.(out.map((m) => m.repo))
    return out
  }

  async readiness(model: string | null): Promise<Readiness> {
    if (!this.options.binary()) return { ready: false, reason: 'needs-endpoint', message: 'The TensorFold engine is missing from this build.' }
    const installed = await this.installed().catch(() => [])
    if (!installed.length) return { ready: false, reason: 'needs-model', message: 'Download a model for the built-in TensorFold engine in Models.' }
    const entry = model ? installed.find((m) => m.repo === model) : installed[0]
    if (!entry) return { ready: false, reason: 'needs-model', message: `${model} is not downloaded. Open Models to get it.` }
    if (entry.source === 'library') {
      try { await this.options.library.resolve(entry.repo) } catch (error) { return { ready: false, reason: 'needs-model', message: (error as Error).message } }
      if (entry.drafter && await this.cachedModelBytes(entry.drafter) == null) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Scan the companion checkpoint again in Models.` }
    }
    if (entry.fitsMemory === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs more memory than this Mac allows. Pick a smaller model in Models.` }
    if (entry.drafter && entry.drafterInstalled === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Download the model again in Models.` }
    return { ready: true, reason: 'ok', message: '' }
  }

  async librarySnapshot(): Promise<EngineModelLibrarySnapshot> {
    const snapshot = await this.options.library.snapshot()
    for (const model of snapshot.models) {
      const drafter = model.repo ? this.requiredDrafter(model.id, model.repo) : null
      if (!drafter || !model.loadable) continue
      const bytes = await this.cachedModelBytes(drafter)
      if (bytes == null) { model.loadable = false; model.reason = `This checkpoint needs its drafter ${drafter}. Download the full model in Models or scan the companion checkpoint too.` }
      else if (model.memoryFloorBytes != null) {
        model.memoryFloorBytes += bytes
        model.fitsMemory = model.memoryFloorBytes <= this.options.memoryLimitFor(model.modelType)
        if (!model.fitsMemory) { model.loadable = false; model.reason = 'This model and its drafter need more model memory than the current hardware allows.' }
      }
    }
    return snapshot
  }
}
