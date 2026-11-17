import type { EngineInstalledModel, EngineEvent, EngineModelGenerationSettings, EngineModelLibrarySnapshot, Readiness } from '../types'
import { FAMILY_TITLES, SUGGESTED_MODELS, drafterFor } from './catalog'
import { parseInfoOutput, parseModelsOutput, runEngineCli, type ModelInfo } from './cli'
import { isLocalModelId, type ModelLibrary } from './library'
import type { ModelRevisions } from './model-revisions'
import { assertRepoId, hasCachedSnapshot, repoCachedBytes, type EngineBinary } from './paths'
import type { EnginePreferenceStore } from './preferences'
import type { LearningCopies } from './learning-copy'
import type { EngineSupervisor } from './supervisor'

// What the engine can load: cached/downloaded checkpoints and linked library checkpoints, the
// argument each one is launched with, its drafter companion, and whether it fits this Mac.

export interface InstalledModelsDeps {
  binary: () => EngineBinary | null
  requireBinary: () => EngineBinary
  env: () => NodeJS.ProcessEnv
  hub: () => string
  library: ModelLibrary
  revisions: ModelRevisions
  preferences: EnginePreferenceStore
  learningCopies: LearningCopies
  modelSettings: (repo: string) => EngineModelGenerationSettings | null
  memoryLimitFor: EngineSupervisor['memoryLimitFor']
  // Whether a download for this repo is queued or running.
  isPulling: (repo: string) => boolean
  emit: (event: EngineEvent) => void
  onInstalled?: (repos: string[]) => void
}

export class InstalledModels {
  private cache: EngineInstalledModel[] | null = null
  private readonly localRepos = new Map<string, string>()

  constructor(private readonly deps: InstalledModelsDeps) {}

  invalidate(): void {
    this.cache = null
  }

  // The canonical repo a library checkpoint resolved to, once it has been resolved.
  canonical(id: string): string | undefined {
    return this.localRepos.get(id)
  }

  async info(repo: string): Promise<ModelInfo> {
    const result = await runEngineCli(this.deps.requireBinary(), ['info', await this.argument(repo)], this.deps.env())
    if (result.code !== 0) throw new Error((result.stdout || result.stderr).trim().split(/\r?\n/)[0] || `tensorfold-native info ${repo} failed.`)
    return parseInfoOutput(result.stdout)
  }

  async argument(repo: string): Promise<string> {
    assertRepoId(repo)
    if (isLocalModelId(repo)) {
      const local = await this.deps.library.resolve(repo)
      if (local.repo && (this.deps.isPulling(local.repo) || this.deps.isPulling(this.requiredDrafter(repo, local.repo) ?? ''))) throw new Error('Wait for the model download to finish before loading it.')
      if (local.repo) this.localRepos.set(repo, local.repo)
      return local.path
    }
    if (this.deps.isPulling(repo)) throw new Error('Wait for the model download to finish before loading it.')
    if (repo === this.deps.preferences.get().learningModel) {
      const owned = await this.deps.learningCopies.existing(repo)
      if (!owned) throw new Error('The private Living Weights model is missing; preserve its profile before repairing it.')
      return owned
    }
    const selected = await this.deps.revisions.selected(repo)
    if (selected) return selected
    if (!await hasCachedSnapshot(this.deps.hub(), repo)) {
      const local = await this.localDrafter(repo)
      if (local) return local.path
    }
    return repo
  }

  async cachedBytes(repo: string): Promise<number | null> {
    try {
      const selected = await this.deps.revisions.selected(repo)
      if (selected) return this.deps.revisions.snapshotBytes(selected)
      if (await hasCachedSnapshot(this.deps.hub(), repo)) return repoCachedBytes(this.deps.hub(), repo)
      return (await this.localDrafter(repo))?.weightsBytes ?? null
    } catch { return null }
  }

  requiredDrafter(model: string, canonical = this.localRepos.get(model) ?? model): string | null {
    return this.deps.modelSettings(model)?.speculativeDecoding === 'off' ? null : drafterFor(canonical)
  }

  private async localDrafter(repo: string) {
    // Resolve only a known companion identity, never an arbitrary repo against a nearby folder.
    if (!SUGGESTED_MODELS.some((model) => drafterFor(model.repo) === repo)) return null
    const entries = (await this.deps.library.snapshot()).models.filter((model) => model.repo === repo && model.drafterOnly && model.complete)
    if (entries.length !== 1) return null
    return this.deps.library.resolve(entries[0].id, true)
  }

  async list(refresh = false): Promise<EngineInstalledModel[]> {
    if (this.cache && !refresh) return this.cache
    if (!this.deps.binary()) return []
    const listed = await runEngineCli(this.deps.requireBinary(), ['models'], this.deps.env())
    const selected = await this.deps.revisions.read()
    const models = new Map(parseModelsOutput(listed.stdout).map((model) => [model.repo, model]))
    for (const repo of Object.keys(selected)) if (!models.has(repo)) models.set(repo, { repo, family: '', modelType: '' })
    const out: EngineInstalledModel[] = []
    for (const model of models.values()) {
      const drafter = this.requiredDrafter(model.repo)
      const drafterBytes = drafter ? await this.cachedBytes(drafter) : null
      const entry: EngineInstalledModel = { ...model, source: selected[model.repo] ? 'download' : 'cache', format: null, contextTokens: null, weightsBytes: null, memoryFloorBytes: null, fitsMemory: null, drafter, drafterInstalled: drafter ? drafterBytes != null : null }
      try {
        const info = await this.info(model.repo)
        entry.family = info.family
        entry.modelType = info.modelType
        entry.format = info.format
        entry.contextTokens = info.contextTokens
        entry.weightsBytes = info.weightsBytes
        entry.memoryFloorBytes = info.memoryFloorBytes + (drafterBytes ?? 0)
        entry.fitsMemory = entry.memoryFloorBytes <= this.deps.memoryLimitFor(info.modelType)
      } catch {
        if (selected[model.repo]) continue // Never present a different legacy ref as the selected model.
        /* Legacy listed but unreadable: keep the row without sizes. */
      }
      out.push(entry)
    }
    for (const local of (await this.deps.library.snapshot()).models.filter((model) => model.loadable)) {
      try {
        const checked = await this.deps.library.resolve(local.id)
        const drafter = checked.repo ? this.requiredDrafter(local.id, checked.repo) : null
        const drafterBytes = drafter ? await this.cachedBytes(drafter) : null
        const memoryFloorBytes = checked.memoryFloorBytes == null ? null : checked.memoryFloorBytes + (drafterBytes ?? 0)
        if (checked.repo) this.localRepos.set(local.id, checked.repo)
        out.push({ repo: local.id, source: 'library', title: local.title, localPath: checked.path, family: FAMILY_TITLES[checked.modelType!] ?? checked.modelType!, modelType: checked.modelType!, format: checked.format, contextTokens: null, weightsBytes: checked.weightsBytes, memoryFloorBytes, fitsMemory: memoryFloorBytes == null ? null : memoryFloorBytes <= this.deps.memoryLimitFor(checked.modelType), drafter, drafterInstalled: drafter ? drafterBytes != null : null })
      } catch { /* Moved or changed checkpoints are unavailable until scanned again. */ }
    }
    this.cache = out
    this.deps.emit({ kind: 'installed', installed: out })
    this.deps.onInstalled?.(out.map((m) => m.repo))
    return out
  }

  async readiness(model: string | null): Promise<Readiness> {
    if (!this.deps.binary()) return { ready: false, reason: 'needs-endpoint', message: 'The TensorFold engine is missing from this build.' }
    const installed = await this.list().catch(() => [])
    if (!installed.length) return { ready: false, reason: 'needs-model', message: 'Download a model for the built-in TensorFold engine in Models.' }
    const entry = model ? installed.find((m) => m.repo === model) : installed[0]
    if (!entry) return { ready: false, reason: 'needs-model', message: `${model} is not downloaded. Open Models to get it.` }
    if (entry.source === 'library') {
      try { await this.deps.library.resolve(entry.repo) } catch (error) { return { ready: false, reason: 'needs-model', message: (error as Error).message } }
      if (entry.drafter && await this.cachedBytes(entry.drafter) == null) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Scan the companion checkpoint again in Models.` }
    }
    if (entry.fitsMemory === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs more memory than this Mac allows. Pick a smaller model in Models.` }
    if (entry.drafter && entry.drafterInstalled === false) return { ready: false, reason: 'needs-model', message: `${entry.repo} needs its drafter ${entry.drafter}. Download the model again in Models.` }
    return { ready: true, reason: 'ok', message: '' }
  }

  async librarySnapshot(): Promise<EngineModelLibrarySnapshot> {
    const snapshot = await this.deps.library.snapshot()
    for (const model of snapshot.models) {
      const drafter = model.repo ? this.requiredDrafter(model.id, model.repo) : null
      if (!drafter || !model.loadable) continue
      const bytes = await this.cachedBytes(drafter)
      if (bytes == null) { model.loadable = false; model.reason = `This checkpoint needs its drafter ${drafter}. Download the full model in Models or scan the companion checkpoint too.` }
      else if (model.memoryFloorBytes != null) {
        model.memoryFloorBytes += bytes
        model.fitsMemory = model.memoryFloorBytes <= this.deps.memoryLimitFor(model.modelType)
        if (!model.fitsMemory) { model.loadable = false; model.reason = 'This model and its drafter need more model memory than the current hardware allows.' }
      }
    }
    return snapshot
  }

  // After the library's folders or scan results change: relist, then answer with the fresh library.
  async libraryChanged(): Promise<EngineModelLibrarySnapshot> {
    this.cache = null
    await this.list(true)
    return this.librarySnapshot()
  }
}
