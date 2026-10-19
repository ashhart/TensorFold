import type { EngineCapabilities, EngineInstalledModel, ModelPreflight } from '../types'
import { FAMILY_TITLES, drafterFor } from './catalog'
import { createHubFetch, preflightRepo, searchHub, type HubDeps } from './hub'
import { assertRepoId, freeDiskBytes, isRevision, repoCachedBytes } from './paths'
import { isLocalModelId } from './library'
import { resolveHubEndpoint } from './hub-network'
import { hubError } from '../../../shared/hub-failure'

interface ModelHubOptions {
  endpoint: string
  fetch?: typeof fetch
  // Live launch environment: HF_HUB_OFFLINE is read per request, so a saved or just-saved
  // override governs the next search or preflight without restarting the app.
  env: () => NodeJS.ProcessEnv
  totalMemoryBytes: number
  capabilities: () => Promise<EngineCapabilities>
  installed: () => Promise<EngineInstalledModel[]>
  cacheDir: () => string
  memory: () => { limit: number; budget: number; limitFor?: (modelType: string | null) => { bytes: number; basis: string } }
}

const OFFLINE = /^(1|true|yes|on)$/i

// A fetch policy and its accepted preflights belong to one immutable endpoint.
export class ModelHub {
  private readonly endpoint: string
  private readonly fetchImpl: typeof fetch
  private readonly preflights = new Map<string, ModelPreflight>()
  constructor(private readonly options: ModelHubOptions) {
    this.endpoint = resolveHubEndpoint(options.endpoint)
    this.fetchImpl = options.fetch ?? createHubFetch(this.endpoint)
  }
  private deps(): HubDeps {
    // The endpoint and fetch stay pinned; only the offline decision tracks the current environment.
    return Object.freeze({ endpoint: this.endpoint, fetch: this.fetchImpl, offline: OFFLINE.test(this.options.env().HF_HUB_OFFLINE ?? '') })
  }
  private assertOnline(): void {
    if (OFFLINE.test(this.options.env().HF_HUB_OFFLINE ?? '')) throw hubError('offline', 'Hugging Face is offline in this session (HF_HUB_OFFLINE is set).')
  }
  search(query: string) { this.assertOnline(); return searchHub(query, this.deps()) }
  cached(key: string): ModelPreflight | undefined { return this.preflights.get(key) }
  invalidate(repo: string): void {
    for (const [key, value] of this.preflights) if (value.repo === repo) this.preflights.delete(key)
  }
  // Verdicts frozen at an older memory limit must not survive a limit or policy change.
  clearPreflights(): void { this.preflights.clear() }
  async preflight(repo: string, revision: string | null): Promise<ModelPreflight> {
    assertRepoId(repo)
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place. Scan their folder instead of downloading them.')
    if (revision != null && !isRevision(revision)) throw new Error('Expected a branch, tag or commit for the revision.')
    this.assertOnline()
    const [caps, installed] = await Promise.all([this.options.capabilities(), this.options.installed()])
    const memory = this.options.memory()
    const result = await preflightRepo(repo, revision, { ...this.deps(),
      families: caps.families, familyTitles: FAMILY_TITLES, totalMemoryBytes: this.options.totalMemoryBytes,
      memoryLimitBytes: memory.limit, memoryBudgetBytes: memory.budget, memoryLimitFor: memory.limitFor,
      installed: new Set(installed.map((m) => m.repo)), drafterFor,
      cachedBytes: (r) => repoCachedBytes(this.options.cacheDir(), r),
      freeDiskBytes: () => freeDiskBytes(this.options.cacheDir()),
    })
    this.preflights.set(JSON.stringify([repo, revision]), result)
    return result
  }
}