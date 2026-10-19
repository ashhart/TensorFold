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
  env: NodeJS.ProcessEnv
  totalMemoryBytes: number
  capabilities: () => Promise<EngineCapabilities>
  installed: () => Promise<EngineInstalledModel[]>
  cacheDir: () => string
  memory: () => { limit: number; budget: number }
}

// A fetch policy and its accepted preflights belong to one immutable endpoint.
export class ModelHub {
  private readonly deps: HubDeps
  private readonly preflights = new Map<string, ModelPreflight>()
  constructor(private readonly options: ModelHubOptions) {
    const endpoint = resolveHubEndpoint(options.endpoint)
    this.deps = Object.freeze({ endpoint, fetch: options.fetch ?? createHubFetch(endpoint),
      offline: /^(1|true|yes|on)$/i.test(options.env.HF_HUB_OFFLINE ?? '') })
  }
  search(query: string) { return searchHub(query, this.deps) }
  cached(key: string): ModelPreflight | undefined { return this.preflights.get(key) }
  invalidate(repo: string): void {
    for (const [key, value] of this.preflights) if (value.repo === repo) this.preflights.delete(key)
  }
  async preflight(repo: string, revision: string | null): Promise<ModelPreflight> {
    assertRepoId(repo)
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place. Scan their folder instead of downloading them.')
    if (revision != null && !isRevision(revision)) throw new Error('Expected a branch, tag or commit for the revision.')
    if (this.deps.offline) throw hubError('offline', 'Hugging Face is offline in this session (HF_HUB_OFFLINE is set).')
    const [caps, installed] = await Promise.all([this.options.capabilities(), this.options.installed()])
    const memory = this.options.memory()
    const result = await preflightRepo(repo, revision, { ...this.deps,
      families: caps.families, familyTitles: FAMILY_TITLES, totalMemoryBytes: this.options.totalMemoryBytes,
      memoryLimitBytes: memory.limit, memoryBudgetBytes: memory.budget,
      installed: new Set(installed.map((m) => m.repo)), drafterFor,
      cachedBytes: (r) => repoCachedBytes(this.options.cacheDir(), r),
      freeDiskBytes: () => freeDiskBytes(this.options.cacheDir()),
    })
    this.preflights.set(JSON.stringify([repo, revision]), result)
    return result
  }
}
