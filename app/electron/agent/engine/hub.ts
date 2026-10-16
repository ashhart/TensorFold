import { readBoundedJsonResponse, readBoundedResponseText } from '../bounded-response'
import { createConnectorFetch } from '../secure-fetch'
import type { HubGated, HubModelSummary, ModelPreflight, PreflightVerdict } from '../types'
import { describeFormat, formatName } from './cli'
import { assertRepoId, isRevision } from './paths'
import { formatGiB } from './supervisor'
import { discardResponseBody } from '../pinned-fetch'

// Hugging Face search and a per-repo preflight: is the family served, is the format one the engine
// reads, is the repo gated (pull sends no token), how big is it, does it fit memory and disk.

export const HF_ORIGIN = 'https://huggingface.co'
export const RUNTIME_MARGIN_BYTES = 2 * 1024 ** 3
const DISK_HEADROOM_BYTES = 1024 ** 3
const TIMEOUT_MS = 15_000

export function createHubFetch(): typeof fetch {
  return createConnectorFetch(HF_ORIGIN, 'Hugging Face')
}

export interface HubDeps {
  fetch: typeof fetch
  offline?: boolean
}

function assertOnline(deps: HubDeps): void {
  if (deps.offline) throw new Error('Hugging Face is offline in this session (HF_HUB_OFFLINE is set).')
}

async function getJson(deps: HubDeps, url: string, label: string): Promise<any> {
  const res = await deps.fetch(url, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(TIMEOUT_MS), redirect: 'manual' })
  if (res.status === 401 || res.status === 404) throw new Error(`${label}: not found on Hugging Face, or private.`)
  if (!res.ok) throw new Error(`${label}: Hugging Face answered HTTP ${res.status}.`)
  return readBoundedJsonResponse(res, label, 8 * 1024 * 1024)
}

function gatedValue(value: unknown): HubGated {
  return value === 'auto' || value === 'manual' ? value : value === true ? 'manual' : false
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function parseSearchResults(body: unknown): HubModelSummary[] {
  if (!Array.isArray(body)) throw new Error('Hugging Face search returned an unexpected shape.')
  const out: HubModelSummary[] = []
  for (const item of body) {
    const repo = typeof item?.id === 'string' ? item.id : typeof item?.modelId === 'string' ? item.modelId : ''
    if (!repo || !/^[^/\s]+\/[^/\s]+$/.test(repo)) continue
    out.push({
      repo,
      downloads: numberOrNull(item.downloads),
      likes: numberOrNull(item.likes),
      lastModified: typeof item.lastModified === 'string' ? item.lastModified : null,
      gated: gatedValue(item.gated),
      modelType: typeof item.config?.model_type === 'string' ? item.config.model_type : null,
      pipelineTag: typeof item.pipeline_tag === 'string' ? item.pipeline_tag : null,
      libraryName: typeof item.library_name === 'string' ? item.library_name : null,
    })
  }
  return out
}

export async function searchHub(query: string, deps: HubDeps, limit = 30): Promise<HubModelSummary[]> {
  assertOnline(deps)
  const q = String(query ?? '').trim()
  if (!q) return []
  if (q.length > 200) throw new Error('Search text is too long.')
  const params = new URLSearchParams({ search: q, limit: String(Math.min(Math.max(1, limit), 50)), sort: 'downloads', direction: '-1' })
  for (const field of ['downloads', 'likes', 'lastModified', 'gated', 'config', 'pipeline_tag', 'library_name']) params.append('expand', field)
  return parseSearchResults(await getJson(deps, `${HF_ORIGIN}/api/models?${params}`, 'Hugging Face search'))
}

function nextLink(header: string | null): string | null {
  const match = /<([^>]+)>;\s*rel="next"/.exec(header ?? '')
  if (!match) return null
  const url = new URL(match[1], HF_ORIGIN)
  return url.origin === HF_ORIGIN ? url.toString() : null
}

// Sum of file sizes at one revision; follows the tree API's Link pagination on huggingface.co only.
export async function treeTotal(deps: HubDeps, repo: string, revision: string): Promise<{ bytes: number; files: number }> {
  let url: string | null = `${HF_ORIGIN}/api/models/${assertRepoId(repo)}/tree/${encodeURIComponent(revision)}?recursive=true`
  let bytes = 0
  let files = 0
  for (let page = 0; url && page < 20; page++) {
    const res: Response = await deps.fetch(url, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(TIMEOUT_MS), redirect: 'manual' })
    if (!res.ok) throw new Error(`${repo}: the file list is unavailable (HTTP ${res.status}).`)
    const next = nextLink(res.headers.get('link'))
    const body = await readBoundedJsonResponse(res, `${repo} file list`, 8 * 1024 * 1024)
    if (!Array.isArray(body)) throw new Error(`${repo}: the file list has an unexpected shape.`)
    for (const entry of body) {
      if (entry?.type !== 'file') continue
      files++
      bytes += Math.max(0, Number(entry.lfs?.size ?? entry.size) || 0)
    }
    url = next
  }
  return { bytes, files }
}

async function readConfig(deps: HubDeps, repo: string, revision: string): Promise<Record<string, any> | null> {
  try {
    let url = new URL(`${HF_ORIGIN}/${repo}/resolve/${encodeURIComponent(revision)}/config.json`)
    const signal = AbortSignal.timeout(TIMEOUT_MS)
    for (let hop = 0; hop <= 5; hop++) {
      const res = await deps.fetch(url.href, { signal, redirect: 'manual' })
      if ([301, 302, 303, 307, 308].includes(res.status)) {
        const location = res.headers.get('location')
        await discardResponseBody(res)
        if (!location || hop === 5) return null
        const next = new URL(location, url)
        if (next.origin !== HF_ORIGIN || next.username || next.password) return null
        url = next
        continue
      }
      if (res.status !== 200) { await discardResponseBody(res); return null }
      const doc = JSON.parse(await readBoundedResponseText(res, `${repo} config.json`, 4 * 1024 * 1024))
      return doc && typeof doc === 'object' && !Array.isArray(doc) ? doc : null
    }
    return null
  } catch {
    return null
  }
}

export interface PreflightContext extends HubDeps {
  families: Record<string, string[]>
  familyTitles?: Record<string, string>
  totalMemoryBytes: number
  memoryBudgetBytes?: number
  memoryLimitBytes: number
  installed: Set<string>
  drafterFor: (repo: string) => string | null
  cachedBytes: (repo: string) => Promise<number>
  freeDiskBytes: () => Promise<number | null>
  now?: () => number
}

export async function preflightRepo(repo: string, revision: string | null, ctx: PreflightContext): Promise<ModelPreflight> {
  assertOnline(ctx)
  assertRepoId(repo)
  const rev = revision ?? 'main'
  if (!isRevision(rev)) throw new Error('Expected a branch, tag or commit for the revision.')
  const info = await getJson(ctx, `${HF_ORIGIN}/api/models/${repo}/revision/${encodeURIComponent(rev)}`, repo)
  if (typeof info?.sha !== 'string' || !/^[0-9a-f]{40}$/.test(info.sha)) throw new Error('The Hub did not resolve the model to an immutable commit. Retry the model checks before downloading.')
  const sha: string = info.sha
  const gated = gatedValue(info?.gated)
  const [tree, config] = await Promise.all([treeTotal(ctx, repo, sha), gated ? Promise.resolve(null) : readConfig(ctx, repo, sha)])
  const modelType: string | null = typeof config?.model_type === 'string' ? config.model_type : typeof info?.config?.model_type === 'string' ? info.config.model_type : null
  const formats = modelType ? ctx.families[modelType] : undefined
  const supported = !!formats && !config?.dflash_config
  const format = config ? formatName(config) : null
  const formatOk = supported && config && formats!.length ? !!format && formats!.includes(format) : null

  const drafterRepo = supported ? ctx.drafterFor(repo) : null
  let drafterRevision: string | null = null
  const drafterTree = drafterRepo ? await (async () => {
    const metadata = await getJson(ctx, `${HF_ORIGIN}/api/models/${drafterRepo}/revision/main`, drafterRepo)
    if (typeof metadata?.sha !== 'string' || !/^[0-9a-f]{40}$/.test(metadata.sha) || gatedValue(metadata.gated)) return null
    drafterRevision = metadata.sha
    return treeTotal(ctx, drafterRepo, drafterRevision!)
  })().catch(() => null) : null
  const totalBytes = tree.bytes + (drafterTree?.bytes ?? 0)
  const memoryFloorBytes = tree.bytes ? totalBytes + RUNTIME_MARGIN_BYTES : null
  const fitsMemory = memoryFloorBytes == null ? null : memoryFloorBytes <= ctx.memoryLimitBytes
  const cachedBytes = Math.min(totalBytes, (await ctx.cachedBytes(repo)) + (drafterRepo ? await ctx.cachedBytes(drafterRepo) : 0))
  const neededDiskBytes = Math.max(0, totalBytes - cachedBytes)
  const freeDiskBytes = await ctx.freeDiskBytes()
  const fitsDisk = freeDiskBytes == null ? null : freeDiskBytes >= neededDiskBytes + DISK_HEADROOM_BYTES
  const installed = ctx.installed.has(repo)

  const reasons: string[] = []
  if (!modelType) reasons.push('Its config.json names no model_type, so no engine family can be matched.')
  else if (config?.dflash_config) reasons.push('This is a DFlash2 drafter; it downloads with the model it drafts for.')
  else if (!supported) reasons.push(`The built-in engine serves no ${modelType} family yet.`)
  if (formatOk === false) reasons.push(`The engine reads ${formats!.map(describeFormat).join(' or ')} for this family; this checkpoint is ${describeFormat(format)}.`)
  if (supported && !gated && formatOk == null) reasons.push('Its config.json format could not be verified; retry the model checks before downloading.')
  if (gated) reasons.push('The repo is gated, and the engine downloads without a Hugging Face token.')
  if (fitsMemory === false) reasons.push(`It needs about ${formatGiB(memoryFloorBytes!)} of memory; the limit on this Mac is ${formatGiB(ctx.memoryLimitBytes)} (70% of ${formatGiB(ctx.memoryBudgetBytes ?? ctx.totalMemoryBytes)}${(ctx.memoryBudgetBytes ?? ctx.totalMemoryBytes) < ctx.totalMemoryBytes ? ' qualification budget' : ''}).`)
  if (fitsDisk === false) reasons.push(`It needs ${formatGiB(neededDiskBytes)} more disk space; ${formatGiB(freeDiskBytes!)} is free.`)
  if (drafterRepo && (!drafterTree?.bytes || !drafterRevision)) reasons.push(`Its drafter ${drafterRepo} could not be resolved and sized at an immutable commit.`)

  let verdict: PreflightVerdict = 'ok'
  if (!supported) verdict = 'unsupported'
  else if (formatOk === false) verdict = 'format'
  else if (gated) verdict = 'gated'
  else if (formatOk !== true) verdict = 'error'
  else if (drafterRepo && (!drafterTree?.bytes || !drafterRevision)) verdict = 'error'
  else if (fitsMemory === false) verdict = 'too-big'
  else if (installed) verdict = 'installed'
  else if (fitsDisk === false) verdict = 'disk'

  return {
    repo,
    revision: revision ?? null,
    resolvedRevision: sha,
    verdict,
    reasons,
    modelType,
    family: modelType && supported ? ctx.familyTitles?.[modelType] ?? modelType : null,
    supported,
    format,
    formatOk,
    gated,
    totalBytes: tree.bytes ? totalBytes : null,
    fileCount: tree.files,
    drafter: drafterRepo ? { repo: drafterRepo, totalBytes: drafterTree?.bytes ?? null, resolvedRevision: drafterRevision } : null,
    memoryFloorBytes,
    memoryLimitBytes: ctx.memoryLimitBytes,
    fitsMemory,
    freeDiskBytes,
    neededDiskBytes,
    fitsDisk,
    cachedBytes,
    installed,
    checkedAt: (ctx.now ?? Date.now)(),
  }
}
