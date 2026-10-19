import { discoverOmlx } from './omlx'
import { createLocalProviderFetch } from './secure-fetch'
import type { Provider, ProviderDiscovery, ProviderDiscoveryKind } from './types'
import { readBoundedJsonResponse } from './bounded-response'

const LOCAL_TIMEOUT_MS = 1800

function uniqueSorted(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map((value) => String(value || '').trim()).filter(Boolean))).sort()
}

function idsFromModelsPayload(json: any): string[] {
  const data: any[] = json?.data ?? json?.models ?? (Array.isArray(json) ? json : [])
  return uniqueSorted(data.map((model) => (typeof model === 'string' ? model : model?.id ?? model?.name ?? model?.model)))
}

async function fetchJson(url: string, apiKey = '', timeoutMs = LOCAL_TIMEOUT_MS, fetchImpl: typeof fetch = fetch): Promise<{ status: number; ok: boolean; json: any; statusText: string }> {
  try {
    const res = await fetchImpl(url, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: AbortSignal.timeout(timeoutMs),
    })
    let json: any = null
    try { json = await readBoundedJsonResponse(res, 'Local provider models') } catch { json = null }
    return { status: res.status, ok: res.ok, json, statusText: res.statusText || '' }
  } catch (e: any) {
    // Node's fetch throws a generic "fetch failed"; the real reason (e.g. connect ECONNREFUSED host:port,
    // ENOTFOUND, timeout) is on e.cause — surface it so the user sees an actionable message.
    return { status: 0, ok: false, json: null, statusText: String(e?.cause?.message || e?.cause?.code || e?.message || e || 'unreachable') }
  }
}

function originFromBaseUrl(baseUrl: string, fallback: string): string {
  try {
    const url = new URL(baseUrl)
    return url.origin
  } catch {
    return fallback
  }
}

export async function fetchOllamaModels(baseUrl = 'http://127.0.0.1:11434/v1', fetchImpl: typeof fetch = fetch): Promise<string[]> {
  fetchImpl = createLocalProviderFetch(baseUrl, fetchImpl)
  const origin = originFromBaseUrl(baseUrl, 'http://127.0.0.1:11434')
  const native = await fetchJson(`${origin}/api/tags`, '', LOCAL_TIMEOUT_MS, fetchImpl)
  if (native.ok) return idsFromModelsPayload(native.json)
  const compat = await fetchJson(`${baseUrl.replace(/\/+$/, '')}/models`, 'ollama', LOCAL_TIMEOUT_MS, fetchImpl)
  if (compat.ok) return idsFromModelsPayload(compat.json)
  if (!native.status && !compat.status) throw new Error(`Couldn't reach Ollama at ${baseUrl} — ${compat.statusText || native.statusText || 'connection refused'}. Is \`ollama serve\` running on that host?`)
  throw new Error(`Ollama HTTP ${native.status || compat.status}`)
}

export async function fetchLmStudioModels(baseUrl = 'http://127.0.0.1:1234/v1', apiKey = '', fetchImpl: typeof fetch = fetch): Promise<string[]> {
  fetchImpl = createLocalProviderFetch(baseUrl, fetchImpl)
  const compat = await fetchJson(`${baseUrl.replace(/\/+$/, '')}/models`, apiKey, LOCAL_TIMEOUT_MS, fetchImpl)
  if (compat.ok) return idsFromModelsPayload(compat.json)
  const origin = originFromBaseUrl(baseUrl, 'http://127.0.0.1:1234')
  const native = await fetchJson(`${origin}/api/v0/models`, apiKey, LOCAL_TIMEOUT_MS, fetchImpl)
  if (native.ok) return idsFromModelsPayload(native.json)
  if (compat.status === 401 || native.status === 401) throw new Error('LM Studio is running but requires an API token.')
  if (!compat.status && !native.status) throw new Error(`Couldn't reach LM Studio at ${baseUrl} — ${compat.statusText || 'connection refused'}. Check the server is running and "Serve on Local Network" is enabled, plus the host/port.`)
  throw new Error(`LM Studio HTTP ${compat.status || native.status}`)
}

export async function fetchOpenAiCompatibleModels(baseUrl: string, apiKey = '', timeoutMs = LOCAL_TIMEOUT_MS, fetchImpl: typeof fetch = fetch): Promise<string[]> {
  fetchImpl = createLocalProviderFetch(baseUrl, fetchImpl)
  const compat = await fetchJson(`${baseUrl.replace(/\/+$/, '')}/models`, apiKey, timeoutMs, fetchImpl)
  if (compat.ok) return idsFromModelsPayload(compat.json)
  if (!compat.status) throw new Error(`Couldn't reach ${baseUrl} — ${compat.statusText || 'connection refused'}.`)
  throw new Error(`Provider models HTTP ${compat.status || compat.statusText || 'unreachable'}`)
}

async function discoverOllama(): Promise<ProviderDiscovery> {
  const baseUrl = 'http://127.0.0.1:11434/v1'
  try {
    const models = await fetchOllamaModels(baseUrl)
    return {
      kind: 'ollama',
      found: true,
      name: 'Ollama (local)',
      source: 'ollama',
      baseUrl,
      apiKey: 'ollama',
      models,
      defaultModel: models[0] ?? null,
      configPath: null,
      message: models.length ? `Discovered Ollama with ${models.length} model${models.length === 1 ? '' : 's'}.` : 'Discovered Ollama, but no models are installed.',
    }
  } catch (e: any) {
    return {
      kind: 'ollama',
      found: false,
      name: 'Ollama (local)',
      source: 'ollama',
      baseUrl,
      apiKey: 'ollama',
      models: [],
      defaultModel: null,
      configPath: null,
      message: String(e?.message || 'Ollama was not reachable on port 11434.'),
    }
  }
}

async function discoverLmStudio(): Promise<ProviderDiscovery> {
  const baseUrl = 'http://127.0.0.1:1234/v1'
  try {
    const models = await fetchLmStudioModels(baseUrl)
    return {
      kind: 'lmstudio',
      found: true,
      name: 'LM Studio (local)',
      source: 'lmstudio',
      baseUrl,
      apiKey: 'lm-studio',
      models,
      defaultModel: models[0] ?? null,
      configPath: null,
      message: models.length ? `Discovered LM Studio with ${models.length} model${models.length === 1 ? '' : 's'}.` : 'Discovered LM Studio, but no models are loaded or downloaded.',
    }
  } catch (e: any) {
    const message = String(e?.message || 'LM Studio was not reachable on port 1234.')
    return {
      kind: 'lmstudio',
      found: message.includes('requires an API token'),
      name: 'LM Studio (local)',
      source: 'lmstudio',
      baseUrl,
      apiKey: '',
      models: [],
      defaultModel: null,
      configPath: null,
      message,
    }
  }
}

async function discoverOmlxProvider(): Promise<ProviderDiscovery> {
  const discovered = await discoverOmlx()
  return {
    kind: 'omlx',
    found: discovered.found,
    name: 'oMLX (local)',
    source: 'omlx',
    baseUrl: discovered.baseUrl,
    apiKey: discovered.apiKey,
    models: discovered.models,
    defaultModel: discovered.defaultModel,
    configPath: discovered.configPath,
    message: discovered.message,
  }
}

function isLoopbackHost(baseUrl: string): boolean {
  try {
    const host = new URL(baseUrl).hostname.replace(/^\[|\]$/g, '').toLowerCase()
    return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '0.0.0.0' || host.endsWith('.localhost')
  } catch { return false }
}

// Probe a remote (LAN) OpenAI-compatible node's /v1/models directly. The local
// ~/.omlx config (and the fixed localhost ports for Ollama/LM Studio) can't
// describe a node on another machine — e.g. a Mac Studio at 192.168.x.x — so when
// the user points the provider at a non-loopback host we ask that node itself.
async function discoverRemoteOpenAiCompatible(kind: ProviderDiscoveryKind, baseUrl: string, apiKey: string): Promise<ProviderDiscovery> {
  const name = kind === 'omlx' ? 'oMLX (network)' : kind === 'ollama' ? 'Ollama (network)' : kind === 'custom' ? 'TensorFold / OpenAI-compatible (network)' : 'LM Studio (network)'
  const base = { kind, name, source: kind === 'custom' ? 'custom' as const : kind, baseUrl, apiKey, configPath: null }
  try {
    const models = await fetchOpenAiCompatibleModels(baseUrl, apiKey)
    return { ...base, found: models.length > 0, models, defaultModel: models[0] ?? null, message: models.length ? `Found ${models.length} model${models.length === 1 ? '' : 's'} at ${baseUrl}.` : `Reached ${baseUrl} but it reported no models.` }
  } catch (e: any) {
    return { ...base, found: false, models: [], defaultModel: null, message: `Could not reach ${baseUrl}: ${e?.message ?? e}` }
  }
}

// Any OpenAI-compatible endpoint, local or networked: a TensorFold engine, vLLM, llama.cpp, a proxy.
async function discoverCustomProvider(baseUrl: string, apiKey: string): Promise<ProviderDiscovery> {
  const base = { kind: 'custom' as const, name: 'TensorFold / OpenAI-compatible', source: 'custom' as const, baseUrl, apiKey, configPath: null }
  try {
    const models = await fetchOpenAiCompatibleModels(baseUrl, apiKey)
    return { ...base, found: models.length > 0, models, defaultModel: models[0] ?? null, message: models.length ? `Found ${models.length} model${models.length === 1 ? '' : 's'} at ${baseUrl}.` : `Reached ${baseUrl} but it reported no models.` }
  } catch (e: any) {
    return { ...base, found: false, models: [], defaultModel: null, message: `Could not reach ${baseUrl}: ${e?.message ?? e}` }
  }
}

export async function discoverLocalProvider(kind: ProviderDiscoveryKind, baseUrl?: string, apiKey = ''): Promise<ProviderDiscovery> {
  // A non-loopback base URL means a networked node — probe it directly instead of
  // the local runtime config (which only knows about this machine).
  const target = (baseUrl ?? '').trim()
  if (kind === 'custom') return discoverCustomProvider(target || 'http://127.0.0.1:8080/v1', apiKey)
  if (target && !isLoopbackHost(target)) return discoverRemoteOpenAiCompatible(kind, target, apiKey)
  if (kind === 'ollama') return discoverOllama()
  if (kind === 'lmstudio') return discoverLmStudio()
  return discoverOmlxProvider()
}

export function isOllamaProvider(provider: Provider): boolean {
  return provider.source === 'ollama' || /(^|\b)ollama(\b|$)/i.test(provider.name) || /:11434(?:\/|$)/.test(provider.baseUrl)
}

export function isLmStudioProvider(provider: Provider): boolean {
  return provider.source === 'lmstudio' || /lm\s*studio/i.test(provider.name) || /:1234(?:\/|$)/.test(provider.baseUrl)
}

export function isOpenRouterProvider(provider: Provider): boolean {
  if (provider.source === 'openrouter') return true
  try {
    return new URL(provider.baseUrl).hostname.endsWith('openrouter.ai')
  } catch {
    return false
  }
}
