import type { Provider, ProviderDiscovery } from './types'

// Pure helpers for local-provider discovery (task #45).
//
// Auto-save on field blur + an in-flight discovery fetch used to race: a blur
// would commit the draft while discovery was still running, and when discovery
// finished it always APPENDED a fresh provider — so rapid blurs + retries could
// spawn ~5 duplicate oMLX entries. These helpers make "find the matching
// provider and update it, otherwise add one" a single, testable operation.

/**
 * True when a base URL points at this machine (loopback). Auto-discovery only
 * works against the local runtime config, so the Discover button is hidden for
 * remote/LAN endpoints like 192.168.x.x (task #46).
 */
export function isLoopbackEndpoint(baseUrl: string | undefined | null): boolean {
  const value = (baseUrl ?? '').trim()
  if (!value) return false
  let host: string
  try {
    host = new URL(/^[a-z]+:\/\//i.test(value) ? value : `http://${value}`).hostname
  } catch {
    return false
  }
  host = host.replace(/^\[|\]$/g, '').toLowerCase()
  return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '0.0.0.0' || host.endsWith('.localhost')
}

/** True when two providers refer to the same local endpoint. */
export function sameLocalEndpoint(a: Pick<Provider, 'baseUrl' | 'source'>, b: Pick<Provider, 'baseUrl' | 'source'>): boolean {
  return normalizeBaseUrl(a.baseUrl) === normalizeBaseUrl(b.baseUrl) && (a.source ?? null) === (b.source ?? null)
}

function normalizeBaseUrl(value: string | undefined | null): string {
  return (value ?? '').trim().replace(/\/+$/, '').toLowerCase()
}

/**
 * Locate an existing provider that a discovery result should merge into.
 * Prefers an explicit id match, then falls back to baseUrl + source equality so
 * repeated discovery of the same node updates rather than duplicates.
 */
export function findDiscoveryTarget(
  providers: Provider[],
  discovered: Pick<Provider, 'baseUrl' | 'source'>,
  providerId?: string,
): Provider | null {
  if (providerId) {
    const byId = providers.find((p) => p.id === providerId)
    if (byId) return byId
  }
  return providers.find((p) => sameLocalEndpoint(p, discovered)) ?? null
}

/**
 * Merge a discovery result into the provider list. When a matching provider
 * exists (by id, or by baseUrl + source) it is updated in place; otherwise the
 * built provider is appended exactly once. This is the dedup guard for #45.
 */
export function mergeDiscoveredProvider(
  providers: Provider[],
  built: Provider,
  providerId?: string,
): { providers: Provider[]; target: Provider; created: boolean } {
  const existing = findDiscoveryTarget(providers, built, providerId)
  if (existing) {
    const merged: Provider = { ...built, id: existing.id, name: existing.name?.trim() || built.name }
    return {
      providers: providers.map((p) => (p.id === existing.id ? merged : p)),
      target: merged,
      created: false,
    }
  }
  return { providers: [...providers, built], target: built, created: true }
}

/** Build the Provider record a discovery result describes (without dedup). */
export function buildDiscoveredProvider(
  discovered: Pick<ProviderDiscovery, 'name' | 'source' | 'baseUrl' | 'apiKey' | 'models'>,
  id: string,
  existing?: Provider | null,
): Provider {
  return {
    ...(existing ?? { id, name: discovered.name, transport: 'chat-completions', auth: 'api-key', models: [], kind: 'local', baseUrl: '', apiKey: '' }),
    id,
    name: existing?.name?.trim() || discovered.name,
    transport: 'chat-completions',
    auth: 'api-key',
    kind: 'local',
    source: discovered.source,
    baseUrl: discovered.baseUrl,
    apiKey: discovered.apiKey,
    models: discovered.models.length ? discovered.models : (existing?.models?.length ? existing.models : ['local-model']),
  }
}
