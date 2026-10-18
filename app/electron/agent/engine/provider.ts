import type { ChatParams } from '../openai'
import type { ModelCallSample } from '../turn-stats'
import { validateEngineModelGeneration } from '../../../shared/model-profiles'
import type { EngineModelGenerationSettings, Provider } from '../types'

// The built-in provider and the hook streamChat uses to reach the running engine. Kept free of
// Electron and child-process imports so the agent loop and its tests can load it.

export const ENGINE_PROVIDER_ID = 'tensorfold'
// Replaced per request by the live loopback endpoint; nothing is ever sent here.
export const ENGINE_PLACEHOLDER_URL = 'http://127.0.0.1/v1'
// A cold first request can prefill a long prompt before any token streams.
export const ENGINE_FIRST_BYTE_MS = 10 * 60_000
export const ENGINE_IDLE_MS = 3 * 60_000

export function builtinEngineProvider(): Provider {
  return {
    id: ENGINE_PROVIDER_ID,
    name: 'TensorFold (built in)',
    transport: 'chat-completions',
    auth: 'api-key',
    baseUrl: ENGINE_PLACEHOLDER_URL,
    apiKey: '',
    models: [],
    kind: 'local',
    source: 'tensorfold',
  }
}

export function isEngineProvider(provider: Pick<Provider, 'source'> | null | undefined): boolean {
  return provider?.source === 'tensorfold'
}

// Settings migration: the built-in provider is added once (after the user's own list), a deletion after
// that sticks, and it is never repointed; only its installed-model list is kept from saved settings.
export function withBuiltinEngineProvider(providers: Provider[], alreadySeeded = false): Provider[] {
  const builtin = builtinEngineProvider()
  const index = providers.findIndex((p) => p.id === ENGINE_PROVIDER_ID)
  if (index < 0) return alreadySeeded ? providers : [...providers, builtin]
  const kept = { ...builtin, models: providers[index].models ?? [] }
  return providers.filter((p, i) => p.id !== ENGINE_PROVIDER_ID || i === index).map((p, i) => (i === index ? kept : p))
}

export interface LiveEndpoint {
  model: string
  baseUrl: string
  apiKey: string
  release?: () => void
  generation?: EngineModelGenerationSettings | null
  greedyOnly?: boolean
}

export type EngineServeHook = (model: string, options: { start: boolean; signal?: AbortSignal; exclusive?: boolean; owner?: symbol }) => Promise<LiveEndpoint>

let serveHook: EngineServeHook | null = null
let performanceRecorder: ((model: string, sample: ModelCallSample) => void) | null = null

export function setEngineServeHook(hook: EngineServeHook | null): void {
  serveHook = hook
}

export function setEnginePerformanceRecorder(recorder: ((model: string, sample: ModelCallSample) => void) | null): void {
  performanceRecorder = recorder
}

export function recordEnginePerformance(model: string, sample: ModelCallSample): void {
  // An optional UI observer must never fail an otherwise completed model request.
  try { performanceRecorder?.(model, sample) } catch { /* observer only */ }
}

export interface EngineRoute {
  params: ChatParams
  fetch: typeof fetch
  firstByteMs: number
  idleMs: number
  release?: () => void
}

function raceAbort(promise: Promise<LiveEndpoint>, signal?: AbortSignal): Promise<LiveEndpoint> {
  if (!signal) return promise
  if (signal.aborted) {
    void promise.then((value) => value.release?.(), () => {})
    return Promise.reject(signal.reason ?? new Error('Aborted.'))
  }
  return new Promise<LiveEndpoint>((resolve, reject) => {
    let aborted = false
    const onAbort = () => { aborted = true; reject(signal.reason ?? new Error('Aborted.')) }
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => { signal.removeEventListener('abort', onAbort); if (aborted) value.release?.(); else resolve(value) },
      (error) => { signal.removeEventListener('abort', onAbort); reject(error) },
    )
  })
}

// Loopback-only fetch for the running engine. undici's own header and body timeouts are raised to
// the first-byte budget; streamChat's idle timer is the finer limit.
export function engineFetch(baseUrl: string, assertRequestAllowed?: () => void): typeof fetch {
  const origin = new URL(baseUrl).origin
  if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(origin)) throw new Error('The built-in engine only listens on 127.0.0.1.')
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
    if (url.origin !== origin) throw new Error('Refusing a built-in engine request to another origin.')
    assertRequestAllowed?.()
    const { Agent } = require('undici') as { Agent: new (options: Record<string, unknown>) => { close(): Promise<void>; destroy(): Promise<void> } }
    const dispatcher = new Agent({ headersTimeout: ENGINE_FIRST_BYTE_MS, bodyTimeout: ENGINE_FIRST_BYTE_MS })
    try {
      const res = await fetch(input, { ...(init ?? {}), redirect: 'error', dispatcher } as RequestInit)
      void dispatcher.close().catch(() => {})
      return res
    } catch (error) {
      void dispatcher.destroy().catch(() => {})
      throw error
    }
  }) as typeof fetch
}

// streamChat's one engine hook: null for every other provider. For the built-in provider it asks the
// supervisor to serve p.model (starting or switching the engine unless `start` is false) and returns
// params pointed at the live endpoint, its fetch, and the longer stream limits.
export async function engineRoute(p: ChatParams, options: { start: boolean } = { start: true }): Promise<EngineRoute | null> {
  if (!isEngineProvider(p.provider)) return null
  if (!serveHook) throw new Error('The built-in TensorFold engine is not available in this process.')
  if (p.signal?.aborted) throw p.signal.reason ?? new Error('Aborted.')
  const live = await raceAbort(serveHook(p.model, { ...options, signal: p.signal, ...(p.modelPreview ? { exclusive: true } : {}), ...(p.modelComparisonOwner ? { owner: p.modelComparisonOwner } : {}) }), p.signal)
  try {
    const provider = { ...p.provider!, baseUrl: live.baseUrl, apiKey: live.apiKey }
    return {
      params: { ...p, provider, baseUrl: live.baseUrl, apiKey: live.apiKey, modelSettings: live.generation ? validateEngineModelGeneration(live.generation) : undefined, greedyOnly: live.greedyOnly === true },
      fetch: engineFetch(live.baseUrl, p.assertRequestAllowed),
      firstByteMs: ENGINE_FIRST_BYTE_MS,
      idleMs: ENGINE_IDLE_MS,
      release: live.release,
    }
  } catch (error) { live.release?.(); throw error }
}
