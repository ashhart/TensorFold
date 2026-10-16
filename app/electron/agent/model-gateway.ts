import type { Provider } from './types'
import { readBoundedJsonResponse } from './bounded-response'

export const AI_GATEWAY_BASE_URL = 'https://ai-gateway.vercel.sh/v1'

export const AI_GATEWAY_DEFAULT_MODELS = [
  'openai/gpt-5.4',
  'openai/gpt-5.5',
  'openai/gpt-4.1',
  'openai/o4-mini',
  'anthropic/claude-sonnet-4.6',
  'anthropic/claude-opus-4.1',
  'xai/grok-4.1-fast-reasoning',
  'google/gemini-2.5-pro',
]

export function isAiGatewayProvider(provider: Pick<Provider, 'source' | 'name' | 'baseUrl'>): boolean {
  if (provider.source === 'ai-gateway') return true
  const text = `${provider.name ?? ''} ${provider.baseUrl ?? ''}`.toLowerCase()
  if (/ai gateway|vercel gateway/.test(text)) return true
  try {
    return new URL(provider.baseUrl).hostname === 'ai-gateway.vercel.sh'
  } catch {
    return false
  }
}

function idsFromModelPayload(payload: unknown): string[] {
  const data = (payload as any)?.data ?? (payload as any)?.models ?? (Array.isArray(payload) ? payload : [])
  if (!Array.isArray(data)) return []
  return data.map((model) => (typeof model === 'string' ? model : model?.id ?? model?.name)).filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
}

export function normalizeGatewayModels(models: string[]): string[] {
  return Array.from(new Set([...AI_GATEWAY_DEFAULT_MODELS, ...models])).filter(Boolean)
}

export async function fetchAiGatewayModels(provider: Provider, fetchImpl: typeof fetch = fetch): Promise<string[]> {
  const baseUrl = (provider.baseUrl || AI_GATEWAY_BASE_URL).replace(/\/+$/, '')
  const res = await fetchImpl(`${baseUrl}/models`, {
    headers: provider.apiKey ? { Authorization: `Bearer ${provider.apiKey}` } : {},
    signal: AbortSignal.timeout(8000),
  })
  if (!res.ok) {
    if ([404, 405, 501].includes(res.status)) return AI_GATEWAY_DEFAULT_MODELS
    throw new Error(`HTTP ${res.status} ${res.statusText || ''}`.trim())
  }
  const payload = await readBoundedJsonResponse(res, 'AI Gateway models')
  return normalizeGatewayModels(idsFromModelPayload(payload))
}
