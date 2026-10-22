// Endpoint-reported capacities, scoped to the effective account and connection configuration.
import { createHmac, randomBytes } from 'node:crypto'
import type { Provider, Settings } from './types'
import { recommendedCompactionThreshold } from './model-context-windows'

const windows = new Map<string, number>()
let identitySalt = randomBytes(32)

export const DEFAULT_HOSTED_CLOUD_LIMIT = 128_000
export const DEFAULT_HOSTED_LOCAL_LIMIT = 24_000

// A process-private digest distinguishes accounts and credential edits without storing secrets in cache keys.
export function hostedContextIdentity(provider: Partial<Provider> | null | undefined): string {
  const id = provider?.id?.trim() ?? ''
  if (!id) return ''
  const endpoint = (provider?.baseUrl ?? '').trim().replace(/\/+$/, '')
  const configuration = [id, endpoint, provider?.transport ?? 'chat-completions', provider?.resolvedAccountId ?? null, provider?.apiKey ?? '']
  return createHmac('sha256', identitySalt).update(JSON.stringify(configuration)).digest('hex')
}

function key(identity: string, model: string): string {
  return `${identity}\u0000${model.trim()}`
}

export function recordHostedContext(identity: string | null | undefined, model: string | null | undefined, tokens: number | null | undefined): void {
  if (!identity || !model || typeof tokens !== 'number' || !Number.isSafeInteger(tokens) || tokens <= 0) return
  windows.set(key(identity, model), tokens)
}

// Partial listings preserve reports for unmentioned models within the same configuration.
export function recordHostedContexts(identity: string | null | undefined, models: readonly string[], tokens: Record<string, number>): void {
  if (!identity) return
  for (const model of models) if (tokens[model] !== undefined) recordHostedContext(identity, model, tokens[model])
}

export function hostedContextWindow(identity: string | null | undefined, model: string | null | undefined): number | null {
  if (!identity || !model) return null
  return windows.get(key(identity, model)) ?? null
}

// Drop every model belonging to one effective configuration.
export function forgetHostedContexts(identity: string | null | undefined): void {
  if (!identity) return
  const prefix = `${identity}\u0000`
  for (const recorded of [...windows.keys()]) if (recorded.startsWith(prefix)) windows.delete(recorded)
}

export function clearHostedContexts(): void {
  windows.clear()
  identitySalt = randomBytes(32)
}

// Rotate identities on saved provider edits so late refreshes cannot repopulate the current configuration.
export function invalidateHostedContexts(previous: readonly Provider[], next: readonly Provider[]): void {
  if (JSON.stringify(previous) !== JSON.stringify(next)) clearHostedContexts()
}

// A reported capacity is authoritative, so the guessed-window floor cannot raise its trigger.
export function hostedWorkingTokens(window: number): number {
  if (!Number.isFinite(window) || window <= 0) return 0
  return Math.min(recommendedCompactionThreshold(window), Math.floor(window))
}

// Explicit choices remain caps even when their numeric values equal the initial defaults.
export function explicitContextCaps(
  cloudLimit: number | undefined, localLimit: number | undefined, isLocal: boolean,
  choices: Pick<Settings, 'contextLimitExplicit' | 'localContextLimitExplicit'> = {},
): number[] {
  const caps: number[] = []
  if (typeof cloudLimit === 'number' && cloudLimit > 0 && (choices.contextLimitExplicit || cloudLimit !== DEFAULT_HOSTED_CLOUD_LIMIT)) caps.push(cloudLimit)
  if (isLocal && typeof localLimit === 'number' && localLimit > 0 && (choices.localContextLimitExplicit || localLimit !== DEFAULT_HOSTED_LOCAL_LIMIT)) caps.push(localLimit)
  return caps
}

// Bound the reported trigger by both capacity and explicit choices.
export function hostedTrigger(window: number, caps: readonly number[]): number {
  const working = hostedWorkingTokens(window)
  return caps.length ? Math.min(working, ...caps) : working
}
