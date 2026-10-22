import { providerWithAccount } from '../../agent/provider-accounts'
import { activeEngineModelSettings, engineProfileWorkingLimit } from '../../../shared/model-profiles'
import type { EngineModelGenerationSettings, EngineModelProfileSet, Provider } from '../../agent/types'
import { getModelContextWindow, recommendedCompactionThreshold } from '../../agent/model-context-windows'
import { store } from '../state'
import { isEngineProvider } from '../../agent/engine/provider'
import { engineContextWindow } from '../../agent/engine/context'
import { DEFAULT_HOSTED_CLOUD_LIMIT, DEFAULT_HOSTED_LOCAL_LIMIT, explicitContextCaps, hostedContextIdentity, hostedContextWindow, hostedTrigger } from '../../agent/hosted-context'

export function parseJsonLoose(s: string): any {
  try { const m = s.match(/\{[\s\S]*\}/); return JSON.parse(m ? m[0] : s) } catch { return null }
}

function isLocalProvider(provider: Provider): boolean {
  if (provider.kind === 'local') return true
  return /(^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$))/.test(provider.baseUrl)
}

function inferredLocalContextTrigger(model: string | null): number {
  const m = (model ?? '').toLowerCase()
  if (/\b(4k|4096)\b/.test(m)) return 3_000
  if (/\b(8k|8192)\b/.test(m)) return 6_000
  if (/\b(16k|16384)\b/.test(m)) return 12_000
  if (/\b(32k|32768)\b/.test(m)) return 24_000
  if (/\b(64k|65536)\b/.test(m)) return 48_000
  return 24_000
}

export interface ContextLimitInputs {
  /** User-configured cloud compaction threshold (store.settings.contextLimit). */
  cloudLimit: number
  /** User-configured local compaction threshold (store.settings.localContextLimit). */
  localLimit: number
  /** Optional explicit per-run override (store.settings.contextLimitOverride). */
  override?: number
  contextLimitExplicit?: boolean
  localContextLimitExplicit?: boolean
  modelSettings?: EngineModelGenerationSettings | null
  engineModelProfiles?: EngineModelProfileSet[]
}

// Resolve a bounded trigger from engine profiles, reported capacities, explicit caps, or fallback guesses.
export function computeContextLimit(provider: Provider, model: string | null, inputs: ContextLimitInputs): number {
  // The built-in engine reports its served window once a model is loaded; that replaces the local guess.
  const engineWindow = isEngineProvider(provider) ? engineContextWindow(model) : null
  const selected = inputs.modelSettings ?? (isEngineProvider(provider) && model ? activeEngineModelSettings({ engineModelProfiles: inputs.engineModelProfiles }, model) : null)
  if (isEngineProvider(provider) && selected) {
    const working = engineProfileWorkingLimit(selected, engineWindow)
    return inputs.override && inputs.override > 0 ? Math.min(inputs.override, working) : working
  }
  if (engineWindow != null) {
    const trigger = recommendedCompactionThreshold(engineWindow)
    return inputs.override && inputs.override > 0 ? Math.min(inputs.override, trigger) : trigger
  }
  // A hosted provider's models endpoint reported the window (vLLM's max_model_len); that beats the
  // guesses too, but not the user's own settings: an explicit context limit still caps it.
  const hostedWindow = hostedContextWindow(hostedContextIdentity(provider), model)
  if (hostedWindow != null) {
    const trigger = hostedTrigger(hostedWindow, explicitContextCaps(inputs.cloudLimit, inputs.localLimit, isLocalProvider(provider), inputs))
    return inputs.override && inputs.override > 0 ? Math.min(inputs.override, trigger) : trigger
  }
  const cloudLimit = inputs.cloudLimit || DEFAULT_HOSTED_CLOUD_LIMIT
  const realWindow = getModelContextWindow(model)
  const modelTrigger = realWindow != null ? recommendedCompactionThreshold(realWindow) : null

  if (inputs.override && inputs.override > 0) {
    if (modelTrigger != null) return Math.min(inputs.override, modelTrigger)
    // Unknown window: a local model still needs its safety net, or a large override (e.g. 128k) lets
    // the transcript overflow a tiny local context window. Clamp to the local limit.
    if (isLocalProvider(provider)) return Math.min(inputs.override, inputs.localLimit || DEFAULT_HOSTED_LOCAL_LIMIT)
    return inputs.override
  }

  if (!isLocalProvider(provider)) {
    return modelTrigger != null ? Math.min(cloudLimit, modelTrigger) : cloudLimit
  }

  const localLimit = inputs.localLimit || DEFAULT_HOSTED_LOCAL_LIMIT
  const candidates = [cloudLimit, localLimit, inferredLocalContextTrigger(model)]
  if (modelTrigger != null) candidates.push(modelTrigger)
  return Math.min(...candidates)
}

export function effectiveContextLimit(provider: Provider, model: string | null, accountId?: string | null): number {
  const selected = accountId === undefined
    ? provider.resolvedAccountId === undefined
      ? (provider.id === store.settings.activeProviderId ? (store.settings.activeProviderAccountId ?? provider.defaultAccountId) : provider.defaultAccountId)
      : provider.resolvedAccountId
    : accountId
  const effective = provider.resolvedAccountId === selected ? provider : providerWithAccount(provider, selected)
  return computeContextLimit(effective, model, {
    cloudLimit: store.settings.contextLimit,
    localLimit: store.settings.localContextLimit,
    contextLimitExplicit: store.settings.contextLimitExplicit,
    localContextLimitExplicit: store.settings.localContextLimitExplicit,
    override: store.settings.contextLimitOverride,
    modelSettings: isEngineProvider(provider) && model ? activeEngineModelSettings(store.settings, model) : null,
  })
}

/**
 * Task #41: multi-model room context floor. In a room with mixed models the
 * whole room should compact to the *lowest* seat's effective limit so no seat
 * silently overflows. Returns 0 for an empty seat list (no floor / compaction off).
 */
export function roomContextLimit(
  seats: Array<{ provider: Provider; model: string | null }>,
  inputs: ContextLimitInputs,
): number {
  if (!seats.length) return 0
  return Math.min(...seats.map((seat) => computeContextLimit(seat.provider, seat.model, inputs)))
}

export { getModelContextWindow }
