import { activeEngineModelSettings, engineProfileWorkingLimit } from '../../../shared/model-profiles'
import type { EngineModelGenerationSettings, EngineModelProfileSet, Provider } from '../../agent/types'
import { getModelContextWindow, recommendedCompactionThreshold } from '../../agent/model-context-windows'
import { store } from '../state'
import { isEngineProvider } from '../../agent/engine/provider'
import { engineContextWindow } from '../../agent/engine/context'

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
  modelSettings?: EngineModelGenerationSettings | null
  engineModelProfiles?: EngineModelProfileSet[]
}

/**
 * Task #41: model-aware compaction trigger.
 *
 * Pure core for {@link effectiveContextLimit}. Resolves the token threshold at
 * which a transcript should compact for a given provider + model, scaling to the
 * model's *real* context window when we know it:
 *
 *  - Cloud providers: min(userCloudSetting, recommendedTrigger(realWindow)).
 *    With the default 128k setting a Sonnet thread still caps at 128k (backward
 *    compatible); raise the setting to 256k and it scales up to the model's
 *    170k recommended trigger instead of an unreachable 256k.
 *  - Local providers: also folds in the local setting and the legacy name-based
 *    inference so a tiny 4k local model compacts early.
 *
 * An explicit `override` is honoured verbatim but still clamped to the model's
 * recommended trigger so a user can never set a limit the model can't reach.
 * Unknown models keep the prior configured behaviour (no real window -> no scaling).
 */
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
  const cloudLimit = inputs.cloudLimit || 128_000
  const realWindow = getModelContextWindow(model)
  const modelTrigger = realWindow != null ? recommendedCompactionThreshold(realWindow) : null

  if (inputs.override && inputs.override > 0) {
    if (modelTrigger != null) return Math.min(inputs.override, modelTrigger)
    // Unknown window: a local model still needs its safety net, or a large override (e.g. 128k) lets
    // the transcript overflow a tiny local context window. Clamp to the local limit.
    if (isLocalProvider(provider)) return Math.min(inputs.override, inputs.localLimit || 24_000)
    return inputs.override
  }

  if (!isLocalProvider(provider)) {
    return modelTrigger != null ? Math.min(cloudLimit, modelTrigger) : cloudLimit
  }

  const localLimit = inputs.localLimit || 24_000
  const candidates = [cloudLimit, localLimit, inferredLocalContextTrigger(model)]
  if (modelTrigger != null) candidates.push(modelTrigger)
  return Math.min(...candidates)
}

export function effectiveContextLimit(provider: Provider, model: string | null): number {
  return computeContextLimit(provider, model, {
    cloudLimit: store.settings.contextLimit,
    localLimit: store.settings.localContextLimit,
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
