import type { Settings } from './types'

export interface AssistantModelSelection {
  providerId: string | null
  providerAccountId: string | null
  model: string | null
}

type AssistantModelSettings = Pick<
  Settings,
  'assistantProviderId' | 'assistantProviderAccountId' | 'assistantModel' | 'activeProviderId' | 'activeProviderAccountId' | 'activeModel'
>

// Per-surface defaults (VST-015): each launch surface can pin its own provider/model.
// 'chat' is the Assistant surface (summaries, reply drafting); 'code' is new coding tasks.
export type ModelSurface = 'chat' | 'code'

type SurfaceModelSettings = AssistantModelSettings & Pick<Settings, 'codeProviderId' | 'codeProviderAccountId' | 'codeModel'>

/** True when a distinct Assistant model is configured (vs. inheriting the active model). */
export function hasAssistantModelOverride(settings: Pick<Settings, 'assistantProviderId' | 'assistantModel'>): boolean {
  return !!(settings.assistantProviderId && settings.assistantModel)
}

/** True when the surface has its own complete provider+model override configured. */
export function hasSurfaceModelOverride(settings: Partial<SurfaceModelSettings>, surface: ModelSurface): boolean {
  return surface === 'chat'
    ? !!(settings.assistantProviderId && settings.assistantModel)
    : !!(settings.codeProviderId && settings.codeModel)
}

/**
 * The model a surface should launch with: its own dedicated provider/model when configured,
 * otherwise the globally-active model. A partial override (provider without model, or vice
 * versa) is NOT an override — falling back beats launching a provider with no model.
 * NOTE: duplicated main-side as surfaceModelOverride (electron/agent/provider-accounts.ts);
 * the parity test in assistant-model.test.ts pins the two to identical precedence.
 */
export function resolveSurfaceModel(settings: SurfaceModelSettings, surface: ModelSurface): AssistantModelSelection {
  if (hasSurfaceModelOverride(settings, surface)) {
    return surface === 'chat'
      ? { providerId: settings.assistantProviderId ?? null, providerAccountId: settings.assistantProviderAccountId ?? null, model: settings.assistantModel ?? null }
      : { providerId: settings.codeProviderId ?? null, providerAccountId: settings.codeProviderAccountId ?? null, model: settings.codeModel ?? null }
  }
  return {
    providerId: settings.activeProviderId,
    providerAccountId: settings.activeProviderAccountId,
    model: settings.activeModel,
  }
}

/**
 * The model the Assistant surface should use — the chat half of resolveSurfaceModel, kept
 * under its original name for existing callers.
 */
export function resolveAssistantModel(settings: AssistantModelSettings): AssistantModelSelection {
  return resolveSurfaceModel({ ...settings, codeProviderId: null, codeProviderAccountId: null, codeModel: null }, 'chat')
}

export function settingsForSurfaceModelPick<T extends SurfaceModelSettings>(settings: T, surface: ModelSurface, selection: AssistantModelSelection): T {
  if (hasSurfaceModelOverride(settings, surface)) return surface === 'code'
    ? { ...settings, codeProviderId: selection.providerId, codeProviderAccountId: selection.providerAccountId, codeModel: selection.model }
    : { ...settings, assistantProviderId: selection.providerId, assistantProviderAccountId: selection.providerAccountId, assistantModel: selection.model }
  return { ...settings, activeProviderId: selection.providerId, activeProviderAccountId: selection.providerAccountId, activeModel: selection.model }
}
