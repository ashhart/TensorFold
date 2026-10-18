import type { EngineModelGenerationSettings, EngineModelProfile, EngineModelProfileSet, SaveEngineModelProfilesInput, Settings } from '../electron/agent/types'

const fields = ['contextTokens', 'maxOutputTokens', 'temperature', 'topP', 'topK', 'minP', 'speculativeDecoding'] as const
function record(value: unknown): value is Record<string, unknown> { return value != null && typeof value === 'object' && !Array.isArray(value) }
function finite(value: unknown, label: string, min: number, max: number, integer = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || integer && !Number.isSafeInteger(value)) throw new Error(`${label} must be ${integer ? 'a whole number' : 'a number'} between ${min} and ${max}.`)
  return value
}
function text(value: unknown, label: string, max: number): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\x00-\x1f\x7f]/.test(value)) throw new Error(`${label} must contain between 1 and ${max} characters.`)
  return value.trim()
}

/** A whitelist shared by IPC, persistence and the request boundary. No arbitrary server flags. */
export function validateEngineModelGeneration(value: unknown): EngineModelGenerationSettings {
  if (!record(value)) throw new Error('Model settings must be an object.')
  const contextTokens = finite(value.contextTokens, 'Context window', 1024, 1_048_576, true)
  const maxOutputTokens = finite(value.maxOutputTokens, 'Maximum output', 1, contextTokens - 1, true)
  if (value.speculativeDecoding !== 'auto' && value.speculativeDecoding !== 'off') throw new Error('Speculative decoding must be automatic or off.')
  return { contextTokens, maxOutputTokens, temperature: finite(value.temperature, 'Temperature', 0, 2), topP: finite(value.topP, 'Top P', 0.0001, 1), topK: finite(value.topK, 'Top K', 0, 1000, true), minP: finite(value.minP, 'Min P', 0, 1), speculativeDecoding: value.speculativeDecoding }
}

export function validateEngineModelProfileSet(value: unknown): EngineModelProfileSet {
  if (!record(value)) throw new Error('Model profiles must be an object.')
  const model = text(value.model, 'Model', 4096)
  const revision = finite(value.revision, 'Profile revision', 0, Number.MAX_SAFE_INTEGER, true)
  if (!Array.isArray(value.profiles) || value.profiles.length > 32) throw new Error('Keep at most 32 profiles for each model.')
  const ids = new Set<string>(), names = new Set<string>()
  const profiles = value.profiles.map((raw): EngineModelProfile => {
    if (!record(raw)) throw new Error('Each model profile must be an object.')
    const id = text(raw.id, 'Profile ID', 100), name = text(raw.name, 'Profile name', 60)
    if (ids.has(id) || names.has(name.toLocaleLowerCase())) throw new Error('Profile names and IDs must be unique within this model.')
    ids.add(id); names.add(name.toLocaleLowerCase())
    return { id, name, ...validateEngineModelGeneration(raw) }
  })
  const activeProfileId = value.activeProfileId == null ? null : text(value.activeProfileId, 'Active profile', 100)
  if (activeProfileId && !ids.has(activeProfileId)) throw new Error('The selected profile no longer exists. Choose another profile.')
  return { model, revision, activeProfileId, profiles }
}

export function validateEngineModelProfileSets(value: unknown): EngineModelProfileSet[] {
  if (value == null) return []
  if (!Array.isArray(value) || value.length > 256) throw new Error('Model profile storage is invalid.')
  const rows = value.map(validateEngineModelProfileSet)
  if (new Set(rows.map((row) => row.model)).size !== rows.length) throw new Error('Model profiles contain duplicate models.')
  return rows
}

export function activeEngineModelSettings(settings: Pick<Settings, 'engineModelProfiles'>, model: string): EngineModelGenerationSettings | null {
  const set = settings.engineModelProfiles?.find((entry) => entry.model === model)
  if (!set?.activeProfileId) return null
  const valid = validateEngineModelProfileSet(set)
  const profile = valid.profiles.find((entry) => entry.id === valid.activeProfileId)
  return profile ? validateEngineModelGeneration(profile) : null
}

export function updateEngineModelProfiles(settings: Settings, input: SaveEngineModelProfilesInput): Settings {
  const current = validateEngineModelProfileSets(settings.engineModelProfiles)
  const existing = current.find((entry) => entry.model === input?.model)
  if (input?.expectedRevision !== (existing?.revision ?? 0)) throw new Error('These model profiles changed elsewhere. Reopen settings and retry your edit.')
  const next = validateEngineModelProfileSet({ ...input, revision: (existing?.revision ?? 0) + 1 })
  return { ...settings, engineModelProfiles: [...current.filter((entry) => entry.model !== next.model), next] }
}

export function defaultEngineModelProfile(contextTokens: number | null | undefined, temperature = 0.1): EngineModelGenerationSettings {
  const context = typeof contextTokens === 'number' && Number.isSafeInteger(contextTokens) && contextTokens >= 1024 ? Math.min(contextTokens, 32768) : 8192
  return { contextTokens: context, maxOutputTokens: Math.min(2048, Math.floor(context / 4)), temperature: Math.min(2, Math.max(0, temperature)), topP: 1, topK: 0, minP: 0, speculativeDecoding: 'auto' }
}

export function engineModelSettingsEqual(a: EngineModelGenerationSettings | null, b: EngineModelGenerationSettings | null): boolean {
  return a === b || !!a && !!b && fields.every((key) => a[key] === b[key])
}

export function loadEngineModelProfiles(value: unknown): Pick<Settings, 'engineModelProfiles' | 'engineModelProfilesRecovery' | 'engineModelProfilesProblem'> {
  try { return { engineModelProfiles: validateEngineModelProfileSets(value) } }
  catch { return { engineModelProfiles: [], engineModelProfilesRecovery: value, engineModelProfilesProblem: 'Saved model profiles contain invalid values and were not applied. Their original data has been retained for recovery.' } }
}

/** Single-Mac native Metal support in 1.0.12; the GUI never launches --speed-up.
 * Flash Next and GLM-5.3-Flash sample on one Mac, while their two-Mac mode remains
 * greedy-only in the engine. GLM-5.3 EXLR8 still requires greedy decoding. */
export function isGreedyOnlyEngineModel(model: { modelType: string }): boolean {
  return model.modelType === 'glm_moe_dsa'
}
export function assertEngineModelSamplingSupported(model: { modelType: string }, generation: EngineModelGenerationSettings): void {
  if (isGreedyOnlyEngineModel(model) && (generation.temperature !== 0 || generation.topP !== 1 || generation.topK !== 0 || generation.minP !== 0)) throw new Error('This model uses greedy decoding in the bundled engine. Set temperature to 0, Top P to 1, Top K to 0 and Min P to 0.')
}

export function engineProfileContextWindow(profile: EngineModelGenerationSettings, servedWindow?: number | null): number {
  const served = typeof servedWindow === 'number' && Number.isSafeInteger(servedWindow) && servedWindow > 0 ? servedWindow : profile.contextTokens
  return Math.min(profile.contextTokens, served)
}
/** Reserve the actual reply cap, including small windows below the legacy 4k compaction floor. */
export function engineProfileWorkingLimit(profile: EngineModelGenerationSettings, servedWindow?: number | null): number {
  const window = engineProfileContextWindow(profile, servedWindow)
  return Math.max(1, Math.min(Math.floor(window * 0.85), window - profile.maxOutputTokens))
}
