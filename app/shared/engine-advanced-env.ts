// Advanced engine environment variables: the values a user passes through to the
// bundled `tensorfold-native` process (load-limit raises, offline mode, engine knobs).
// Shared so the main process validates exactly what the settings UI offers.

export type EngineEnvOverrides = Record<string, string>

const KEY_PATTERN = /^[A-Z][A-Z0-9_]{0,63}$/
const MAX_OVERRIDES = 50
const MAX_VALUE_LENGTH = 2048

// The app or the OS owns these; overriding them from settings would fight the
// launcher (spawn order, credentials) or the process loader (dynamic linker).
const DENIED_KEYS = new Set(['HOME', 'PATH', 'SHELL', 'USER', 'LOGNAME', 'HF_ENDPOINT', 'TENSORFOLD_API_KEY', 'TENSORFOLD_NO_UPDATE_CHECK'])

export function isDeniedEngineEnvKey(key: string): boolean {
  return DENIED_KEYS.has(key) || /^(DYLD_|LD_|NODE_OPTIONS|ELECTRON_)/.test(key) || /(_API_KEY|_TOKEN|_SECRET|_PASSWORD)$/.test(key)
}

function denied(key: string): never {
  throw new Error(`${key} is managed by TensorFold or the system and cannot be set here.`)
}

// Strict normalisation at the IPC boundary: an object of UPPER_SNAKE names to plain
// single-line strings. Anything else is a caller bug or a hostile renderer.
export function normalizeEngineEnvOverrides(value: unknown): EngineEnvOverrides {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a set of environment variables.')
  const entries = Object.entries(value as Record<string, unknown>)
  if (entries.length > MAX_OVERRIDES) throw new Error(`Set at most ${MAX_OVERRIDES} variables.`)
  const next: EngineEnvOverrides = {}
  for (const [key, raw] of entries) {
    if (!KEY_PATTERN.test(key)) throw new Error('Variable names use letters, digits and underscores, and start with a letter.')
    if (isDeniedEngineEnvKey(key)) denied(key)
    if (typeof raw !== 'string' || !raw.length) throw new Error(`${key} needs a value.`)
    if (raw.length > MAX_VALUE_LENGTH || raw.includes('\0')) throw new Error(`${key} needs a value of at most ${MAX_VALUE_LENGTH} characters.`)
    next[key] = raw
  }
  return next
}

// Lenient read of a persisted map: keep only entries the strict validator would accept,
// so a hand-edited preferences file can never poison the engine environment.
export function readEngineEnvOverrides(value: unknown): EngineEnvOverrides {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) return {}
  const next: EngineEnvOverrides = {}
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!KEY_PATTERN.test(key) || isDeniedEngineEnvKey(key) || typeof raw !== 'string' || !raw.length || raw.length > MAX_VALUE_LENGTH || raw.includes('\0')) continue
    next[key] = raw
  }
  return next
}

// The engine reads one load-limit variable per big-model family: GB above 0 replaces its
// default 70%-of-RAM load limit. The app-side memory floor check must not veto a model
// the user has deliberately allowed, so it honours the same variables.
export const ENGINE_LOAD_LIMIT_KEYS = ['GLM_LOAD_LIMIT_GB', 'FLASHNEXT_LOAD_LIMIT_GB'] as const

export interface EngineLoadLimitOverride {
  key: string
  gb: number
  bytes: number
}

const GIB = 1024 ** 3

export function engineLoadLimitOverride(env: Record<string, string | undefined>): EngineLoadLimitOverride | null {
  let best: EngineLoadLimitOverride | null = null
  for (const key of ENGINE_LOAD_LIMIT_KEYS) {
    const raw = env[key]
    if (typeof raw !== 'string') continue
    const gb = Number(raw)
    if (!Number.isFinite(gb) || gb <= 0) continue
    const bytes = Math.floor(gb * GIB)
    if (!best || bytes > best.bytes) best = { key, gb, bytes }
  }
  return best
}

// Known knobs offered as quick-add chips. The first two are the reported pain point;
// the rest are the engine's own declared environment (see `capabilities --json`).
export const ENGINE_ENV_PRESETS: Array<{ key: string; placeholder: string; desc: string }> = [
  { key: 'GLM_LOAD_LIMIT_GB', placeholder: '100', desc: 'GB cap for loading GLM-5.3, replacing the 70% of RAM limit. Also raises the app-side memory check.' },
  { key: 'FLASHNEXT_LOAD_LIMIT_GB', placeholder: '100', desc: 'GB cap for loading Flash Next models, replacing the 70% of RAM limit. Also raises the app-side memory check.' },
  { key: 'HF_HUB_OFFLINE', placeholder: '1', desc: 'Set 1 to keep the engine off the network; model downloads need the connection back on.' },
  { key: 'HF_HOME', placeholder: '/Volumes/Models/hf', desc: 'Where Hugging Face caches downloads. TensorFold keeps its own model folders in Settings → Model folders.' },
  { key: 'TENSORFOLD_REQUEST_LOG', placeholder: '/tmp/engine-requests.log', desc: 'Engine knob: append per-request records to a file for diagnostics.' },
  { key: 'TENSORFOLD_NO_LIVE', placeholder: '1', desc: 'Engine knob: disable live features of the served model.' },
  { key: 'TENSORFOLD_SEED_SALT', placeholder: 'salt', desc: 'Engine knob: mix a salt into request seeds.' },
]