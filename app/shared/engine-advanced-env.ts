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
    if (/[\r\n]/.test(raw)) throw new Error(`${key} needs a single-line value.`)
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
    if (Object.keys(next).length >= MAX_OVERRIDES) break
    if (!KEY_PATTERN.test(key) || isDeniedEngineEnvKey(key) || typeof raw !== 'string' || !raw.length || raw.length > MAX_VALUE_LENGTH || raw.includes('\0') || /[\r\n]/.test(raw)) continue
    next[key] = raw
  }
  return next
}

// The engine reads one load-limit variable per big-model family. The native code
// (zig/src/families/glm/load_plan.zig, zig/src/native/cache_fit.zig) applies each variable only
// to its own family's checkpoints, chosen by the config.json model_type, never the repository name.
// The app-side admission floor must not veto a model the user has deliberately allowed for its family.
export const ENGINE_LOAD_LIMIT_KEYS = ['GLM_LOAD_LIMIT_GB', 'FLASHNEXT_LOAD_LIMIT_GB'] as const

// glm5_next is GLM-5.3-Flash on this Mac and glm_moe_dsa its expert-parallel sibling; qwen4_exp
// is Qwen3.8 Flash Next (zig/src/cli/hub.zig family list). Checkpoints choose their loader by these
// config.json model_type values, so the app maps variables the same way.
const LOAD_LIMIT_FAMILIES: Record<(typeof ENGINE_LOAD_LIMIT_KEYS)[number], string[]> = {
  GLM_LOAD_LIMIT_GB: ['glm5_next', 'glm_moe_dsa'],
  FLASHNEXT_LOAD_LIMIT_GB: ['qwen4_exp'],
}

export interface EngineLoadLimitOverride {
  key: string
  gb: number
  bytes: number
  // True when the request reaches or exceeds the byte cap (the native engine caps further at its GPU working set).
  capped: boolean
}

// Mirrors the native parse (std.fmt.parseFloat over a space/tab trim, verified against Zig 0.17):
// decimal GB, above zero, finite. parseFloat reads decimal exponents and underscore digit
// separators, hex floats (0x64 is 100; the exponent after p is decimal digits), and the
// case-insensitive inf/infinity/nan specials, but not trailing junk ("12GB") — the specials
// and junk both fall out as invalid or non-finite below. An underscore is only valid between
// two digits of the same digit run, so "1_", "1_.5" and "100__0" are refused; empty or
// whitespace-only text means "unset" and keeps the 70% default. Native measures GB as decimal
// 1e9 and caps the request at min(RAM, GPU working set); capBytes carries the app-visible RAM
// bound, so a value like 1e308 cannot overflow the byte result.
const DIGITS = String.raw`\d+(?:_\d+)*`
const HEX_DIGITS = String.raw`[0-9a-fA-F]+(?:_[0-9a-fA-F]+)*`
const LOAD_LIMIT_TEXT = new RegExp(
  `^[+-]?(?:` +
  // Decimal float: digits with optional fraction and an e/E exponent.
  String.raw`(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:[eE][+-]?${DIGITS})?|` +
  // Hex float: 0x mantissa (hex digits), optional fraction, optional p/P exponent (decimal digits —
  // parseScientific scans base 10 even in hex, so 0x1p1e is refused while 0x1e2 reads as hex 482).
  String.raw`0[xX](?:${HEX_DIGITS}(?:\.(?:${HEX_DIGITS})?)?|\.${HEX_DIGITS})(?:[pP][+-]?${DIGITS})?|` +
  // Specials, case-insensitive.
  String.raw`[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN]` +
  `)$`,
)

export function engineLoadLimitOverride(env: Record<string, string | undefined>, modelType: string | null, capBytes: number): EngineLoadLimitOverride | null {
  const key = modelType ? ENGINE_LOAD_LIMIT_KEYS.find((candidate) => LOAD_LIMIT_FAMILIES[candidate].includes(modelType)) : undefined
  if (!key) return null
  const raw = env[key]
  if (typeof raw !== 'string') return null
  // Native trims exactly " \t"; other whitespace (a trailing newline, say) leaves junk that fails the parse.
  const text = raw.replace(/^[ \t]+|[ \t]+$/g, '')
  if (!text.length) return null
  if (!LOAD_LIMIT_TEXT.test(text)) return null
  const gb = loadLimitGb(text)
  if (gb == null || !Number.isFinite(gb) || gb <= 0 || !(capBytes > 0)) return null
  const requested = gb * 1e9
  const capped = !Number.isFinite(requested) || requested >= capBytes
  return { key, gb, bytes: Math.min(Math.floor(requested), capBytes), capped }
}

// The numeric value the native parser would read. Decimal text goes through Number; hex text
// cannot (Number('0x1p6') is NaN, and stripping underscores first would misread 0x1_0 as 10),
// so its mantissa digits are read in base 16 — underscores dropped positionally — and the
// optional p/P exponent scales by a power of two, exactly as std.fmt.parseFloat's hex path does.
function loadLimitGb(text: string): number | null {
  const negative = text.startsWith('-')
  const body = text.replace(/^[+-]/, '')
  if (/^0[xX]/.test(body)) {
    const digits = body.slice(2).replace(/_/g, '')
    const [mantissaText, exponentText = ''] = digits.split(/[pP]/, 2)
    const [integerPart, fractionPart = ''] = mantissaText.split('.', 2)
    let mantissa = parseInt(integerPart || '0', 16)
    if (fractionPart) mantissa += parseInt(fractionPart, 16) / 16 ** fractionPart.length
    const exponent = exponentText.length ? parseInt(exponentText.replace(/_/g, ''), 10) : 0
    const value = mantissa * 2 ** exponent
    return negative ? -value : value
  }
  const value = Number(body.replaceAll('_', ''))
  return Number.isNaN(value) ? null : negative ? -value : value
}

// Known knobs offered as quick-add chips. The first two are the reported pain point;
// the rest are the engine's own declared environment (see `capabilities --json`).
export const ENGINE_ENV_PRESETS: Array<{ key: string; placeholder: string; desc: string }> = [
  { key: 'GLM_LOAD_LIMIT_GB', placeholder: '100', desc: 'GB cap for loading GLM-5.3, replacing the 70% of RAM limit. Also raises the app-side memory check.' },
  { key: 'FLASHNEXT_LOAD_LIMIT_GB', placeholder: '100', desc: 'GB cap for loading Flash Next models, replacing the 70% of RAM limit. Also raises the app-side memory check.' },
  { key: 'HF_HUB_OFFLINE', placeholder: '1', desc: 'Set 1 to keep the engine off the network; model downloads need the connection back on.' },
  { key: 'HF_HOME', placeholder: '/Volumes/Models/hf', desc: 'Where Hugging Face caches downloads. TensorFold keeps its own model folders in Settings → Model folders.' },
  { key: 'TENSORFOLD_REQUEST_LOG', placeholder: '/tmp/engine-requests.log', desc: 'Engine knob: append per-request records to a file for diagnostics.' },
  { key: 'TENSORFOLD_NO_LIVE', placeholder: '1', desc: 'Set 1 to hide the engine live text display in its terminal; the served model is unchanged.' },
  { key: 'TENSORFOLD_SEED_SALT', placeholder: 'salt', desc: 'Engine knob: mix a salt into request seeds.' },
]
