// A control vector the operator chooses on this Mac: the file the engine reads,
// whether it is in force, and the strength it is applied with. The engine takes
// the same two things as --glp and --glp-alpha.

// The strength has to be above zero: the engine refuses anything else. A strength
// past a few stops the model answering, so the app keeps the range narrow.
export const CONTROL_VECTOR_ALPHA_MAX = 16
export const CONTROL_VECTOR_ALPHA_DEFAULT = 1

export interface EngineControlVector {
  enabled: boolean
  file: string | null
  alpha: number
}

export const CONTROL_VECTOR_DEFAULTS: EngineControlVector = { enabled: false, file: null, alpha: CONTROL_VECTOR_ALPHA_DEFAULT }

function inRange(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= CONTROL_VECTOR_ALPHA_MAX
}

// The saved shape, read from disk: anything malformed falls back to the default,
// and a vector with no file cannot be in force.
export function readControlVector(doc: unknown): EngineControlVector {
  const raw = (doc ?? {}) as { enabled?: unknown; file?: unknown; alpha?: unknown }
  const file = typeof raw.file === 'string' && raw.file.length > 0 ? raw.file : null
  const alpha = inRange(raw.alpha) ? raw.alpha : CONTROL_VECTOR_ALPHA_DEFAULT
  return { enabled: raw.enabled === true && file != null, file, alpha }
}

// A strength the operator named, checked against what the engine accepts.
export function checkedAlpha(value: unknown): number {
  if (!inRange(value)) throw new Error(`The strength must be above 0 and at most ${CONTROL_VECTOR_ALPHA_MAX}.`)
  return value
}

// The operator's request read over the choice they already made: a field that is
// not named keeps its current value, so changing the strength leaves the vector in
// force, and turning it off leaves the strength alone.
export function mergeControlVector(current: EngineControlVector, input: { enabled?: unknown; alpha?: unknown }): { enabled: boolean; alpha: number } {
  const enabled = input?.enabled === undefined ? current.enabled : input.enabled === true
  const alpha = input?.alpha === undefined || input?.alpha === null ? current.alpha : checkedAlpha(input.alpha)
  return { enabled, alpha }
}
