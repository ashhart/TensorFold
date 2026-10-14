// Schema-aware repair for tool-call arguments that parsed as valid JSON but landed on
// the wrong primitive type — a real gap with non-frontier/local models, which sometimes
// emit a string "42" where a number is expected, a string "true" where a boolean is
// expected, or a bare scalar where the schema wants an array. JSON.parse can't catch
// this (the JSON itself is well-formed); this is a second, schema-driven repair pass.
//
// Deliberately conservative: only coerces the handful of unambiguous shapes described
// below. Anything ambiguous (a non-numeric string against a "number" property, an empty
// string, null/undefined, an object) is left exactly as-is for downstream code to handle
// — never throws, never drops a field, never invents a value.

function schemaProperties(schema: Record<string, unknown>): Record<string, any> {
  const props = (schema as any)?.properties
  return props && typeof props === 'object' && !Array.isArray(props) ? props : {}
}

function propertyType(propSchema: unknown): string | undefined {
  if (!propSchema || typeof propSchema !== 'object') return undefined
  const t = (propSchema as any).type
  return typeof t === 'string' ? t : undefined
}

// A "clean" numeric string: non-empty after trimming, and Number(...) on the trimmed
// value yields a finite number. Number('') and Number('   ') are 0 — explicitly
// rejected here so an empty/blank string is never silently coerced to 0.
function coercibleNumber(value: string): number | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  return Number.isFinite(n) ? n : undefined
}

function coercibleBoolean(value: string): boolean | undefined {
  const lower = value.trim().toLowerCase()
  if (lower === 'true') return true
  if (lower === 'false') return false
  return undefined
}

/**
 * Repair well-formed-but-mistyped tool-call arguments against the tool's JSON-Schema
 * `parameters`. Pure: never mutates `args`, always returns a new object. Keys without a
 * matching `schema.properties` entry — and values that don't cleanly coerce — pass
 * through untouched (including values that already have the right type).
 */
export function coerceToolArgs(args: Record<string, any>, schema: Record<string, unknown>): Record<string, any> {
  const props = schemaProperties(schema)
  const out: Record<string, any> = { ...args }

  for (const key of Object.keys(out)) {
    const propSchema = props[key]
    if (propSchema === undefined) continue
    const type = propertyType(propSchema)
    const value = out[key]

    if ((type === 'number' || type === 'integer') && typeof value === 'string') {
      const n = coercibleNumber(value)
      if (n !== undefined) out[key] = n
      continue
    }

    if (type === 'boolean' && typeof value === 'string') {
      const b = coercibleBoolean(value)
      if (b !== undefined) out[key] = b
      continue
    }

    if (type === 'array' && !Array.isArray(value) && (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')) {
      out[key] = [value]
      continue
    }
  }

  return out
}
