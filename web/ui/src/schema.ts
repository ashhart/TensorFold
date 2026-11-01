// A small JSON Schema check for the Chat's workshop: pure, so node's tests load it (`tests/workshop.test.mjs`).

// `true` allows anything and `false` nothing, as JSON Schema says, wherever a sub-schema goes
export type Schema = { type?: string | string[]; properties?: Record<string, Schema | boolean>; required?: string[]; items?: Schema | boolean
  enum?: unknown[]; additionalProperties?: boolean | Schema; description?: string; minimum?: number; maximum?: number
  const?: unknown; pattern?: string; minLength?: number; maxLength?: number; minItems?: number; maxItems?: number
  exclusiveMinimum?: number; exclusiveMaximum?: number; [other: string]: unknown }
// the keywords `check` reads; a schema using others is said to be checked only in part
const KNOWN = new Set(['type', 'properties', 'required', 'items', 'enum', 'additionalProperties', 'description', 'minimum', 'maximum',
  'const', 'pattern', 'minLength', 'maxLength', 'minItems', 'maxItems', 'exclusiveMinimum', 'exclusiveMaximum', 'title', '$schema', 'default', 'examples'])

/** Keywords of ``schema`` (and its parts) that `check` does not read, such as oneOf or $ref. */
export function unchecked(schema: Schema | boolean | undefined, out = new Set<string>()): Set<string> {
  if (!schema || typeof schema !== 'object') return out
  for (const k of Object.keys(schema)) if (!KNOWN.has(k)) out.add(k)
  for (const sub of Object.values(schema.properties ?? {})) unchecked(sub, out)
  if (schema.items) unchecked(schema.items, out)
  if (typeof schema.additionalProperties === 'object') unchecked(schema.additionalProperties, out)
  return out
}
/** JSON equality: objects compare key by key, whatever their order. */
export function equal(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) return a.length === (b as unknown[]).length && a.every((x, i) => equal(x, (b as unknown[])[i]))
  const ka = Object.keys(a), kb = Object.keys(b)
  return ka.length === kb.length && ka.every((k) => Object.hasOwn(b, k) && equal((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
}
/** A small JSON Schema check: types, required keys, enums, const, patterns, items, extra keys, lengths and bounds.
 *  Returns what is wrong; `unchecked` says which keywords of a schema it does not read. */
export function check(value: unknown, schema: Schema | boolean | undefined, path = '$'): string[] {
  if (schema === false) return [`${path}: no value is allowed here`]
  if (!schema || schema === true) return []
  const errs: string[] = []
  const types = schema.type == null ? [] : Array.isArray(schema.type) ? schema.type : [schema.type]
  const kind = value === null ? 'null' : Array.isArray(value) ? 'array' : Number.isInteger(value) ? 'integer' : typeof value
  if (types.length && !types.some((t) => t === kind || (t === 'number' && kind === 'integer'))) return [`${path}: ${kind}, expected ${types.join(' or ')}`]
  if (schema.enum && !schema.enum.some((e) => equal(e, value))) errs.push(`${path}: ${JSON.stringify(value)} is not one of ${schema.enum.map((e) => JSON.stringify(e)).join(', ')}`)
  if ('const' in schema && !equal(schema.const, value)) errs.push(`${path}: ${JSON.stringify(value)}, expected ${JSON.stringify(schema.const)}`)
  if (typeof value === 'number') {
    if (schema.minimum != null && value < schema.minimum) errs.push(`${path}: below ${schema.minimum}`)
    if (schema.maximum != null && value > schema.maximum) errs.push(`${path}: above ${schema.maximum}`)
    if (schema.exclusiveMinimum != null && value <= schema.exclusiveMinimum) errs.push(`${path}: not above ${schema.exclusiveMinimum}`)
    if (schema.exclusiveMaximum != null && value >= schema.exclusiveMaximum) errs.push(`${path}: not below ${schema.exclusiveMaximum}`)
  }
  if (typeof value === 'string') {
    if (schema.minLength != null && [...value].length < schema.minLength) errs.push(`${path}: shorter than ${schema.minLength}`)
    if (schema.maxLength != null && [...value].length > schema.maxLength) errs.push(`${path}: longer than ${schema.maxLength}`)
    if (schema.pattern != null) { try { if (!new RegExp(schema.pattern, 'u').test(value)) errs.push(`${path}: does not match ${schema.pattern}`) } catch { errs.push(`${path}: the schema's pattern is not a valid expression`) } }
  }
  if (Array.isArray(value)) {
    if (schema.minItems != null && value.length < schema.minItems) errs.push(`${path}: fewer than ${schema.minItems} items`)
    if (schema.maxItems != null && value.length > schema.maxItems) errs.push(`${path}: more than ${schema.maxItems} items`)
  }
  if (kind === 'object' && value) {
    const obj = value as Record<string, unknown>
    for (const k of schema.required ?? []) if (!Object.hasOwn(obj, k)) errs.push(`${path}.${k}: missing`)   // not toString
    for (const [k, v] of Object.entries(obj)) {
      const sub = schema.properties && Object.hasOwn(schema.properties, k) ? schema.properties[k] : undefined
      if (sub !== undefined) errs.push(...check(v, sub, `${path}.${k}`))
      else if (schema.additionalProperties === false) errs.push(`${path}.${k}: not expected`)
      else if (schema.additionalProperties !== undefined && schema.additionalProperties !== true) errs.push(...check(v, schema.additionalProperties, `${path}.${k}`))
    }
  }
  if (kind === 'array' && schema.items !== undefined) (value as unknown[]).forEach((v, i) => errs.push(...check(v, schema.items, `${path}[${i}]`)))
  return errs
}

/** A JSON reply: parsed (a fenced block is read too) and, with a schema, checked. */
export function checkJson(text: string, schema: Schema | null): { ok: boolean; detail: string } {
  const t = text.trim().replace(/^```(?:json)?\s*/, '').replace(/\s*```$/, '')
  let v: unknown
  try { v = JSON.parse(t) } catch (e) { return { ok: false, detail: `not JSON: ${(e as Error).message}` } }
  if (!schema) return { ok: true, detail: 'valid JSON' }
  const errs = check(v, schema)
  const partly = [...unchecked(schema)]
  return errs.length ? { ok: false, detail: errs.slice(0, 4).join('; ') }
    : { ok: true, detail: partly.length ? `valid JSON; matches what was checked, but ${partly.join(', ')} not checked here` : 'valid JSON, matches the schema' }
}
