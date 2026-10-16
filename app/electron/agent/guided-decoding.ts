// Schema-constrained decoding + validate-then-retry.
//
// Two orthogonal jobs, both PURE (no I/O, no side effects — everything injectable):
//
//   1. REQUEST SHAPING — `guidedDecodingBody()` returns the extra fields to spread onto the
//      /chat/completions body (openai.ts streamChat) so a provider that *natively* constrains
//      output to a JSON Schema does so. OpenAI/Azure/vLLM/LM Studio speak `response_format`
//      with `{ type: 'json_schema', json_schema: {...} }`; vLLM/SGLang/Aphrodite also accept
//      the vendor extension `guided_json`; Ollama accepts a top-level `format` object. We emit
//      the dialect the provider understands and NOTHING for providers that can't (so the body
//      is unchanged and the model is merely *asked* to match the schema in-prompt).
//
//   2. VALIDATE-THEN-RETRY — even a "guided" provider can drift (and most local backends only
//      best-effort constrain), so `validateAgainstSchema()` checks a parsed tool-call/structured
//      object against the same JSON Schema, and `retryDirective()` turns a mismatch into a
//      concrete corrective message + a `shouldRetry` flag the loop can feed back to the model.
//
// The validator is a deliberately SMALL, dependency-free subset of JSON Schema draft-07 —
// enough to catch the mistakes models actually make (missing required keys, wrong types, bad
// enums, extra keys under additionalProperties:false, out-of-range numbers, array bounds). It
// is NOT a spec-complete validator; inject `validate` in `runGuardedDecode` to swap in Ajv if a
// caller needs full coverage.

export type JsonSchema = {
  type?: JsonSchemaType | JsonSchemaType[]
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  enum?: unknown[]
  const?: unknown
  additionalProperties?: boolean | JsonSchema
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  minLength?: number
  maxLength?: number
  pattern?: string
  minItems?: number
  maxItems?: number
  anyOf?: JsonSchema[]
  oneOf?: JsonSchema[]
  allOf?: JsonSchema[]
  nullable?: boolean
  [k: string]: unknown
}

export type JsonSchemaType = 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array' | 'null'

export interface SchemaError {
  // JSON-pointer-ish path to the offending value, e.g. `/user/age` or `` for the root.
  path: string
  message: string
}

export interface ValidationResult {
  valid: boolean
  errors: SchemaError[]
}

// ---------------------------------------------------------------------------
// 1. REQUEST SHAPING
// ---------------------------------------------------------------------------

// How a provider wants a JSON-Schema constraint expressed on the request body.
//   - 'response_format'  → OpenAI/Azure/vLLM/LM Studio: { response_format: { type: 'json_schema', json_schema: { name, schema, strict } } }
//   - 'guided_json'      → vLLM/SGLang/Aphrodite vendor extension: { guided_json: <schema> }
//   - 'ollama_format'    → Ollama native constrained decoding: { format: <schema> }
//   - 'none'             → provider cannot natively constrain; emit nothing (prompt-only).
export type GuidedDialect = 'response_format' | 'guided_json' | 'ollama_format' | 'none'

export interface GuidedSpec {
  // A JSON-Schema-shaped constraint the output object must satisfy.
  schema: JsonSchema
  // Label used in the OpenAI `json_schema.name` slot (must be a short identifier). Default 'output'.
  name?: string
  // OpenAI `strict` flag — asks the provider to reject any output that doesn't match. Default true.
  strict?: boolean
  // Override the auto-detected dialect (e.g. force 'none' to disable native constraining).
  dialect?: GuidedDialect
}

// Minimal shape of the Provider fields we consult — kept structural so the module doesn't take
// a hard dependency on the full Provider type (which lives behind the type-only boundary).
export interface GuidedProviderInfo {
  transport?: string | null
  source?: string | null
  baseUrl?: string | null
}

const OPENAI_HOSTS = /(^|\.)openai\.com$|(^|\.)azure\.com$|(^|\.)openrouter\.ai$/i

// Decide which native constrained-decoding dialect (if any) a provider speaks. Pure — driven
// only by declared transport/source and the base URL host. CLI transports (codex/claude) never
// take a `response_format`, so they always resolve to 'none' (prompt-only guidance).
export function detectGuidedDialect(provider: GuidedProviderInfo | undefined | null): GuidedDialect {
  if (!provider) return 'none'
  const transport = provider.transport ?? 'chat-completions'
  if (transport === 'codex-cli' || transport === 'claude-cli' || transport === 'opencode' || transport === 'kilo' || transport === 'pi-cli' || transport === 'cursor-cli' || transport === 'gemini-cli') return 'none'

  const source = (provider.source ?? '').toLowerCase()
  if (source === 'ollama') return 'ollama_format'
  if (source === 'lmstudio') return 'response_format'
  if (source === 'openai' || source === 'openrouter' || source === 'ai-gateway') return 'response_format'

  // Fall back to host sniffing for 'custom' sources.
  const host = safeHost(provider.baseUrl)
  if (host && OPENAI_HOSTS.test(host)) return 'response_format'
  // Local OpenAI-compatible servers (vLLM/SGLang) commonly bind loopback; guided_json is their
  // widely-supported vendor extension and is ignored by servers that don't grok it.
  if (host && /^(localhost|127\.0\.0\.1|::1|0\.0\.0\.0)$/i.test(host)) return 'guided_json'

  // Unknown remote OpenAI-compatible endpoint: response_format is the standard, safest bet.
  return 'response_format'
}

function safeHost(url: string | null | undefined): string | null {
  if (!url) return null
  try { return new URL(url).hostname } catch { return null }
}

// Build the object to spread onto the /chat/completions request body so the provider natively
// constrains output to `spec.schema`. Returns `{}` when the provider can't (dialect 'none') so
// the caller can `...guidedDecodingBody(spec, provider)` unconditionally without a branch.
export function guidedDecodingBody(
  spec: GuidedSpec,
  provider: GuidedProviderInfo | undefined | null,
): Record<string, unknown> {
  const dialect = spec.dialect ?? detectGuidedDialect(provider)
  if (dialect === 'none') return {}
  if (dialect === 'guided_json') return { guided_json: spec.schema }
  if (dialect === 'ollama_format') return { format: spec.schema }
  // response_format (OpenAI json_schema)
  return {
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: sanitizeSchemaName(spec.name ?? 'output'),
        schema: spec.schema,
        strict: spec.strict ?? true,
      },
    },
  }
}

// OpenAI requires json_schema.name to match ^[a-zA-Z0-9_-]+$ (<=64 chars). Coerce anything else.
function sanitizeSchemaName(name: string): string {
  const cleaned = name.replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 64)
  return cleaned || 'output'
}

// ---------------------------------------------------------------------------
// 2. VALIDATION
// ---------------------------------------------------------------------------

function typeOf(value: unknown): JsonSchemaType {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  const t = typeof value
  if (t === 'number') return Number.isInteger(value) ? 'integer' : 'number'
  if (t === 'boolean') return 'boolean'
  if (t === 'object') return 'object'
  return 'string'
}

function matchesType(value: unknown, wanted: JsonSchemaType): boolean {
  const actual = typeOf(value)
  if (wanted === 'number') return actual === 'number' || actual === 'integer'
  if (wanted === 'integer') return actual === 'integer'
  return actual === wanted
}

function joinPath(base: string, key: string | number): string {
  return `${base}/${key}`
}

// Validate `value` against `schema`, collecting errors. Pure and synchronous. Draft-07 subset
// (see module header).
export function validateAgainstSchema(value: unknown, schema: JsonSchema, path = ''): ValidationResult {
  const errors: SchemaError[] = []
  walk(value, schema, path, errors)
  return { valid: errors.length === 0, errors }
}

function walk(value: unknown, schema: JsonSchema, path: string, errors: SchemaError[]): void {
  if (!schema || typeof schema !== 'object') return

  // const / enum
  if ('const' in schema && !deepEqual(value, schema.const)) {
    errors.push({ path, message: `must equal ${JSON.stringify(schema.const)}` })
  }
  if (Array.isArray(schema.enum) && !schema.enum.some((e) => deepEqual(e, value))) {
    errors.push({ path, message: `must be one of ${JSON.stringify(schema.enum)}` })
  }

  // nullable shortcut (OpenAPI-style) — allow null and skip further checks on null.
  if (value === null && (schema.nullable === true || allowsNull(schema.type))) return

  // type
  if (schema.type != null) {
    const wanted = Array.isArray(schema.type) ? schema.type : [schema.type]
    if (!wanted.some((t) => matchesType(value, t))) {
      errors.push({ path, message: `expected type ${wanted.join('|')} but got ${typeOf(value)}` })
      // A wrong container type makes deeper checks meaningless; stop this branch.
      return
    }
  }

  const actual = typeOf(value)

  if (actual === 'string') validateString(value as string, schema, path, errors)
  if (actual === 'number' || actual === 'integer') validateNumber(value as number, schema, path, errors)
  if (actual === 'array') validateArray(value as unknown[], schema, path, errors)
  if (actual === 'object') validateObject(value as Record<string, unknown>, schema, path, errors)

  // Combinators.
  if (Array.isArray(schema.allOf)) {
    for (const sub of schema.allOf) walk(value, sub, path, errors)
  }
  if (Array.isArray(schema.anyOf)) {
    const anyOk = schema.anyOf.some((sub) => validateAgainstSchema(value, sub, path).valid)
    if (!anyOk) errors.push({ path, message: 'did not match any of the allowed schemas (anyOf)' })
  }
  if (Array.isArray(schema.oneOf)) {
    const matches = schema.oneOf.filter((sub) => validateAgainstSchema(value, sub, path).valid).length
    if (matches !== 1) errors.push({ path, message: `must match exactly one schema (oneOf), matched ${matches}` })
  }
}

function allowsNull(type: JsonSchema['type']): boolean {
  if (type == null) return false
  return Array.isArray(type) ? type.includes('null') : type === 'null'
}

function validateString(value: string, schema: JsonSchema, path: string, errors: SchemaError[]): void {
  if (typeof schema.minLength === 'number' && value.length < schema.minLength) {
    errors.push({ path, message: `shorter than minLength ${schema.minLength}` })
  }
  if (typeof schema.maxLength === 'number' && value.length > schema.maxLength) {
    errors.push({ path, message: `longer than maxLength ${schema.maxLength}` })
  }
  if (typeof schema.pattern === 'string') {
    let re: RegExp | null = null
    try { re = new RegExp(schema.pattern) } catch { re = null }
    if (re && !re.test(value)) errors.push({ path, message: `does not match pattern ${schema.pattern}` })
  }
}

function validateNumber(value: number, schema: JsonSchema, path: string, errors: SchemaError[]): void {
  if (typeof schema.minimum === 'number' && value < schema.minimum) {
    errors.push({ path, message: `less than minimum ${schema.minimum}` })
  }
  if (typeof schema.maximum === 'number' && value > schema.maximum) {
    errors.push({ path, message: `greater than maximum ${schema.maximum}` })
  }
  if (typeof schema.exclusiveMinimum === 'number' && value <= schema.exclusiveMinimum) {
    errors.push({ path, message: `must be greater than ${schema.exclusiveMinimum}` })
  }
  if (typeof schema.exclusiveMaximum === 'number' && value >= schema.exclusiveMaximum) {
    errors.push({ path, message: `must be less than ${schema.exclusiveMaximum}` })
  }
}

function validateArray(value: unknown[], schema: JsonSchema, path: string, errors: SchemaError[]): void {
  if (typeof schema.minItems === 'number' && value.length < schema.minItems) {
    errors.push({ path, message: `fewer than minItems ${schema.minItems}` })
  }
  if (typeof schema.maxItems === 'number' && value.length > schema.maxItems) {
    errors.push({ path, message: `more than maxItems ${schema.maxItems}` })
  }
  if (schema.items) {
    value.forEach((item, i) => walk(item, schema.items!, joinPath(path, i), errors))
  }
}

function validateObject(value: Record<string, unknown>, schema: JsonSchema, path: string, errors: SchemaError[]): void {
  if (Array.isArray(schema.required)) {
    for (const key of schema.required) {
      if (!(key in value)) errors.push({ path: joinPath(path, key), message: 'is required but missing' })
    }
  }
  const props = schema.properties ?? {}
  for (const [key, sub] of Object.entries(props)) {
    if (key in value) walk(value[key], sub, joinPath(path, key), errors)
  }
  if (schema.additionalProperties === false) {
    for (const key of Object.keys(value)) {
      if (!(key in props)) errors.push({ path: joinPath(path, key), message: 'is not an allowed property (additionalProperties:false)' })
    }
  } else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
    const extra = schema.additionalProperties as JsonSchema
    for (const key of Object.keys(value)) {
      if (!(key in props)) walk(value[key], extra, joinPath(path, key), errors)
    }
  }
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b) return false
  if (a && b && typeof a === 'object') {
    if (Array.isArray(a) !== Array.isArray(b)) return false
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) return false
      return a.every((x, i) => deepEqual(x, b[i]))
    }
    const ak = Object.keys(a as object), bk = Object.keys(b as object)
    if (ak.length !== bk.length) return false
    return ak.every((k) => deepEqual((a as any)[k], (b as any)[k]))
  }
  return false
}

// ---------------------------------------------------------------------------
// 3. RETRY DIRECTIVE
// ---------------------------------------------------------------------------

export interface RetryDirective {
  // Whether the caller should send the model another turn to fix the output.
  shouldRetry: boolean
  // A corrective message to feed back to the model (empty when shouldRetry is false).
  message: string
  // The structured errors that drove the decision (empty when valid).
  errors: SchemaError[]
  // 1-based count of attempts already spent (echoed for the caller's bookkeeping).
  attempt: number
}

export interface RetryOptions {
  // 1-based index of the attempt that just produced `value`. Default 1.
  attempt?: number
  // Give up (shouldRetry:false) once this many attempts have been made. Default 2.
  maxAttempts?: number
  // Label for the object in the corrective message (e.g. the tool name). Default 'output'.
  label?: string
}

// Build the corrective directive from a validation result. Pure. When the value is valid, or the
// attempt budget is spent, `shouldRetry` is false. Otherwise `message` names each violation with
// its path so the model can target the exact fix.
export function retryDirective(result: ValidationResult, opts: RetryOptions = {}): RetryDirective {
  const attempt = Math.max(1, Math.floor(opts.attempt ?? 1))
  const maxAttempts = Math.max(1, Math.floor(opts.maxAttempts ?? 2))
  const label = opts.label ?? 'output'

  if (result.valid) return { shouldRetry: false, message: '', errors: [], attempt }
  if (attempt >= maxAttempts) {
    return { shouldRetry: false, message: '', errors: result.errors, attempt }
  }

  const lines = result.errors.map((e) => `  - ${e.path || '(root)'}: ${e.message}`).join('\n')
  const message =
    `Your ${label} did not match the required JSON schema. Fix these problems and reply with ` +
    `ONLY the corrected JSON:\n${lines}`
  return { shouldRetry: true, message, errors: result.errors, attempt }
}

// ---------------------------------------------------------------------------
// 4. ORCHESTRATION HELPER (DI)
// ---------------------------------------------------------------------------

export interface GuardedDecodeOptions<T> {
  // The JSON Schema the produced object must satisfy.
  schema: JsonSchema
  // Produce/parse a candidate object for a given attempt. `directive` is null on the first call
  // and carries the previous mismatch on retries so the caller can re-prompt the model.
  produce: (attempt: number, directive: RetryDirective | null) => Promise<T> | T
  // Max attempts (INCLUDING the first). Default 2.
  maxAttempts?: number
  // Label for corrective messages (e.g. the tool name). Default 'output'.
  label?: string
  // Injectable validator (default: the built-in subset validator). Swap in Ajv for full coverage.
  validate?: (value: unknown, schema: JsonSchema) => ValidationResult
}

export interface GuardedDecodeResult<T> {
  // The last produced value (valid if `ok`, else the final failing candidate).
  value: T
  // Whether the final value satisfied the schema.
  ok: boolean
  // The final directive (its errors describe the residual mismatch when !ok).
  directive: RetryDirective
  // How many attempts were actually made.
  attempts: number
}

// Run produce → validate → (retry with corrective directive) up to `maxAttempts` times. Every
// dependency is injected, so this is fully unit-testable with a scripted `produce`. Returns the
// last candidate and whether it ultimately validated.
export async function runGuardedDecode<T>(opts: GuardedDecodeOptions<T>): Promise<GuardedDecodeResult<T>> {
  const maxAttempts = Math.max(1, Math.floor(opts.maxAttempts ?? 2))
  const validate = opts.validate ?? ((v, s) => validateAgainstSchema(v, s))
  const label = opts.label ?? 'output'

  let directive: RetryDirective | null = null
  let value: T = undefined as unknown as T
  let attempt = 0

  while (attempt < maxAttempts) {
    attempt++
    value = await opts.produce(attempt, directive)
    const result = validate(value, opts.schema)
    directive = retryDirective(result, { attempt, maxAttempts, label })
    if (!directive.shouldRetry) {
      return { value, ok: result.valid, directive, attempts: attempt }
    }
  }

  // Reached only if the loop boundary coincided with a still-retryable directive; the value is
  // the final (invalid) candidate.
  return { value, ok: false, directive: directive!, attempts: attempt }
}
