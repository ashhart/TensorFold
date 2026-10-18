// Run: jiti electron/agent/guided-decoding.test.ts
import assert from 'node:assert/strict'
import {
  detectGuidedDialect,
  guidedDecodingBody,
  validateAgainstSchema,
  retryDirective,
  runGuardedDecode,
  type JsonSchema,
} from './guided-decoding.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('guided-decoding tests:')

// ---- detectGuidedDialect ----
{
  assert.equal(detectGuidedDialect(null), 'none')
  assert.equal(detectGuidedDialect({ transport: 'codex-cli' }), 'none')
  assert.equal(detectGuidedDialect({ transport: 'claude-cli' }), 'none')
  assert.equal(detectGuidedDialect({ source: 'ollama' }), 'ollama_format')
  assert.equal(detectGuidedDialect({ source: 'lmstudio' }), 'response_format')
  assert.equal(detectGuidedDialect({ source: 'openai' }), 'response_format')
  assert.equal(detectGuidedDialect({ source: 'openrouter' }), 'response_format')
  assert.equal(detectGuidedDialect({ baseUrl: 'https://api.openai.com/v1' }), 'response_format')
  assert.equal(detectGuidedDialect({ baseUrl: 'http://localhost:8000/v1' }), 'guided_json')
  assert.equal(detectGuidedDialect({ baseUrl: 'http://127.0.0.1:8000/v1' }), 'guided_json')
  assert.equal(detectGuidedDialect({ baseUrl: 'https://my-gateway.example.com/v1' }), 'response_format')
  ok('detectGuidedDialect: CLI→none, ollama→format, lmstudio/openai→response_format, loopback→guided_json')
}

// ---- guidedDecodingBody: dialect-specific shapes ----
{
  const schema: JsonSchema = { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] }

  // response_format (openai)
  const rf = guidedDecodingBody({ schema, name: 'my output!' }, { source: 'openai' }) as any
  assert.equal(rf.response_format.type, 'json_schema')
  assert.deepEqual(rf.response_format.json_schema.schema, schema)
  assert.equal(rf.response_format.json_schema.strict, true, 'strict defaults true')
  assert.equal(rf.response_format.json_schema.name, 'my_output', 'name sanitized to [A-Za-z0-9_-]')

  // guided_json (loopback vLLM)
  const gj = guidedDecodingBody({ schema }, { baseUrl: 'http://127.0.0.1:8000/v1' }) as any
  assert.deepEqual(gj, { guided_json: schema })

  // ollama format
  const of = guidedDecodingBody({ schema }, { source: 'ollama' }) as any
  assert.deepEqual(of, { format: schema })

  // none → empty object (spreadable no-op) for CLI transports
  assert.deepEqual(guidedDecodingBody({ schema }, { transport: 'claude-cli' }), {})

  // explicit dialect override wins
  assert.deepEqual(guidedDecodingBody({ schema, dialect: 'none' }, { source: 'openai' }), {})

  // strict:false honored
  const strictOff = guidedDecodingBody({ schema, strict: false }, { source: 'openai' }) as any
  assert.equal(strictOff.response_format.json_schema.strict, false)
  ok('guidedDecodingBody: emits response_format/guided_json/ollama-format/{} + name-sanitize + strict + override')
}

// ---- validateAgainstSchema: happy path ----
{
  const schema: JsonSchema = {
    type: 'object',
    required: ['name', 'age'],
    properties: {
      name: { type: 'string', minLength: 1 },
      age: { type: 'integer', minimum: 0, maximum: 150 },
      role: { enum: ['admin', 'user'] },
    },
    additionalProperties: false,
  }
  const good = validateAgainstSchema({ name: 'Ada', age: 30, role: 'admin' }, schema)
  assert.equal(good.valid, true)
  assert.equal(good.errors.length, 0)
  ok('validateAgainstSchema: valid object passes with zero errors')
}

// ---- validateAgainstSchema: the mistakes models make ----
{
  const schema: JsonSchema = {
    type: 'object',
    required: ['name', 'age'],
    properties: {
      name: { type: 'string', minLength: 1 },
      age: { type: 'integer', minimum: 0, maximum: 150 },
      role: { enum: ['admin', 'user'] },
    },
    additionalProperties: false,
  }

  // missing required
  const missing = validateAgainstSchema({ name: 'Ada' }, schema)
  assert.equal(missing.valid, false)
  assert.ok(missing.errors.some((e) => e.path === '/age' && /required/.test(e.message)))

  // wrong type (string age)
  const wrongType = validateAgainstSchema({ name: 'Ada', age: 'thirty' }, schema)
  assert.ok(wrongType.errors.some((e) => e.path === '/age' && /expected type/.test(e.message)))

  // out of range
  const oor = validateAgainstSchema({ name: 'Ada', age: 999 }, schema)
  assert.ok(oor.errors.some((e) => e.path === '/age' && /maximum/.test(e.message)))

  // bad enum
  const badEnum = validateAgainstSchema({ name: 'Ada', age: 30, role: 'wizard' }, schema)
  assert.ok(badEnum.errors.some((e) => e.path === '/role' && /one of/.test(e.message)))

  // extra prop under additionalProperties:false
  const extra = validateAgainstSchema({ name: 'Ada', age: 30, hacker: true }, schema)
  assert.ok(extra.errors.some((e) => e.path === '/hacker' && /not an allowed property/.test(e.message)))

  // non-integer where integer required
  const nonInt = validateAgainstSchema({ name: 'Ada', age: 30.5 }, schema)
  assert.ok(nonInt.errors.some((e) => e.path === '/age'))
  ok('validateAgainstSchema: catches missing/wrong-type/range/enum/extra-prop/non-integer with paths')
}

// ---- validateAgainstSchema: nested arrays & items ----
{
  const schema: JsonSchema = {
    type: 'object',
    properties: {
      tags: { type: 'array', minItems: 1, maxItems: 3, items: { type: 'string' } },
    },
    required: ['tags'],
  }
  assert.equal(validateAgainstSchema({ tags: ['a', 'b'] }, schema).valid, true)
  const tooMany = validateAgainstSchema({ tags: ['a', 'b', 'c', 'd'] }, schema)
  assert.ok(tooMany.errors.some((e) => /maxItems/.test(e.message)))
  const badItem = validateAgainstSchema({ tags: ['a', 5] }, schema)
  assert.ok(badItem.errors.some((e) => e.path === '/tags/1' && /expected type/.test(e.message)))
  const empty = validateAgainstSchema({ tags: [] }, schema)
  assert.ok(empty.errors.some((e) => /minItems/.test(e.message)))
  ok('validateAgainstSchema: array minItems/maxItems + per-item type with indexed paths')
}

// ---- validateAgainstSchema: null / nullable + combinators ----
{
  const nullable: JsonSchema = { type: ['string', 'null'] }
  assert.equal(validateAgainstSchema(null, nullable).valid, true)
  assert.equal(validateAgainstSchema('x', nullable).valid, true)
  assert.equal(validateAgainstSchema(5, nullable).valid, false)

  const anyOf: JsonSchema = { anyOf: [{ type: 'string' }, { type: 'integer' }] }
  assert.equal(validateAgainstSchema('x', anyOf).valid, true)
  assert.equal(validateAgainstSchema(3, anyOf).valid, true)
  assert.equal(validateAgainstSchema(true, anyOf).valid, false)
  ok('validateAgainstSchema: null-union + anyOf combinator')
}

// ---- retryDirective ----
{
  const schema: JsonSchema = { type: 'object', required: ['a'], properties: { a: { type: 'string' } } }
  const bad = validateAgainstSchema({}, schema)

  const d1 = retryDirective(bad, { attempt: 1, maxAttempts: 2, label: 'search_web args' })
  assert.equal(d1.shouldRetry, true)
  assert.ok(d1.message.includes('search_web args'))
  assert.ok(d1.message.includes('/a'))
  assert.ok(d1.message.includes('ONLY'))
  assert.equal(d1.errors.length, bad.errors.length)

  // budget exhausted → no more retry, but errors still surfaced
  const d2 = retryDirective(bad, { attempt: 2, maxAttempts: 2 })
  assert.equal(d2.shouldRetry, false)
  assert.equal(d2.message, '')
  assert.ok(d2.errors.length > 0)

  // valid → never retry
  const good = validateAgainstSchema({ a: 'ok' }, schema)
  const d3 = retryDirective(good, { attempt: 1, maxAttempts: 3 })
  assert.equal(d3.shouldRetry, false)
  assert.equal(d3.errors.length, 0)
  ok('retryDirective: first-mismatch retries with pathful message; exhausted/valid do not')
}

// ---- runGuardedDecode: DI orchestration, retries then succeeds ----
{
  const schema: JsonSchema = { type: 'object', required: ['n'], properties: { n: { type: 'integer' } } }
  const attempts: Array<{ attempt: number; hadDirective: boolean }> = []

  const res = await runGuardedDecode<any>({
    schema,
    maxAttempts: 3,
    label: 'plan',
    produce: (attempt, directive) => {
      attempts.push({ attempt, hadDirective: !!directive })
      // First attempt is wrong (string), second is corrected.
      return attempt === 1 ? { n: 'nope' } : { n: 42 }
    },
  })

  assert.equal(res.ok, true)
  assert.deepEqual(res.value, { n: 42 })
  assert.equal(res.attempts, 2)
  assert.equal(attempts[0].hadDirective, false, 'first produce has no prior directive')
  assert.equal(attempts[1].hadDirective, true, 'retry produce receives the corrective directive')
  ok('runGuardedDecode: re-prompts on mismatch and returns the corrected value')
}

// ---- runGuardedDecode: exhausts attempts, reports failure ----
{
  const schema: JsonSchema = { type: 'object', required: ['n'], properties: { n: { type: 'integer' } } }
  let calls = 0
  const res = await runGuardedDecode<any>({
    schema,
    maxAttempts: 2,
    produce: () => { calls++; return { n: 'still wrong' } },
  })
  assert.equal(res.ok, false)
  assert.equal(res.attempts, 2)
  assert.equal(calls, 2)
  assert.ok(res.directive.errors.length > 0)
  ok('runGuardedDecode: exhausts maxAttempts and returns ok:false with residual errors')
}

// ---- runGuardedDecode: injected validator is honored ----
{
  const schema: JsonSchema = { type: 'string' }
  let validatorCalls = 0
  const res = await runGuardedDecode<any>({
    schema,
    produce: () => ({ anything: true }),
    validate: (v) => { validatorCalls++; return { valid: true, errors: [] } }, // always-valid injected validator
  })
  assert.equal(res.ok, true)
  assert.equal(validatorCalls, 1, 'stops after first success')
  ok('runGuardedDecode: uses the injected validator (Ajv swap-in point)')
}

console.log(`\n${passed} guided-decoding checks passed.`)
