// Run: npx jiti electron/agent/otel-genai.test.ts
import assert from 'node:assert/strict'
import {
  mapRunToSpans,
  mapTurnSpan,
  mapToolCallSpan,
  buildRunOtlpPayload,
  buildOtlpPayload,
  emitSpans,
  OTLP_ERROR_BODY_MAX_BYTES,
  emitRun,
  msToUnixNano,
  deriveTraceId,
  deriveSpanId,
  type RunSpanInput,
  type HttpSender,
  type OtelKeyValue,
} from './otel-genai'

let passed = 0
const ok = (name: string) => {
  passed++
  console.log(`  ✓ ${name}`)
}

const attr = (attrs: OtelKeyValue[], key: string) => attrs.find((a) => a.key === key)?.value

console.log('otel-genai tests:')

const run: RunSpanInput = {
  runId: 'run-42',
  threadId: 'thread-9',
  system: 'openai',
  startMs: 1000,
  endMs: 5000,
  turns: [
    {
      id: 't1',
      system: 'openai',
      requestModel: 'gpt-4o',
      responseModel: 'gpt-4o-2024',
      temperature: 0.7,
      maxTokens: 512,
      usage: { inputTokens: 120, outputTokens: 45 },
      responseId: 'resp_1',
      finishReason: 'stop',
      startMs: 1010,
      endMs: 2000,
    },
  ],
  toolCalls: [
    { id: 'c1', toolName: 'read_file', callId: 'call_abc', toolType: 'function', parentTurnId: 't1', startMs: 2010, endMs: 2100 },
    { id: 'c2', toolName: 'mcp__repo__search', callId: 'call_def', startMs: 2110, endMs: 2200, error: 'boom' },
  ],
}

// --- id generation ---
{
  const tid = deriveTraceId('run:run-42')
  const sid = deriveSpanId('run:run-42')
  assert.equal(tid.length, 32, 'trace id is 16 bytes / 32 hex')
  assert.equal(sid.length, 16, 'span id is 8 bytes / 16 hex')
  assert.match(tid, /^[0-9a-f]{32}$/)
  assert.match(sid, /^[0-9a-f]{16}$/)
  assert.equal(deriveTraceId('run:run-42'), tid, 'trace id is deterministic')
  assert.notEqual(deriveTraceId('run:other'), tid, 'distinct seeds → distinct trace ids')
  ok('id gen: deterministic, correctly-sized, distinct')
}

// --- msToUnixNano ---
{
  assert.equal(msToUnixNano(1000), '1000000000')
  assert.equal(msToUnixNano(0), '0')
  assert.equal(msToUnixNano(-5), '0', 'negative clamps to 0')
  assert.equal(msToUnixNano(1.5), '1500000', 'sub-ms → nanos')
  // large value keeps integer precision (BigInt path)
  assert.equal(msToUnixNano(1_700_000_000_000), '1700000000000000000')
  ok('msToUnixNano: ms→nanos with BigInt precision + guards')
}

// --- turn span mapping ---
{
  const span = mapTurnSpan(run.turns![0], { traceId: 'aa', parentSpanId: 'bb' })
  assert.equal(span.name, 'chat gpt-4o', 'span name = "{op} {model}"')
  assert.equal(span.kind, 3, 'model turn is CLIENT')
  assert.equal(span.parentSpanId, 'bb')
  assert.equal((attr(span.attributes, 'gen_ai.operation.name') as any).stringValue, 'chat')
  assert.equal((attr(span.attributes, 'gen_ai.system') as any).stringValue, 'openai')
  assert.equal((attr(span.attributes, 'gen_ai.request.model') as any).stringValue, 'gpt-4o')
  assert.equal((attr(span.attributes, 'gen_ai.response.model') as any).stringValue, 'gpt-4o-2024')
  assert.equal((attr(span.attributes, 'gen_ai.usage.input_tokens') as any).intValue, 120)
  assert.equal((attr(span.attributes, 'gen_ai.usage.output_tokens') as any).intValue, 45)
  assert.equal((attr(span.attributes, 'gen_ai.request.temperature') as any).doubleValue, 0.7)
  assert.equal((attr(span.attributes, 'gen_ai.request.max_tokens') as any).intValue, 512)
  const finish = attr(span.attributes, 'gen_ai.response.finish_reasons') as any
  assert.equal(finish.arrayValue.values[0].stringValue, 'stop', 'finish_reasons is an array')
  assert.equal(span.status.code, 1, 'no error → OK status')
  ok('mapTurnSpan: gen_ai.* attributes + CLIENT kind + OK status')
}

// --- tool call span mapping ---
{
  const okCall = mapToolCallSpan(run.toolCalls![0], { traceId: 'aa', parentSpanId: 'p1' })
  assert.equal(okCall.name, 'execute_tool read_file')
  assert.equal(okCall.kind, 1, 'tool call is INTERNAL')
  assert.equal((attr(okCall.attributes, 'gen_ai.operation.name') as any).stringValue, 'execute_tool')
  assert.equal((attr(okCall.attributes, 'gen_ai.tool.name') as any).stringValue, 'read_file')
  assert.equal((attr(okCall.attributes, 'gen_ai.tool.call.id') as any).stringValue, 'call_abc')
  assert.equal((attr(okCall.attributes, 'gen_ai.tool.type') as any).stringValue, 'function')

  const errCall = mapToolCallSpan(run.toolCalls![1], { traceId: 'aa' })
  assert.equal(errCall.status.code, 2, 'errored tool call → ERROR status')
  assert.equal(errCall.status.message, 'boom')
  assert.equal(attr(errCall.attributes, 'gen_ai.tool.call.id') !== undefined, true)
  ok('mapToolCallSpan: execute_tool span + error status propagation')
}

// --- whole-run mapping + parenting ---
{
  const spans = mapRunToSpans(run)
  assert.equal(spans.length, 4, 'root + 1 turn + 2 tool calls')
  const root = spans[0]
  assert.equal(root.name, 'gen_ai.run run-42')
  assert.equal(root.parentSpanId, undefined, 'root has no parent')
  assert.equal((attr(root.attributes, 'gen_ai.run.id') as any).stringValue, 'run-42')
  assert.equal((attr(root.attributes, 'gen_ai.conversation.id') as any).stringValue, 'thread-9')

  // all spans share one trace id
  assert.ok(spans.every((s) => s.traceId === root.traceId), 'one trace id per run')

  const turn = spans[1]
  assert.equal(turn.parentSpanId, root.spanId, 'turn parents onto root')

  const c1 = spans.find((s) => s.name === 'execute_tool read_file')!
  const c2 = spans.find((s) => s.name === 'execute_tool mcp__repo__search')!
  assert.equal(c1.parentSpanId, turn.spanId, 'tool call parents onto its turn')
  assert.equal(c2.parentSpanId, root.spanId, 'tool call with no parentTurnId falls back to root')

  // deterministic ids across two mappings
  const again = mapRunToSpans(run)
  assert.deepEqual(again.map((s) => s.spanId), spans.map((s) => s.spanId), 'span ids are deterministic')
  ok('mapRunToSpans: root+children, shared trace id, correct parenting, deterministic')
}

// --- OTLP envelope ---
{
  const payload = buildRunOtlpPayload(run, { serviceName: 'vontra-studio', serviceVersion: '2.0', attributes: { 'deployment.environment': 'test', foo: 3, flag: true } })
  const rs = payload.resourceSpans[0]
  const svc = rs.resource.attributes.find((a) => a.key === 'service.name')!
  assert.equal((svc.value as any).stringValue, 'vontra-studio')
  const ver = rs.resource.attributes.find((a) => a.key === 'service.version')!
  assert.equal((ver.value as any).stringValue, '2.0')
  const env = rs.resource.attributes.find((a) => a.key === 'deployment.environment')!
  assert.equal((env.value as any).stringValue, 'test')
  const foo = rs.resource.attributes.find((a) => a.key === 'foo')!
  assert.equal((foo.value as any).intValue, 3)
  const flag = rs.resource.attributes.find((a) => a.key === 'flag')!
  assert.equal((flag.value as any).boolValue, true)
  assert.equal(rs.scopeSpans[0].scope.name, 'vontra.otel.genai')
  assert.equal(rs.scopeSpans[0].spans.length, 4)
  // round-trips through JSON (fully serializable)
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(payload)))
  ok('buildRunOtlpPayload: resource + scope envelope, typed attrs, JSON-serializable')
}

// --- emitter: success path (DI sender) ---
{
  type Captured = { url: string; body: string; headers: Record<string, string> }
  let captured: Captured | null = null
  let responseCancelled = false
  const responseBody = new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array([1])) },
    cancel() { responseCancelled = true },
  })
  const send: HttpSender = async (url, init) => {
    captured = { url, body: init.body, headers: init.headers }
    return { status: 200, ok: true, body: responseBody }
  }
  const res = await emitRun(run, {
    endpoint: 'http://collector:4318/v1/traces',
    send,
    headers: { authorization: 'Bearer xyz' },
    resource: { serviceName: 'vontra' },
  })
  assert.equal(res.ok, true)
  assert.equal(res.status, 200)
  assert.equal(res.spanCount, 4)
  assert.ok(captured, 'sender was invoked')
  const cap = captured as Captured
  assert.equal(cap.url, 'http://collector:4318/v1/traces')
  assert.equal(cap.headers['content-type'], 'application/json')
  assert.equal(cap.headers['authorization'], 'Bearer xyz')
  const parsed = JSON.parse(cap.body)
  assert.equal(parsed.resourceSpans[0].scopeSpans[0].spans.length, 4, 'body carries all spans')
  assert.equal(responseCancelled, true, 'successful collector response bodies must release their socket')
  ok('emitRun: posts OTLP/JSON via injected sender with headers and releases the response body')
}

// --- emitter: HTTP-error path ---
{
  const send: HttpSender = async () => new Response('rate limited', { status: 429 })
  const res = await emitSpans(mapRunToSpans(run), { endpoint: 'http://x/v1/traces', send })
  assert.equal(res.ok, false)
  assert.equal(res.status, 429)
  assert.match(res.error!, /HTTP 429/)
  assert.match(res.error!, /rate limited/)
  assert.ok(res.payload, 'failed result still returns the payload that was attempted')
  ok('emitSpans: HTTP error → ok:false with status + body, never throws')
}

// --- emitter: hostile oversized HTTP-error body ---
{
  let cancelled = false
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('x'.repeat(OTLP_ERROR_BODY_MAX_BYTES + 1)))
    },
    cancel() { cancelled = true },
  })
  const send: HttpSender = async () => ({ status: 500, ok: false, headers: new Headers({ 'content-type': 'text/plain' }), body })
  const res = await emitSpans(mapRunToSpans(run), { endpoint: 'http://x/v1/traces', send })
  assert.equal(res.ok, false)
  assert.equal(res.error, 'OTLP export failed: HTTP 500')
  assert.equal(cancelled, true)
  ok('emitSpans: oversized collector error bodies are cancelled before allocation')
}

// --- emitter: transport-throw path ---
{
  const send: HttpSender = async () => {
    throw new Error('ECONNREFUSED')
  }
  const res = await emitSpans(mapRunToSpans(run), { endpoint: 'http://x/v1/traces', send })
  assert.equal(res.ok, false)
  assert.equal(res.status, 0)
  assert.equal(res.error, 'ECONNREFUSED')
  ok('emitSpans: transport throw is caught → ok:false, no exception escapes')
}

// --- empty run ---
{
  const spans = mapRunToSpans({ runId: 'empty', startMs: 0, endMs: 1 })
  assert.equal(spans.length, 1, 'run with no turns/calls → just the root span')
  const payload = buildOtlpPayload(spans)
  assert.equal(payload.resourceSpans[0].resource.attributes[0].key, 'service.name')
  assert.equal((payload.resourceSpans[0].resource.attributes[0].value as any).stringValue, 'vontra', 'default service.name')
  ok('mapRunToSpans: empty run yields a single root span')
}

console.log(`\notel-genai: ${passed} checks passed`)
