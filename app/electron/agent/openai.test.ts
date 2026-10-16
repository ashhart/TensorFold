// Run: npx tsx electron/agent/openai.test.ts
import assert from 'node:assert'
import {
  delegatedCliHeartbeatText,
  PROVIDER_JSON_MAX_BYTES,
  PROVIDER_SSE_FRAME_MAX_CHARS,
  PROVIDER_TOOL_ARGUMENTS_MAX_CHARS,
  readWithIdleTimeout,
  streamChat,
  withDelegatedCliHeartbeat,
} from './openai.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

{
  await assert.rejects(
    readWithIdleTimeout(() => new Promise<never>(() => {}), 10),
    /idle for 10ms/,
  )
  const aborted = new AbortController()
  aborted.abort()
  await assert.rejects(readWithIdleTimeout(async () => 'late', 1_000, aborted.signal), (error: any) => error?.name === 'AbortError')
  assert.equal(await readWithIdleTimeout(async () => 'ready', 100), 'ready')
  ok('provider stream reads time out and honor abort instead of hanging forever')
}

{
  const aborted = new AbortController()
  aborted.abort()
  const unhandled: unknown[] = []
  const onUnhandled = (error: unknown) => { unhandled.push(error) }
  process.on('unhandledRejection', onUnhandled)
  try {
    await assert.rejects(
      readWithIdleTimeout(() => Promise.reject(new DOMException('Fixture read aborted.', 'AbortError')), 100, aborted.signal),
      (error: any) => error?.name === 'AbortError',
    )
    await new Promise<void>((resolve) => setImmediate(resolve))
    assert.deepStrictEqual(unhandled, [], 'an already-aborted read must still observe its rejected promise')
    const activeFailure = new Error('Fixture active read failure.')
    await assert.rejects(readWithIdleTimeout(() => Promise.reject(activeFailure), 100), (error) => error === activeFailure)
    const rejectingSource = async function* () { throw activeFailure }
    for await (const _chunk of withDelegatedCliHeartbeat(rejectingSource(), { transport: 'codex-cli', name: 'Fixture' }, { signal: aborted.signal })) { /* already cancelled */ }
    await new Promise<void>((resolve) => setImmediate(resolve))
    assert.deepStrictEqual(unhandled, [], 'an already-aborted delegated iterator must still observe its rejected next promise')
    await assert.rejects(async () => {
      for await (const _chunk of withDelegatedCliHeartbeat(rejectingSource(), { transport: 'codex-cli', name: 'Fixture' })) { /* expected failure */ }
    }, (error) => error === activeFailure)
  } finally {
    process.removeListener('unhandledRejection', onUnhandled)
  }
  ok('already-aborted reads/iterators observe rejection while active failures propagate unchanged')
}

async function collectContent() {
  let out = ''
  for await (const chunk of streamChat({
    baseUrl: 'http://127.0.0.1:9000/v1',
    apiKey: '',
    model: 'local-model',
    messages: [{ role: 'user', content: 'hello' }],
  })) {
    if (chunk.type === 'content') out += chunk.text
  }
  return out
}

function sseResponse(parts: string[]): Response {
  const body = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder()
      for (const p of parts) controller.enqueue(enc.encode(p))
      controller.close()
    },
  })
  return new Response(body as any, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

async function collectChunks(): Promise<any[]> {
  const chunks: any[] = []
  for await (const chunk of streamChat({
    baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', model: 'local-model',
    messages: [{ role: 'user', content: 'hello' }],
  })) chunks.push(chunk)
  return chunks
}

async function* delayedContent(delayMs: number): AsyncGenerator<any> {
  await new Promise((resolve) => setTimeout(resolve, delayMs))
  yield { type: 'content', text: 'done' }
}

async function main() {
  console.log('OpenAI-compatible transport tests:')
  const originalFetch = globalThis.fetch
  try {
    for (const transport of ['opencode', 'kilo'] as const) {
      let requests = 0
      globalThis.fetch = async () => { requests++; throw new Error('transport must not be reached') }
      await assert.rejects(async () => {
        for await (const _chunk of streamChat({
          provider: { id: transport, name: transport, kind: 'cloud', transport, serverUrl: 'http://127.0.0.1:4096', baseUrl: '', apiKey: '', models: ['m'] } as any,
          model: 'm', messages: [{ role: 'user', content: 'do not submit' }], approvalMode: 'readonly', cwd: '/tmp',
        })) { /* exhaust */ }
      }, /cannot confine, so it can't be used for planning-only work/)
      assert.equal(requests, 0, `${transport} rejection happens before transport creation or prompt submission`)
    }
    ok('OpenCode and Kilo fail closed before prompt submission (external engines plan only)')

    globalThis.fetch = async () => new Response(JSON.stringify({
      choices: [{ message: { content: 'local response' }, finish_reason: 'stop' }],
    }), { status: 200, headers: { 'content-type': 'application/json' } })
    assert.strictEqual(await collectContent(), 'local response')
    ok('accepts non-streaming JSON chat completions from local providers')

    globalThis.fetch = async () => new Response(`{"padding":"${'x'.repeat(PROVIDER_JSON_MAX_BYTES)}"}`, {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
    await assert.rejects(collectChunks(), /exceeds the .*byte limit/)
    ok('rejects oversized non-streaming provider JSON')

    globalThis.fetch = async () => sseResponse([`data: ${'x'.repeat(PROVIDER_SSE_FRAME_MAX_CHARS + 1)}\n\n`])
    await assert.rejects(collectChunks(), /frame exceeded the safety limit/)
    ok('cancels an oversized provider SSE frame')

    const argumentFragment = 'a'.repeat(64 * 1024)
    const argumentFrames = Array.from({ length: Math.ceil(PROVIDER_TOOL_ARGUMENTS_MAX_CHARS / argumentFragment.length) + 1 }, () =>
      `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: argumentFragment } }] } }] })}\n\n`,
    )
    globalThis.fetch = async () => sseResponse(argumentFrames)
    await assert.rejects(collectChunks(), /tool-call arguments exceeded the safety limit/)
    ok('caps aggregate tool-call arguments across individually valid SSE frames')

    // Tool-call stream that ends with [DONE] and NO truthy finish_reason (vLLM/llama.cpp-style).
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\\"path\\":\\"x\\"}"}}]}}]}\n\n',
      'data: [DONE]\n\n',
    ])
    let chunks = await collectChunks()
    const tc = chunks.find((c) => c.type === 'tool_calls')
    assert.ok(tc, 'tool_calls flushed even without a finish_reason chunk')
    assert.strictEqual(tc.toolCalls[0].function.name, 'read_file')
    assert.ok(chunks.some((c) => c.type === 'finish'), 'a terminal finish is always emitted')
    ok('flushes accumulated tool calls + finish when the stream ends without finish_reason')

    // A mid-stream error frame (no choices) and finish_reason "error" both fail the turn.
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"delta":{"content":"par"}}]}\n\n',
      'data: {"error":{"message":"CUDA out of memory","type":"internal_error"}}\n\n',
    ])
    await assert.rejects(collectChunks(), /Provider error: CUDA out of memory/)
    globalThis.fetch = async () => sseResponse(['data: {"choices":[{"delta":{},"finish_reason":"error"}]}\n\n'])
    await assert.rejects(collectChunks(), /Provider error: .*finish_reason "error"/)
    ok('error frames and finish_reason "error" throw instead of ending as a blank reply')

    // [DONE] ends the read even when the server keeps the body open afterwards.
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) {
        const enc = new TextEncoder()
        controller.enqueue(enc.encode('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'))
      },
    }) as any, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    chunks = await collectChunks()
    assert.strictEqual(chunks.filter((c) => c.type === 'content').map((c) => c.text).join(''), 'ok')
    assert.strictEqual(chunks.filter((c) => c.type === 'finish').length, 1)
    ok('[DONE] finishes the turn without waiting for the body to close')

    // Trusted local servers are asked for usage; the usage frame after finish_reason is read, not dropped.
    let sentBody: any = null
    globalThis.fetch = async (_url: any, init?: any) => {
      sentBody = JSON.parse(String(init?.body ?? '{}'))
      return sseResponse([
        'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
        'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\n',
        'data: [DONE]\n\n',
      ])
    }
    chunks = await collectChunks()
    assert.deepStrictEqual(sentBody.stream_options, { include_usage: true })
    const usage = chunks.find((c) => c.type === 'usage')
    assert.strictEqual(usage?.completionTokens, 2)
    assert.strictEqual(usage?.promptTokens, 12)
    assert.ok(usage.firstTokenAt >= usage.requestAt && usage.lastTokenAt >= usage.firstTokenAt, 'token times are measured from the stream')
    assert.strictEqual(chunks.filter((c) => c.type === 'content').map((c) => c.text).join(''), 'hello')
    assert.strictEqual(chunks.at(-1)?.type, 'usage', 'usage is the last chunk')
    ok('trusted local streams request usage and report real token counts and times')

    // The built-in engine (v1.0.7 chat.zig/openai.zig) puts `tensorfold` timings on the finish chunk and,
    // with include_usage, sends usage (with cached prompt tokens) in its own frame after it.
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":"hi there"},"finish_reason":null}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"exact_mode":"exact",'
        + '"tensorfold":{"enable_thinking":false,"engine":"zig-metal","tokens_per_second":212.5,"seconds":0.4,"prefill_seconds":0.25,"prefill_widths":[512],"time_to_first_token":0.31,"sampling":"greedy","drafts":true},'
        + '"speculative":{"rounds":20,"drafted":120,"accepted":80}}\n\n',
      'data: {"choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":86,"total_tokens":1286,"prompt_tokens_details":{"cached_tokens":400},"completion_tokens_details":{"reasoning_tokens":0}}}\n\n',
      'data: [DONE]\n\n',
    ])
    chunks = await collectChunks()
    const engineUsage = chunks.find((c) => c.type === 'usage')
    assert.deepStrictEqual(engineUsage.engine, { decodeTokensPerSecond: 212.5, decodeSeconds: 0.4, prefillSeconds: 0.25, firstTokenSeconds: 0.31 })
    assert.strictEqual(engineUsage.promptTokens, 1200)
    assert.strictEqual(engineUsage.cachedTokens, 400)
    assert.strictEqual(engineUsage.completionTokens, 86)
    assert.strictEqual(chunks.at(-1)?.type, 'usage', 'usage stays the last chunk')

    // Without include_usage the engine sends usage on the finish chunk itself; a null prefill stays null.
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"tensorfold":{"tokens_per_second":0,"seconds":0,"prefill_seconds":null,"time_to_first_token":null},'
        + '"usage":{"prompt_tokens":50,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":50}}}\n\n',
      'data: [DONE]\n\n',
    ])
    chunks = await collectChunks()
    const cachedUsage = chunks.find((c) => c.type === 'usage')
    assert.deepStrictEqual(cachedUsage.engine, { decodeTokensPerSecond: 0, decodeSeconds: 0, prefillSeconds: null, firstTokenSeconds: null })
    assert.strictEqual(cachedUsage.cachedTokens, 50)

    // Any other server: no `tensorfold` object means no engine timings, so the client's own clock is used.
    globalThis.fetch = async () => sseResponse(['data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],"tensorfold":"nope","usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n', 'data: [DONE]\n\n'])
    chunks = await collectChunks()
    assert.strictEqual(chunks.find((c) => c.type === 'usage').engine, null)
    assert.strictEqual(chunks.find((c) => c.type === 'usage').cachedTokens, null)
    ok('engine reply timings and cached prompt tokens ride on the usage chunk')

    // External providers are never sent stream_options (some cloud APIs reject unknown fields).
    sentBody = null
    globalThis.fetch = async (_url: any, init?: any) => {
      sentBody = JSON.parse(String(init?.body ?? '{}'))
      return sseResponse(['data: {"choices":[{"delta":{"content":"plan"},"finish_reason":"stop"}]}\n\n', 'data: [DONE]\n\n'])
    }
    for await (const _chunk of streamChat({
      provider: { id: 'cloud', name: 'Cloud', kind: 'cloud', transport: 'chat-completions', baseUrl: 'https://1.1.1.1/v1', apiKey: 'k', models: ['m'] } as any,
      baseUrl: 'https://1.1.1.1/v1', apiKey: 'k', model: 'm', messages: [{ role: 'user', content: 'plan it' }],
    })) { /* exhaust */ }
    assert.ok(sentBody, 'external request reached the mocked transport')
    assert.strictEqual(sentBody.stream_options, undefined)
    ok('external providers are not asked for stream usage')

    // A server that ignores include_usage and holds the body open costs only the short tail.
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'))
      },
    }) as any, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    const tailStarted = Date.now()
    chunks = await collectChunks()
    assert.ok(Date.now() - tailStarted < 5_000, 'no full idle timeout after finish')
    assert.strictEqual(chunks.find((c) => c.type === 'usage')?.completionTokens, null, 'no usage means null, never an estimate')
    ok('a missing usage frame ends the stream after a short tail with no token count')

    // One frame carrying a whole large tool call stays under the frame cap.
    const bigArgs = JSON.stringify({ path: 'x', content: 'b'.repeat(PROVIDER_TOOL_ARGUMENTS_MAX_CHARS - 64) })
    globalThis.fetch = async () => sseResponse([
      `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_big', function: { name: 'write_file', arguments: bigArgs } }] }, finish_reason: 'tool_calls' }] })}\n\n`,
    ])
    chunks = await collectChunks()
    assert.strictEqual(chunks.find((c) => c.type === 'tool_calls')?.toolCalls[0].function.arguments.length, bigArgs.length)
    ok('a single frame may carry a tool call up to the argument cap')

    // Final event with finish_reason but NO trailing newline (server closes right after it).
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
    ])
    chunks = await collectChunks()
    assert.strictEqual(chunks.filter((c) => c.type === 'finish').length, 1, 'exactly one finish (no double-emit)')
    assert.strictEqual(chunks.filter((c) => c.type === 'content').map((c) => c.text).join(''), 'hi', 'trailing un-newlined event is parsed')
    ok('parses a final event with no trailing newline')

    // Two PARALLEL tool calls from a backend (Ollama/vLLM/LM Studio-style) that never sends
    // `index` on tool_call deltas at all. Each call starts with an {id, function.name} chunk,
    // followed by argument-only continuation chunks with no id/name/index.
    globalThis.fetch = async () => sseResponse([
      'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_a","function":{"name":"read_file","arguments":""}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"path\\":"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"a.txt\\"}"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_b","function":{"name":"write_file","arguments":""}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"path\\":"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"b.txt\\"}"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
      'data: [DONE]\n\n',
    ])
    chunks = await collectChunks()
    const parallelToolCalls = chunks.find((c) => c.type === 'tool_calls')
    assert.ok(parallelToolCalls, 'tool_calls chunk emitted')
    assert.strictEqual(parallelToolCalls.toolCalls.length, 2, 'two index-less parallel tool calls stay separate, not merged into one')
    const [first, second] = parallelToolCalls.toolCalls
    assert.strictEqual(first.function.name, 'read_file')
    assert.strictEqual(first.function.arguments, '{"path":"a.txt"}')
    assert.strictEqual(second.function.name, 'write_file')
    assert.strictEqual(second.function.arguments, '{"path":"b.txt"}')
    ok('keeps index-less parallel tool-call deltas as separate ToolCall entries instead of collapsing onto index 0')

    // Actual parallel framing: both calls appear at stable array positions in every delta.
    // A single global "current call" corrupts this shape by appending both continuation
    // fragments to B; position continuity must preserve A and B independently.
    globalThis.fetch = async () => sseResponse([
      `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [
        { id: 'call_a', function: { name: 'read_file', arguments: '' } },
        { id: 'call_b', function: { name: 'write_file', arguments: '' } },
      ] } }] })}\n\n`,
      `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [
        { function: { arguments: '{"path":' } },
        { function: { arguments: '{"path":' } },
      ] } }] })}\n\n`,
      `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [
        { function: { arguments: '"a.txt"}' } },
        { function: { arguments: '"b.txt"}' } },
      ] } }] })}\n\n`,
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
    ])
    chunks = await collectChunks()
    const interleaved = chunks.find((c) => c.type === 'tool_calls')
    assert.deepStrictEqual(
      interleaved.toolCalls.map((call: any) => [call.id, call.function.name, call.function.arguments]),
      [
        ['call_a', 'read_file', '{"path":"a.txt"}'],
        ['call_b', 'write_file', '{"path":"b.txt"}'],
      ],
    )
    ok('preserves array-position continuity for truly interleaved index-less parallel tool calls')

    const delegatedChunks: any[] = []
    for await (const chunk of withDelegatedCliHeartbeat(
      delayedContent(25),
      { transport: 'codex-cli', name: 'Codex CLI OAuth' } as any,
      { heartbeatMs: 5 },
    )) {
      delegatedChunks.push(chunk)
    }
    assert.ok(delegatedChunks.some((c) => c.type === 'reasoning' && /Codex is still working/.test(c.text)), 'delegated CLI silence emits a visible progress pulse')
    assert.strictEqual(delegatedChunks.filter((c) => c.type === 'content').map((c) => c.text).join(''), 'done', 'real CLI output still streams after progress pulses')
    assert.match(delegatedCliHeartbeatText('Claude', 65_000, 15_000), /Claude is still working \(1m 5s elapsed; no live CLI event for 15s\)\./)
    ok('delegated CLI streams emit heartbeat reasoning while the external process is silent')
  } finally {
    globalThis.fetch = originalFetch
  }
  console.log(`\n${passed} checks passed.`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
