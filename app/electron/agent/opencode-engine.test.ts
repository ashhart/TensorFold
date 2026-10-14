// Run: jiti electron/agent/opencode-engine.test.ts
//
// OpenCode engine (VST-033). Shapes verified against installed opencode 1.17's OpenAPI +
// live SSE probe. Fake transport → no server spawned.
import assert from 'node:assert/strict'
import {
  isOpenCodeProtocol,
  openCodeModelField,
  openCodePromptBody,
  parseOpenCodeSseLine,
  chunkFromOpenCodeEvent,
  createOpenCodeHttpTransport,
  openCodeTerminalOutcome,
  runOpenCodeTurn,
  type OpenCodeEvent,
  type OpenCodeTransport,
  type OpenCodeStreamChunk,
} from './opencode-engine'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('opencode-engine tests:')

// --- shared-protocol helper (Kilo is an OpenCode fork on the same server) ---
{
  assert.equal(isOpenCodeProtocol('opencode'), true)
  assert.equal(isOpenCodeProtocol('kilo'), true, 'Kilo drives the same engine')
  assert.equal(isOpenCodeProtocol('codex-cli'), false)
  assert.equal(isOpenCodeProtocol('chat-completions'), false)
  assert.equal(isOpenCodeProtocol(undefined), false)
  ok('isOpenCodeProtocol: opencode + kilo only')
}

// --- model id split ---
{
  assert.deepEqual(openCodeModelField('anthropic/claude-3-5-sonnet'), { providerID: 'anthropic', modelID: 'claude-3-5-sonnet' })
  assert.deepEqual(openCodeModelField('openrouter/meta-llama/llama-3'), { providerID: 'openrouter', modelID: 'meta-llama/llama-3' }, 'only the FIRST slash splits')
  assert.deepEqual(openCodeModelField('bare-model'), { providerID: '', modelID: 'bare-model' }, 'no slash → empty provider, server default')
  ok('openCodeModelField: first-slash split, bare id tolerated')
}

// --- prompt body ---
{
  assert.deepEqual(openCodePromptBody('hello'), { parts: [{ type: 'text', text: 'hello' }] }, 'no model → server default (no model field)')
  assert.deepEqual(openCodePromptBody('hi', { model: 'opencode-default' }).model, undefined, 'the sentinel default is not sent as a model')
  const full = openCodePromptBody('hi', { model: 'anthropic/opus', agent: 'build', system: 'be terse' })
  assert.deepEqual(full.model, { providerID: 'anthropic', modelID: 'opus' })
  assert.equal(full.agent, 'build')
  assert.equal(full.system, 'be terse')
  ok('openCodePromptBody: parts + optional model/agent/system')
}

// --- SSE line parsing ---
{
  assert.deepEqual(parseOpenCodeSseLine('data: {"type":"server.connected","properties":{}}'), { type: 'server.connected', properties: {} })
  assert.deepEqual(parseOpenCodeSseLine('data: {"type":"session.idle","properties":{"sessionID":"ses_1"}}'), { type: 'session.idle', properties: { sessionID: 'ses_1' } })
  assert.equal(parseOpenCodeSseLine(': keep-alive comment'), null)
  assert.equal(parseOpenCodeSseLine(''), null)
  assert.equal(parseOpenCodeSseLine('data: not-json'), null, 'a torn/partial frame is skipped, never throws')
  assert.equal(parseOpenCodeSseLine('data: {"noType":true}'), null)
  ok('parseOpenCodeSseLine: data frames only, junk skipped')
}

// --- event → chunk translation, session scoping ---
{
  const ev = (type: string, props: any): OpenCodeEvent => ({ type, properties: props })
  assert.deepEqual(chunkFromOpenCodeEvent(ev('session.next.text.delta', { sessionID: 'ses_1', delta: 'hi' }), 'ses_1'), { type: 'content', text: 'hi' })
  assert.deepEqual(chunkFromOpenCodeEvent(ev('session.next.reasoning.delta', { sessionID: 'ses_1', delta: 'think' }), 'ses_1'), { type: 'reasoning', text: 'think' })
  assert.equal(chunkFromOpenCodeEvent(ev('session.next.text.delta', { sessionID: 'ses_OTHER', delta: 'hi' }), 'ses_1'), null, 'another session\'s delta is ignored on the shared stream')
  assert.equal(chunkFromOpenCodeEvent(ev('session.next.text.delta', { sessionID: 'ses_1', delta: '' }), 'ses_1'), null, 'empty delta yields no chunk')
  assert.equal(chunkFromOpenCodeEvent(ev('session.next.tool.called', { sessionID: 'ses_1' }), 'ses_1'), null, 'non-text events yield no chunk')
  ok('chunkFromOpenCodeEvent: text/reasoning deltas, session-scoped')
}

// --- terminal outcome ---
{
  const ev = (type: string, props: any): OpenCodeEvent => ({ type, properties: props })
  assert.deepEqual(openCodeTerminalOutcome(ev('session.idle', { sessionID: 'ses_1' }), 'ses_1'), { kind: 'idle' })
  assert.equal((openCodeTerminalOutcome(ev('session.error', { sessionID: 'ses_1', error: { data: { message: 'boom' } } }), 'ses_1') as any).kind, 'error')
  assert.equal((openCodeTerminalOutcome(ev('session.error', { sessionID: 'ses_1', error: { data: { message: 'boom' } } }), 'ses_1') as any).message, 'boom')
  assert.equal((openCodeTerminalOutcome(ev('session.next.step.failed', { sessionID: 'ses_1', error: { name: 'ProviderAuthError' } }), 'ses_1') as any).message, 'ProviderAuthError')
  assert.equal(openCodeTerminalOutcome(ev('session.idle', { sessionID: 'ses_OTHER' }), 'ses_1'), null, 'another session going idle is not our terminal')
  assert.equal(openCodeTerminalOutcome(ev('session.next.text.delta', { sessionID: 'ses_1' }), 'ses_1'), null)
  ok('openCodeTerminalOutcome: idle/error classified, session-scoped')
}

// --- production HTTP transport: cancellation and deadlines ---
{
  let requestSignalAborted = false
  const transport = createOpenCodeHttpTransport({
    serverUrl: 'http://127.0.0.1:4096',
    requestTimeoutMs: 1_000,
    fetchImpl: (async (_url, init) => await new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal
      signal?.addEventListener('abort', () => {
        requestSignalAborted = true
        reject(Object.assign(new Error('fetch aborted'), { name: 'AbortError' }))
      }, { once: true })
    })) as typeof fetch,
  })
  const controller = new AbortController()
  const pending = transport.createSession('TensorFold Studio', controller.signal)
  setTimeout(() => controller.abort(), 5)
  await assert.rejects(pending, (error: any) => error?.name === 'AbortError')
  assert.equal(requestSignalAborted, true, 'run cancellation reaches the underlying create-session fetch')

  requestSignalAborted = false
  const promptController = new AbortController()
  const promptPending = transport.postPromptAsync('ses_1', openCodePromptBody('hello'), promptController.signal)
  setTimeout(() => promptController.abort(), 5)
  await assert.rejects(promptPending, (error: any) => error?.name === 'AbortError')
  assert.equal(requestSignalAborted, true, 'run cancellation reaches the underlying prompt fetch')

  requestSignalAborted = false
  const abortController = new AbortController()
  const abortPending = transport.abort('ses_1', abortController.signal)
  setTimeout(() => abortController.abort(), 5)
  await assert.rejects(abortPending, (error: any) => error?.name === 'AbortError')
  assert.equal(requestSignalAborted, true, 'run cancellation reaches the underlying abort fetch when supplied')
  ok('HTTP create/prompt/abort: cancellation interrupts hung requests')
}

{
  const transport = createOpenCodeHttpTransport({
    serverUrl: 'http://127.0.0.1:4096',
    requestTimeoutMs: 10,
    abortTimeoutMs: 10,
    fetchImpl: (async () => await new Promise<Response>(() => {})) as typeof fetch,
  })
  await assert.rejects(transport.postPromptAsync('ses_1', openCodePromptBody('hello')), /OpenCode prompt request timed out after 10ms/)
  await assert.rejects(transport.abort('ses_1'), /OpenCode abort request timed out after 10ms/)
  ok('HTTP prompt/abort: hung requests have hard deadlines')
}

{
  let cancelled = false
  const transport = createOpenCodeHttpTransport({
    serverUrl: 'http://127.0.0.1:4096',
    idleTimeoutMs: 10,
    fetchImpl: (async () => new Response(new ReadableStream<Uint8Array>({
      start() {},
      cancel() { cancelled = true },
    }), { headers: { 'content-type': 'text/event-stream' } })) as typeof fetch,
  })
  const events = transport.events()
  await assert.rejects(events.next(), /OpenCode event stream was idle for 10ms/)
  assert.equal(cancelled, true, 'idle SSE body is cancelled before the transport fails')
  ok('HTTP events: a half-open SSE read fails at the idle deadline')
}

{
  const transport = createOpenCodeHttpTransport({
    serverUrl: 'http://127.0.0.1:4096',
    requestTimeoutMs: 1_000,
    fetchImpl: (async () => await new Promise<Response>(() => {})) as typeof fetch,
  })
  const controller = new AbortController()
  const events = transport.events(controller.signal)
  const pending = events.next()
  setTimeout(() => controller.abort(), 5)
  await assert.rejects(pending, (error: any) => error?.name === 'AbortError')
  ok('HTTP events: run cancellation interrupts a hung SSE connect')
}

// --- full turn driver with a fake transport ---
function fakeTransport(script: OpenCodeEvent[], opts: { createId?: string; onAbort?: () => void; onPrompt?: (id: string, body: any) => void } = {}): OpenCodeTransport {
  return {
    async createSession() { return opts.createId ?? 'ses_new' },
    async postPromptAsync(id, body) { opts.onPrompt?.(id, body) },
    async abort() { opts.onAbort?.() },
    async *events() { for (const e of script) yield e },
  }
}

async function collect(gen: AsyncGenerator<OpenCodeStreamChunk>): Promise<OpenCodeStreamChunk[]> {
  const out: OpenCodeStreamChunk[] = []
  for await (const c of gen) out.push(c)
  return out
}

{
  // The real transport's events() is a lazy async generator: GET /event and its MIME/read
  // setup do not happen until next(). Prompt submission must wait for that setup to yield
  // the initial server.connected frame, otherwise a fast first delta can be lost.
  const order: string[] = []
  let releaseReady!: () => void
  const readyGate = new Promise<void>((resolve) => { releaseReady = resolve })
  const transport: OpenCodeTransport = {
    async createSession() { order.push('create'); return 'ses_ready' },
    async postPromptAsync() {
      order.push('prompt')
      assert.ok(order.includes('subscription:ready'), 'prompt observes a live SSE reader')
    },
    async abort() {},
    async *events() {
      order.push('subscription:start')
      await readyGate
      order.push('subscription:ready')
      yield { type: 'server.connected', properties: {} }
      yield { type: 'session.next.text.delta', properties: { sessionID: 'ses_ready', delta: 'captured' } }
      yield { type: 'session.idle', properties: { sessionID: 'ses_ready' } }
    },
  }
  const pending = collect(runOpenCodeTurn({ transport, prompt: 'race-safe' }))
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
  assert.deepEqual(order, ['subscription:start'], 'create and prompt stay blocked until the lazy subscription is actually ready')
  releaseReady()
  const chunks = await pending
  assert.deepEqual(order, ['subscription:start', 'subscription:ready', 'create', 'prompt'])
  assert.equal((chunks.find((chunk) => chunk.type === 'content') as any)?.text, 'captured', 'priming does not discard the first turn delta')
  ok('runOpenCodeTurn: awaits lazy subscription readiness before prompt')
}

{
  // Fresh session: create → cursor → deltas → idle.
  const script: OpenCodeEvent[] = [
    { type: 'server.connected', properties: {} },
    { type: 'session.next.reasoning.delta', properties: { sessionID: 'ses_new', delta: 'planning' } },
    { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: 'Hello' } },
    { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: ' world' } },
    { type: 'session.idle', properties: { sessionID: 'ses_new' } },
    { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: 'AFTER IDLE' } },
  ]
  let prompted: any = null
  const chunks = await collect(runOpenCodeTurn({ transport: fakeTransport(script, { onPrompt: (_id, body) => { prompted = body } }), prompt: 'hi', model: 'anthropic/opus' }))
  const cursor = chunks.find((c) => c.type === 'resume_cursor') as any
  assert.equal(cursor?.cursor.sessionId, 'ses_new', 'a fresh session emits its cursor')
  const text = chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join('')
  assert.equal(text, 'Hello world', 'text deltas relayed until idle; post-idle delta dropped')
  assert.ok(chunks.some((c) => c.type === 'reasoning'), 'reasoning relayed')
  assert.deepEqual(prompted.model, { providerID: 'anthropic', modelID: 'opus' }, 'the model reached the prompt as an object')
  ok('runOpenCodeTurn: fresh session streams to idle, emits cursor')
}

{
  // Resume: no create, reuse the cursor's session id.
  let created = 0
  const transport: OpenCodeTransport = {
    async createSession() { created++; return 'ses_should_not_be_used' },
    async postPromptAsync() {},
    async abort() {},
    async *events() {
      yield { type: 'session.next.text.delta', properties: { sessionID: 'ses_kept', delta: 'resumed' } }
      yield { type: 'session.idle', properties: { sessionID: 'ses_kept' } }
    },
  }
  const chunks = await collect(runOpenCodeTurn({ transport, prompt: 'again', resumeCursor: { sessionId: 'ses_kept' } }))
  assert.equal(created, 0, 'resume never creates a new session')
  assert.equal((chunks.find((c) => c.type === 'content') as any).text, 'resumed')
  ok('runOpenCodeTurn: resume reuses the session, no create')
}

{
  const controller = new AbortController()
  let createSignal: AbortSignal | undefined
  let promptSignal: AbortSignal | undefined
  const transport: OpenCodeTransport = {
    async createSession(_title, signal) { createSignal = signal; return 'ses_signal' },
    async postPromptAsync(_id, _body, signal) { promptSignal = signal },
    async abort() {},
    async *events() { yield { type: 'session.idle', properties: { sessionID: 'ses_signal' } } },
  }
  await collect(runOpenCodeTurn({ transport, prompt: 'signal', signal: controller.signal }))
  assert.equal(createSignal, controller.signal)
  assert.equal(promptSignal, controller.signal)
  ok('runOpenCodeTurn: the run AbortSignal reaches create and prompt')
}

{
  // Error event → throw.
  const script: OpenCodeEvent[] = [
    { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: 'partial' } },
    { type: 'session.error', properties: { sessionID: 'ses_new', error: { data: { message: 'provider auth failed' } } } },
  ]
  await assert.rejects(collect(runOpenCodeTurn({ transport: fakeTransport(script), prompt: 'hi' })), /provider auth failed/)
  ok('runOpenCodeTurn: a session.error aborts the turn with its message')
}

{
  // Abort → server-side abort called, generator stops.
  let aborted = false
  const controller = new AbortController()
  const transport: OpenCodeTransport = {
    async createSession() { return 'ses_new' },
    async postPromptAsync() {},
    async abort() { aborted = true },
    async *events() {
      yield { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: 'one' } }
      controller.abort()
      yield { type: 'session.next.text.delta', properties: { sessionID: 'ses_new', delta: 'two' } }
    },
  }
  const chunks = await collect(runOpenCodeTurn({ transport, prompt: 'hi', signal: controller.signal }))
  const text = chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join('')
  assert.equal(text, 'one', 'streaming stops at the abort')
  assert.equal(aborted, true, 'the server-side turn is aborted on cancel')
  ok('runOpenCodeTurn: abort stops streaming and aborts server-side')
}

{
  // A caller may stop consuming without aborting its signal (navigation/engine teardown).
  // Returning the generator after prompt submission must still interrupt the live server turn.
  let aborted = 0
  const transport: OpenCodeTransport = {
    async createSession() { return 'ses_return' },
    async postPromptAsync() {},
    async abort() { aborted++ },
    async *events() {
      yield { type: 'server.connected', properties: {} }
      yield { type: 'session.next.text.delta', properties: { sessionID: 'ses_return', delta: 'partial' } }
      await new Promise<void>(() => {})
    },
  }
  const turn = runOpenCodeTurn({ transport, prompt: 'stop early' })
  assert.equal((await turn.next()).value?.type, 'resume_cursor')
  assert.deepEqual(await turn.next(), { done: false, value: { type: 'content', text: 'partial' } })
  await turn.return(undefined)
  assert.equal(aborted, 1, 'consumer return interrupts the submitted nonterminal turn')
  ok('runOpenCodeTurn: consumer return aborts a nonterminal submitted prompt')
}

console.log(`\n${passed} passed`)
