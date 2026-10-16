import assert from 'node:assert/strict'
import {
  CodexAppServerSession,
  JsonRpcStdio,
  chunkFromNotification,
  isRecoverableThreadResumeError,
  getOrCreateCodexAppServerSession,
  closeCodexAppServerSession,
  closeAllCodexAppServerSessions,
  pooledCodexAppServerSessionKeys,
  streamCodexAppServer,
  latestUserMessageText,
  withProtocolIdleTimeout,
  type CodexStdioTransport,
  type CodexStreamChunk,
} from './codex-app-server'

// A fake stdio transport. Tests push inbound lines via `emit(...)`; outbound
// writes are recorded in `sent` (parsed). An optional auto-responder lets a test
// script a JSON-RPC reply (and follow-up notifications) for a given method.
class FakeTransport implements CodexStdioTransport {
  sent: any[] = []
  killed = false
  private lineCb: ((line: string) => void) | null = null
  // method -> (id, params) => respond/notify
  responders = new Map<string, (id: number, params: any) => void>()

  write(line: string): void {
    const msg = JSON.parse(line)
    this.sent.push(msg)
    if (msg.id != null && msg.method) {
      const responder = this.responders.get(msg.method)
      if (responder) responder(msg.id, msg.params)
    }
  }
  onLine(cb: (line: string) => void): void {
    this.lineCb = cb
  }
  kill(): void {
    this.killed = true
  }
  // Push an inbound raw line (response or notification) to the client.
  emit(obj: unknown): void {
    this.lineCb?.(JSON.stringify(obj))
  }
  respond(method: string, fn: (id: number, params: any) => void): void {
    this.responders.set(method, fn)
  }
  // Find the last request sent for a method.
  lastRequest(method: string): any {
    return [...this.sent].reverse().find((m) => m.method === method)
  }
}

async function collect(gen: AsyncGenerator<CodexStreamChunk>): Promise<CodexStreamChunk[]> {
  const out: CodexStreamChunk[] = []
  for await (const c of gen) out.push(c)
  return out
}

// --- pure helpers ---
assert.deepEqual(chunkFromNotification('item/agentMessage/delta', { delta: 'hi' }), {
  type: 'content',
  text: 'hi',
})
assert.deepEqual(chunkFromNotification('item/reasoning/textDelta', { delta: 'thinking' }), {
  type: 'reasoning',
  text: 'thinking',
})
assert.equal(chunkFromNotification('item/agentMessage/delta', { delta: '' }), null)
assert.equal(chunkFromNotification('turn/started', {}), null)

assert.equal(isRecoverableThreadResumeError(new Error('thread not found')), true)
assert.equal(isRecoverableThreadResumeError(new Error('no such thread')), true)
assert.equal(isRecoverableThreadResumeError(new Error('rate limited')), false)
assert.equal(isRecoverableThreadResumeError(new Error('file not found')), false)

await assert.rejects(withProtocolIdleTimeout(new Promise(() => {}), 10, 'test protocol'), /test protocol was idle for 10ms/)

// --- JsonRpcStdio: request/response id correlation + notification routing ---
async function testRpcFraming() {
  const t = new FakeTransport()
  t.respond('ping', (id) => t.emit({ jsonrpc: '2.0', id, result: { pong: true } }))
  const rpc = new JsonRpcStdio(t)
  const result = await rpc.request('ping', { x: 1 })
  assert.deepEqual(result, { pong: true })
  // outbound framing carries jsonrpc + incrementing id
  assert.equal(t.sent[0].jsonrpc, '2.0')
  assert.equal(t.sent[0].id, 1)

  let routed: any = null
  rpc.on('note', (p) => { routed = p })
  t.emit({ jsonrpc: '2.0', method: 'note', params: { hello: 'world' } })
  assert.deepEqual(routed, { hello: 'world' })

  // error responses reject
  t.respond('boom', (id) => t.emit({ jsonrpc: '2.0', id, error: { code: -1, message: 'kaboom' } }))
  await assert.rejects(rpc.request('boom'), /kaboom/)
}

// --- initialize -> startThread -> runTurn yields content, ends on turn/completed ---
async function testFullTurn() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) =>
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-abc' } } }),
  )
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-1' } } })
    // stream reasoning then content, then complete
    t.emit({ jsonrpc: '2.0', method: 'turn/started', params: { turn: { id: 'turn-1' } } })
    t.emit({
      jsonrpc: '2.0',
      method: 'item/reasoning/textDelta',
      params: { turnId: 'turn-1', delta: 'plan ' },
    })
    t.emit({
      jsonrpc: '2.0',
      method: 'item/agentMessage/delta',
      params: { turnId: 'turn-1', delta: 'Hello' },
    })
    t.emit({
      jsonrpc: '2.0',
      method: 'item/agentMessage/delta',
      params: { turnId: 'turn-1', delta: ' world' },
    })
    t.emit({
      jsonrpc: '2.0',
      method: 'turn/completed',
      params: { turn: { id: 'turn-1', status: 'completed' } },
    })
  })

  const session = new CodexAppServerSession({ transport: t })
  await session.initialize()
  // initialized notification went out (no id)
  assert.ok(t.sent.some((m) => m.method === 'initialized' && m.id == null))

  const threadId = await session.startThread({ cwd: '/repo' })
  assert.equal(threadId, 'thread-abc')
  assert.deepEqual(session.resumeCursor, { threadId: 'thread-abc' })

  const chunks = await collect(session.runTurn('hi there', { model: 'gpt-5-codex' }))
  assert.deepEqual(chunks, [
    { type: 'reasoning', text: 'plan ' },
    { type: 'content', text: 'Hello' },
    { type: 'content', text: ' world' },
  ])
  // turn/start carried the prompt as structured text input + threadId + model
  const turnReq = t.lastRequest('turn/start')
  assert.equal(turnReq.params.threadId, 'thread-abc')
  assert.deepEqual(turnReq.params.input, [{ type: 'text', text: 'hi there' }])
  assert.equal(turnReq.params.model, 'gpt-5-codex')
}

async function testSilentTurnTimeout() {
  const t = new FakeTransport()
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-silent' } } }))
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-silent' } } })
    // Deliberately never emit a delta, completion, error, or process close.
  })
  const session = new CodexAppServerSession({ transport: t })
  await session.startThread({})
  await assert.rejects(collect(session.runTurn('go', { idleTimeoutMs: 15 })), /idle for 15ms/)
  assert.equal(t.killed, true, 'a silent session is killed so the pool cannot reuse it')
  assert.equal(session.isClosed, true)
}

// --- runTurn throws when the turn completes with status failed ---
async function testFailedTurn() {
  const t = new FakeTransport()
  t.respond('thread/start', (id) =>
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-x' } } }),
  )
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-9' } } })
    t.emit({
      jsonrpc: '2.0',
      method: 'turn/completed',
      params: { turn: { id: 'turn-9', status: 'failed', error: { message: 'model exploded' } } },
    })
  })
  const session = new CodexAppServerSession({ transport: t })
  await session.startThread()
  await assert.rejects(collect(session.runTurn('x')), /model exploded/)
}

// --- resumeThread sends thread/resume with the cursor's threadId ---
async function testResume() {
  const t = new FakeTransport()
  t.respond('thread/resume', (id, params) => {
    assert.equal(params.threadId, 'prior-thread')
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'prior-thread' } } })
  })
  const session = new CodexAppServerSession({ transport: t })
  const tid = await session.resumeThread('prior-thread', { cwd: '/repo' })
  assert.equal(tid, 'prior-thread')
  assert.ok(t.lastRequest('thread/resume'))
}

// --- resume falls back to thread/start on a recoverable "thread not found" ---
async function testResumeFallback() {
  const t = new FakeTransport()
  t.respond('thread/resume', (id) =>
    t.emit({ jsonrpc: '2.0', id, error: { code: -1, message: 'thread not found' } }),
  )
  t.respond('thread/start', (id) =>
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'fresh-thread' } } }),
  )
  const session = new CodexAppServerSession({ transport: t })
  const tid = await session.resumeThread('gone-thread')
  assert.equal(tid, 'fresh-thread')
  assert.ok(t.lastRequest('thread/start'))
}

// --- interrupt sends turn/interrupt for the active turn ---
async function testInterrupt() {
  const t = new FakeTransport()
  t.respond('thread/start', (id) =>
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-i' } } }),
  )
  // turn/start replies but never completes — interrupt while pending
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-int' } } })
  })
  t.respond('turn/interrupt', (id, params) => {
    assert.equal(params.threadId, 'thread-i')
    assert.equal(params.turnId, 'turn-int')
    t.emit({ jsonrpc: '2.0', id, result: {} })
    // After interrupt, the server completes the turn so runTurn can finish.
    t.emit({
      jsonrpc: '2.0',
      method: 'turn/completed',
      params: { turn: { id: 'turn-int', status: 'completed' } },
    })
  })
  const session = new CodexAppServerSession({ transport: t })
  await session.startThread()
  const gen = session.runTurn('long task')
  // Kick the generator to send turn/start and register handlers.
  const first = gen.next()
  await new Promise((r) => setTimeout(r, 0))
  await session.interrupt()
  await first
  // Drain remainder.
  await collect((async function* () {
    let n = await first
    while (!n.done) {
      yield n.value
      n = await gen.next()
    }
  })())
  assert.ok(t.lastRequest('turn/interrupt'))
}

// --- close kills the transport ---
async function testClose() {
  const t = new FakeTransport()
  const session = new CodexAppServerSession({ transport: t })
  session.close()
  assert.equal(t.killed, true)
}

// --- latestUserMessageText: the newest user-role message only ---
function testLatestUserMessageText() {
  assert.equal(latestUserMessageText([
    { role: 'system', content: 'sys' } as any,
    { role: 'user', content: 'first' } as any,
    { role: 'assistant', content: 'reply' } as any,
    { role: 'user', content: 'second' } as any,
  ]), 'second')
  assert.equal(latestUserMessageText([{ role: 'system', content: 'sys' } as any]), '')
  assert.equal(latestUserMessageText([]), '')
}

// --- session pool: reuse, fresh-start vs resume, manual close/evict ---
async function testPoolReuse() {
  let starts = 0
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) => { starts++; t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'pool-thread' } } }) })

  const first = await getOrCreateCodexAppServerSession('key-1', () => t, { cwd: '/repo' }, null)
  assert.equal(first.needsFullHistory, true, 'a brand-new thread (no resume cursor) needs the full history on its first turn')
  assert.equal(starts, 1)

  const second = await getOrCreateCodexAppServerSession('key-1', () => t, { cwd: '/repo' }, null)
  assert.equal(second.session, first.session, 'the SAME key reuses the SAME live session, not a fresh process')
  assert.equal(second.needsFullHistory, false, 'a reused pooled session does not need the full history again')
  assert.equal(starts, 1, 'reuse does not call thread/start a second time')

  assert.deepEqual(pooledCodexAppServerSessionKeys().includes('key-1'), true)
  closeCodexAppServerSession('key-1')
  assert.deepEqual(pooledCodexAppServerSessionKeys().includes('key-1'), false, 'closing evicts the key from the pool')

  // After eviction, the SAME key spawns a genuinely fresh session again.
  const third = await getOrCreateCodexAppServerSession('key-1', () => t, { cwd: '/repo' }, null)
  assert.notEqual(third.session, first.session)
  assert.equal(starts, 2)
  closeAllCodexAppServerSessions()
}

async function testPoolResume() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/resume', (id, params) => {
    assert.equal(params.threadId, 'old-thread')
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'old-thread' } } })
  })
  const { needsFullHistory } = await getOrCreateCodexAppServerSession('key-resume', () => t, { cwd: '/repo' }, { threadId: 'old-thread' })
  assert.equal(needsFullHistory, false, 'a resumed thread already has its history server-side -- no need to re-send it')
  assert.ok(t.lastRequest('thread/resume'))
  closeAllCodexAppServerSessions()
}

// --- streamCodexAppServer: end-to-end via a fake transport -- fresh turn sends the full
// flattened history, a second turn on the SAME session key sends only the newest message,
// and a resume_cursor chunk is yielded once the turn completes. ---
async function testStreamCodexAppServer() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'stream-thread' } } }))
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-s1' } } })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'turn-s1', delta: 'ok' } })
    t.emit({ jsonrpc: '2.0', method: 'turn/completed', params: { turn: { id: 'turn-s1', status: 'completed' } } })
  })

  const provider = { id: 'p1', cliPath: undefined, cliEnv: {}, cliApprovalPolicy: undefined, cliSandbox: undefined } as any
  const messages1 = [
    { role: 'system', content: 'be terse' } as any,
    { role: 'user', content: 'hello there' } as any,
  ]
  const chunks1: any[] = []
  for await (const c of streamCodexAppServer({ provider, model: 'codex-cli-default', messages: messages1, cwd: '/repo', sessionKey: 'stream-key', resumeCursor: null, spawnTransport: () => t })) {
    chunks1.push(c)
  }
  assert.ok(chunks1.some((c) => c.type === 'content' && c.text === 'ok'))
  const cursorChunk1 = chunks1.find((c) => c.type === 'resume_cursor')
  assert.deepEqual(cursorChunk1?.cursor, { threadId: 'stream-thread' }, 'the resume cursor is yielded once the turn completes')
  const firstTurnReq = t.lastRequest('turn/start')
  assert.match(firstTurnReq.params.input[0].text, /be terse/, 'the FIRST turn on a fresh thread sends the full flattened history')
  assert.match(firstTurnReq.params.input[0].text, /hello there/)

  // Second turn, SAME sessionKey -> pooled reuse -> only the newest user message.
  const messages2 = [...messages1, { role: 'assistant', content: 'ok' } as any, { role: 'user', content: 'now do X' } as any]
  const chunks2: any[] = []
  for await (const c of streamCodexAppServer({ provider, model: 'codex-cli-default', messages: messages2, cwd: '/repo', sessionKey: 'stream-key', resumeCursor: null, spawnTransport: () => t })) {
    chunks2.push(c)
  }
  const secondTurnReq = t.lastRequest('turn/start')
  assert.deepEqual(secondTurnReq.params.input, [{ type: 'text', text: 'now do X' }], 'a SUBSEQUENT turn on the same pooled session sends ONLY the newest message, not the full history again')

  closeAllCodexAppServerSessions()
}

// --- rate limits: account/rateLimits/updated during a turn becomes a rate_limits chunk ---
async function testRateLimits() {
  const t = new FakeTransport()
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-rl' } } }))
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-1' } } })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'turn-1', delta: 'hi' } })
    // The real codex notification: a sparse rolling rate-limit update.
    t.emit({ jsonrpc: '2.0', method: 'account/rateLimits/updated', params: { rateLimits: { primary: { usedPercent: 61, resetsAt: 123, windowDurationMins: 300 }, secondary: { usedPercent: 22 }, planType: 'plus' } } })
    t.emit({ jsonrpc: '2.0', method: 'turn/completed', params: { turn: { id: 'turn-1', status: 'completed' } } })
  })
  const session = new CodexAppServerSession({ transport: t })
  await session.startThread({})
  const chunks = await collect(session.runTurn('go'))
  const rate = chunks.find((c) => c.type === 'rate_limits') as { type: 'rate_limits'; snapshot: any } | undefined
  assert.ok(rate, 'the rate-limit notification surfaces as a chunk')
  assert.equal(rate!.snapshot.primary.usedPercent, 61)
  assert.equal(rate!.snapshot.secondary.usedPercent, 22)
  assert.equal(rate!.snapshot.planType, 'plus')
  // Content still streams normally alongside it.
  assert.ok(chunks.some((c) => c.type === 'content' && c.text === 'hi'))
}

// --- live steering: a mid-turn cue becomes turn/steer with the expectedTurnId precondition ---
async function testSteering() {
  const { createSteeringChannel } = await import('./steering')
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-s' } } }))
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-9' } } })
    t.emit({ jsonrpc: '2.0', method: 'turn/started', params: { turn: { id: 'turn-9' } } })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'turn-9', delta: 'working' } })
    // The turn stays open until the steer arrives (the responder below completes it).
  })
  t.respond('turn/steer', (id, params) => {
    assert.equal(params.threadId, 'thread-s')
    assert.equal(params.expectedTurnId, 'turn-9', 'steer carries the ACTIVE turn id precondition')
    assert.deepEqual(params.input, [{ type: 'text', text: 'change course' }])
    t.emit({ jsonrpc: '2.0', id, result: {} })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'turn-9', delta: ' steered' } })
    t.emit({ jsonrpc: '2.0', method: 'turn/completed', params: { turn: { id: 'turn-9', status: 'completed' } } })
  })

  const session = new CodexAppServerSession({ transport: t })
  await session.initialize()
  await session.startThread({ cwd: '/repo' })

  // No active turn yet → steer refuses locally without a request.
  assert.equal(await session.steer('too early'), false, 'steer before a turn is refused')
  assert.equal(t.lastRequest('turn/steer'), undefined)

  const channel = createSteeringChannel()
  let offer: Promise<boolean> | null = null
  const chunks: CodexStreamChunk[] = []
  for await (const chunk of session.runTurn('go', { steering: channel })) {
    chunks.push(chunk)
    if (!offer && chunk.type === 'content') {
      assert.equal(channel.canAccept, true, 'channel is ready while the turn is active')
      offer = channel.offer({ id: 'cue-1', text: 'change course' })
    }
  }
  assert.equal(await offer!, true, 'the server-acked steer resolves true')
  assert.deepEqual(chunks, [
    { type: 'content', text: 'working' },
    { type: 'content', text: ' steered' },
  ])
  assert.equal(channel.canAccept, false, 'the channel disconnects when the turn ends')

  // A steer refused server-side (turn already over → error response) resolves false.
  const t2 = new FakeTransport()
  t2.respond('initialize', (id) => t2.emit({ jsonrpc: '2.0', id, result: {} }))
  t2.respond('thread/start', (id) => t2.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'thread-r' } } }))
  t2.respond('turn/start', (id) => {
    t2.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-r' } } })
    t2.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { delta: 'quick' } })
  })
  t2.respond('turn/steer', (id) => {
    t2.emit({ jsonrpc: '2.0', id, error: { code: -32600, message: 'turn is no longer active' } })
    t2.emit({ jsonrpc: '2.0', method: 'turn/completed', params: { turn: { id: 'turn-r', status: 'completed' } } })
  })
  const session2 = new CodexAppServerSession({ transport: t2 })
  await session2.initialize()
  await session2.startThread({})
  const channel2 = createSteeringChannel()
  let offer2: Promise<boolean> | null = null
  for await (const chunk of session2.runTurn('go', { steering: channel2 })) {
    if (!offer2 && chunk.type === 'content') offer2 = channel2.offer({ id: 'cue-2', text: 'late' })
  }
  assert.equal(await offer2!, false, 'a server-refused steer resolves false (cue stays visible as not applied)')
}

async function main() {
  await testRpcFraming()
  await testFullTurn()
  await testSilentTurnTimeout()
  await testFailedTurn()
  await testRateLimits()
  await testSteering()
  await testResume()
  await testResumeFallback()
  await testInterrupt()
  await testClose()
  testLatestUserMessageText()
  await testPoolReuse()
  await testPoolResume()
  await testStreamCodexAppServer()
  console.log('codex-app-server.test.ts: all assertions passed')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
