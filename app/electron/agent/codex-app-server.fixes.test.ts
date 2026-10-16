// Run: jiti electron/agent/codex-app-server.fixes.test.ts
// C-provider-transport HIGH bugs:
//  (1) getOrCreateCodexAppServerSession leaked the spawned child when
//      initialize/startThread/resumeThread rejected (never pooled, never closed).
//  (2) A pooled thread ignored a later turn's changed sandbox/approvalPolicy/cwd,
//      so switching a conversation to read-only had no effect.
//  (3) Aborting a streamCodexAppServer turn never sent turn/interrupt, leaving the
//      codex turn running server-side.
import assert from 'node:assert/strict'
import {
  getOrCreateCodexAppServerSession,
  closeAllCodexAppServerSessions,
  pooledCodexAppServerSessionKeys,
  streamCodexAppServer,
  type CodexStdioTransport,
} from './codex-app-server'

let ok = 0

// A fake transport mirroring codex-app-server.test.ts's, with optional auto-responders.
class FakeTransport implements CodexStdioTransport {
  sent: any[] = []
  killed = false
  private lineCb: ((line: string) => void) | null = null
  responders = new Map<string, (id: number, params: any) => void>()

  write(line: string): void {
    const msg = JSON.parse(line)
    this.sent.push(msg)
    if (msg.id != null && msg.method) {
      const responder = this.responders.get(msg.method)
      if (responder) responder(msg.id, msg.params)
    }
  }
  onLine(cb: (line: string) => void): void { this.lineCb = cb }
  kill(): void { this.killed = true }
  emit(obj: unknown): void { this.lineCb?.(JSON.stringify(obj)) }
  respond(method: string, fn: (id: number, params: any) => void): void { this.responders.set(method, fn) }
  lastRequest(method: string): any { return [...this.sent].reverse().find((m) => m.method === method) }
}

// (1) A start failure must close the freshly-spawned transport AND not pool it.
async function testStartFailureClosesAndDoesNotPool() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  // thread/start rejects with a hard (non-recoverable) protocol error.
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, error: { code: -1, message: 'boom protocol error' } }))

  await assert.rejects(
    getOrCreateCodexAppServerSession('leak-key', () => t, { cwd: '/repo' }, null),
    /boom protocol error/,
  )
  assert.equal(t.killed, true, 'the spawned transport is closed (not orphaned) on a start failure')
  assert.equal(pooledCodexAppServerSessionKeys().includes('leak-key'), false, 'a failed start is never pooled')
  closeAllCodexAppServerSessions()
  ok++
}

// (2) A changed sandbox/approval/cwd on a pooled key rebuilds a fresh thread.
async function testConfigChangeRebuildsThread() {
  let starts = 0
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id, params) => {
    starts++
    t.emit({ jsonrpc: '2.0', id, result: { thread: { id: `thread-${starts}`, sandbox: params?.sandbox } } })
  })

  const first = await getOrCreateCodexAppServerSession(
    'cfg-key', () => t, { cwd: '/repo', sandbox: 'workspace-write', approvalPolicy: 'on-request' }, null,
  )
  assert.equal(starts, 1)

  // Same config -> reuse, no new thread/start.
  const same = await getOrCreateCodexAppServerSession(
    'cfg-key', () => t, { cwd: '/repo', sandbox: 'workspace-write', approvalPolicy: 'on-request' }, null,
  )
  assert.equal(same.session, first.session, 'unchanged config reuses the pooled session')
  assert.equal(starts, 1, 'unchanged config does not rebuild')

  // Switch to read-only -> must rebuild a fresh thread (the old one held workspace-write).
  const readonly = await getOrCreateCodexAppServerSession(
    'cfg-key', () => t, { cwd: '/repo', sandbox: 'read-only', approvalPolicy: 'never' }, null,
  )
  assert.notEqual(readonly.session, first.session, 'a changed sandbox/approval rebuilds a NEW session')
  assert.equal(starts, 2, 'a changed sandbox triggers a fresh thread/start')
  assert.equal(t.lastRequest('thread/start').params.sandbox, 'read-only', 'the new thread carries the new sandbox')

  closeAllCodexAppServerSessions()
  ok++
}

// (3) Aborting the stream sends turn/interrupt to codex.
async function testAbortSendsInterrupt() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'abort-thread' } } }))
  // turn/start replies and streams one chunk, but never completes on its own.
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-a' } } })
    t.emit({ jsonrpc: '2.0', method: 'turn/started', params: { turn: { id: 'turn-a' } } })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'turn-a', delta: 'working' } })
  })
  let interruptParams: any = null
  t.respond('turn/interrupt', (id, params) => {
    interruptParams = params
    t.emit({ jsonrpc: '2.0', id, result: {} })
    // codex acks the interrupt by completing the turn so runTurn can settle.
    t.emit({ jsonrpc: '2.0', method: 'turn/completed', params: { turn: { id: 'turn-a', status: 'completed' } } })
  })

  const controller = new AbortController()
  const provider = { id: 'p', cliEnv: {} } as any
  const gen = streamCodexAppServer({
    provider,
    model: 'codex-cli-default',
    messages: [{ role: 'user', content: 'do work' } as any],
    cwd: '/repo',
    sessionKey: 'abort-key',
    resumeCursor: null,
    signal: controller.signal,
    spawnTransport: () => t,
  })

  // Pull the first real chunk, then abort and stop iterating (mirrors loop.ts breaking on abort).
  const first = await gen.next()
  assert.equal(first.done, false)
  assert.deepEqual(first.value, { type: 'content', text: 'working' })
  controller.abort()
  // Returning the generator (as the consumer's for-await break does) must trigger the interrupt.
  await gen.return(undefined as any)
  // Let any microtasks flush.
  await new Promise((r) => setTimeout(r, 0))

  assert.ok(interruptParams, 'turn/interrupt was sent to codex on abort')
  assert.equal(interruptParams.threadId, 'abort-thread')
  assert.equal(interruptParams.turnId, 'turn-a')

  closeAllCodexAppServerSessions()
  ok++
}

// (4) A server that acknowledges neither interrupt nor turn completion is
// evicted after the bounded grace period, so cancellation cannot hang forever.
async function testAbortReapsWedgedServer() {
  const t = new FakeTransport()
  t.respond('initialize', (id) => t.emit({ jsonrpc: '2.0', id, result: { capabilities: {} } }))
  t.respond('thread/start', (id) => t.emit({ jsonrpc: '2.0', id, result: { thread: { id: 'wedged-thread' } } }))
  t.respond('turn/start', (id) => {
    t.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'wedged-turn' } } })
    t.emit({ jsonrpc: '2.0', method: 'turn/started', params: { turn: { id: 'wedged-turn' } } })
    t.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { turnId: 'wedged-turn', delta: 'working' } })
  })
  // Deliberately ignore turn/interrupt: no response and no turn/completed.
  t.respond('turn/interrupt', () => {})

  const controller = new AbortController()
  const gen = streamCodexAppServer({
    provider: { id: 'p', cliEnv: {} } as any,
    model: 'codex-cli-default',
    messages: [{ role: 'user', content: 'do work' } as any],
    cwd: '/repo',
    sessionKey: 'wedged-abort-key',
    signal: controller.signal,
    interruptGraceMs: 25,
    spawnTransport: () => t,
  })
  assert.equal((await gen.next()).done, false)
  const pending = gen.next()
  controller.abort()
  assert.equal((await pending).done, true, 'forced closure settles the suspended stream as cancellation')
  assert.equal(t.killed, true, 'the wedged app-server transport is forcibly closed')
  assert.equal(pooledCodexAppServerSessionKeys().includes('wedged-abort-key'), false, 'the poisoned session is evicted')
  ok++
}

async function main() {
  await testStartFailureClosesAndDoesNotPool()
  await testConfigChangeRebuildsThread()
  await testAbortSendsInterrupt()
  await testAbortReapsWedgedServer()
  console.log(`codex-app-server.fixes.test.ts: all ${ok} assertions passed`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
