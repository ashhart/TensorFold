// Run: jiti electron/agent/codex-app-server.death.test.ts
//
// Process-death propagation (full-review-2026-06-30 Medium #5): a codex app-server
// process dying mid-turn must WAKE the suspended runTurn generator with a failure —
// not leave it hanging on a notification that will never arrive — and a pooled
// session for that key must be rebuilt, not reused while poisoned.
import assert from 'node:assert/strict'
import {
  CodexAppServerSession,
  getOrCreateCodexAppServerSession,
  closeAllCodexAppServerSessions,
  type CodexStdioTransport,
} from './codex-app-server'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('codex app-server death-propagation tests:')

class FakeTransport implements CodexStdioTransport {
  sent: any[] = []
  killed = false
  private lineCb: ((line: string) => void) | null = null
  private closeCb: ((reason: string) => void) | null = null
  responders = new Map<string, (id: number, params: any) => void>()

  write(line: string): void {
    const msg = JSON.parse(line)
    this.sent.push(msg)
    if (msg.id != null && msg.method) this.responders.get(msg.method)?.(msg.id, msg.params)
  }
  onLine(cb: (line: string) => void): void { this.lineCb = cb }
  onClose(cb: (reason: string) => void): void { this.closeCb = cb }
  kill(): void { this.killed = true }
  emit(obj: unknown): void { this.lineCb?.(JSON.stringify(obj)) }
  respond(method: string, fn: (id: number, params: any) => void): void { this.responders.set(method, fn) }
  // Simulate the underlying process dying.
  die(reason: string): void { this.closeCb?.(reason) }
}

function scriptHappySession(transport: FakeTransport): void {
  transport.respond('initialize', (id) => transport.emit({ jsonrpc: '2.0', id, result: {} }))
  transport.respond('thread/start', (id) => transport.emit({ jsonrpc: '2.0', id, result: { thread: { id: 't1' } } }))
  transport.respond('turn/start', (id) => {
    transport.emit({ jsonrpc: '2.0', id, result: { turn: { id: 'turn-1' } } })
    transport.emit({ jsonrpc: '2.0', method: 'item/agentMessage/delta', params: { delta: 'partial', turnId: 'turn-1' } })
  })
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_res, rej) => setTimeout(() => rej(new Error(`TIMEOUT: ${label} did not settle in ${ms}ms — the turn is hanging`)), ms)),
  ])
}

// 1. Process death mid-turn wakes the suspended generator with the close reason.
{
  const transport = new FakeTransport()
  scriptHappySession(transport)
  const session = new CodexAppServerSession({ transport })
  await session.initialize()
  await session.startThread({})

  const gen = session.runTurn('do work')
  const first = await gen.next()
  assert.deepEqual(first.value, { type: 'content', text: 'partial' })

  // The next pull suspends on the wake signal; kill the process while suspended.
  const pending = gen.next()
  transport.die('codex app-server exited (SIGKILL) — panic: out of memory')
  const outcome = await withTimeout(pending.then(() => 'resolved', (error: Error) => error.message), 1000, 'runTurn after process death')
  assert.ok(typeof outcome === 'string' && outcome.includes('SIGKILL'), 'generator throws the close reason instead of hanging')
  assert.equal(session.isClosed, true, 'session reports closed after process death')
  ok('runTurn mid-stream: process death throws with the exit reason (no hang)')
}

// 2. An in-flight request (no response yet) rejects on death.
{
  const transport = new FakeTransport()
  transport.respond('initialize', (id) => transport.emit({ jsonrpc: '2.0', id, result: {} }))
  // thread/start never responds — the request stays pending until death.
  const session = new CodexAppServerSession({ transport })
  await session.initialize()
  const pending = session.startThread({})
  transport.die('codex app-server exited (1)')
  const message = await withTimeout(pending.then(() => 'resolved', (error: Error) => error.message), 1000, 'pending request after death')
  assert.ok(typeof message === 'string' && message.includes('exited'), 'pending request rejects with the close reason')
  ok('in-flight request: rejected on process death')
}

// 3. A poisoned pooled session is evicted and rebuilt on the next turn.
{
  closeAllCodexAppServerSessions()
  const transports: FakeTransport[] = []
  const spawnTransport = () => {
    const transport = new FakeTransport()
    scriptHappySession(transport)
    transports.push(transport)
    return transport
  }
  const first = await getOrCreateCodexAppServerSession('thread-death', spawnTransport, {}, null)
  assert.equal(first.needsFullHistory, true)
  transports[0].die('codex app-server exited (SIGSEGV)')
  assert.equal(first.session.isClosed, true)

  const second = await getOrCreateCodexAppServerSession('thread-death', spawnTransport, {}, null)
  assert.notEqual(second.session, first.session, 'pool rebuilt a fresh session instead of reusing the dead one')
  assert.equal(second.session.isClosed, false)
  assert.equal(transports.length, 2, 'a second transport was spawned for the rebuild')
  closeAllCodexAppServerSessions()
  ok('pool: dead session evicted and rebuilt on next use')
}

// 4. Explicit close() (app quit) also wakes an active turn.
{
  const transport = new FakeTransport()
  scriptHappySession(transport)
  const session = new CodexAppServerSession({ transport })
  await session.initialize()
  await session.startThread({})
  const gen = session.runTurn('long turn')
  await gen.next()
  const pending = gen.next()
  session.close()
  const outcome = await withTimeout(pending.then(() => 'resolved', (error: Error) => error.message), 1000, 'runTurn after explicit close')
  assert.ok(typeof outcome === 'string' && outcome.includes('closed'), 'explicit close wakes the turn with a failure')
  assert.equal(transport.killed, true, 'explicit close still kills the child')
  ok('close(): active turn woken with a failure, child killed')
}

console.log(`${passed} passed`)
