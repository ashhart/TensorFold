// Run: jiti electron/agent/acp-server-concurrency.test.ts
//
// Covers two fixes in acp-server.ts that operate at the handler-logic layer (no SDK
// transport needed): (1) a second session/prompt must abort AND AWAIT the prior turn so
// two runAgent() calls never mutate the same per-session Thread concurrently; (2) idle
// sessions/threads are evicted (TTL + cap) instead of leaking for the process lifetime.
import assert from 'node:assert/strict'
import { createAcpAgentHandlers, type AcpServerDeps, type AcpTurnDeps } from './acp-server'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function makeHandlers(over: Partial<AcpServerDeps> = {}, runTurn?: AcpServerDeps['runTurn']) {
  const deps: AcpServerDeps = {
    cwd: '/repo',
    runTurn: runTurn ?? (async () => {}),
    ...over,
  }
  return createAcpAgentHandlers(deps)
}

// A throwaway AgentContext: prompt() only uses client.notify/client.request, neither of
// which our fake runTurn invokes, so a no-op stub is sufficient.
const fakeClient = { notify: () => {}, request: async () => ({ outcome: { outcome: 'cancelled' } }) } as any

async function testSecondPromptAwaitsPriorTurn() {
  let active = 0
  let maxConcurrent = 0
  let firstStarted!: () => void
  const firstStartedGate = new Promise<void>((r) => { firstStarted = r })
  let releaseFirst!: () => void
  const firstRelease = new Promise<void>((r) => { releaseFirst = r })
  const order: string[] = []

  const runTurn = async (params: AcpTurnDeps) => {
    active++
    maxConcurrent = Math.max(maxConcurrent, active)
    order.push(`start:${params.promptText}`)
    if (params.promptText === 'one') {
      firstStarted()
      await firstRelease // hold turn #1 open until the test releases it
    }
    order.push(`end:${params.promptText}`)
    active--
  }

  const h = makeHandlers({}, runTurn)
  const { sessionId } = await h.newSession({ cwd: '/repo' } as any)

  // Fire prompt #1 (will block), then prompt #2 while #1 is still mid-turn.
  const p1 = h.prompt({ sessionId, prompt: [{ type: 'text', text: 'one' }] } as any, fakeClient)
  await firstStartedGate // ensure #1 is actually running before issuing #2
  const p2 = h.prompt({ sessionId, prompt: [{ type: 'text', text: 'two' }] } as any, fakeClient)

  // Give the event loop a few ticks: #2 must NOT have started runTurn yet (it must await #1).
  await new Promise((r) => setTimeout(r, 20))
  assert.ok(!order.includes('start:two'), 'second prompt must not start its turn until the first completes')

  releaseFirst()
  await Promise.all([p1, p2])

  assert.strictEqual(maxConcurrent, 1, 'the two turns must never run concurrently against the shared Thread')
  assert.deepStrictEqual(order, ['start:one', 'end:one', 'start:two', 'end:two'], 'turns ran strictly serialized in order')
  ok('a second prompt aborts AND awaits the prior turn (no concurrent runAgent on one Thread)')
}

async function testIdleSessionsEvicted() {
  const closed: string[] = []
  // Tiny TTL so the second newSession reaps the first as idle.
  const h = makeHandlers({ sessionIdleTtlMs: 5, onSessionClosed: (id) => closed.push(id) })
  const a = await h.newSession({ cwd: '/repo' } as any)
  await new Promise((r) => setTimeout(r, 15))
  const b = await h.newSession({ cwd: '/repo' } as any) // reapSessions() runs here

  assert.ok(!h.sessions.has(a.sessionId), 'the idle session past its TTL was evicted')
  assert.ok(h.sessions.has(b.sessionId), 'the fresh session is retained')
  assert.deepStrictEqual(closed, [a.sessionId], 'onSessionClosed fired so the per-session Thread can be forgotten')
  ok('idle sessions (and their threads) are evicted, not leaked')
}

async function testSessionCapEvictsOldest() {
  const closed: string[] = []
  // maxSessions=2, no idle TTL pressure (long TTL): a 3rd session forces out the oldest idle one.
  const h = makeHandlers({ maxSessions: 2, sessionIdleTtlMs: 60_000, onSessionClosed: (id) => closed.push(id) })
  const a = await h.newSession({ cwd: '/repo' } as any)
  await new Promise((r) => setTimeout(r, 2))
  const b = await h.newSession({ cwd: '/repo' } as any)
  await new Promise((r) => setTimeout(r, 2))
  const c = await h.newSession({ cwd: '/repo' } as any)

  assert.ok(!h.sessions.has(a.sessionId), 'oldest session evicted once the cap is exceeded')
  assert.ok(h.sessions.has(b.sessionId) && h.sessions.has(c.sessionId), 'the two most recent sessions are retained')
  assert.deepStrictEqual(closed, [a.sessionId], 'eviction forgets exactly the evicted session')
  ok('session cap evicts the oldest idle session')
}

async function main() {
  console.log('acp-server concurrency/eviction tests:')
  await testSecondPromptAwaitsPriorTurn()
  await testIdleSessionsEvicted()
  await testSessionCapEvictsOldest()
  console.log(`\n${passed} checks passed.`)
}

main().catch((e) => { console.error(e); process.exit(1) })
