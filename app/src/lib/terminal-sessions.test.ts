import assert from 'node:assert'
import type { CliSession } from './types'
import { reconcileSessionsForOrigin, setSessions, removeSession, renameSession, nextActiveAfterClose, summarizeTerminalActivity, terminalSessionKey, terminalSessionRoute, updateSessionStatus } from './terminal-sessions'

function sess(id: string, title = id): CliSession {
  return { id, projectId: 'p1', cliId: 'shell', title, status: 'idle', createdAt: 0 }
}

// setSessions replaces a project's list without touching other projects
{
  const map = { p0: [sess('a')] }
  const next = setSessions(map, 'p1', [sess('x'), sess('y')])
  assert.deepStrictEqual(next.p1.map((s) => s.id), ['x', 'y'])
  assert.deepStrictEqual(next.p0.map((s) => s.id), ['a'], 'other projects untouched')
  assert.notStrictEqual(next, map, 'returns a new map (immutable)')
}

// Existing sessions route by immutable origin, never by a project's mutable execution target.
{
  assert.deepStrictEqual(terminalSessionRoute(sess('local')), { kind: 'local' })
  assert.deepStrictEqual(terminalSessionRoute({ ...sess('remote-a'), hostLocation: 'orchestrator', orchestratorHostId: 'host-a' }), { kind: 'orchestrator', hostId: 'host-a' })
  assert.deepStrictEqual(terminalSessionRoute({ ...sess('legacy-remote'), hostLocation: 'orchestrator' }), { kind: 'unresolved' }, 'legacy remote rows fail closed')
  assert.deepStrictEqual(terminalSessionRoute(undefined), { kind: 'unresolved' }, 'missing rows cannot fall through to a local call')
}

// Each list update owns only its origin. A global retarget/local refresh must not hide remote A,
// and a remote B refresh must not erase either local or remote A sessions.
{
  const local = { ...sess('local'), hostLocation: 'local' as const }
  const remoteA = { ...sess('remote-a'), hostLocation: 'orchestrator' as const, orchestratorHostId: 'host-a' }
  const legacy = { ...sess('legacy'), hostLocation: 'orchestrator' as const }
  let map: Record<string, CliSession[]> = { p1: [local, remoteA, legacy] }

  map = reconcileSessionsForOrigin(map, 'p1', [{ ...sess('local-new'), hostLocation: 'orchestrator', orchestratorHostId: 'spoofed' }], null)
  assert.deepStrictEqual(map.p1.map((s) => s.id), ['remote-a', 'legacy', 'local-new'])
  assert.equal(map.p1.at(-1)?.hostLocation, 'local', 'the receiving local channel, not session-controlled metadata, stamps the origin')

  map = reconcileSessionsForOrigin(map, 'p1', [{ ...sess('remote-b'), hostLocation: 'local' }], 'host-b')
  assert.deepStrictEqual(map.p1.map((s) => s.id), ['remote-a', 'legacy', 'local-new', 'remote-b'])
  assert.deepStrictEqual(terminalSessionRoute(map.p1.at(-1)), { kind: 'orchestrator', hostId: 'host-b' })

  map = reconcileSessionsForOrigin(map, 'p1', [], 'host-a')
  assert.deepStrictEqual(map.p1.map((s) => s.id), ['legacy', 'local-new', 'remote-b'], 'an empty host-A list clears only host A')
}

// removeSession drops one session; missing project is a no-op
{
  const map = { p1: [sess('a'), sess('b')] }
  assert.deepStrictEqual(removeSession(map, 'p1', 'a').p1.map((s) => s.id), ['b'])
  assert.strictEqual(removeSession(map, 'nope', 'a'), map, 'unknown project returns same ref')
}

// renameSession updates only the matching session's title
{
  const map = { p1: [sess('a', 'Shell'), sess('b', 'Codex')] }
  const next = renameSession(map, 'p1', 'a', 'Build')
  assert.strictEqual(next.p1[0].title, 'Build')
  assert.strictEqual(next.p1[1].title, 'Codex', 'sibling untouched')
}

// nextActiveAfterClose: pick the session that takes the closed slot, else the previous, else first, else null
{
  const list = [sess('a'), sess('b'), sess('c')]
  assert.strictEqual(nextActiveAfterClose(list, 'b'), 'c', 'closing middle → the one that slides into its index')
  assert.strictEqual(nextActiveAfterClose(list, 'c'), 'b', 'closing last → previous')
  assert.strictEqual(nextActiveAfterClose(list, 'a'), 'b', 'closing first → next')
  assert.strictEqual(nextActiveAfterClose([sess('a')], 'a'), null, 'closing the only one → null')
}

{
  const idle = sess('idle')
  const busy = { ...sess('busy'), status: 'busy' as const }
  const attention = { ...sess('attention'), status: 'attention' as const }
  const done = { ...sess('done'), status: 'done' as const }
  assert.deepStrictEqual(summarizeTerminalActivity([idle, busy, attention, done, busy]), { total: 3, running: 1, attention: 1 })
  const map = { p1: [idle], p2: [sess('other')] }
  const updated = updateSessionStatus(map, 'idle', 'busy')
  assert.equal(updated.p1[0].status, 'busy')
  assert.equal(updated.p2[0].status, 'idle')
  assert.strictEqual(updateSessionStatus(updated, 'missing', 'done'), updated)
}

// Independently generated IDs may collide across local/host A/host B. Status events update only
// the matching origin tuple, never every row sharing the raw session id.
{
  let map: Record<string, CliSession[]> = { p1: [] }
  map = reconcileSessionsForOrigin(map, 'p1', [sess('collision')], null)
  map = reconcileSessionsForOrigin(map, 'p1', [sess('collision')], 'host-a')
  map = reconcileSessionsForOrigin(map, 'p1', [sess('collision')], 'host-b')
  assert.deepStrictEqual(map.p1.map(terminalSessionKey), ['local:collision', 'orchestrator:host-a:collision', 'orchestrator:host-b:collision'], 'origin list reconciliation preserves raw-id collisions')
  const fromA = updateSessionStatus(map, 'collision', 'busy', 'host-a')
  assert.deepStrictEqual(fromA.p1.map((session) => session.status), ['idle', 'busy', 'idle'])
  const localUpdate = updateSessionStatus(fromA, 'collision', 'attention', null)
  assert.deepStrictEqual(localUpdate.p1.map((session) => session.status), ['attention', 'busy', 'idle'])
  assert.deepStrictEqual(summarizeTerminalActivity(localUpdate.p1), { total: 3, running: 1, attention: 1 }, 'activity counts the same raw id independently per origin')
}

console.log('terminal-sessions: all checks passed')
