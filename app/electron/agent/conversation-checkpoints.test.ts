import assert from 'node:assert/strict'
import {
  captureConversationCheckpoint,
  restoreConversationCheckpoint,
  forkConversationCheckpoint,
  fingerprintConversation,
  nextForkBranch,
  sequentialIdFactory,
  ConversationCheckpointLog,
  MAIN_BRANCH,
  type ConversationState,
  type CheckpointDeps,
} from './conversation-checkpoints'

let passed = 0
function ok(msg: string) {
  passed++
  console.log(`ok ${passed} - ${msg}`)
}

// Deterministic deps: fixed monotonic clock + sequential ids.
function makeDeps(): CheckpointDeps {
  let t = 1_000
  return { clock: { now: () => (t += 10) }, newId: sequentialIdFactory('t') }
}

function state(turn: number): ConversationState {
  return {
    apiMessages: [
      { role: 'user', content: `q${turn}` },
      { role: 'assistant', content: `a${turn}` },
    ],
    timeline: [
      { id: `u${turn}`, kind: 'user', text: `q${turn}` },
      { id: `m${turn}`, kind: 'assistant', text: `a${turn}` },
    ],
    goal: `goal-${turn}`,
    goalStatus: 'active',
    plan: [{ id: 'p1', text: 'step', done: false }],
    status: 'idle',
  }
}

// ── capture: counts + fingerprint + fields ───────────────────────────────────
{
  const deps = makeDeps()
  const cp = captureConversationCheckpoint(deps, state(1), { label: 'first' })
  assert.equal(cp.messageCount, 2)
  assert.equal(cp.timelineCount, 2)
  assert.equal(cp.branch, MAIN_BRANCH)
  assert.equal(cp.parentId, null)
  assert.equal(cp.label, 'first')
  assert.equal(cp.fileSha, null)
  assert.equal(cp.createdAt, 1_010)
  assert.match(cp.fingerprint, /^[0-9a-f]{64}$/)
  ok('capture records counts, root parent, label, clock, sha256 fingerprint')
}

// ── fingerprint is order-stable + content-sensitive ──────────────────────────
{
  const a = state(1)
  const b: ConversationState = {
    // reversed key order, same content
    status: 'idle',
    plan: [{ done: false, text: 'step', id: 'p1' }],
    goalStatus: 'active',
    goal: 'goal-1',
    timeline: [
      { kind: 'user', id: 'u1', text: 'q1' },
      { kind: 'assistant', id: 'm1', text: 'a1' },
    ],
    apiMessages: [
      { content: 'q1', role: 'user' },
      { content: 'a1', role: 'assistant' },
    ],
  }
  assert.equal(fingerprintConversation(a), fingerprintConversation(b))
  const c = state(1)
  c.apiMessages.push({ role: 'user', content: 'more' })
  assert.notEqual(fingerprintConversation(a), fingerprintConversation(c))
  ok('fingerprint is key-order-stable and changes when content changes')
}

// ── immutability: snapshot is frozen + isolated from live mutation ───────────
{
  const deps = makeDeps()
  const live = state(1)
  const cp = captureConversationCheckpoint(deps, live)
  // mutating the live state after capture must not touch the snapshot
  live.apiMessages.push({ role: 'user', content: 'leaked?' })
  ;(live.timeline[0] as any).text = 'mutated'
  assert.equal(cp.messageCount, 2)
  assert.equal((cp.state.timeline[0] as any).text, 'q1')
  // the snapshot itself is deep-frozen
  assert.equal(Object.isFrozen(cp), true)
  assert.equal(Object.isFrozen(cp.state), true)
  assert.equal(Object.isFrozen(cp.state.apiMessages), true)
  assert.throws(() => {
    ;(cp.state.apiMessages as any).push({ role: 'user', content: 'x' })
  })
  ok('snapshot is deep-frozen and isolated from later live mutation')
}

// ── restore returns a fresh mutable clone (rewind) ───────────────────────────
{
  const deps = makeDeps()
  const cp = captureConversationCheckpoint(deps, state(5))
  const restored = restoreConversationCheckpoint(cp)
  assert.equal(fingerprintConversation(restored), cp.fingerprint)
  // mutable + independent of snapshot and of other restores
  restored.apiMessages.push({ role: 'user', content: 'edit' })
  assert.equal(cp.state.apiMessages.length, 2)
  const restored2 = restoreConversationCheckpoint(cp)
  assert.equal(restored2.apiMessages.length, 2)
  assert.notEqual(restored, restored2)
  ok('restore yields a fresh mutable clone that does not alias the snapshot')
}

// ── fork creates an alternate branch off a past snapshot ─────────────────────
{
  const deps = makeDeps()
  const root = captureConversationCheckpoint(deps, state(1))
  const forked = forkConversationCheckpoint(deps, root, [MAIN_BRANCH])
  assert.equal(forked.branch, 'fork-1')
  assert.equal(forked.checkpoint.parentId, root.id)
  assert.equal(forked.checkpoint.branch, 'fork-1')
  // same content as fork point, but a distinct snapshot id
  assert.equal(forked.checkpoint.fingerprint, root.fingerprint)
  assert.notEqual(forked.checkpoint.id, root.id)
  // seed state is mutable and matches
  assert.equal(fingerprintConversation(forked.state), root.fingerprint)
  // original snapshot untouched
  assert.equal(root.branch, MAIN_BRANCH)
  ok('fork re-homes a past snapshot on a new branch without disturbing the original')
}

// ── nextForkBranch avoids collisions ─────────────────────────────────────────
{
  assert.equal(nextForkBranch([MAIN_BRANCH]), 'fork-1')
  assert.equal(nextForkBranch([MAIN_BRANCH, 'fork-1', 'fork-2']), 'fork-3')
  ok('nextForkBranch picks the first free fork-N name')
}

// ── log: capture chain, dedup, rewind, fork, lineage ─────────────────────────
{
  const log = new ConversationCheckpointLog(makeDeps())
  const c1 = log.capture(state(1))
  const c2 = log.capture(state(2))
  const c3 = log.capture(state(3))
  assert.equal(log.list().length, 3)
  assert.equal(c2.parentId, c1.id)
  assert.equal(c3.parentId, c2.id)
  assert.equal(log.head()!.id, c3.id)

  // dedup: capturing identical-to-HEAD state adds nothing
  const dup = log.capture(state(3))
  assert.equal(dup.id, c3.id)
  assert.equal(log.list().length, 3)

  // rewind moves HEAD back and returns that state; later snapshots remain
  const rewound = log.rewind(c1.id)
  assert.equal(fingerprintConversation(rewound), c1.fingerprint)
  assert.equal(log.head()!.id, c1.id)
  assert.equal(log.list().length, 3)

  // fork off c2 opens a new branch and sets HEAD to the fork root
  const fk = log.fork(c2.id)
  assert.equal(fk.branch, 'fork-1')
  assert.equal(log.head()!.id, fk.checkpoint.id)
  assert.equal(log.branches().sort().join(','), 'fork-1,main')
  assert.equal(log.list().length, 4)

  // capturing after a fork continues on the fork branch
  const c4 = log.capture(state(9))
  assert.equal(c4.branch, 'fork-1')
  assert.equal(c4.parentId, fk.checkpoint.id)

  // lineage walks parents back to root
  const chain = log.lineage(c4.id).map((c) => c.id)
  assert.deepEqual(chain, [c4.id, fk.checkpoint.id, c2.id, c1.id])

  // unknown id throws on rewind/fork
  assert.throws(() => log.rewind('nope'))
  assert.throws(() => log.fork('nope'))
  ok('log captures a chain, dedups, rewinds, forks a branch, and walks lineage')
}

// ── fileSha correlation carries through capture + fork ───────────────────────
{
  const deps = makeDeps()
  const cp = captureConversationCheckpoint(deps, state(1), { fileSha: 'abc123' })
  assert.equal(cp.fileSha, 'abc123')
  const fk = forkConversationCheckpoint(deps, cp, [MAIN_BRANCH])
  assert.equal(fk.checkpoint.fileSha, 'abc123')
  ok('fileSha correlation to a workspace checkpoint carries through fork')
}

console.log(`\n${passed} conversation-checkpoint assertions passed`)
