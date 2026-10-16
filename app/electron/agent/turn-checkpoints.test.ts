// Run: jiti electron/agent/turn-checkpoints.test.ts
//
// Turn checkpoints, plan lineage, handoff (VST-036): the stitch over workspace-checkpoints +
// conversation-checkpoints + plan-gate. Pure — no git repo, no live engine.
import assert from 'node:assert/strict'
import { captureConversationCheckpoint, sequentialIdFactory, type CheckpointDeps, type ConversationState } from './conversation-checkpoints'
import { TurnCheckpointLedger, linkPlanImplementation, buildHandoff } from './turn-checkpoints'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('turn-checkpoints tests:')

function harness(startAt = 1_000) {
  let clock = startAt
  const deps: CheckpointDeps = { clock: { now: () => clock }, newId: sequentialIdFactory('cvk') }
  return { deps, advance: (ms: number) => { clock += ms } }
}

function state(messages: Array<{ role: string; content: unknown }>, extra: Partial<ConversationState> = {}): ConversationState {
  return { apiMessages: messages, timeline: messages.map((m, i) => ({ id: `t${i}`, kind: m.role })), ...extra }
}

// --- record + list + get ---
{
  const { deps } = harness()
  const ledger = new TurnCheckpointLedger(deps)
  const cp1 = captureConversationCheckpoint(deps, state([{ role: 'user', content: 'hi' }]), { label: 'turn 1' })
  const t1 = ledger.record({ threadId: 'th1', turn: 1, conversation: cp1, fileRef: 'refs/vontra/checkpoints/th1/1', fileSha: 'abc123' })
  assert.equal(t1.turn, 1)
  assert.equal(t1.fileRef, 'refs/vontra/checkpoints/th1/1')
  assert.equal(t1.fileSha, 'abc123')
  assert.equal(t1.conversationId, cp1.id, 'unifies the conversation snapshot id')
  assert.equal(t1.messageCount, 1)
  const cp2 = captureConversationCheckpoint(deps, state([{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }]))
  ledger.record({ threadId: 'th1', turn: 2, conversation: cp2, fileRef: 'refs/vontra/checkpoints/th1/2', fileSha: 'def456', planId: 'plan-7' })
  assert.deepEqual(ledger.list().map((c) => c.turn), [1, 2], 'list is turn-ordered')
  assert.equal(ledger.get(2)?.planId, 'plan-7', 'a frozen plan id rides on the turn')
  assert.equal(ledger.get(99), null)
  ok('record: unifies file ref + conversation snapshot + plan per turn')
}

// --- revert restores files + conversation (spec test #1) ---
{
  const { deps } = harness()
  const ledger = new TurnCheckpointLedger(deps)
  const early = state([{ role: 'user', content: 'first' }, { role: 'assistant', content: 'A' }])
  const cpEarly = captureConversationCheckpoint(deps, early)
  ledger.record({ threadId: 'th1', turn: 1, conversation: cpEarly, fileRef: 'ref-1', fileSha: 'sha1' })
  const later = state([...early.apiMessages, { role: 'user', content: 'second' }, { role: 'assistant', content: 'B' }])
  ledger.record({ threadId: 'th1', turn: 2, conversation: captureConversationCheckpoint(deps, later), fileRef: 'ref-2', fileSha: 'sha2' })

  const revert = ledger.revertTo(1)
  assert.equal(revert.fileRef, 'ref-1', 'revert points the file restore at turn 1\'s git ref')
  assert.deepEqual(revert.conversation.apiMessages.map((m: any) => m.content), ['first', 'A'], 'AND restores the conversation to turn 1 (later messages gone)')
  assert.throws(() => ledger.revertTo(5), /No turn checkpoint/, 'reverting to an unknown turn throws, never silently no-ops')
  ok('revertTo: files + conversation restored together, atomically')
}

// --- fork isolation (spec test #2) ---
{
  const { deps } = harness()
  const ledger = new TurnCheckpointLedger(deps)
  const base = state([{ role: 'user', content: 'shared' }, { role: 'assistant', content: 'shared reply' }])
  ledger.record({ threadId: 'th1', turn: 1, conversation: captureConversationCheckpoint(deps, base), fileRef: 'ref-1', fileSha: 'sha1' })

  const fork = ledger.forkFrom(1, { label: 'alternate' })
  assert.notEqual(fork.branch, 'main', 'fork opens a new branch')
  assert.equal(fork.fileRef, 'ref-1', 'the fork carries the same file ref to diverge from')
  assert.deepEqual(fork.conversation.apiMessages.map((m: any) => m.content), ['shared', 'shared reply'], 'fork starts as a copy of the fork point')

  // Mutating the fork's returned state must NOT affect the ledger's turn-1 checkpoint.
  fork.conversation.apiMessages.push({ role: 'user', content: 'only in fork' } as any)
  const reverted = ledger.revertTo(1)
  assert.equal(reverted.conversation.apiMessages.length, 2, 'the parent turn-1 checkpoint is untouched by fork mutation')
  ok('forkFrom: fork is an isolated clone; mutating it never touches the parent')
}

// --- plan → implementation lineage (spec test #3) ---
{
  const link = linkPlanImplementation({ planId: 'plan-1', planThreadId: 'plan-thread', implementationThreadId: 'impl-thread', ticketId: 'TK-42' }, { now: () => 5_000 })
  assert.equal(link.planId, 'plan-1')
  assert.equal(link.planThreadId, 'plan-thread')
  assert.equal(link.implementationThreadId, 'impl-thread')
  assert.equal(link.ticketId, 'TK-42', 'a Kanban ticket integrates into the lineage')
  assert.equal(link.createdAt, 5_000)
  const noTicket = linkPlanImplementation({ planId: 'p', planThreadId: 'a', implementationThreadId: 'b' })
  assert.equal(noTicket.ticketId, null, 'ticket is optional')
  assert.throws(() => linkPlanImplementation({ planId: 'p', planThreadId: 'same', implementationThreadId: 'same' }), /own plan thread/, 'a plan cannot be implemented in its own thread')
  assert.throws(() => linkPlanImplementation({ planId: '', planThreadId: 'a', implementationThreadId: 'b' }), /requires/, 'ids are required')
  ok('linkPlanImplementation: plan→implementation link with optional ticket, guarded')
}

// --- cross-engine handoff transcript continuity (spec test #4) ---
{
  const conv = state([
    { role: 'user', content: 'Refactor the auth module' },
    { role: 'assistant', content: 'Started; extracted a helper' },
    { role: 'user', content: 'Now add tests' },
    { role: 'assistant', content: '' }, // empty streaming placeholder — must be dropped
  ])
  const payload = buildHandoff({ fromThreadId: 'th1', fromEngine: 'codex', toEngine: 'claude', atTurn: 2, conversation: conv }, { now: () => 9_000 })
  assert.equal(payload.fromEngine, 'codex')
  assert.equal(payload.toEngine, 'claude')
  assert.equal(payload.transcript.length, 3, 'empty messages are dropped from the carried transcript')
  const seed = payload.seedMessage.content as string
  assert.ok(seed.includes('handed off from codex'), 'the seed frames the handoff for the receiving engine')
  assert.ok(seed.includes('Refactor the auth module') && seed.includes('Now add tests'), 'the seed carries the full prior transcript (continuity)')
  assert.ok(seed.includes('[user]') && seed.includes('[assistant]'), 'roles are preserved in the flattened transcript')
  assert.throws(() => buildHandoff({ fromThreadId: 'th1', fromEngine: 'codex', toEngine: 'codex', atTurn: 1, conversation: conv }), /different target engine/, 'handoff to the same engine is refused')
  ok('buildHandoff: seed carries full prior transcript so the new engine continues seamlessly')

  // Strong-form handoff (VST-049): the structured transcript preserves roles so
  // the receiving thread can seed its apiMessages directly (not just via the
  // flattened seedMessage). Verify the transcript carries valid roles + content.
  const roles = payload.transcript.map((m) => m.role)
  assert.deepEqual(roles, ['user', 'assistant', 'user'], 'structured transcript preserves role order')
  for (const m of payload.transcript) {
    assert.ok(typeof m.content === 'string' && m.content.length > 0, `transcript message content is a non-empty string: ${JSON.stringify(m.role)}`)
  }
  ok('buildHandoff: structured transcript preserves roles for direct apiMessages seeding')
}

// --- handoff handles array (multimodal) content ---
{
  const conv: ConversationState = {
    apiMessages: [{ role: 'user', content: [{ type: 'text', text: 'look at this' }] as any }],
    timeline: [],
  }
  const payload = buildHandoff({ fromThreadId: 'th1', fromEngine: 'a', toEngine: 'b', atTurn: 1, conversation: conv })
  assert.ok((payload.seedMessage.content as string).includes('look at this'), 'array/multimodal content flattens into the seed')
  ok('buildHandoff: multimodal content flattens')
}

// --- serialize → restore round-trips revert AND fork ---
{
  const { deps } = harness()
  const ledger = new TurnCheckpointLedger(deps)
  ledger.record({ threadId: 'th1', turn: 1, conversation: captureConversationCheckpoint(deps, state([{ role: 'user', content: 'one' }])), fileRef: 'ref-1', fileSha: 'sha1' })
  ledger.record({ threadId: 'th1', turn: 2, conversation: captureConversationCheckpoint(deps, state([{ role: 'user', content: 'one' }, { role: 'assistant', content: 'two' }])), fileRef: 'ref-2', fileSha: 'sha2', planId: 'plan-9' })

  // Disk round-trip: JSON stringify/parse is exactly what the persister does.
  const blob = JSON.parse(JSON.stringify(ledger.serialize()))
  const revived = TurnCheckpointLedger.restore(blob, { clock: deps.clock, newId: () => `fresh_${Math.random().toString(36).slice(2)}` })

  assert.deepEqual(revived.list().map((t) => t.turn), [1, 2], 'both turns survive the round-trip')
  assert.equal(revived.get(2)?.planId, 'plan-9', 'plan id survives')
  assert.equal(revived.get(1)?.fileRef, 'ref-1', 'file ref survives')

  const revert = revived.revertTo(1)
  assert.deepEqual(revert.conversation.apiMessages.map((m: any) => m.content), ['one'], 'a restored ledger still reverts files+conversation')

  const fork = revived.forkFrom(2)
  assert.equal(fork.fileRef, 'ref-2', 'a restored ledger still forks with the file ref')
  fork.conversation.apiMessages.push({ role: 'user', content: 'divergent' })
  assert.equal(revived.revertTo(2).conversation.apiMessages.length, 2, 'fork isolation holds after restore')
  ok('serialize/restore: disk round-trip preserves revert + fork behavior')
}

// --- serialize bounds turns; restore skips garbage and never clobbers live entries ---
{
  const { deps } = harness()
  const ledger = new TurnCheckpointLedger(deps)
  for (let turn = 1; turn <= 5; turn++) {
    ledger.record({ threadId: 'th1', turn, conversation: captureConversationCheckpoint(deps, state([{ role: 'user', content: `m${turn}` }])) })
  }
  assert.deepEqual(ledger.serialize({ maxTurns: 2 }).entries.map((e) => e.turn.turn), [4, 5], 'maxTurns keeps the newest N turns')

  // Corrupt / hostile blobs restore to an empty ledger instead of throwing.
  assert.equal(TurnCheckpointLedger.restore(null).list().length, 0)
  assert.equal(TurnCheckpointLedger.restore('junk').list().length, 0)
  assert.equal(TurnCheckpointLedger.restore({ entries: 'nope' }).list().length, 0)
  const partial = TurnCheckpointLedger.restore({
    entries: [
      { turn: { turn: 'NaN' }, conversation: { id: 'c1' } },        // bad turn number
      { turn: { turn: 3 }, conversation: null },                     // missing conversation
      { turn: { turn: 3 }, conversation: { id: 'c3', state: { apiMessages: [{ role: 'user', content: 'ok' }], timeline: [] } } }, // good
    ],
  })
  assert.deepEqual(partial.list().map((t) => t.turn), [3], 'malformed entries are skipped, valid ones restore')
  assert.deepEqual(partial.revertTo(3).conversation.apiMessages.map((m: any) => m.content), ['ok'])

  // A live entry recorded since boot wins over the persisted copy of the same turn.
  const live = new TurnCheckpointLedger(deps)
  live.record({ threadId: 'th1', turn: 1, conversation: captureConversationCheckpoint(deps, state([{ role: 'user', content: 'live' }])) })
  live.restoreEntries(ledger.serialize())
  assert.deepEqual(live.revertTo(1).conversation.apiMessages.map((m: any) => m.content), ['live'], 'restore never clobbers a live turn')
  assert.equal(live.list().length, 5, 'missing turns are filled in from disk')
  ok('serialize/restore: bounded, corruption-safe, live-entries-win merge')
}

console.log(`\n${passed} passed`)
