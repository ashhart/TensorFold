import assert from 'node:assert/strict'
import {
  buildSystemPromptSnapshot,
  messagesWithSystemPromptSnapshot,
  refreshSystemPromptSnapshot,
  renderSystemPromptSnapshot,
  systemPromptFingerprint,
} from './system-prompt-snapshot'

const sections = {
  base: 'base prompt',
  persona: 'persona block',
  capabilities: 'capability block',
  goal: 'goal block',
}

const first = buildSystemPromptSnapshot(sections, 1000)
const rebuiltLater = buildSystemPromptSnapshot({ ...sections }, 2000)

assert.equal(first.fingerprint, rebuiltLater.fingerprint)
assert.equal(first.createdAt, 1000)
assert.equal(rebuiltLater.createdAt, 2000)
assert.equal(
  renderSystemPromptSnapshot(first),
  'persona block\n\nbase prompt\n\ngoal block\n\ncapability block',
)

const changedFingerprint = systemPromptFingerprint({
  ...sections,
  capabilities: 'different capability block',
})
assert.notEqual(changedFingerprint, first.fingerprint)
console.log('ok system prompt snapshot is stable and changes on section edits')

const reused = refreshSystemPromptSnapshot(first, { ...sections }, 3000)
assert.equal(reused, first)

const refreshed = refreshSystemPromptSnapshot(first, { ...sections, goal: 'new goal' }, 4000)
assert.notEqual(refreshed, first)
assert.equal(refreshed.createdAt, 4000)
assert.equal(refreshed.updatedAt, 4000)
assert.equal(renderSystemPromptSnapshot(refreshed), 'persona block\n\nbase prompt\n\nnew goal\n\ncapability block')
console.log('ok system prompt snapshot refreshes only when section bytes change')

const messages = [
  { role: 'system' as const, content: 'stored base prompt' },
  { role: 'user' as const, content: 'hello' },
]
const renderedMessages = messagesWithSystemPromptSnapshot(messages, first)
assert.equal(renderedMessages[0].content, renderSystemPromptSnapshot(first))
assert.equal(messages[0].content, 'stored base prompt')
assert.equal(renderedMessages[1], messages[1])
console.log('ok system prompt snapshot renders over the model system message without mutating history')

// Provider/model switch preserves context (VST-016): a failover re-enters runAgent on the
// SAME thread with a different provider/model. Only the rendered system head may change
// (capabilities/persona re-resolve); the conversation — user turns, assistant tool_calls,
// tool results — and the compaction ledger must survive byte-identical, same order.
{
  const thread = {
    apiMessages: [
      { role: 'system', content: 'stored base prompt' },
      { role: 'user', content: 'fix the flaky test' },
      { role: 'assistant', content: null, tool_calls: [{ id: 't1', type: 'function', function: { name: 'read_file', arguments: '{"path":"a.ts"}' } }] },
      { role: 'tool', tool_call_id: 't1', name: 'read_file', content: 'file body' },
      { role: 'assistant', content: 'found it' },
    ],
    compactions: [{ id: 'c1', epoch: 1 }],
  }
  const beforeSwitch = JSON.stringify(thread.apiMessages.slice(1))

  const onModelA = refreshSystemPromptSnapshot(null, { base: 'base', persona: 'p', goal: '', capabilities: 'tools for gpt-5.5' })
  const renderedA = messagesWithSystemPromptSnapshot(thread.apiMessages, onModelA)

  // Failover: new provider/model → capabilities section changes, snapshot refreshes.
  const onModelB = refreshSystemPromptSnapshot(onModelA, { base: 'base', persona: 'p', goal: '', capabilities: 'tools for claude-sonnet-5' })
  assert.notEqual(onModelB.fingerprint, onModelA.fingerprint, 'the switch produced a new system head')
  const renderedB = messagesWithSystemPromptSnapshot(thread.apiMessages, onModelB)

  assert.equal(JSON.stringify(thread.apiMessages.slice(1)), beforeSwitch, 'stored conversation untouched by both renders')
  assert.equal(JSON.stringify(renderedB.slice(1)), beforeSwitch, 'the new provider receives the SAME conversation')
  assert.notEqual(renderedB[0].content, renderedA[0].content, 'only the system head differs across the switch')
  assert.deepEqual(thread.compactions, [{ id: 'c1', epoch: 1 }], 'compaction ledger (and its epoch) survives the switch')
  console.log('ok provider/model switch rewrites only the system head — conversation and epoch survive')
}
