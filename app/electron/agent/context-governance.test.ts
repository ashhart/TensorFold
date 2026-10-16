// Run: jiti electron/agent/context-governance.test.ts
//
// Context governance (VST-016): bounded fragments with provenance/priority/epoch/token
// cost and VISIBLE truncation; bounded tool output with managed spill files; the context
// epoch derived from the compaction ledger.
import assert from 'node:assert/strict'
import {
  boundToolOutput,
  buildContextFragment,
  contextEpoch,
  estimateTextTokens,
  fragmentStat,
  FRAGMENT_CHAR_BUDGETS,
  toolOutputSpillPath,
  TOOL_OUTPUT_HEAD_CHARS,
  TOOL_OUTPUT_MAX_CHARS,
  TOOL_OUTPUT_TAIL_CHARS,
} from './context-governance'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('context-governance tests:')

{
  // A huge project-memory block is bounded by its priority's budget and SAYS it was cut.
  const huge = 'M'.repeat(60_000)
  const fragment = buildContextFragment({ source: 'project_memory', text: huge, epoch: 2 })
  assert.equal(fragment.priority, 'high')
  assert.equal(fragment.epoch, 2)
  assert.equal(fragment.truncated, true)
  assert.equal(fragment.originalChars, 60_000)
  assert.ok(fragment.chars < 60_000 && fragment.chars >= FRAGMENT_CHAR_BUDGETS.high, 'bounded near the budget')
  assert.match(fragment.text, /\[Context fragment truncated by TensorFold: showing 24000 of 60000 chars from project_memory/)
  assert.equal(fragment.estTokens, Math.ceil(fragment.text.length / 4), 'token cost matches the ~4 chars/token heuristic')

  const small = buildContextFragment({ source: 'channel_prompt', text: 'be brief', epoch: 0 })
  assert.equal(small.truncated, false)
  assert.equal(small.text, 'be brief')

  // 'pinned' content is never cut by this layer.
  const pinned = buildContextFragment({ source: 'goal', text: 'G'.repeat(100_000), epoch: 1 })
  assert.equal(pinned.truncated, false)
  assert.equal(pinned.chars, 100_000)

  // The persistable stat carries everything except the text.
  const stat = fragmentStat(fragment, 'VONTRA.md')
  assert.equal((stat as { text?: string }).text, undefined)
  assert.deepEqual({ source: stat.source, truncated: stat.truncated, note: stat.note }, { source: 'project_memory', truncated: true, note: 'VONTRA.md' })
  ok('fragments: priority budgets bound, truncation is visible in-band, pinned never cut, stats are text-free')
}

{
  // The context epoch is the compaction ledger length — 0 before any fold.
  assert.equal(contextEpoch({}), 0)
  assert.equal(contextEpoch({ compactions: [] }), 0)
  assert.equal(contextEpoch({ compactions: [{}, {}] as never }), 2)
  // Fragments built after a fold carry the new epoch.
  const thread = { compactions: [] as unknown[] }
  const before = buildContextFragment({ source: 'persona', text: 'p', epoch: contextEpoch(thread) })
  thread.compactions.push({ id: 'c1' })
  const after = buildContextFragment({ source: 'persona', text: 'p', epoch: contextEpoch(thread) })
  assert.equal(before.epoch, 0)
  assert.equal(after.epoch, 1)
  ok('epoch: derived from the compaction ledger; new fragments carry the post-fold epoch')
}

{
  // Huge tool output: head+tail kept, the middle spills to a managed file that the note NAMES.
  const output = `START${'x'.repeat(40_000)}THE-ERROR-IS-HERE`
  const bounded = boundToolOutput({ tool: 'run_command', output, spillRelPath: 'vontra-output://abc-001-run_command.txt', spillWritten: true })
  assert.equal(bounded.truncated, true)
  assert.equal(bounded.totalChars, output.length)
  assert.ok(bounded.content.startsWith('START'), 'head preserved')
  assert.ok(bounded.content.endsWith('THE-ERROR-IS-HERE'), 'tail preserved — build errors live at the end')
  assert.ok(bounded.content.length <= TOOL_OUTPUT_HEAD_CHARS + TOOL_OUTPUT_TAIL_CHARS + 400, 'bounded')
  assert.match(bounded.content, /Full output saved to vontra-output:\/\/abc-001-run_command\.txt — use read_file with offset\/limit/)
  assert.deepEqual(bounded.spill, { relPath: 'vontra-output://abc-001-run_command.txt', totalChars: output.length, totalLines: 1 })

  // Spill write failed → the note NEVER names a file that does not exist.
  const dropped = boundToolOutput({ tool: 'run_command', output, spillRelPath: 'vontra-output://abc-002-run_command.txt', spillWritten: false })
  assert.equal(dropped.spill, null)
  assert.match(dropped.content, /The middle was dropped \(no spill file available\)/)
  assert.ok(!dropped.content.includes('abc-002'), 'no phantom file named')

  // Under the cap → byte-identical pass-through.
  const small = boundToolOutput({ tool: 'read_file', output: 'tiny' })
  assert.deepEqual(small, { content: 'tiny', truncated: false, totalChars: 4, spill: null })
  assert.ok(TOOL_OUTPUT_MAX_CHARS >= TOOL_OUTPUT_HEAD_CHARS + TOOL_OUTPUT_TAIL_CHARS, 'cap covers head+tail')
  ok('tool output: head+tail bound with a named managed file; honest fallback when the spill fails; small outputs untouched')
}

{
  const path = toolOutputSpillPath('a1b2c3d4-e5f6', 7, 'run_command')
  assert.equal(path, 'vontra-output://a1b2c3d4-e5f6-007-run_command.txt')
  assert.match(toolOutputSpillPath('../../evil', 1, 'x/../y'), /^vontra-output:\/\/evil-001-xy\.txt$/, 'path pieces are sanitized')
  ok('spill paths: run-scoped, sequenced, sanitized')
}

{
  assert.equal(estimateTextTokens(''), 0)
  assert.equal(estimateTextTokens('abcd'), 1)
  assert.equal(estimateTextTokens('abcde'), 2)
  ok('token estimate mirrors the compaction heuristic (~4 chars/token, ceil)')
}

console.log(`  ${passed} passed`)
