// Run: npx jiti src/lib/turn-blocks.test.ts
import assert from 'node:assert/strict'
import {
  buildTurnRows,
  currentStepLabel,
  editTotals,
  isEstimatedThroughputMeta,
  isMachineryNotice,
  needsUser,
  shortToolMeta,
  turnFileEdits,
  turnActivityState,
  workRowLabel,
  type TurnRow,
} from './turn-blocks'
import type { TimelineItem } from './types'

let n = 0
const at = (s: number) => 1_000_000 + s * 1000
const user = (text: string, s = 0): TimelineItem => ({ id: `u${n++}`, kind: 'user', text, createdAt: at(s) })
const assistant = (text: string, s = 0): TimelineItem => ({ id: `a${n++}`, kind: 'assistant', text, createdAt: at(s) })
const tool = (name: string, extra: Partial<TimelineItem> = {}, s = 0): TimelineItem => ({ id: `t${n++}`, kind: 'tool', tool: name, title: name, state: 'ok', createdAt: at(s), ...extra })
const notice = (extra: Partial<TimelineItem>, s = 0): TimelineItem => ({ id: `n${n++}`, kind: 'notice', createdAt: at(s), ...extra })
const shape = (rows: TurnRow[]) => rows.map((row) => row.type === 'item' ? `${row.item.kind}:${row.item.id}` : row.type)

// A finished code turn: work folds, the answer and its edits stay visible.
{
  const u = user('add a toggle', 0)
  const interim = assistant('Let me look first.', 1)
  const reasoning: TimelineItem = { id: 'r1', kind: 'reasoning', text: 'plan', createdAt: at(1) }
  const read = tool('read_file', { path: 'src/App.tsx' }, 2)
  const edit = tool('edit_file', { path: 'src/App.tsx', diff: [{ type: 'hunk', text: '@@' }, { type: 'del', text: 'a' }, { type: 'add', text: 'b' }, { type: 'add', text: 'c' }, { type: 'ctx', text: 'd' }] }, 3)
  const run = tool('run_command', { command: 'npm test', changedPaths: ['coverage/out.json'] }, 5)
  const contract = notice({ title: 'Run contract', text: 'Run contract saved: code on local with m.' }, 0)
  const answer = assistant('Done.', 8)
  const receipt = notice({ title: 'Run receipt', text: 'Run receipt: verified.', runReceipt: { startedAt: at(0), finishedAt: at(12) } as any }, 12)
  const rows = buildTurnRows([u, contract, reasoning, interim, read, edit, run, answer, receipt])
  assert.deepEqual(shape(rows), [`user:${u.id}`, 'work', `assistant:${answer.id}`, 'edits'])
  const work = rows.find((row) => row.type === 'work')!.turn
  assert.deepEqual(work.work.map((item) => item.id), [contract.id, reasoning.id, interim.id, read.id, edit.id, run.id, receipt.id])
  assert.equal(work.index, 1)
  assert.equal(work.steps, 3)
  assert.equal(work.answerId, answer.id)
  assert.equal(workRowLabel(work, { now: at(20) }), 'Worked for 12s · 3 steps')
  assert.deepEqual(work.edits, [
    { path: 'src/App.tsx', additions: 2, deletions: 1, counted: true },
    { path: 'coverage/out.json', additions: 0, deletions: 0, counted: false },
  ])
  assert.deepEqual(editTotals(work.edits), { files: 2, additions: 2, deletions: 1 })
}

// Anything that needs the user stays out of the collapsed row.
{
  const u = user('go')
  const awaiting = tool('run_command', { state: 'awaiting', command: 'rm -rf build' })
  const err: TimelineItem = { id: 'e1', kind: 'error', text: 'boom' }
  const question = tool('ask_user', { state: 'pending' })
  const setting = tool('app_action', { appAction: { kind: 'change_setting', payload: { key: 'theme', value: 'dark' } } })
  const rows = buildTurnRows([u, tool('read_file'), awaiting, question, setting, err])
  assert.deepEqual(shape(rows).slice(0, 2), [`user:${u.id}`, 'work'])
  for (const item of [awaiting, question, setting, err]) assert.ok(shape(rows).includes(`${item.kind}:${item.id}`), `${item.id} visible`)
  assert.ok(needsUser(awaiting) && needsUser(question) && needsUser(setting))
  assert.ok(!needsUser(tool('ask_user', { state: 'ok' })))
}

// Live turn: the streaming text shows; edits wait for the end; the row names the current step.
{
  const u = user('go', 0)
  const running = tool('edit_file', { state: 'running', path: 'src/theme.ts' }, 3)
  const rows = buildTurnRows([user('earlier', -60), assistant('old answer', -50), u, tool('read_file', {}, 1), running], { running: true })
  const live = rows.filter((row) => row.type === 'work').at(-1)!.turn
  assert.equal(live.running, true)
  assert.equal(live.index, 2)
  assert.equal(currentStepLabel(live, 'tool'), 'Editing theme.ts')
  assert.equal(workRowLabel(live, { now: at(14), status: 'tool' }), 'Working · 14s · Editing theme.ts')
  assert.equal(workRowLabel(live, { now: at(14), status: 'awaiting' }), 'Working · 14s · Waiting for approval')
  assert.ok(!rows.some((row) => row.type === 'edits' && row.turn === live))
  // A live turn shows its row even before the first event arrives.
  const fresh = buildTurnRows([user('hello')], { running: true })
  assert.deepEqual(shape(fresh).slice(1), ['work'])
}

// Text after the last tool call is the answer; text before it is interim work.
{
  const u = user('q')
  const first = assistant('Checking.')
  const second = assistant('Here it is.')
  const rows = buildTurnRows([u, first, tool('search'), second])
  assert.ok(shape(rows).includes(`assistant:${second.id}`))
  assert.ok(!shape(rows).includes(`assistant:${first.id}`))
  // A plain chat answer has no work row at all.
  assert.deepEqual(shape(buildTurnRows([u, second])), [`user:${u.id}`, `assistant:${second.id}`])
}

// Notices: run bookkeeping folds; anything else stays a visible note. Items before the first message form turn 0.
{
  assert.ok(isMachineryNotice(notice({ title: 'Guard', text: 'Guardrail: stop.' })))
  assert.ok(isMachineryNotice(notice({ title: 'Verify-on-stop', text: 'Code changes need fresh verification.' })))
  assert.ok(isMachineryNotice(notice({ text: 'Provider hiccup (429) — retrying in 1.0s (attempt 1 of 3).' })))
  assert.ok(isMachineryNotice(notice({ runContract: {} as any, text: 'x' })))
  assert.ok(!isMachineryNotice(notice({ text: 'Model changed to GPT-5.' })))
  assert.ok(!isMachineryNotice(notice({ title: 'Budget', text: 'Spend is near the cap.' })))
  const pre = notice({ text: 'Handed off from another thread.' })
  const rows = buildTurnRows([pre, user('hi'), assistant('hello')])
  assert.equal(rows[0].type === 'item' && rows[0].item.id, pre.id)
  assert.equal(rows[0].turn.index, 0)
}

// Thought-only and timestamp-free turns.
{
  const u: TimelineItem = { id: 'u-x', kind: 'user', text: 'why' }
  const rows = buildTurnRows([u, { id: 'r-x', kind: 'reasoning', text: 'hmm' }, { id: 'a-x', kind: 'assistant', text: 'because' }])
  assert.equal(workRowLabel(rows[1].turn, { now: 0 }), 'Thought')
  const timed = buildTurnRows([user('why', 0), { id: 'r-y', kind: 'reasoning', text: 'hmm', createdAt: at(1) }, assistant('because', 5)])
  assert.equal(workRowLabel(timed[1].turn, { now: 0 }), 'Thought for 5s')
}

// Edits ignore denied tools and merge repeated paths.
{
  const edits = turnFileEdits([
    tool('edit_file', { path: './a.ts', diff: [{ type: 'add', text: 'x' }] }),
    tool('edit_file', { path: 'a.ts', diff: [{ type: 'del', text: 'y' }], changedPaths: ['a.ts'] }),
    tool('write_file', { path: 'b.ts', state: 'denied', diff: [{ type: 'add', text: 'z' }] }),
  ])
  assert.deepEqual(edits, [{ path: 'a.ts', additions: 1, deletions: 1, counted: true }])
}

// Display helpers.
{
  assert.equal(shortToolMeta('exit 0 · 240ms · sandbox=build-test · macos-sandbox-exec · artifacts=sbox-1 · redacted=2'), 'exit 0 · 240ms')
  assert.equal(shortToolMeta('62 lines'), '62 lines')
  assert.equal(shortToolMeta(undefined), '')
  assert.ok(isEstimatedThroughputMeta('~212 tok/s'))
  assert.ok(!isEstimatedThroughputMeta('stopped — repeating itself'))
}

console.log('turn-blocks: ok')

{
  const rows = buildTurnRows([user('Check the changes'), tool('run_command'), assistant('Finished.')])
  const turn = rows.find((row) => row.type === 'work')!.turn
  const withReceipt = (verdict: string) => ({ ...turn, work: [...turn.work, notice({ runReceipt: { verdict } as any })] })
  assert.equal(turnActivityState({ ...turn, running: true }, 'awaiting'), 'waiting')
  assert.equal(turnActivityState({ ...turn, running: true }, 'tool'), 'running')
  assert.equal(turnActivityState(turn), 'completed')
  assert.equal(turnActivityState(withReceipt('cancelled')), 'stopped')
  assert.equal(turnActivityState(withReceipt('failed')), 'failed')
  assert.equal(turnActivityState(withReceipt('blocked')), 'review')
  assert.equal(turnActivityState(withReceipt('unverified')), 'review')
  assert.equal(turnActivityState(withReceipt('verified')), 'completed')
  assert.equal(turnActivityState({ ...turn, work: [tool('run_command', { state: 'running' })] }), 'interrupted')
  assert.equal(turnActivityState({ ...turn, work: [tool('run_command', { state: 'error' })] }), 'review')
}

{
  const question = tool('ask_user', { state: 'awaiting', question: { id: 'q-visible', state: 'pending' } as any })
  const rows = buildTurnRows([user('Help me choose'), question], { running: true })
  const work = rows.find((row) => row.type === 'work')!.turn
  assert.equal(currentStepLabel(work, 'awaiting'), 'Waiting for your answer')
  assert.ok(shape(rows).includes(`tool:${question.id}`), 'a pending question stays visible outside collapsed work')
  const answered = { ...question, state: 'ok' as const, question: { ...question.question!, state: 'answered' as const, answer: 'Amber' } }
  const settled = buildTurnRows([user('Help me choose'), answered, assistant('Amber it is.')])
  assert.ok(shape(settled).includes(`tool:${question.id}`), 'the answer remains readable in the conversation after completion')
}
