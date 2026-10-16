// Run: jiti src/lib/thread-row.test.ts
//
// Sidebar thread rows: run state, diff stats, place label, day groups and compact ages.
import assert from 'node:assert/strict'
import { compactAge, countNeedsYou, groupThreads, threadDiffStats, threadPlace, threadRowState } from './thread-row'
import type { Job, Thread, TimelineItem } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-row tests:')

const idle = { id: 't1', status: 'idle' as const }

{
  assert.equal(threadRowState({ thread: idle, run: 'tool' }), 'running')
  assert.equal(threadRowState({ thread: idle, run: 'streaming' }), 'running')
  assert.equal(threadRowState({ thread: idle, run: 'awaiting' }), 'needs-you')
  assert.equal(threadRowState({ thread: idle, run: 'tool', approvals: 1 }), 'needs-you')
  assert.equal(threadRowState({ thread: idle, run: 'error' }), 'failed')
  assert.equal(threadRowState({ thread: idle, run: 'idle' }), 'done')
  assert.equal(threadRowState({ thread: idle }), 'done')
  ok('live run status maps to running, needs you, failed and done')
}

{
  assert.equal(threadRowState({ thread: { id: 't1', status: 'running' } }), 'done')
  assert.equal(threadRowState({ thread: { id: 't1', status: 'error' } }), 'failed')
  ok('a persisted running flag alone is not trusted; a persisted error still shows')
}

{
  const job = (threadId: string, resolvedAt: number | null) => ({ pendingApprovals: [{ id: 'a', callId: 'c', threadId, itemId: null, tool: 'run_command', requestedAt: 1, resolvedAt }] }) as unknown as Pick<Job, 'pendingApprovals'>
  assert.equal(threadRowState({ thread: idle, jobs: [job('t1', null)] }), 'needs-you')
  assert.equal(threadRowState({ thread: idle, jobs: [job('t1', 5)] }), 'done')
  assert.equal(threadRowState({ thread: idle, jobs: [job('other', null)] }), 'done')
  ok('an unresolved job approval on this thread means it needs you')
}

{
  assert.equal(countNeedsYou(['needs-you', 'running', 'needs-you', 'done']), 2)
  ok('counts threads that need you')
}

const item = (p: Partial<TimelineItem>): TimelineItem => ({ id: Math.random().toString(36).slice(2), kind: 'tool', ...p } as TimelineItem)

{
  const timeline = [
    item({ kind: 'user', text: 'go' }),
    item({ tool: 'edit_file', path: 'a.ts', state: 'ok', diff: [{ type: 'hunk', text: '@@' }, { type: 'del', text: 'x' }, { type: 'add', text: 'y' }, { type: 'add', text: 'z' }] }),
    item({ tool: 'write_file', path: 'b.ts', state: 'ok', diff: [{ type: 'add', text: '1' }] }),
    item({ tool: 'edit_file', path: 'c.ts', state: 'denied', diff: [{ type: 'add', text: 'no' }] }),
    item({ tool: 'read_file', path: 'd.ts', state: 'ok' }),
  ]
  assert.deepEqual(threadDiffStats({ timeline }), { files: 2, additions: 3, deletions: 1 })
  assert.equal(threadDiffStats({ timeline: [item({ kind: 'user', text: 'hi' })] }), null)
  ok('diff stats count only successful edits with diffs')
}

{
  const project = { id: 'p1', name: 'Zora-Agent' }
  assert.equal(threadPlace({ kind: 'task', cwd: '/x/web-dashboard', projectId: 'p1' }, project), 'Zora-Agent')
  assert.equal(threadPlace({ kind: 'task', cwd: '/x/web-dashboard/', projectId: null }, null), 'web-dashboard')
  assert.equal(threadPlace({ kind: 'task', cwd: '', projectId: null }, null), 'Chat')
  assert.equal(threadPlace({ kind: 'chat', cwd: '', projectId: null }, null), 'Chat')
  ok('place is the project, else the folder, else Chat')
}

{
  const now = new Date(2026, 9, 12, 15, 0, 0).getTime()
  const at = (d: number, h: number) => new Date(2026, 9, d, h, 0, 0).getTime()
  const threads = [
    { id: 'a', updatedAt: at(12, 9) },
    { id: 'b', updatedAt: at(11, 23) },
    { id: 'c', updatedAt: at(3, 10) },
    { id: 'd', updatedAt: at(12, 14) },
    { id: 'e', updatedAt: at(1, 10) },
  ] as Pick<Thread, 'id' | 'updatedAt'>[]
  const groups = groupThreads(threads, new Set(['e']), now)
  assert.deepEqual(groups.map((g) => [g.label, g.threads.map((t) => t.id)]), [
    ['Pinned', ['e']],
    ['Today', ['d', 'a']],
    ['Yesterday', ['b']],
    ['Earlier', ['c']],
  ])
  assert.deepEqual(groupThreads([{ id: 'x', updatedAt: at(12, 1) }], new Set(), now).map((g) => g.key), ['today'])
  ok('groups by pinned, today, yesterday and earlier, newest first, without empty groups')
}

{
  const now = 1_000_000_000_000
  assert.equal(compactAge(now - 10_000, now), 'now')
  assert.equal(compactAge(now - 4 * 60_000, now), '4m')
  assert.equal(compactAge(now - 3 * 3_600_000, now), '3h')
  assert.equal(compactAge(now - 2 * 86_400_000, now), '2d')
  assert.equal(compactAge(now - 15 * 86_400_000, now), '2w')
  assert.equal(compactAge(now + 5_000, now), 'now')
  ok('compact ages')
}

console.log(`\n${passed} thread-row tests passed`)
