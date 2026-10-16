// Run: jiti src/lib/thread-changes.test.ts
//
// Changes tab model: files grouped by the turn that last touched them, joined with git status,
// and the unified diff parser behind the diff view.
import assert from 'node:assert/strict'
import { changeTotals, parseUnifiedDiff, relativeToCwd, threadChangeTurns } from './thread-changes'
import type { GitStatus, TimelineItem } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-changes tests:')

const item = (p: Partial<TimelineItem>): TimelineItem => ({ id: Math.random().toString(36).slice(2), kind: 'tool', ...p } as TimelineItem)

{
  assert.equal(relativeToCwd('/repo/app/src/a.ts', '/repo/app'), 'src/a.ts')
  assert.equal(relativeToCwd('/repo/app/src/a.ts', '/repo/app/'), 'src/a.ts')
  assert.equal(relativeToCwd('./src/a.ts', '/repo/app'), 'src/a.ts')
  assert.equal(relativeToCwd('src/a.ts', ''), 'src/a.ts')
  ok('tool paths become repo-relative')
}

const git: GitStatus = {
  branch: 'main',
  files: [
    { path: 'src/App.tsx', status: 'M', additions: 3, deletions: 1 },
    { path: 'src/theme.ts', status: 'A', additions: 20, deletions: 0 },
    { path: 'package-lock.json', status: 'M', additions: 4, deletions: 4 },
  ],
  additions: 27,
  deletions: 5,
}

{
  const timeline = [
    item({ kind: 'user', text: 'Add a toggle' }),
    item({ tool: 'edit_file', path: '/repo/src/App.tsx', state: 'ok', diff: [{ type: 'add', text: 'a' }] }),
    item({ kind: 'user', text: 'Now persist it' }),
    item({ tool: 'write_file', path: 'src/theme.ts', state: 'ok', diff: [{ type: 'add', text: 'b' }] }),
    item({ tool: 'edit_file', path: 'src/denied.ts', state: 'denied' }),
    item({ tool: 'run_command', command: 'npm i', state: 'ok', changedPaths: ['package-lock.json'] }),
  ]
  const turns = threadChangeTurns({ timeline }, git, '/repo')
  assert.deepEqual(turns.map((t) => [t.turn, t.prompt, t.files.map((f) => f.path)]), [
    [2, 'Now persist it', ['package-lock.json', 'src/theme.ts']],
    [1, 'Add a toggle', ['src/App.tsx']],
  ])
  assert.deepEqual(turns[1].files[0], { path: 'src/App.tsx', status: 'M', additions: 3, deletions: 1, editDiff: [{ type: 'add', text: 'a' }] })
  assert.deepEqual(changeTotals(turns), { files: 3, additions: 27, deletions: 5 })
  ok('groups by turn, newest first, using git counts and changedPaths')
}

{
  const timeline = [
    item({ kind: 'user', text: 'one' }),
    item({ tool: 'edit_file', path: 'src/App.tsx', state: 'ok', diff: [{ type: 'add', text: 'old' }] }),
    item({ kind: 'user', text: 'two' }),
    item({ tool: 'edit_file', path: 'src/App.tsx', state: 'ok', diff: [{ type: 'del', text: 'x' }, { type: 'add', text: 'new' }] }),
  ]
  const turns = threadChangeTurns({ timeline }, null, '')
  assert.equal(turns.length, 1)
  assert.equal(turns[0].turn, 2)
  assert.deepEqual(turns[0].files[0], { path: 'src/App.tsx', status: null, additions: 1, deletions: 1, editDiff: [{ type: 'del', text: 'x' }, { type: 'add', text: 'new' }] })
  ok('a file edited in two turns lists once under its latest turn; no git falls back to the edit diff')
}

{
  const turns = threadChangeTurns({ timeline: [item({ kind: 'user', text: 'hi' })] }, git, '/repo')
  assert.equal(turns.length, 1)
  assert.equal(turns[0].key, 'other')
  assert.equal(turns[0].files.length, 3)
  assert.deepEqual(threadChangeTurns({ timeline: [] }, null, ''), [])
  ok('working-tree changes the thread did not make go in a separate group')
}

{
  const timeline = [item({ kind: 'user', text: 'x' }), item({ tool: 'edit_file', path: 'App.tsx', state: 'ok', diff: [{ type: 'add', text: 'a' }] })]
  const turns = threadChangeTurns({ timeline }, git, '/repo/src')
  assert.equal(turns[0].files[0].path, 'src/App.tsx')
  assert.equal(turns.find((t) => t.key === 'other')?.files.some((f) => f.path === 'src/App.tsx'), false)
  ok('a thread in a repo subfolder matches git paths by suffix')
}

{
  const text = [
    'diff --git a/src/App.tsx b/src/App.tsx',
    'index 1..2 100644',
    '--- a/src/App.tsx',
    '+++ b/src/App.tsx',
    '@@ -18,3 +18,4 @@ export function App() {',
    '-  return <main>',
    '+  return <main className="x">',
    '+    <Dock />',
    '     <Thread />',
    '\\ No newline at end of file',
    'diff --git a/b.ts b/b.ts',
    '--- /dev/null',
    '+++ b/b.ts',
    '@@ -0,0 +1 @@',
    '+export {}',
    '',
  ].join('\n')
  assert.deepEqual(parseUnifiedDiff(text), [
    { type: 'hunk', text: '@@ -18,3 +18,4 @@ export function App() {' },
    { type: 'del', text: '  return <main>', oldNo: 18 },
    { type: 'add', text: '  return <main className="x">', newNo: 18 },
    { type: 'add', text: '    <Dock />', newNo: 19 },
    { type: 'ctx', text: '    <Thread />', oldNo: 19, newNo: 20 },
    { type: 'hunk', text: '@@ -0,0 +1 @@' },
    { type: 'add', text: 'export {}', newNo: 1 },
  ])
  assert.deepEqual(parseUnifiedDiff(''), [])
  ok('parses unified diffs, skipping headers and no-newline markers')
}

console.log(`\n${passed} thread-changes tests passed`)
