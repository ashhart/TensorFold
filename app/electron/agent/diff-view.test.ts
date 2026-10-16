// Run: jiti electron/agent/diff-view.test.ts
//
// Diff-view depth (VST-040): whitespace-ignore diff, per-turn navigation, file-tree filter,
// ask-why reference payload. Pure (composes diff.ts).
import assert from 'node:assert/strict'
import {
  computeDiffView,
  computeDiffIgnoringWhitespace,
  diffTurns,
  stepDiffTurn,
  buildDiffFileTree,
  buildAskWhyReference,
} from './diff-view'
import type { TimelineItem } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('diff-view tests:')

// --- whitespace-ignore ---
{
  // Only indentation changed → an ignore-whitespace diff shows no add/del content lines.
  const before = 'function f() {\nreturn 1\n}'
  const after = 'function f() {\n    return 1\n}'
  const normal = computeDiffView(before, after)
  assert.ok(normal.some((l) => l.type === 'add' || l.type === 'del'), 'a normal diff sees the reindent')
  const ignored = computeDiffIgnoringWhitespace(before, after)
  assert.equal(ignored.filter((l) => l.type === 'add' || l.type === 'del').length, 0, 'ignore-whitespace hides a pure reindent')

  // A real content change still shows under ignore-whitespace.
  const realChange = computeDiffView('return 1', 'return 2', { ignoreWhitespace: true })
  assert.ok(realChange.some((l) => l.type === 'add' || l.type === 'del'), 'a real change survives ignore-whitespace')
  ok('computeDiffView: ignoreWhitespace hides reindents, keeps real changes')
}

// --- per-turn navigation ---
{
  const timeline: TimelineItem[] = [
    { id: 'u1', kind: 'user' } as TimelineItem,
    { id: 't1', kind: 'tool', tool: 'edit_file', args: { path: 'src/a.ts' } } as TimelineItem,
    { id: 'a1', kind: 'assistant' } as TimelineItem,
    { id: 'u2', kind: 'user' } as TimelineItem,
    { id: 't2', kind: 'tool', tool: 'run_command', command: 'ls' } as TimelineItem, // not a write
    { id: 'u3', kind: 'user' } as TimelineItem,
    { id: 't3', kind: 'tool', tool: 'write_file', args: { path: 'src/b.ts' } } as TimelineItem,
  ]
  const turns = diffTurns(timeline)
  assert.deepEqual(turns.map((t) => t.turn), [1, 3], 'only turns with file writes are diff turns (turn 2 ran a command)')
  assert.deepEqual(turns.map((t) => t.files[0]), ['src/a.ts', 'src/b.ts'])

  assert.equal(stepDiffTurn(turns, 0, 'next')?.turn, 1, 'next from 0 → first diff turn')
  assert.equal(stepDiffTurn(turns, 1, 'next')?.turn, 3, 'next skips the non-diff turn 2')
  assert.equal(stepDiffTurn(turns, 3, 'next'), null, 'no next past the last')
  assert.equal(stepDiffTurn(turns, 3, 'prev')?.turn, 1)
  assert.equal(stepDiffTurn(turns, 1, 'prev'), null, 'no prev before the first')
  ok('diffTurns + stepDiffTurn: per-turn diff navigation skips non-diff turns')
}

// --- file tree + filter ---
{
  const paths = ['src/app/main.ts', 'src/app/util.ts', 'src/lib/store.ts', 'README.md']
  const tree = buildDiffFileTree(paths)
  const top = tree.map((n) => n.name)
  assert.deepEqual(top, ['README.md', 'src'], 'sorted top level: files + dirs')
  const src = tree.find((n) => n.name === 'src')!
  assert.equal(src.isDir, true)
  assert.deepEqual(src.children.map((n) => n.name), ['app', 'lib'])
  const app = src.children.find((n) => n.name === 'app')!
  assert.deepEqual(app.children.map((n) => n.path), ['src/app/main.ts', 'src/app/util.ts'])
  assert.equal(app.children[0].isDir, false)

  // Filter narrows to matching leaves (keeping ancestors).
  const filtered = buildDiffFileTree(paths, 'store')
  assert.deepEqual(filtered.map((n) => n.name), ['src'])
  assert.deepEqual(filtered[0].children.map((n) => n.name), ['lib'], 'only the matching branch survives the filter')
  assert.deepEqual(buildDiffFileTree(paths, 'zzz'), [], 'no matches → empty tree')
  ok('buildDiffFileTree: nested tree, sorted, case-insensitive filter keeps ancestors')
}

// --- ask-why reference payload ---
{
  const diff = [
    { type: 'hunk', text: '@@ -1,2 +1,2 @@' },
    { type: 'del', text: 'const x = 1' },
    { type: 'add', text: 'const x = 2' },
    { type: 'ctx', text: 'return x' },
  ] as any
  const ref = buildAskWhyReference({ file: 'src/x.ts', diff, turn: 4 })
  assert.equal(ref.file, 'src/x.ts')
  assert.equal(ref.turn, 4)
  assert.ok(ref.prompt.includes('src/x.ts') && ref.prompt.includes('turn 4'), 'names the file + turn')
  assert.ok(ref.prompt.includes('+ const x = 2') && ref.prompt.includes('- const x = 1'), 'embeds the hunk excerpt as a diff block')
  assert.ok(ref.prompt.includes('```diff'), 'wraps the excerpt in a diff fence')

  // Empty diff → a sensible file-level ask, no fence.
  const bare = buildAskWhyReference({ file: '', diff: [] })
  assert.ok(bare.prompt.includes('this file') && !bare.prompt.includes('```'), 'empty diff → file-level ask, no fence')
  assert.equal(bare.turn, null)
  ok('buildAskWhyReference: file + turn + bounded hunk excerpt; graceful when empty')
}

console.log(`\n${passed} passed`)
