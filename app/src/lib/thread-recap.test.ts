// Run: jiti src/lib/thread-recap.test.ts
//
// Generated thread recap (VST-040): deterministic summary from the timeline — turns, files
// changed/read, commands, last error, headline.
import assert from 'node:assert/strict'
import { buildThreadRecap } from './thread-recap'
import type { TimelineItem } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-recap tests:')

const item = (p: Partial<TimelineItem>): TimelineItem => ({ id: Math.random().toString(36).slice(2), kind: 'notice', ...p } as TimelineItem)

{
  const recap = buildThreadRecap([])
  assert.equal(recap.text, '')
  assert.equal(recap.turns, 0)
  assert.equal(recap.headline, null)
  ok('empty timeline → empty recap')
}

{
  const timeline = [
    item({ kind: 'user', text: 'add a dark mode toggle' }),
    item({ kind: 'tool', tool: 'read_file', path: 'src/App.tsx' }),
    item({ kind: 'tool', tool: 'edit_file', path: 'src/App.tsx' }),
    item({ kind: 'tool', tool: 'write_file', path: 'src/useTheme.ts' }),
    item({ kind: 'tool', tool: 'run_command', command: 'npm test' }),
    item({ kind: 'assistant', text: 'Added a persistent dark-mode toggle. Tests pass.' }),
  ]
  const recap = buildThreadRecap(timeline)
  assert.equal(recap.turns, 1)
  assert.deepEqual(recap.filesChanged, ['src/App.tsx', 'src/useTheme.ts'], 'distinct edited files, sorted')
  assert.deepEqual(recap.filesRead, ['src/App.tsx'])
  assert.equal(recap.commands, 1)
  assert.equal(recap.headline, 'Added a persistent dark-mode toggle.')
  assert.match(recap.text, /1 turn · 2 files changed · 1 command run/)
  assert.match(recap.text, /\*\*Changed:\*\* src\/App\.tsx, src\/useTheme\.ts/)
  ok('recap: counts turns/files/commands, headline is the newest assistant sentence')
}

{
  const timeline = [
    item({ kind: 'user', text: 'fix the build' }),
    item({ kind: 'assistant', text: 'First attempt.' }),
    item({ kind: 'error', text: 'TypeError: cannot read property x of undefined at foo.ts:10' }),
    item({ kind: 'assistant', text: 'Second, corrected attempt that resolves it.' }),
  ]
  const recap = buildThreadRecap(timeline)
  assert.equal(recap.headline, 'Second, corrected attempt that resolves it.', 'headline follows the LATEST assistant message')
  assert.match(recap.lastError ?? '', /TypeError/)
  assert.match(recap.text, /\*\*Last error:\*\*/)
  ok('recap: last error surfaced, headline tracks the freshest assistant text')
}

{
  const recap = buildThreadRecap([item({ kind: 'assistant', text: '```ts\nconst x = 1\n```\nDone wiring the config.' })])
  assert.equal(recap.headline, 'Done wiring the config.')
  ok('recap: headline skips fenced code')
}

console.log(`  ${passed} passed`)
