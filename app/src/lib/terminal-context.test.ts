// Run: jiti src/lib/terminal-context.test.ts
//
// Composer terminal-context injection (VST-039): ANSI stripping, tail bounding, fenced
// reference building.
import assert from 'node:assert/strict'
import { buildTerminalReference, stripAnsi, terminalTail } from './terminal-context'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('terminal-context tests:')

{
  const colored = '\x1b[32mPASS\x1b[0m src/app.test.ts\n\x1b[1m\x1b[31mFAIL\x1b[0m other.test.ts'
  assert.equal(stripAnsi(colored), 'PASS src/app.test.ts\nFAIL other.test.ts')
  // OSC title set + cursor moves are stripped too.
  assert.equal(stripAnsi('\x1b]0;my title\x07done\x1b[2K'), 'done')
  ok('stripAnsi: colour, OSC, cursor codes removed; text preserved')
}

{
  const raw = Array.from({ length: 100 }, (_, i) => `line ${i + 1}`).join('\n') + '\n\n\n'
  const tail = terminalTail(raw, { maxLines: 5 })
  assert.equal(tail, 'line 96\nline 97\nline 98\nline 99\nline 100', 'keeps the last 5 lines, drops trailing blanks')

  const big = 'x'.repeat(9000)
  const capped = terminalTail(big, { maxChars: 100 })
  assert.ok(capped.length <= 100, 'char cap enforced')
  assert.ok(capped.startsWith('…'), 'front-trimmed with an ellipsis (recent output wins)')

  assert.equal(terminalTail('   \n  \n'), '', 'all-blank scrollback yields nothing')
  ok('terminalTail: line + char bounds, blank-safe')
}

{
  const ref = buildTerminalReference('npm test', 'PASS a\n\x1b[31mFAIL\x1b[0m b\n')
  assert.match(ref, /Regarding recent output from the .npm test. terminal/)
  assert.match(ref, /```\nPASS a\nFAIL b\n```/, 'ANSI-stripped output in a fence')

  const empty = buildTerminalReference('shell', '   ')
  assert.match(empty, /no recent output captured/)
  assert.ok(!empty.includes('```'), 'no empty fence for empty output')
  ok('buildTerminalReference: named, fenced, honest on empty')
}

console.log(`  ${passed} passed`)
