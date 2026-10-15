// Run: jiti electron/agent/tool-patch.test.ts
import assert from 'node:assert/strict'
import { replaceLiteral } from './tool-patch.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('tool-patch tests:')

{
  // The whole point: the replacement is taken LITERALLY. String.prototype.replace expands
  // $&, $`, $', $1..$9 and $$ inside the replacement argument — which silently corrupts any
  // edit whose new text contains those tokens (regex replacement strings, shell $1/$$, jQuery
  // $-identifiers, template literals). replaceLiteral must write the bytes verbatim.
  assert.equal(replaceLiteral('a X b', 'X', '$&'), 'a $& b', '$& must not expand to the match')
  assert.equal(replaceLiteral('foo', 'foo', '$1'), '$1', '$1 must not expand to a (nonexistent) group')
  assert.equal(replaceLiteral('a', 'a', '$$'), '$$', '$$ must not collapse to a single $')
  assert.equal(replaceLiteral('x y', 'y', '$`'), 'x $`', '$` must not expand to the prefix')
  assert.equal(replaceLiteral('x y', 'x', "$'"), "$' y", "$' must not expand to the suffix")
  assert.equal(replaceLiteral('const p = price * $rate', '$rate', 'rate'), 'const p = price * rate', 'ordinary replacement still works')
  ok('replaceLiteral: replacement string is never $-expanded')
}

{
  // Matches String.replace string-pattern semantics: only the first occurrence is replaced.
  assert.equal(replaceLiteral('a a a', 'a', 'b'), 'b a a', 'replaces only the first occurrence')
  assert.equal(replaceLiteral('no match here', 'zzz', 'X'), 'no match here', 'no match leaves input unchanged')
  ok('replaceLiteral: first-occurrence-only, no-op on miss')
}

console.log(`\n${passed} passed`)
