// Run: jiti electron/agent/edit-syntax-gate.test.ts
import assert from 'node:assert/strict'
import { checkEditedSyntax, isGatedCodePath } from './edit-syntax-gate.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('edit-syntax-gate tests:')

// ---- isGatedCodePath: only real code extensions are gated ----
{
  for (const p of ['a.ts', 'b.tsx', 'c.mts', 'd.js', 'e.jsx', 'f.mjs', 'g.cjs', 'h.json']) {
    assert.equal(isGatedCodePath(p), true, `${p} should be gated`)
  }
  for (const p of ['readme.md', 'styles.css', 'notes.txt', 'data.yaml', 'script.py', 'Dockerfile', 'x.svg', 'noext']) {
    assert.equal(isGatedCodePath(p), false, `${p} should NOT be gated`)
  }
  ok('isGatedCodePath gates ts/tsx/js/jsx/json only')
}

// ---- valid TypeScript passes ----
{
  const r = await checkEditedSyntax('src/x.ts', 'export const add = (a: number, b: number): number => a + b\n')
  assert.equal(r.checked, true)
  assert.equal(r.ok, true)
  ok('valid TypeScript passes the gate')
}

// ---- valid TSX (with JSX) passes ----
{
  const r = await checkEditedSyntax('src/C.tsx', 'export const C = () => <div className="x">{1 + 2}</div>\n')
  assert.equal(r.ok, true)
  assert.equal(r.checked, true)
  ok('valid TSX with JSX passes')
}

// ---- broken TypeScript (unbalanced brace) is caught ----
{
  const r = await checkEditedSyntax('src/x.ts', 'export function f() {\n  return 1\n// missing close brace\n')
  assert.equal(r.checked, true)
  assert.equal(r.ok, false, 'unbalanced brace must fail')
  assert.ok(typeof r.error === 'string' && r.error.length > 0, 'a syntax error message is returned')
  ok('broken TypeScript (unbalanced brace) is caught with a message')
}

// ---- broken JS (dangling paren) is caught ----
{
  const r = await checkEditedSyntax('src/x.js', 'const y = (1 + \n')
  assert.equal(r.ok, false)
  ok('broken JS (dangling paren) is caught')
}

// ---- valid JSON passes; broken JSON fails ----
{
  const good = await checkEditedSyntax('pkg.json', '{"a":1,"b":[2,3]}')
  assert.equal(good.ok, true)
  assert.equal(good.checked, true)
  const bad = await checkEditedSyntax('pkg.json', '{"a":1,,}')
  assert.equal(bad.ok, false, 'malformed JSON must fail')
  assert.ok(bad.error && bad.error.length > 0)
  ok('valid JSON passes, malformed JSON fails')
}

// ---- TYPE errors are NOT syntax errors: the gate must not block them ----
{
  // `const n: number = "string"` is a type error but syntactically valid TS — esbuild transpiles it fine.
  const r = await checkEditedSyntax('src/x.ts', 'const n: number = "a string, type-wrong but parseable"\nexport default n\n')
  assert.equal(r.ok, true, 'type errors are not the gate\'s job — only syntax')
  ok('type errors pass (gate is syntax-only, not a type checker)')
}

// ---- non-code files are skipped (checked:false, ok:true), never blocked ----
{
  const md = await checkEditedSyntax('README.md', '# Title\n\nthis is { not ] valid code (((')
  assert.equal(md.checked, false)
  assert.equal(md.ok, true)
  const py = await checkEditedSyntax('script.py', 'def f(:\n  pass')
  assert.equal(py.checked, false)
  assert.equal(py.ok, true)
  ok('non-code files are skipped, never blocked')
}

// ---- empty / whitespace content is allowed (deletion / clearing a file) ----
{
  const empty = await checkEditedSyntax('src/x.ts', '')
  assert.equal(empty.ok, true)
  const ws = await checkEditedSyntax('src/x.ts', '   \n  ')
  assert.equal(ws.ok, true)
  ok('empty / whitespace-only content is allowed')
}

// ---- oversized content is skipped (fail-open, no perf cliff) ----
{
  const huge = 'const x = 1\n'.repeat(120_000) // > ~1MB
  const r = await checkEditedSyntax('src/x.ts', huge)
  assert.equal(r.checked, false, 'oversized files are skipped, not parsed')
  assert.equal(r.ok, true)
  ok('oversized content is skipped (fail-open)')
}

console.log(`\n${passed} checks passed.`)
