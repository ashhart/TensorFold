// Run: jiti electron/agent/tool-paths.test.ts
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, symlinkSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { realpathSync } from 'node:fs'
import { safeResolve } from './tool-paths.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('tool-paths tests:')

const root = realpathSync.native(mkdtempSync(path.join(tmpdir(), 'vontra-safe-')))
const outside = realpathSync.native(mkdtempSync(path.join(tmpdir(), 'vontra-out-')))
try {
  // ordinary in-workspace paths resolve, lexical escapes are rejected
  assert.equal(safeResolve(root, 'a/b.txt'), path.join(root, 'a/b.txt'))
  assert.throws(() => safeResolve(root, '../escape.txt'), /escapes the workspace/)
  ok('resolves in-workspace paths and rejects lexical ".." escapes')

  // DANGLING symlink final component: existsSync follows the link and reports "missing", so
  // the ancestor-walk climbs to the (in-workspace) parent and passes — but a write would
  // follow the link to its out-of-workspace destination. Must be rejected (gap-audit).
  symlinkSync(path.join(outside, 'nope.txt'), path.join(root, 'dangling'))
  assert.throws(() => safeResolve(root, 'dangling'), /symlink/i, 'dangling symlink to outside rejected')

  // symlink to an existing outside file: rejected
  writeFileSync(path.join(outside, 'secret.txt'), 'x')
  symlinkSync(path.join(outside, 'secret.txt'), path.join(root, 'tolink'))
  assert.throws(() => safeResolve(root, 'tolink'), /symlink/i, 'symlink to outside file rejected')
  ok('rejects a symlink final component that escapes the workspace (dangling or live)')

  // symlink pointing INSIDE the workspace stays allowed (writes land in-workspace)
  mkdirSync(path.join(root, 'sub'))
  writeFileSync(path.join(root, 'sub', 'real.txt'), 'y')
  symlinkSync(path.join(root, 'sub', 'real.txt'), path.join(root, 'inlink'))
  assert.doesNotThrow(() => safeResolve(root, 'inlink'), 'in-workspace symlink allowed')
  ok('allows a symlink that resolves inside the workspace')

  console.log(`\n${passed} passed`)
} finally {
  rmSync(root, { recursive: true, force: true })
  rmSync(outside, { recursive: true, force: true })
}
