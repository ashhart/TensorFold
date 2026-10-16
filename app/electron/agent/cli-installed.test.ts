// Run: jiti electron/agent/cli-installed.test.ts
import assert from 'node:assert/strict'
import { builtinClis, cliBinaryPresent, installedClis, makeCustomCli } from './cli-registry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-installed tests:')

// A fake filesystem: only these absolute paths "exist". Probe dirs are fixed so the test never
// depends on the real machine's PATH.
const DIRS = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']
const present = new Set(['/opt/homebrew/bin/claude', '/usr/local/bin/codex', '/usr/bin/git'])
const exists = (p: string) => present.has(p)

{
  const shell = builtinClis().find((c) => c.id === 'shell')!
  assert.equal(cliBinaryPresent(shell, { dirs: DIRS, exists }), true, 'plain shell is always present (no command to resolve)')
  ok('cliBinaryPresent: a plain shell needs no binary')
}

{
  const claude = builtinClis().find((c) => c.id === 'claude')!
  const gemini = builtinClis().find((c) => c.id === 'gemini')!
  assert.equal(cliBinaryPresent(claude, { dirs: DIRS, exists }), true, 'claude resolves in /opt/homebrew/bin')
  assert.equal(cliBinaryPresent(gemini, { dirs: DIRS, exists }), false, 'gemini is not on the fake PATH')
  ok('cliBinaryPresent: installed => true, missing => false')
}

{
  // The launch command may carry flags; only the first word (the binary) is probed.
  const flagged = makeCustomCli('Flagged', 'codex --full-auto')
  assert.equal(cliBinaryPresent(flagged, { dirs: DIRS, exists }), true, 'probes the first word, ignores flags')
  ok('cliBinaryPresent: resolves the binary word, not the whole command line')
}

{
  // An absolute/relative path command is taken at face value against the fake fs.
  const abs = makeCustomCli('Abs', '/usr/bin/git status')
  const absMissing = makeCustomCli('AbsMissing', '/nope/tool')
  assert.equal(cliBinaryPresent(abs, { dirs: DIRS, exists }), true, 'absolute path that exists => true')
  assert.equal(cliBinaryPresent(absMissing, { dirs: DIRS, exists }), false, 'absolute path that is missing => false')
  ok('cliBinaryPresent: path-style commands are checked directly')
}

{
  const filtered = installedClis(builtinClis(), { dirs: DIRS, exists })
  const ids = filtered.map((c) => c.id)
  assert.ok(ids.includes('shell'), 'shell survives filtering (always kept)')
  assert.ok(ids.includes('claude') && ids.includes('codex'), 'installed CLIs are kept')
  assert.ok(!ids.includes('cursor') && !ids.includes('gemini') && !ids.includes('grok') && !ids.includes('kilo') && !ids.includes('pi') && !ids.includes('opencode'), 'uninstalled CLIs are dropped')
  ok('installedClis: keeps shell + installed binaries, drops the rest')
}

{
  // Order is preserved (shell first, as the registry declares it).
  const filtered = installedClis(builtinClis(), { dirs: DIRS, exists })
  assert.equal(filtered[0].id, 'shell', 'declaration order is preserved')
  ok('installedClis: preserves declaration order')
}

{
  // Even with an empty filesystem the shell is still offered (the launcher fallback).
  const filtered = installedClis(builtinClis(), { dirs: DIRS, exists: () => false })
  assert.deepEqual(filtered.map((c) => c.id), ['shell'], 'nothing installed => only the shell')
  ok('installedClis: degrades to shell-only when no binaries are found')
}

console.log(`\n${passed} passed`)
