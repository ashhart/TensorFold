import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { auxiliaryToolContext, resolveRunInstructionGraph } from './loop'

const root = await mkdtemp(path.join(os.tmpdir(), 'vontra-ssh-instructions-'))
try {
  await writeFile(path.join(root, 'AGENTS.md'), 'LOCAL LOOKALIKE INJECTION')
  const local = await resolveRunInstructionGraph(root, null, undefined)
  assert.match(local?.text ?? '', /LOCAL LOOKALIKE INJECTION/, 'control proves the local resolver sees the file')
  const remote = await resolveRunInstructionGraph(root, null, { kind: 'ssh' } as any)
  assert.equal(remote, null, 'SSH run omits host-local instructions')
} finally {
  await rm(root, { recursive: true, force: true })
}

const signal = new AbortController().signal
const backend = { kind: 'ssh', rootCwd: '/srv/repo' } as any
const child = auxiliaryToolContext('/srv/repo', { backend }, signal)
assert.equal(child.backend, backend, 'child research tools retain the parent SSH backend')
assert.equal(child.cwd, '/srv/repo')
assert.equal(child.signal, signal)

const auxiliary = readFileSync(new URL('./run-loop-auxiliary.ts', import.meta.url), 'utf8')
const compaction = readFileSync(new URL('./run-loop-compaction.ts', import.meta.url), 'utf8')
for (const [source, start, end, label] of [
  [auxiliary, 'async function verifyGoal', 'const EXPLORE_TOOL_NAMES', 'goal verifier'],
  [auxiliary, 'async function runDelegateChild', 'async function runExplore', 'delegate child'],
  [auxiliary, 'async function runOracle', 'const DELEGATE_TASK_MAX_CONCURRENT', 'oracle'],
  [compaction, 'async function summarizeTranscript', 'export async function compactThread', 'compaction summarizer'],
] as const) {
  const from = source.indexOf(start)
  const to = source.indexOf(end, from)
  assert.ok(from >= 0 && to > from, `${label} implementation is present`)
  const region = source.slice(from, to)
  assert.match(region, /approvalMode:\s*'readonly'/, `${label} model call is explicitly read-only`)
}
const source = readFileSync(new URL('./run-loop-tool-turn.ts', import.meta.url), 'utf8')
assert.match(source, /runExplore\([^\n]+deps\.toolContext,\s*assertRequestAllowed\)/, 'explore receives parent tool context and its live request guard')
assert.match(source, /runDelegateChild\([^\n]+deps\.toolContext,\s*assertRequestAllowed\)/, 'delegate children receive parent tool context and its live request guard')
assert.match(source, /runOracle\([^\n]+deps\.toolContext,\s*assertRequestAllowed\)/, 'oracle receives parent tool context and its live request guard')

console.log('auxiliary model and SSH instruction boundaries are fail-closed')
