// Run: jiti electron/agent/claude-cli.resume.test.ts
//
// Claude Code session resume + live steering (VST-033 claude parity with codex-app-server):
// cursor capture from stream-json events, --resume arg building, latest-user-only prompt on
// resume, the safe fresh-run fallback when the persisted session is gone, and mid-turn
// steering over the open stream-json stdin under BOTH possible CLI semantics (steer merged
// into the current turn vs queued as a follow-up turn — the docs don't pin which).
import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createSteeringChannel } from './steering'
import {
  buildClaudeArgs,
  claudeSessionIdFromEvent,
  claudeUserMessageLine,
  cleanupStaleClaudeSystemPrompts,
  createClaudeSystemPromptFile,
  isRecoverableClaudeResumeError,
  latestUserBody,
  shouldRetryFreshAfterResumeFailure,
  streamClaudeCli,
  type ClaudeCliChunk,
  type ClaudeCliParams,
} from './claude-cli'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('claude-cli resume tests:')

const provider = { id: 'claude', name: 'Claude CLI', kind: 'cloud', baseUrl: '', apiKey: '', models: [], transport: 'claude-cli' } as unknown as Provider

// --- pure helpers ---

{
  assert.equal(claudeSessionIdFromEvent({ type: 'system', subtype: 'init', session_id: 'abc-123' }), 'abc-123')
  assert.equal(claudeSessionIdFromEvent({ type: 'result', is_error: false, result: 'done', session_id: 'abc-123' }), 'abc-123')
  assert.equal(claudeSessionIdFromEvent({ type: 'assistant', message: {} }), null)
  assert.equal(claudeSessionIdFromEvent({ session_id: '   ' }), null)
  ok('claudeSessionIdFromEvent: captures ids from init/result, ignores blanks')
}

{
  assert.equal(isRecoverableClaudeResumeError('No conversation found with session ID: abc'), true)
  assert.equal(isRecoverableClaudeResumeError('Session abc-123 not found'), true)
  assert.equal(isRecoverableClaudeResumeError('failed to resume conversation'), true)
  assert.equal(isRecoverableClaudeResumeError('Claude CLI is not authenticated. Run `claude login`.'), false)
  assert.equal(isRecoverableClaudeResumeError('Tool execution failed: ENOENT'), false)
  ok('isRecoverableClaudeResumeError: dead-session messages only')
}

{
  assert.equal(shouldRetryFreshAfterResumeFailure({ resumed: true, streamedContent: false, message: 'No conversation found with session ID: x' }), true)
  assert.equal(shouldRetryFreshAfterResumeFailure({ resumed: true, streamedContent: true, message: 'No conversation found with session ID: x' }), false, 'content already streamed → never retry (duplication risk)')
  assert.equal(shouldRetryFreshAfterResumeFailure({ resumed: false, streamedContent: false, message: 'No conversation found' }), false, 'fresh runs never retry')
  assert.equal(shouldRetryFreshAfterResumeFailure({ resumed: true, streamedContent: false, message: '401 unauthorized' }), false, 'auth failures propagate')
  ok('shouldRetryFreshAfterResumeFailure: retry only resumed+silent+dead-session')
}

{
  const messages = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'first ask' },
    { role: 'assistant', content: 'answer' },
    { role: 'user', content: 'newest ask' },
  ] as any
  assert.equal(latestUserBody(messages), 'newest ask')
  assert.equal(latestUserBody([{ role: 'assistant', content: 'x' }] as any), '')
  ok('latestUserBody: newest user message only')
}

{
  const base: ClaudeCliParams = { provider, model: 'claude-opus', messages: [] }
  const fresh = buildClaudeArgs(base, '/private/system-prompt.txt', null)
  assert.ok(fresh.includes('--append-system-prompt-file'), 'fresh run carries only a system-prompt file reference')
  assert.ok(!fresh.includes('SYSTEM PROMPT'), 'system-prompt content never enters argv')
  assert.ok(!fresh.includes('--resume'), 'fresh run has no --resume')
  const inputAt = fresh.indexOf('--input-format')
  assert.ok(inputAt >= 0 && fresh[inputAt + 1] === 'stream-json', 'stdin speaks stream-json (the live-steer seam)')
  const resumed = buildClaudeArgs(base, '/private/system-prompt.txt', { sessionId: 'sess-9' })
  const resumeIndex = resumed.indexOf('--resume')
  assert.ok(resumeIndex >= 0 && resumed[resumeIndex + 1] === 'sess-9', 'resume run passes --resume <sessionId>')
  assert.ok(!resumed.includes('--append-system-prompt-file'), 'resume run does NOT re-append the system prompt')
  ok('buildClaudeArgs: fresh system prompt is file-backed; resume replaces it; stream-json input')
}

{
  const root = mkdtempSync(path.join(tmpdir(), 'vontra-claude-prompt-test-'))
  const dead = path.join(root, 'vontra-claude-prompt-111-dead')
  const live = path.join(root, 'vontra-claude-prompt-222-live')
  mkdirSync(dead, { mode: 0o700 })
  mkdirSync(live, { mode: 0o700 })
  await cleanupStaleClaudeSystemPrompts(root, { currentPid: 222, isProcessAlive: (pid) => pid === 222 })
  assert.equal(existsSync(dead), false, 'a dead process prompt directory is removed on the next run')
  assert.equal(existsSync(live), true, 'a live process prompt directory is never removed')

  const handoff = await createClaudeSystemPromptFile('private system instructions', root, 222)
  assert.match(handoff.path, /vontra-claude-prompt-222-/, 'the private directory is PID-stamped')
  assert.equal(statSync(handoff.path).mode & 0o777, 0o600, 'the temporary prompt file is owner read/write only')
  assert.equal(readFileSync(handoff.path, 'utf8'), 'private system instructions')
  await handoff.cleanup()
  assert.equal(existsSync(handoff.path), false, 'cleanup removes the prompt file and its private directory')
  await handoff.cleanup()
  rmSync(root, { recursive: true, force: true })
  ok('createClaudeSystemPromptFile: PID-stamped 0600 handoff, stale cleanup, idempotent cleanup')
}

{
  const line = claudeUserMessageLine('steer this way')
  assert.ok(line.endsWith('\n'), 'one JSONL record per message')
  const parsed = JSON.parse(line)
  assert.equal(parsed.type, 'user')
  assert.equal(parsed.message.role, 'user')
  assert.deepEqual(parsed.message.content, [{ type: 'text', text: 'steer this way' }])
  ok('claudeUserMessageLine: stream-json user message shape')
}

// --- fake-binary integration: resume-dead-session → fresh fallback, cursor captured ---

{
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-claude-system-file-'))
  const fakeBin = path.join(dir, 'claude')
  const systemSecret = 'SYSTEM_SECRET_not_for_argv_env_or_stdin'
  writeFileSync(fakeBin, `#!/usr/bin/env node
const fs = require('node:fs')
const path = require('node:path')
const dir = __dirname
const args = process.argv.slice(2)
fs.writeFileSync(path.join(dir, 'args.json'), JSON.stringify(args))
fs.writeFileSync(path.join(dir, 'env-leak.txt'), Object.values(process.env).includes(${JSON.stringify(systemSecret)}) ? 'yes' : 'no')
const at = args.indexOf('--append-system-prompt-file')
if (at < 0 || !args[at + 1]) process.exit(9)
const promptPath = args[at + 1]
fs.writeFileSync(path.join(dir, 'prompt-path.txt'), promptPath)
fs.writeFileSync(path.join(dir, 'prompt-mode.txt'), String(fs.statSync(promptPath).mode & 0o777))
fs.writeFileSync(path.join(dir, 'prompt-content.txt'), fs.readFileSync(promptPath, 'utf8'))
let stdin = ''
process.stdin.on('data', (chunk) => { stdin += String(chunk) })
process.stdin.on('end', () => {
  fs.writeFileSync(path.join(dir, 'stdin.txt'), stdin)
  process.stdout.write(JSON.stringify({ type: 'system', subtype: 'init', session_id: 'file-1' }) + '\\n')
  process.stdout.write(JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'safe reply' } } }) + '\\n')
  process.stdout.write(JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'safe reply', session_id: 'file-1' }) + '\\n')
})
`)
  chmodSync(fakeBin, 0o755)

  const chunks: ClaudeCliChunk[] = []
  for await (const chunk of streamClaudeCli({
    provider: { ...provider, cliPath: fakeBin } as Provider,
    model: 'claude-code-default',
    messages: [{ role: 'system', content: systemSecret }, { role: 'user', content: 'ordinary user body' }] as any,
  })) chunks.push(chunk)

  const args = JSON.parse(readFileSync(path.join(dir, 'args.json'), 'utf8')) as string[]
  const promptPath = readFileSync(path.join(dir, 'prompt-path.txt'), 'utf8')
  const stdin = readFileSync(path.join(dir, 'stdin.txt'), 'utf8')
  assert.equal(args.some((arg) => arg.includes(systemSecret)), false, 'system instructions are absent from argv')
  assert.equal(readFileSync(path.join(dir, 'env-leak.txt'), 'utf8'), 'no', 'system instructions are absent from env')
  assert.equal(stdin.includes(systemSecret), false, 'system instructions are absent from user-message stdin')
  assert.equal(stdin.includes('ordinary user body'), true, 'the user message still uses stream-json stdin')
  assert.equal(readFileSync(path.join(dir, 'prompt-content.txt'), 'utf8'), systemSecret)
  assert.equal(Number(readFileSync(path.join(dir, 'prompt-mode.txt'), 'utf8')), 0o600)
  assert.equal(existsSync(promptPath), false, 'the file-backed handoff is removed after the child settles')
  assert.equal(chunks.filter((chunk) => chunk.type === 'content').map((chunk: any) => chunk.text).join(''), 'safe reply')
  ok('streamClaudeCli: system prompt is 0600 file-only and cleaned after completion')
}

{
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-claude-resume-'))
  const fakeBin = path.join(dir, 'claude')
  // A fake `claude` that errors on --resume DEAD (dead-session shape) and otherwise streams
  // an init event (session id), one text delta, and a clean result.
  writeFileSync(fakeBin, `#!/bin/sh
cat > /dev/null
case "$*" in
  *"--resume DEAD"*)
    echo '{"type":"result","subtype":"success","is_error":true,"result":"No conversation found with session ID: DEAD"}'
    exit 1
    ;;
  *)
    echo '{"type":"system","subtype":"init","session_id":"fresh-42"}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello from fresh"}}}'
    echo '{"type":"result","subtype":"success","is_error":false,"result":"hello from fresh","session_id":"fresh-42"}'
    exit 0
    ;;
esac
`)
  chmodSync(fakeBin, 0o755)

  const params: ClaudeCliParams = {
    provider: { ...provider, cliPath: fakeBin } as Provider,
    model: 'claude-code-default',
    messages: [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'continue please' },
    ] as any,
    resumeCursor: { sessionId: 'DEAD' },
  }
  const chunks: ClaudeCliChunk[] = []
  for await (const chunk of streamClaudeCli(params)) chunks.push(chunk)

  const content = chunks.filter((chunk) => chunk.type === 'content').map((chunk: any) => chunk.text).join('')
  assert.equal(content, 'hello from fresh', 'dead-session resume fell back to a fresh run')
  const cursor = chunks.find((chunk) => chunk.type === 'resume_cursor') as any
  assert.ok(cursor, 'a resume cursor chunk is emitted at the end of a successful run')
  assert.equal(cursor.cursor.sessionId, 'fresh-42', 'the NEW session id is captured for next time')
  ok('streamClaudeCli: dead resume falls back fresh once and captures the new cursor')
}

// A healthy resume keeps the same session and still re-emits the cursor.
{
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-claude-resume-ok-'))
  const fakeBin = path.join(dir, 'claude')
  writeFileSync(fakeBin, `#!/bin/sh
cat > /dev/null
case "$*" in
  *"--resume LIVE"*)
    echo '{"type":"system","subtype":"init","session_id":"LIVE"}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"resumed reply"}}}'
    echo '{"type":"result","subtype":"success","is_error":false,"result":"resumed reply","session_id":"LIVE"}'
    exit 0
    ;;
  *)
    echo '{"type":"result","subtype":"success","is_error":true,"result":"should have resumed"}'
    exit 1
    ;;
esac
`)
  chmodSync(fakeBin, 0o755)

  const chunks: ClaudeCliChunk[] = []
  for await (const chunk of streamClaudeCli({
    provider: { ...provider, cliPath: fakeBin } as Provider,
    model: 'claude-code-default',
    messages: [{ role: 'user', content: 'next step' }] as any,
    resumeCursor: { sessionId: 'LIVE' },
  })) chunks.push(chunk)

  const content = chunks.filter((chunk) => chunk.type === 'content').map((chunk: any) => chunk.text).join('')
  assert.equal(content, 'resumed reply')
  const cursor = chunks.find((chunk) => chunk.type === 'resume_cursor') as any
  assert.equal(cursor?.cursor.sessionId, 'LIVE', 'healthy resume keeps the same session id')
  ok('streamClaudeCli: healthy resume streams on the same session')
}

// --- live steering over the open stream-json stdin ---
//
// The CLI's steering semantics aren't documented, so BOTH are scripted: a steer merged into
// the current turn (one result; the grace window closes stdin) and a steer queued as a
// follow-up turn (a second turn's events cancel the grace; close after its result).

function fakeSteerClaude(handleBody: string): Provider {
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-claude-steer-'))
  const bin = path.join(dir, 'claude')
  writeFileSync(bin, `#!/usr/bin/env node
const fs = require('node:fs')
fs.writeFileSync(__dirname + '/args.txt', process.argv.slice(2).join(' '))
const emit = (o) => process.stdout.write(JSON.stringify(o) + '\\n')
const delta = (text) => emit({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text } } })
let users = 0
let buf = ''
process.stdin.on('data', (d) => {
  buf += String(d)
  let i
  while ((i = buf.indexOf('\\n')) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1)
    if (!line.trim()) continue
    fs.appendFileSync(__dirname + '/stdin.txt', line + '\\n')
    const msg = JSON.parse(line)
    if (msg.type === 'user') { users++; handle(users) }
  }
})
process.stdin.on('end', () => process.exit(0))
function handle(nth) {
${handleBody}
}
`)
  chmodSync(bin, 0o755)
  return { ...provider, cliPath: bin } as Provider
}

async function runWithSteer(fake: Provider, steerText: string): Promise<{ content: string; accepted: boolean; stdinLines: any[] }> {
  const channel = createSteeringChannel()
  let offer: Promise<boolean> | null = null
  const chunks: ClaudeCliChunk[] = []
  for await (const chunk of streamClaudeCli({
    provider: fake,
    model: 'claude-code-default',
    messages: [{ role: 'user', content: 'start work' }] as any,
    steering: channel,
    steerGraceMs: 80,
  })) {
    chunks.push(chunk)
    if (!offer && chunk.type === 'content') {
      assert.equal(channel.canAccept, true, 'live turn is steerable')
      offer = channel.offer({ id: 'cue-1', text: steerText })
    }
  }
  const stdinRaw = readFileSync(path.join(path.dirname((fake as any).cliPath), 'stdin.txt'), 'utf8')
  return {
    content: chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join(''),
    accepted: await offer!,
    stdinLines: stdinRaw.trim().split('\n').map((l) => JSON.parse(l)),
  }
}

{
  // MERGED semantics: the steer folds into the current turn — one result, then silence.
  // The grace window must close stdin so the run ends instead of hanging on the watchdog.
  const fake = fakeSteerClaude(`
  if (nth === 1) {
    emit({ type: 'system', subtype: 'init', session_id: 'steer-1' })
    delta('working...')
    // Turn stays open: the single result comes only after the steer arrives.
  }
  if (nth === 2) {
    delta(' merged-steer')
    emit({ type: 'result', subtype: 'success', is_error: false, result: 'working... merged-steer', session_id: 'steer-1' })
  }
`)
  const t0 = Date.now()
  const run = await runWithSteer(fake, 'change course')
  assert.equal(run.accepted, true, 'the steer write is accepted')
  assert.equal(run.content, 'working... merged-steer')
  assert.equal(run.stdinLines.length, 2, 'initial message + steer message on stdin')
  assert.equal(run.stdinLines[1].message.content[0].text, 'change course')
  assert.ok(Date.now() - t0 < 10_000, 'the grace window closed the run promptly (no watchdog hang)')
  ok('streamClaudeCli: merged-semantics steer — grace window ends the run cleanly')
}

{
  // QUEUED semantics: the steer becomes a follow-up turn — its events cancel the grace
  // and both turns' content streams before stdin closes at the second result.
  const fake = fakeSteerClaude(`
  if (nth === 1) {
    emit({ type: 'system', subtype: 'init', session_id: 'steer-2' })
    delta('turn one')
    // result #1 is emitted only once the steer has arrived, so the race is deterministic.
  }
  if (nth === 2) {
    emit({ type: 'result', subtype: 'success', is_error: false, result: 'turn one', num_turns: 1, session_id: 'steer-2' })
    emit({ type: 'system', subtype: 'status', session_id: 'steer-2' })
    delta(' + turn two from steer')
    emit({ type: 'result', subtype: 'success', is_error: false, result: 'turn two', num_turns: 2, session_id: 'steer-2' })
  }
`)
  const run = await runWithSteer(fake, 'follow up now')
  assert.equal(run.accepted, true)
  assert.equal(run.content, 'turn one + turn two from steer', 'both turns stream')
  assert.equal(run.stdinLines.length, 2)
  ok('streamClaudeCli: queued-semantics steer — follow-up turn streams before close')
}

{
  // No steer sent: the first result closes stdin immediately (no grace hang).
  const fake = fakeSteerClaude(`
  if (nth === 1) {
    emit({ type: 'system', subtype: 'init', session_id: 'plain-1' })
    delta('plain answer')
    emit({ type: 'result', subtype: 'success', is_error: false, result: 'plain answer', session_id: 'plain-1' })
  }
`)
  const channel = createSteeringChannel()
  const t0 = Date.now()
  const chunks: ClaudeCliChunk[] = []
  for await (const chunk of streamClaudeCli({
    provider: fake,
    model: 'claude-code-default',
    messages: [{ role: 'user', content: 'quick one' }] as any,
    steering: channel,
    steerGraceMs: 5_000,
  })) chunks.push(chunk)
  assert.equal(chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join(''), 'plain answer')
  assert.ok(Date.now() - t0 < 4_000, 'no pending steer → no grace wait at all')
  ok('streamClaudeCli: steerable run without a steer closes at the first result')
}

{
  // Child env is an allowlist, not the launch environment: an ambient secret must not be
  // printable by the CLI (a read-only `printenv` is legitimate in a room seat), while
  // allowlisted basics and the provider's own cliEnv still arrive.
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-claude-env-'))
  const fakeBin = path.join(dir, 'claude')
  writeFileSync(fakeBin, `#!/bin/sh
printenv > "$(dirname "$0")/child-env.txt"
cat > /dev/null
echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"env ok"}}}'
echo '{"type":"result","subtype":"success","is_error":false,"result":"env ok","session_id":"env-1"}'
exit 0
`)
  chmodSync(fakeBin, 0o755)
  process.env.VONTRA_TEST_UNRELATED_API_KEY = 'sk-launch-env-sentinel'
  try {
    const chunks: ClaudeCliChunk[] = []
    for await (const chunk of streamClaudeCli({
      provider: { ...provider, cliPath: fakeBin, cliEnv: { VONTRA_TEST_PROVIDER_TOKEN: 'provider-configured' } } as Provider,
      model: 'claude-code-default',
      messages: [{ role: 'user', content: 'check env' }] as any,
    })) chunks.push(chunk)
    assert.equal(chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join(''), 'env ok')
  } finally {
    delete process.env.VONTRA_TEST_UNRELATED_API_KEY
  }
  const childEnv = readFileSync(path.join(dir, 'child-env.txt'), 'utf8')
  assert.doesNotMatch(childEnv, /VONTRA_TEST_UNRELATED_API_KEY|sk-launch-env-sentinel/, 'an ambient launch-env secret never reaches the CLI child')
  assert.match(childEnv, /^VONTRA_TEST_PROVIDER_TOKEN=provider-configured$/m, 'provider cliEnv passes through verbatim')
  assert.match(childEnv, /^PATH=/m, 'allowlisted basics (PATH) still reach the child')
  ok('streamClaudeCli: child env is allowlist + provider cliEnv, never the ambient secret')
}

console.log(`${passed} passed`)
