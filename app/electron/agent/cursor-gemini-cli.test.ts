// Run: jiti electron/agent/cursor-gemini-cli.test.ts
//
// Cursor Agent + Gemini CLI engines (VST-033). Neither binary is installed on this machine;
// shapes come from the published CLI references (cursor.com/docs/cli, gemini-cli docs) and
// the fakes emit exactly those documented shapes.
import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import {
  buildCursorArgs,
  cursorSessionIdFromEvent,
  cursorTextFromAssistant,
  cursorStatusFromEvent,
  cursorResultError,
  cursorFailureMessage,
  cursorSpawnEnv,
  streamCursorAgent,
  type CursorCliChunk,
} from './cursor-agent-cli'
import {
  geminiApprovalMode,
  buildGeminiArgs,
  geminiSpawnEnv,
  parseGeminiOutput,
  geminiFailureMessage,
  streamGeminiCli,
  type GeminiCliChunk,
} from './gemini-cli'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cursor+gemini engine tests:')

const cursorProvider = { id: 'cursor', name: 'Cursor', kind: 'cloud', baseUrl: '', apiKey: '', models: [], transport: 'cursor-cli' } as unknown as Provider
const geminiProvider = { id: 'gemini', name: 'Gemini', kind: 'cloud', baseUrl: '', apiKey: '', models: [], transport: 'gemini-cli' } as unknown as Provider

// ---- cursor pure helpers ----

{
  const fresh = buildCursorArgs({ provider: cursorProvider, model: 'gpt-5.2', messages: [] }, null)
  assert.deepEqual(fresh.slice(0, 4), ['-p', '--output-format', 'stream-json', '--stream-partial-output'])
  assert.ok(fresh.includes('--model') && fresh.includes('gpt-5.2'))
  assert.ok(!fresh.includes('--resume'))
  assert.ok(fresh.includes('--mode=ask'), 'an unspecified mode fails closed to Cursor read-only ask mode')
  assert.ok(buildCursorArgs({ provider: cursorProvider, model: 'cursor-default', messages: [], approvalMode: 'readonly' }, null).includes('--mode=ask'))
  assert.ok(buildCursorArgs({ provider: cursorProvider, model: 'cursor-default', messages: [], approvalMode: 'suggest' }, null).includes('--mode=ask'))
  assert.ok(!buildCursorArgs({ provider: cursorProvider, model: 'cursor-default', messages: [], approvalMode: 'auto' }, null).includes('--mode=ask'), 'only explicit Full Auto may use writable Cursor print mode')
  const resumed = buildCursorArgs({ provider: cursorProvider, model: 'cursor-default', messages: [] }, { sessionId: 'chat-7' })
  assert.ok(!resumed.includes('--model'), 'the cursor-default sentinel omits --model')
  const at = resumed.indexOf('--resume')
  assert.ok(at >= 0 && resumed[at + 1] === 'chat-7')
  const keyed = buildCursorArgs({ provider: { ...cursorProvider, apiKey: 'ck-1' } as Provider, model: 'cursor-default', messages: [] }, null)
  assert.ok(!keyed.includes('--api-key') && !keyed.includes('ck-1'), 'the key never enters process argv')
  const spawnEnv = cursorSpawnEnv(
    { ...cursorProvider, apiKey: 'ck-1', cliEnv: { EXTRA: 'kept' } } as Provider,
    { PATH: '/usr/bin:/bin', UNRELATED_API_KEY: 'sk-launch-env-sentinel' },
  )
  assert.equal(spawnEnv.CURSOR_API_KEY, 'ck-1')
  assert.equal(spawnEnv.EXTRA, 'kept', 'provider cliEnv passes through verbatim')
  assert.equal(spawnEnv.PATH, '/usr/bin:/bin', 'allowlisted basics pass through')
  assert.equal(spawnEnv.UNRELATED_API_KEY, undefined, 'ambient launch-env secrets never reach the child')
  ok('buildCursorArgs: print+stream-json+partials and model/resume; API key is environment-only')
}

{
  assert.equal(cursorSessionIdFromEvent({ type: 'system', subtype: 'init', session_id: 'ses-1' }), 'ses-1')
  assert.equal(cursorSessionIdFromEvent({ type: 'assistant', session_id: 'ses-1' }), 'ses-1', 'any event can carry the id')
  assert.equal(cursorSessionIdFromEvent({ type: 'assistant' }), null)
  ok('cursorSessionIdFromEvent: first non-empty session_id wins')
}

{
  const assistant = (text: string) => ({ type: 'assistant', message: { role: 'assistant', content: [{ type: 'text', text }] } })
  // Cumulative partials: each event extends the accumulated text → emit the suffix.
  assert.equal(cursorTextFromAssistant(assistant('Hello'), ''), 'Hello')
  assert.equal(cursorTextFromAssistant(assistant('Hello world'), 'Hello'), ' world')
  // Exact repeat (final full-text event / result echo) → nothing.
  assert.equal(cursorTextFromAssistant(assistant('Hello world'), 'Hello world'), '')
  // Incremental partials: new text does NOT extend the accumulator → emit whole.
  assert.equal(cursorTextFromAssistant(assistant(' world'), 'Hello'), ' world')
  // A stale re-send of an earlier fragment → nothing (already contained).
  assert.equal(cursorTextFromAssistant(assistant('Hello'), 'Hello world'), '')
  ok('cursorTextFromAssistant: cumulative AND incremental partials dedup correctly')
}

{
  assert.equal(cursorStatusFromEvent({ type: 'tool_call', subtype: 'started', tool_call: { readToolCall: { args: { path: '/secret' } } } }), 'Cursor using read', 'tool NAME only — args never surface')
  assert.equal(cursorStatusFromEvent({ type: 'system', subtype: 'init' }), 'Cursor session ready')
  assert.equal(cursorStatusFromEvent({ type: 'assistant' }), '')
  ok('cursorStatusFromEvent: name-only status lines')
}

{
  assert.equal(cursorResultError({ type: 'result', is_error: true, result: 'auth expired' }), 'auth expired')
  assert.equal(cursorResultError({ type: 'result', is_error: false, result: 'fine' }), null)
  assert.match(cursorFailureMessage({ resultError: 'auth expired', exitCode: 0, stderr: '' }) || '', /auth expired/, 'a result error fails even on exit 0')
  assert.equal(cursorFailureMessage({ resultError: null, exitCode: 0, stderr: '' }), null)
  ok('cursor result/failure classification')
}

// ---- cursor fake-binary integration ----

function fakeBin(name: string, scriptBody: string): string {
  const dir = mkdtempSync(path.join(tmpdir(), `vontra-${name}-`))
  const bin = path.join(dir, name)
  writeFileSync(bin, `#!/bin/sh\n${scriptBody}`)
  chmodSync(bin, 0o755)
  return bin
}

async function waitFor(check: () => boolean, timeoutMs = 4_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!check()) {
    if (Date.now() >= deadline) throw new Error('condition did not become true before timeout')
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
}

function processExists(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch { return false }
}

{
  const bin = fakeBin('cursor-agent', `
echo "$@" > "$(dirname "$0")/args.txt"
printenv > "$(dirname "$0")/child-env.txt"
cat > "$(dirname "$0")/prompt.txt"
echo '{"type":"system","subtype":"init","session_id":"cur-ses-1","model":"m","cwd":"/x"}'
echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hel"}]},"session_id":"cur-ses-1"}'
echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello there"}]},"session_id":"cur-ses-1"}'
echo '{"type":"result","subtype":"success","is_error":false,"result":"Hello there","session_id":"cur-ses-1"}'
exit 0
`)
  const chunks: CursorCliChunk[] = []
  ;(async () => {})()
  process.env.VONTRA_TEST_UNRELATED_API_KEY = 'sk-launch-env-sentinel'
  const run = async () => {
    try {
      for await (const chunk of streamCursorAgent({ provider: { ...cursorProvider, cliPath: bin, cliEnv: { VONTRA_TEST_PROVIDER_TOKEN: 'provider-configured' } } as Provider, model: 'cursor-default', messages: [{ role: 'user', content: 'hi' }] as any })) chunks.push(chunk)
    } finally {
      delete process.env.VONTRA_TEST_UNRELATED_API_KEY
    }
  }
  await run()
  const text = chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join('')
  assert.equal(text, 'Hello there', 'cumulative partials + final result dedup to the message once')
  assert.equal((chunks.find((c) => c.type === 'resume_cursor') as any)?.cursor.sessionId, 'cur-ses-1')
  const args = readFileSync(path.join(path.dirname(bin), 'args.txt'), 'utf8')
  assert.doesNotMatch(args, /hi/, 'the prompt never enters process argv')
  assert.equal(readFileSync(path.join(path.dirname(bin), 'prompt.txt'), 'utf8'), 'hi', 'the prompt is handed off over stdin')
  const childEnv = readFileSync(path.join(path.dirname(bin), 'child-env.txt'), 'utf8')
  assert.doesNotMatch(childEnv, /VONTRA_TEST_UNRELATED_API_KEY|sk-launch-env-sentinel/, 'an ambient launch-env secret never reaches the CLI child')
  assert.match(childEnv, /^VONTRA_TEST_PROVIDER_TOKEN=provider-configured$/m, 'provider cliEnv passes through verbatim')
  assert.match(childEnv, /^PATH=/m, 'allowlisted basics (PATH) still reach the child')
  ok('streamCursorAgent: streams, dedups the result echo, captures the cursor')
}

{
  const bin = fakeBin('cursor-agent', `
echo '{"type":"result","subtype":"error","is_error":true,"result":"Not logged in","session_id":"s"}'
exit 0
`)
  await assert.rejects(
    (async () => { for await (const _ of streamCursorAgent({ provider: { ...cursorProvider, cliPath: bin } as Provider, model: 'cursor-default', messages: [{ role: 'user', content: 'hi' }] as any })) { /* drain */ } })(),
    /Not logged in/,
    'an is_error result fails the run even on exit 0',
  )
  ok('streamCursorAgent: error results propagate')
}

// ---- gemini pure helpers ----

{
  assert.equal(geminiApprovalMode('readonly'), 'plan')
  assert.equal(geminiApprovalMode('auto'), 'auto_edit')
  assert.equal(geminiApprovalMode('suggest'), 'default')
  assert.equal(geminiApprovalMode(undefined), 'default')
  assert.notEqual(geminiApprovalMode('auto'), 'yolo')
  ok('geminiApprovalMode: readonly→plan, auto→guarded auto_edit, never yolo')
}

{
  const args = buildGeminiArgs({ provider: geminiProvider, model: 'gemini-2.5-pro', messages: [] })
  assert.equal(args.includes('-p'), false, 'the flattened prompt never enters argv')
  assert.equal(args.includes('do the thing'), false)
  assert.ok(args.includes('--output-format') && args.includes('json'))
  assert.ok(args.includes('-m') && args.includes('gemini-2.5-pro'))
  assert.ok(!buildGeminiArgs({ provider: geminiProvider, model: 'gemini-default', messages: [] }).includes('-m'), 'sentinel omits -m')
  const env = geminiSpawnEnv(
    { ...geminiProvider, apiKey: 'gemini-secret', cliEnv: { EXTRA: 'kept', GEMINI_API_KEY: 'stale' } } as Provider,
    { PATH: '/usr/bin:/bin', UNRELATED_API_KEY: 'sk-launch-env-sentinel' },
  )
  assert.equal(env.GEMINI_API_KEY, 'gemini-secret', 'TensorFold Studio raw key uses Gemini CLI\'s documented env handoff')
  assert.equal(env.EXTRA, 'kept')
  assert.equal(env.PATH, '/usr/bin:/bin', 'allowlisted basics pass through')
  assert.equal(env.UNRELATED_API_KEY, undefined, 'ambient launch-env secrets never reach the child')
  assert.equal(args.includes('gemini-secret'), false, 'API key never enters argv')
  ok('buildGeminiArgs/geminiSpawnEnv: stdin prompt, json mode, model, environment-only key')
}

{
  assert.deepEqual(parseGeminiOutput('{"response":"Hi!","stats":{"tokens":5}}'), { response: 'Hi!', errorMessage: null })
  assert.deepEqual(parseGeminiOutput('Loaded config\n{"response":"","error":{"message":"quota exceeded","code":429}}'), { response: '', errorMessage: 'quota exceeded' }, 'stray preamble lines are skipped; error.message wins')
  assert.equal(parseGeminiOutput('{"response":"x","error":{"weird":true}}').errorMessage?.includes('weird'), true, 'unknown error shapes degrade to a bounded JSON slice')
  assert.deepEqual(parseGeminiOutput('not json at all'), { response: '', errorMessage: null })
  ok('parseGeminiOutput: documented shape + defensive error/preamble handling')
}

{
  assert.equal(geminiFailureMessage({ exitCode: 0, stderr: '', errorMessage: null }), null)
  assert.match(geminiFailureMessage({ exitCode: 0, stderr: '', errorMessage: 'quota exceeded' }) || '', /quota exceeded/, 'an in-band error fails even on exit 0')
  assert.match(geminiFailureMessage({ exitCode: 53, stderr: '', errorMessage: null }) || '', /turn limit/, 'documented exit 53 is labeled')
  assert.match(geminiFailureMessage({ exitCode: 42, stderr: 'bad flag', errorMessage: null }) || '', /input error.*bad flag/s)
  ok('geminiFailureMessage: in-band errors + documented exit codes')
}

// ---- gemini fake-binary integration ----

{
  const bin = fakeBin('gemini', `
cat > "$(dirname "$0")/stdin.txt"
echo "$@" > "$(dirname "$0")/args.txt"
printf '%s' "$GEMINI_API_KEY" > "$(dirname "$0")/key.txt"
printenv > "$(dirname "$0")/child-env.txt"
echo '{"response":"The answer is 4.","stats":{"totalTokens":12}}'
exit 0
`)
  const chunks: GeminiCliChunk[] = []
  const rawKey = 'gemini-key-not-in-argv-or-stdin'
  process.env.VONTRA_TEST_UNRELATED_API_KEY = 'sk-launch-env-sentinel'
  try {
    for await (const chunk of streamGeminiCli({ provider: { ...geminiProvider, cliPath: bin, apiKey: rawKey } as Provider, model: 'gemini-default', messages: [{ role: 'user', content: 'what is 2+2' }] as any, approvalMode: 'readonly' })) chunks.push(chunk)
  } finally {
    delete process.env.VONTRA_TEST_UNRELATED_API_KEY
  }
  assert.equal(chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join(''), 'The answer is 4.')
  const args = require('node:fs').readFileSync(path.join(path.dirname(bin), 'args.txt'), 'utf8')
  const stdin = require('node:fs').readFileSync(path.join(path.dirname(bin), 'stdin.txt'), 'utf8')
  assert.ok(args.includes('--approval-mode plan'), 'readonly ran in plan mode')
  assert.equal(args.includes('what is 2+2'), false, 'the flattened prompt is absent from argv')
  assert.equal(args.includes(rawKey), false, 'the API key is absent from argv')
  assert.ok(stdin.includes('what is 2+2'), 'the flattened prompt rides only on stdin')
  assert.equal(stdin.includes(rawKey), false, 'the API key is absent from stdin')
  assert.equal(require('node:fs').readFileSync(path.join(path.dirname(bin), 'key.txt'), 'utf8'), rawKey, 'the raw key arrives only via GEMINI_API_KEY')
  const childEnv = require('node:fs').readFileSync(path.join(path.dirname(bin), 'child-env.txt'), 'utf8')
  assert.doesNotMatch(childEnv, /VONTRA_TEST_UNRELATED_API_KEY|sk-launch-env-sentinel/, 'an ambient launch-env secret never reaches the CLI child')
  assert.match(childEnv, /^PATH=/m, 'allowlisted basics (PATH) still reach the child')
  ok('streamGeminiCli: stdin-only prompt and environment-only key yield the JSON response')
}

{
  const bin = fakeBin('gemini', `
echo '{"response":"","error":{"message":"API key invalid"}}'
exit 1
`)
  await assert.rejects(
    (async () => { for await (const _ of streamGeminiCli({ provider: { ...geminiProvider, cliPath: bin } as Provider, model: 'gemini-default', messages: [{ role: 'user', content: 'hi' }] as any })) { /* drain */ } })(),
    /API key invalid/,
  )
  ok('streamGeminiCli: in-band errors surface with their message')
}

{
  const bin = fakeBin('gemini', `
trap '' TERM
echo $$ > "$(dirname "$0")/pid.txt"
echo ready > "$(dirname "$0")/ready.txt"
while :; do sleep 1; done
`)
  const dir = path.dirname(bin)
  const gen = streamGeminiCli({
    provider: { ...geminiProvider, cliPath: bin } as Provider,
    model: 'gemini-default',
    messages: [{ role: 'user', content: 'hi' }] as any,
  })
  assert.deepEqual(await gen.next(), { done: false, value: { type: 'reasoning', text: 'Gemini working…\n' } })
  await waitFor(() => existsSync(path.join(dir, 'ready.txt')))
  const pid = Number(readFileSync(path.join(dir, 'pid.txt'), 'utf8'))
  try {
    await gen.return(undefined as any)
    await waitFor(() => !processExists(pid))
    assert.equal(processExists(pid), false, 'the cancelled SIGTERM-ignoring Gemini group is reaped')
  } finally {
    if (processExists(pid)) {
      try { process.kill(-pid, 'SIGKILL') } catch { try { process.kill(pid, 'SIGKILL') } catch { /* gone */ } }
    }
  }
  ok('streamGeminiCli: consumer return reaps a SIGTERM-ignoring process group')
}

console.log(`\n${passed} passed`)
