// Run: jiti electron/agent/pi-cli.test.ts
//
// Pi engine (VST-033), rpc mode. Protocol facts from installed pi 0.80.3's shipped
// docs/rpc.md + a live rpc probe on this machine: JSONL commands over stdin
// (prompt/steer), id-correlated response lines, message_update deltas, agent_end
// completion, stdin EOF → clean exit, strict LF-only framing (readline is explicitly
// non-compliant because it also splits U+2028/U+2029).
import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createSteeringChannel } from './steering'
import {
  assertPiCredentialHandoff,
  buildPiArgs,
  cleanupStalePiSystemPrompts,
  chunkFromPiEvent,
  createPiSystemPromptFile,
  latestPiUserBody,
  piFailureMessage,
  piThinkingLevel,
  piToolsForApproval,
  splitJsonlChunk,
  streamPiCli,
  type PiCliChunk,
  type PiCliParams,
} from './pi-cli'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('pi-cli tests:')

const provider = { id: 'pi', name: 'Pi', kind: 'cloud', baseUrl: '', apiKey: '', models: [], transport: 'pi-cli' } as unknown as Provider

{
  const root = mkdtempSync(path.join(tmpdir(), 'vontra-pi-prompt-test-'))
  const dead = path.join(root, 'vontra-pi-system-111-dead')
  const live = path.join(root, 'vontra-pi-system-222-live')
  mkdirSync(dead); mkdirSync(live)
  await cleanupStalePiSystemPrompts(root, { currentPid: 222, isProcessAlive: (pid) => pid === 222 })
  assert.equal(existsSync(dead), false, 'a dead process prompt is removed on the next creation/start')
  assert.equal(existsSync(live), true, 'a live process prompt is never removed')
  const prompt = await createPiSystemPromptFile('private system', root, 222)
  assert.equal(readFileSync(prompt.path, 'utf8'), 'private system')
  assert.equal(statSync(prompt.path).mode & 0o777, 0o600)
  await prompt.cleanup(); await prompt.cleanup()
  assert.equal(existsSync(prompt.path), false, 'cleanup is idempotent')
  rmSync(root, { recursive: true, force: true })
  ok('private Pi system prompt handoff removes dead-process leftovers and cleans owner-only files')
}

// --- pure helpers ---

{
  assert.equal(piThinkingLevel('low'), 'low')
  assert.equal(piThinkingLevel('medium'), 'medium')
  assert.equal(piThinkingLevel(undefined), 'medium')
  assert.equal(piThinkingLevel('xhigh'), 'xhigh')
  assert.equal(piThinkingLevel('max'), 'xhigh', "'max' clamps to pi's xhigh ceiling")
  assert.equal(piThinkingLevel('off'), 'off')
  assert.equal(piThinkingLevel('minimal'), 'minimal')
  ok('piThinkingLevel: verified pi levels, max clamps to xhigh')
}

{
  assert.deepEqual(piToolsForApproval('readonly'), ['read'], 'read-only mode restricts pi to its read tool')
  assert.equal(piToolsForApproval('suggest'), null)
  assert.equal(piToolsForApproval('auto'), null)
  ok('piToolsForApproval: readonly → read-only allowlist, else pi defaults')
}

{
  const base: PiCliParams = { provider, model: 'anthropic/claude-opus', messages: [] }
const fresh = buildPiArgs(base, 'SYSTEM', 'minted-uuid', false)
  assert.deepEqual(fresh.slice(0, 4), ['--mode', 'rpc', '--thinking', 'medium'])
  assert.ok(!fresh.includes('-p'), 'rpc mode replaces print mode')
  assert.ok(fresh.includes('--model') && fresh.includes('anthropic/claude-opus'))
  assert.ok(fresh.includes('--append-system-prompt'), 'fresh runs carry the system prompt')
  const mintedAt = fresh.indexOf('--session-id')
  assert.ok(mintedAt >= 0 && fresh[mintedAt + 1] === 'minted-uuid', 'fresh runs pass the MINTED session id (create-if-missing)')

  const resumed = buildPiArgs(base, 'SYSTEM', 'ses-9', true)
  const at = resumed.indexOf('--session-id')
  assert.ok(at >= 0 && resumed[at + 1] === 'ses-9', 'resume passes the cursor session id')
  assert.ok(!resumed.includes('--append-system-prompt'), 'resume does not re-append the system prompt')

  const sentinel = buildPiArgs({ ...base, model: 'pi-default' }, '', 'x', false)
  assert.ok(!sentinel.includes('--model'), 'the pi-default sentinel omits --model')

  const readonly = buildPiArgs({ ...base, approvalMode: 'readonly' }, '', 'x', false)
  const toolsAt = readonly.indexOf('--tools')
  assert.ok(toolsAt >= 0 && readonly[toolsAt + 1] === 'read')

  assert.throws(
    () => buildPiArgs({ ...base, provider: { ...provider, apiKey: 'sk-x' } as Provider }, '', 'x', false),
    /without exposing it in the process argument list/,
    'raw API-key mode fails closed instead of putting the secret in argv',
  )
  const envKeyedProvider = { ...provider, apiKey: 'sk-not-in-argv', cliEnv: { OPENAI_API_KEY: 'sk-env' } } as Provider
  assert.doesNotThrow(() => assertPiCredentialHandoff(envKeyedProvider))
  const keyed = buildPiArgs({ ...base, provider: envKeyedProvider }, '', 'x', false)
  assert.ok(!keyed.includes('--api-key') && !keyed.includes('sk-not-in-argv') && !keyed.includes('sk-env'))
  ok('buildPiArgs: rpc mode, session/system/tools; credentials use explicit environment auth only')
}

{
  // Strict LF framing: U+2028 inside a JSON string must NOT split the record (this is the
  // exact readline pitfall pi's rpc docs call out).
  const tricky = JSON.stringify({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'line separator' } })
  const one = splitJsonlChunk('', tricky + '\n')
  assert.equal(one.lines.length, 1)
  assert.equal(JSON.parse(one.lines[0]).assistantMessageEvent.delta, 'line separator')

  // Torn tail buffers; \r\n tolerated; blank lines dropped.
  const first = splitJsonlChunk('', '{"a":1}\r\n\n{"b":')
  assert.deepEqual(first.lines, ['{"a":1}'])
  assert.equal(first.rest, '{"b":')
  const second = splitJsonlChunk(first.rest, '2}\n')
  assert.deepEqual(second.lines, ['{"b":2}'])
  assert.equal(second.rest, '')
  ok('splitJsonlChunk: LF-only records, U+2028 survives, torn tails buffer')
}

{
  assert.deepEqual(chunkFromPiEvent({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'Hello' } }), { type: 'content', text: 'Hello' })
  assert.deepEqual(chunkFromPiEvent({ type: 'message_update', assistantMessageEvent: { type: 'thinking_delta', delta: 'hmm' } }), { type: 'reasoning', text: 'hmm' })
  assert.equal(chunkFromPiEvent({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: '' } }), null)
  const tool = chunkFromPiEvent({ type: 'tool_execution_start', toolCallId: 't1', toolName: 'bash', args: { command: 'rm -rf /' } })
  assert.deepEqual(tool, { type: 'reasoning', text: 'Pi using bash\n' }, 'tool status carries the NAME only — never args')
  assert.deepEqual(chunkFromPiEvent({ type: 'tool_execution_end', toolName: 'bash', isError: true, result: 'secret output' }), { type: 'reasoning', text: 'Pi tool bash failed\n' })
  assert.equal(chunkFromPiEvent({ type: 'tool_execution_end', toolName: 'bash', isError: false }), null, 'successful tool ends are silent')
  assert.equal(chunkFromPiEvent({ type: 'agent_end', messages: [] }), null)
  ok('chunkFromPiEvent: deltas, tool status lines (name only), noise dropped')
}

{
  assert.equal(piFailureMessage({ exitCode: 0, stderr: '' }), null)
  assert.match(piFailureMessage({ exitCode: 1, stderr: 'boom' }) || '', /exited with code 1.*boom/s)
  assert.match(piFailureMessage({ exitCode: 0, stderr: '', retryFinalError: 'rate limited' }) || '', /gave up after retries: rate limited/, 'a failed retry loop is a failure even on exit 0')
  assert.match(piFailureMessage({ exitCode: 0, stderr: '', promptRejected: 'no provider' }) || '', /rejected the prompt: no provider/, 'a refused prompt command is a failure even on exit 0')
  ok('piFailureMessage: exit code, stderr, retry final error, prompt rejection')
}

{
  const messages = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'reply' },
    { role: 'user', content: 'newest' },
  ] as any
  assert.equal(latestPiUserBody(messages), 'newest')
  ok('latestPiUserBody: newest user message only')
}

// --- fake-binary integration (a node script speaking the rpc protocol) ---

function fakePi(handleBody: string): Provider {
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-pi-'))
  const bin = path.join(dir, 'pi')
  writeFileSync(bin, `#!/usr/bin/env node
const fs = require('node:fs')
fs.writeFileSync(__dirname + '/args.txt', process.argv.slice(2).join(' '))
fs.writeFileSync(__dirname + '/env.json', JSON.stringify(process.env))
const systemAt = process.argv.indexOf('--append-system-prompt')
if (systemAt >= 0) {
  const systemFile = process.argv[systemAt + 1]
  fs.writeFileSync(__dirname + '/system.txt', fs.readFileSync(systemFile, 'utf8'))
  fs.writeFileSync(__dirname + '/system-mode.txt', String(fs.statSync(systemFile).mode & 0o777))
}
const emit = (o) => process.stdout.write(JSON.stringify(o) + '\\n')
const record = (line) => fs.appendFileSync(__dirname + '/stdin.txt', line + '\\n')
let buf = ''
process.stdin.on('data', (d) => {
  buf += String(d)
  let i
  while ((i = buf.indexOf('\\n')) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1)
    if (!line.trim()) continue
    record(line)
    handle(JSON.parse(line))
  }
})
process.stdin.on('end', () => process.exit(0))
function handle(cmd) {
${handleBody}
}
`)
  chmodSync(bin, 0o755)
  return { ...provider, cliPath: bin } as Provider
}

function fakeDir(fake: Provider): string {
  return path.dirname((fake as any).cliPath)
}

async function collect(params: PiCliParams): Promise<PiCliChunk[]> {
  const out: PiCliChunk[] = []
  for await (const chunk of streamPiCli(params)) out.push(chunk)
  return out
}

const HAPPY = `
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'agent_start' })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'thinking_delta', delta: 'planning' } })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'Hello' } })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: ' pi' } })
    emit({ type: 'agent_end', messages: [] })
  }
`

{
  // Fresh run: minted session id in args, prompt over stdin, deltas stream, EOF exit.
  const fake = fakePi(HAPPY)
  const chunks = await collect({
    provider: fake,
    model: 'pi-default',
    messages: [{ role: 'system', content: 'PRIVATE SYSTEM INSTRUCTION' }, { role: 'user', content: 'hi there' }] as any,
    newSessionId: () => 'minted-42',
  })
  const text = chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join('')
  assert.equal(text, 'Hello pi')
  assert.ok(chunks.some((c) => c.type === 'reasoning'))
  const cursor = chunks.find((c) => c.type === 'resume_cursor') as any
  assert.equal(cursor?.cursor.sessionId, 'minted-42', 'the minted id becomes the cursor')
  const args = readFileSync(path.join(fakeDir(fake), 'args.txt'), 'utf8')
  assert.ok(args.includes('--mode rpc'), 'spawned in rpc mode')
  assert.ok(args.includes('--session-id minted-42'))
  assert.doesNotMatch(args, /PRIVATE SYSTEM INSTRUCTION/, 'the system prompt never enters argv')
  assert.equal(readFileSync(path.join(fakeDir(fake), 'system.txt'), 'utf8'), 'PRIVATE SYSTEM INSTRUCTION')
  assert.equal(readFileSync(path.join(fakeDir(fake), 'system-mode.txt'), 'utf8'), String(0o600), 'temporary prompt file is owner-only')
  const stdin = readFileSync(path.join(fakeDir(fake), 'stdin.txt'), 'utf8')
  const prompt = JSON.parse(stdin.trim().split('\n')[0])
  assert.equal(prompt.type, 'prompt')
  assert.equal(prompt.message, 'hi there', 'the prompt rides as an rpc command, not argv')
  ok('streamPiCli: fresh rpc run streams and mints the session cursor')
}

{
  // Child env is an allowlist, not the launch environment: an ambient secret must never be
  // printable by the CLI (a read-only `printenv` is legitimate in a room seat), while
  // allowlisted basics and the provider's own cliEnv still arrive.
  const fake = fakePi(HAPPY)
  process.env.VONTRA_TEST_UNRELATED_API_KEY = 'sk-launch-env-sentinel'
  try {
    await collect({
      provider: { ...fake, cliEnv: { VONTRA_TEST_PROVIDER_TOKEN: 'provider-configured' } } as Provider,
      model: 'pi-default',
      messages: [{ role: 'user', content: 'check env' }] as any,
      newSessionId: () => 'env-1',
    })
  } finally {
    delete process.env.VONTRA_TEST_UNRELATED_API_KEY
  }
  const childEnv = JSON.parse(readFileSync(path.join(fakeDir(fake), 'env.json'), 'utf8'))
  assert.equal(childEnv.VONTRA_TEST_UNRELATED_API_KEY, undefined, 'an ambient launch-env secret never reaches the CLI child')
  assert.equal(childEnv.VONTRA_TEST_PROVIDER_TOKEN, 'provider-configured', 'provider cliEnv passes through verbatim')
  assert.equal(typeof childEnv.PATH, 'string', 'allowlisted basics (PATH) still reach the child')
  ok('streamPiCli: child env is allowlist + provider cliEnv, never the ambient secret')
}

{
  // Resume: cursor session id in args; latest-user-only body in the prompt command.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'resumed' } })
    emit({ type: 'agent_end', messages: [] })
  }
`)
  const chunks = await collect({
    provider: fake,
    model: 'pi-default',
    messages: [
      { role: 'user', content: 'first ask' },
      { role: 'assistant', content: 'answer' },
      { role: 'user', content: 'follow up' },
    ] as any,
    resumeCursor: { sessionId: 'pi-ses-KEPT' },
  })
  const args = readFileSync(path.join(fakeDir(fake), 'args.txt'), 'utf8')
  assert.ok(args.includes('--session-id pi-ses-KEPT'), 'resume reattaches the exact session id')
  assert.ok(!args.includes('--append-system-prompt'))
  const stdin = readFileSync(path.join(fakeDir(fake), 'stdin.txt'), 'utf8')
  assert.ok(stdin.includes('follow up') && !stdin.includes('first ask'), 'only the newest user message goes over on resume')
  assert.equal((chunks.find((c) => c.type === 'resume_cursor') as any)?.cursor.sessionId, 'pi-ses-KEPT')
  ok('streamPiCli: resume passes --session-id with latest-user-only prompt')
}

{
  // Live steering: a cue offered mid-run becomes a `steer` command, acked by the server,
  // and the run completes only after the steered continuation.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'working...' } })
    // Turn stays open: completion only comes after a steer arrives.
  }
  if (cmd.type === 'steer') {
    emit({ id: cmd.id, type: 'response', command: 'steer', success: true })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: ' steered!' } })
    emit({ type: 'agent_end', messages: [] })
  }
`)
  const channel = createSteeringChannel()
  let offerResult: Promise<boolean> | null = null
  const chunks: PiCliChunk[] = []
  for await (const chunk of streamPiCli({
    provider: fake,
    model: 'pi-default',
    messages: [{ role: 'user', content: 'go' }] as any,
    steering: channel,
  })) {
    chunks.push(chunk)
    if (!offerResult && chunk.type === 'content') {
      assert.equal(channel.canAccept, true, 'engine is steerable while the turn runs')
      // Offer out-of-band exactly like the loop's pump — never blocking consumption.
      offerResult = channel.offer({ id: 'cue-1', text: 'change course' })
    }
  }
  assert.equal(await offerResult!, true, 'the server ack resolves the offer true')
  const text = chunks.filter((c) => c.type === 'content').map((c: any) => c.text).join('')
  assert.equal(text, 'working... steered!')
  const fakeStdin = readFileSync(path.join(fakeDir(fake), 'stdin.txt'), 'utf8')
  const steerCmd = fakeStdin.trim().split('\n').map((l) => JSON.parse(l)).find((c) => c.type === 'steer')
  assert.equal(steerCmd?.message, 'change course')
  ok('streamPiCli: mid-run cue → rpc steer command, acked and applied')
}

{
  // A steer the server never acks resolves false (refused), and the run still completes.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'quick answer' } })
    emit({ type: 'agent_end', messages: [] })
  }
  // steer commands are swallowed silently
`)
  const channel = createSteeringChannel()
  let offerResult: Promise<boolean> | null = null
  const chunks: PiCliChunk[] = []
  for await (const chunk of streamPiCli({
    provider: fake,
    model: 'pi-default',
    messages: [{ role: 'user', content: 'go' }] as any,
    steering: channel,
    steerAckTimeoutMs: 120,
  })) {
    chunks.push(chunk)
    if (!offerResult && chunk.type === 'content') offerResult = channel.offer({ id: 'cue-1', text: 'too late' })
  }
  assert.equal(await offerResult!, false, 'no ack within the timeout → refused')
  assert.ok(chunks.some((c) => c.type === 'resume_cursor'), 'the run itself still completes')
  ok('streamPiCli: unacked steer resolves false without wedging the run')
}

{
  // Prompt rejection: response success:false fails the run even though pi exits 0.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: false, error: 'no provider configured' })
  }
`)
  await assert.rejects(collect({ provider: fake, model: 'pi-default', messages: [{ role: 'user', content: 'hi' }] as any }), /rejected the prompt: no provider configured/)
  ok('streamPiCli: a refused prompt command throws with the server reason')
}

{
  // Failure: nonzero exit surfaces stderr.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    process.stderr.write('provider auth failed')
    process.exit(1)
  }
`)
  await assert.rejects(collect({ provider: fake, model: 'pi-default', messages: [{ role: 'user', content: 'hi' }] as any }), /exited with code 1.*provider auth failed/s)
  ok('streamPiCli: nonzero exit throws with stderr detail')
}

{
  // Silent success (agent_end with no streamed text) is a failure, not an empty answer.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'agent_end', messages: [] })
  }
`)
  await assert.rejects(collect({ provider: fake, model: 'pi-default', messages: [{ role: 'user', content: 'hi' }] as any }), /finished without a final message/)
  ok('streamPiCli: agent_end with no streamed text is a failure')
}

{
  // Exit 0 WITHOUT agent_end (process died politely mid-run) is also a failure.
  const fake = fakePi(`
  if (cmd.type === 'prompt') {
    emit({ id: cmd.id, type: 'response', command: 'prompt', success: true })
    emit({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'partial' } })
    process.exit(0)
  }
`)
  await assert.rejects(collect({ provider: fake, model: 'pi-default', messages: [{ role: 'user', content: 'hi' }] as any }), /finished without a final message/)
  ok('streamPiCli: exit without agent_end is a failure even with partial text')
}

console.log(`\n${passed} passed`)
