import assert from 'node:assert/strict'
import { applyClaudePromptEffortPrefix, claudeEffort, claudePermissionMode, claudeResultError, claudeFailureMessage, statusFromEvent, textFromEvent } from './claude-cli'
import { cliSpawnEnv } from './cli-spawn-env'

// Claude Code's stream-json reports run errors in a stdout `result` event (is_error:true),
// NOT on stderr. These shapes are captured from the real `claude -p --output-format stream-json`.

// 1) A successful run has no result error.
assert.equal(claudeResultError({ type: 'result', subtype: 'success', is_error: false, result: 'PONG' }), null,
  'a non-error result yields no error')
assert.equal(claudeResultError({ type: 'assistant', message: {} }), null, 'non-result events yield no error')

// 2) An errored result surfaces its message (even though subtype is "success").
assert.equal(
  claudeResultError({ type: 'result', subtype: 'success', is_error: true, result: 'Not logged in · Please run /login' }),
  'Not logged in · Please run /login',
  'an errored result surfaces the CLI message',
)

// 3) Failure messaging: auth errors get the friendly hint, regardless of which channel carried them.
assert.match(
  claudeFailureMessage({ resultError: 'Not logged in · Please run /login', exitCode: 1, stderr: '' }) || '',
  /not authenticated/i, '"Not logged in" is detected as an auth failure')
assert.match(
  claudeFailureMessage({ resultError: 'Failed to authenticate. API Error: 401 Invalid authentication credentials', exitCode: 1, stderr: '' }) || '',
  /not authenticated/i, 'a 401 invalid-credentials result is detected as an auth failure')

// 4) Non-auth result error is surfaced verbatim — including the exit-0 silent-failure case.
assert.match(
  claudeFailureMessage({ resultError: 'Tool execution failed: ENOENT', exitCode: 0, stderr: '' }) || '',
  /Tool execution failed: ENOENT/, 'a result error is surfaced even when the process exits 0')

// 5) A clean run (exit 0, no result error) is NOT a failure.
assert.equal(claudeFailureMessage({ resultError: null, exitCode: 0, stderr: '' }), null, 'clean run is not a failure')

// 6) Falls back to exit code + stderr when there is no structured result error.
assert.match(
  claudeFailureMessage({ resultError: null, exitCode: 1, stderr: 'boom' }) || '',
  /exited with code 1.*boom/s, 'falls back to exit code + stderr')

// 7) Dedup: `--include-partial-messages` emits text twice — incremental `content_block_delta`
// `text_delta` events AND a trailing `assistant` event with the full accumulated block. Once any
// delta has streamed, the `assistant` event's full text must be dropped or the answer is duplicated
// (e.g. "Hello world" -> "Hello worldHello world").
const helloDelta = { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hello world' } } }
const helloAssistant = { type: 'assistant', message: { content: [{ type: 'text', text: 'Hello world' }] } }

// The streamed delta is always emitted.
assert.equal(textFromEvent(helloDelta, false), 'Hello world', 'a text_delta yields its incremental text')

// Before any delta has streamed, an assistant event carries the full text (non-streaming fallback).
assert.equal(textFromEvent(helloAssistant, false), 'Hello world', 'assistant event yields full text when nothing streamed')

// After a delta has streamed, the assistant event must be suppressed (it only re-states the deltas).
assert.equal(textFromEvent(helloAssistant, true), '', 'assistant event is suppressed once deltas streamed (no duplication)')

// Simulate the streamClaudeCli accumulation: deltas then a trailing assistant event.
let streamed = false
let content = ''
for (const event of [helloDelta, helloAssistant]) {
  const text = textFromEvent(event, streamed)
  if (text) { streamed = true; content += text }
}
assert.equal(content, 'Hello world', 'accumulated content is not duplicated by the trailing assistant event')
assert.equal(textFromEvent({ type: 'result', is_error: false, result: 'Fallback final' }, false), 'Fallback final', 'result event is a final-message fallback when no deltas streamed')
assert.equal(textFromEvent({ type: 'result', is_error: false, result: 'Fallback final' }, true), '', 'result fallback is suppressed once deltas streamed')

// 8) Runtime approval mode must win safely over saved provider defaults. Auto accepts edits but
// never maps to Claude's unguarded bypassPermissions mode.
assert.equal(claudePermissionMode({ cliPermissionMode: 'default' } as any, 'auto'), 'acceptEdits', 'Full Auto uses Claude guarded edit mode')
assert.equal(claudePermissionMode({ cliPermissionMode: 'bypassPermissions' } as any, 'auto'), 'acceptEdits', 'saved bypassPermissions cannot leak into Full Auto')
assert.equal(claudePermissionMode({ cliPermissionMode: 'bypassPermissions' } as any, 'readonly'), 'plan', 'readonly overrides saved bypass permission mode')
assert.equal(claudePermissionMode({ cliPermissionMode: 'acceptEdits' } as any, 'suggest'), 'acceptEdits', 'suggest keeps the explicit provider permission mode')

// 9) Streaming status events are translated into visible activity rows instead of leaving the
// TensorFold Studio chat stuck on a generic "writing" state.
assert.equal(
  statusFromEvent({ type: 'stream_event', event: { type: 'content_block_start', content_block: { type: 'tool_use', name: 'Bash' } } }),
  'Claude using Bash',
  'tool_use starts become readable status',
)
assert.equal(
  statusFromEvent({ type: 'stream_event', event: { type: 'content_block_start', content_block: { type: 'thinking' } } }),
  'Claude thinking',
  'thinking blocks become readable status',
)

// 10) Claude effort values must match the local CLI surface. `claude --help` supports
// low/medium/high/xhigh/max; prompt/provider modes resolve to xhigh where applicable.
assert.equal(claudeEffort('minimal'), 'low', 'minimal is lowered to Claude low')
assert.equal(claudeEffort('low'), 'low', 'low passes through')
assert.equal(claudeEffort('medium'), 'medium', 'medium is the fallback/default')
assert.equal(claudeEffort('high'), 'high', 'high passes through')
assert.equal(claudeEffort('xhigh'), 'xhigh', 'xhigh passes through instead of being downgraded')
assert.equal(claudeEffort('extra-high'), 'xhigh', 'extra-high alias maps to xhigh')
assert.equal(claudeEffort('max'), 'max', 'max passes through instead of being downgraded')
assert.equal(claudeEffort('ultrathink'), 'xhigh', 'ultrathink executes at xhigh')
assert.equal(claudeEffort('ultracode'), 'xhigh', 'ultracode executes at xhigh until workflow mode is wired')

assert.equal(applyClaudePromptEffortPrefix('inspect the repo', 'ultrathink'), 'Ultrathink:\ninspect the repo', 'ultrathink prefixes the prompt')
assert.equal(applyClaudePromptEffortPrefix('Ultrathink:\ninspect the repo', 'ultrathink'), 'Ultrathink:\ninspect the repo', 'ultrathink prefix is idempotent')
assert.equal(applyClaudePromptEffortPrefix('inspect the repo', 'xhigh'), 'inspect the repo', 'plain xhigh does not alter the prompt')

// 11) cliSpawnEnv: a CLI child never inherits the launch environment wholesale. Room seats run
// the CLI autonomously and a read-only `printenv` is a legitimate command, so ambient secrets
// would be printable into a reply broadcast to other providers.
{
  const inherited = {
    PATH: '/usr/bin:/bin',
    HOME: '/home/u',
    XDG_CONFIG_HOME: '/home/u/.config',
    HTTPS_PROXY: 'http://proxy:8080',
    NODE_EXTRA_CA_CERTS: '/etc/corp.pem',
    SSH_AUTH_SOCK: '/tmp/ssh.sock',
    UNRELATED_API_KEY: 'sk-launch-env-secret',
    GITHUB_TOKEN: 'ghp-launch-env-secret',
    AWS_SECRET_ACCESS_KEY: 'aws-launch-env-secret',
    VONTRA_SECRET_KEY_BASE64: 'dm9udHJh',
  }
  const env = cliSpawnEnv({ PROVIDER_CLI_TOKEN: 'provider-configured' }, inherited)
  assert.equal(env.PATH, '/usr/bin:/bin', 'allowlisted basics pass through')
  assert.equal(env.HOME, '/home/u')
  assert.equal(env.XDG_CONFIG_HOME, '/home/u/.config')
  assert.equal(env.HTTPS_PROXY, 'http://proxy:8080', 'proxy vars are connectivity, not credentials')
  assert.equal(env.NODE_EXTRA_CA_CERTS, '/etc/corp.pem')
  assert.equal(env.SSH_AUTH_SOCK, '/tmp/ssh.sock', 'git ssh agent stays available to the CLI')
  assert.equal(env.UNRELATED_API_KEY, undefined, 'ambient API keys never reach the child')
  assert.equal(env.GITHUB_TOKEN, undefined, 'ambient tokens never reach the child')
  assert.equal(env.AWS_SECRET_ACCESS_KEY, undefined, 'ambient cloud secrets never reach the child')
  assert.equal(env.VONTRA_SECRET_KEY_BASE64, undefined, 'the managed store key never reaches the child')
  assert.equal(env.PROVIDER_CLI_TOKEN, 'provider-configured', 'provider cliEnv extras pass verbatim')
  assert.equal(Object.keys(env).length, 7, 'nothing else leaks through')
  const overridden = cliSpawnEnv({ PATH: '/provider/path' }, inherited)
  assert.equal(overridden.PATH, '/provider/path', 'provider cliEnv wins over an inherited value')
  assert.deepEqual(cliSpawnEnv({}, {}), {}, 'an empty inherited env yields an empty child env')
}

console.log('claude-cli outcome tests passed')
