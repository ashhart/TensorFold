import assert from 'node:assert/strict'
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { buildCodexArgs, codexExecApprovalPolicy, codexModelAndEffort, statusFromEvent, streamCodexCli } from './codex-cli'

// model + effort + speed(service_tier) all flow into the `codex exec` args
const args = buildCodexArgs({
  provider: { cliPath: 'codex', cliApprovalPolicy: 'never' } as any,
  model: 'gpt-5.5',
  messages: [],
  reasoningEffort: 'high',
  serviceTier: 'flex',
})
const joined = args.join(' ')
assert.ok(args[0] === 'exec', 'first arg is exec')
assert.ok(args.includes('-m') && args.includes('gpt-5.5'), 'passes the chosen model')
assert.ok(joined.includes('model_reasoning_effort="high"'), 'passes the effort level')
assert.ok(joined.includes('service_tier="flex"'), 'passes the speed (service tier)')

// the sentinel default model is NOT forwarded as -m; speed still applies
const args2 = buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [], serviceTier: 'fast' })
assert.ok(!args2.includes('-m'), 'default model is left to the CLI')
assert.ok(args2.join(' ').includes('service_tier="fast"'), 'speed applies even with default model')

const hostileTier = 'fast"\nmodel="attacker'
const hostileArgs = buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [], serviceTier: hostileTier })
const tierOverride = hostileArgs.find((arg) => arg.startsWith('service_tier=')) ?? ''
assert.ok(hostileArgs.includes(`service_tier=${JSON.stringify(hostileTier)}`), 'service tier is encoded as one escaped TOML basic string')
assert.ok(!tierOverride.includes('\n'), 'raw control characters cannot break the config override')

// no serviceTier -> no service_tier arg (CLI/config decides)
const args3 = buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [] })
assert.ok(!args3.join(' ').includes('service_tier'), 'omitted when speed not set')

// `codex exec` (>= 0.125) removed the top-level `--ask-for-approval` flag — passing it makes
// `codex exec` exit 2 ("unexpected argument '--ask-for-approval' found"), breaking every run.
// The approval policy must instead flow through the supported `-c approval_policy="..."` override.
const argsApproval = buildCodexArgs({ provider: { cliApprovalPolicy: 'never' } as any, model: 'codex-cli-default', messages: [] })
assert.ok(!argsApproval.includes('--ask-for-approval'), 'must not pass the removed --ask-for-approval flag to codex exec')
assert.ok(argsApproval.join(' ').includes('approval_policy="never"'), 'approval policy flows through -c approval_policy config override')

const argsApproval2 = buildCodexArgs({ provider: { cliApprovalPolicy: 'on-request' } as any, model: 'codex-cli-default', messages: [] })
assert.ok(argsApproval2.join(' ').includes('approval_policy="never"'), 'on-request is coerced away because codex exec cannot bridge approval prompts')

// Full Auto keeps Codex inside workspace-write; it must never emit the dangerous bypass flag.
const guardedAutoArgs = buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [], approvalMode: 'auto' })
assert.ok(guardedAutoArgs.join(' ').includes('-s workspace-write'), 'Full Auto keeps Codex workspace sandbox active')
assert.ok(!guardedAutoArgs.includes('--dangerously-bypass-approvals-and-sandbox'), 'Full Auto never emits Codex danger-full-access bypass')
assert.ok(buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [], approvalMode: 'readonly' }).join(' ').includes('-s read-only'), 'readonly maps to -s read-only')
assert.ok(buildCodexArgs({ provider: {} as any, model: 'codex-cli-default', messages: [], approvalMode: 'suggest' }).join(' ').includes('-s workspace-write'), 'suggest keeps -s workspace-write')

const savedDefaults = { cliSandbox: 'workspace-write', cliApprovalPolicy: 'on-request' } as any
const autoArgs = buildCodexArgs({ provider: savedDefaults, model: 'codex-cli-default', messages: [], approvalMode: 'auto' }).join(' ')
assert.ok(autoArgs.includes('-s workspace-write'), 'Full Auto forces the guarded workspace-write sandbox')
assert.ok(!autoArgs.includes('--dangerously-bypass-approvals-and-sandbox'), 'saved defaults cannot reintroduce the bypass flag')
assert.ok(!autoArgs.includes('approval_policy="on-request"'), 'Full auto never leaves codex exec in hidden approval mode')
assert.equal(codexExecApprovalPolicy(savedDefaults, 'readonly'), 'never', 'readonly cannot leave codex exec waiting for approval either')

// The effort-preset "models" map to a REAL codex model + reasoning effort. codex has no
// gpt-5.5-extra-high / gpt-5.5-codex-high model — passing them as -m 400s ("model is not supported").
assert.deepEqual(codexModelAndEffort('gpt-5.5-extra-high'), { model: 'gpt-5.5', effort: 'xhigh' }, 'extra-high → gpt-5.5 @ xhigh')
assert.deepEqual(codexModelAndEffort('gpt-5.5-codex-high'), { model: 'gpt-5.5', effort: 'high' }, 'codex-high → gpt-5.5 @ high')
assert.deepEqual(codexModelAndEffort('gpt-5.5', 'low'), { model: 'gpt-5.5', effort: 'low' }, 'a real model uses the effort selector')
assert.deepEqual(codexModelAndEffort('gpt-5.4', 'xhigh'), { model: 'gpt-5.4', effort: 'xhigh' }, 'xhigh effort flows through for a real model')

// buildCodexArgs translates the alias to a runnable command (real model + effort), never the alias id.
const aliasArgs = buildCodexArgs({ provider: {} as any, model: 'gpt-5.5-extra-high', messages: [] }).join(' ')
assert.ok(aliasArgs.includes('-m gpt-5.5') && !aliasArgs.includes('gpt-5.5-extra-high'), 'forwards the real model, not the alias id')
assert.ok(aliasArgs.includes('model_reasoning_effort="xhigh"'), 'forwards the aliased effort')

assert.equal(
  statusFromEvent({ type: 'item.started', item: { type: 'command_execution', command: 'ssh pgx uptime' } }),
  'Codex running shell: ssh pgx uptime',
  'command_execution events become visible shell activity',
)
assert.equal(
  statusFromEvent({ type: 'item.completed', item: { type: 'tool_call', name: 'read_file' } }),
  'Codex finished read_file',
  'tool completion events get a readable label',
)
assert.equal(statusFromEvent({ type: 'item.started', item: { type: 'agent_message' } }), '', 'agent_message status is suppressed because content deltas render the answer')

// Child env is an allowlist, not the launch environment: an ambient secret must never be
// printable by the CLI (a read-only `printenv` is legitimate in a room seat), while
// allowlisted basics and the provider's own cliEnv still arrive.
{
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-codex-env-'))
  const bin = path.join(dir, 'codex')
  writeFileSync(bin, `#!/bin/sh
printenv > "$(dirname "$0")/child-env.txt"
cat > /dev/null
echo '{"type":"response.output_text.delta","delta":"env ok"}'
exit 0
`)
  chmodSync(bin, 0o755)
  process.env.VONTRA_TEST_UNRELATED_API_KEY = 'sk-launch-env-sentinel'
  try {
    let text = ''
    for await (const chunk of streamCodexCli({
      provider: { cliPath: bin, cliEnv: { VONTRA_TEST_PROVIDER_TOKEN: 'provider-configured' } } as any,
      model: 'codex-cli-default',
      messages: [{ role: 'user', content: 'check env' }] as any,
    })) if (chunk.type === 'content') text += chunk.text
    assert.equal(text, 'env ok')
  } finally {
    delete process.env.VONTRA_TEST_UNRELATED_API_KEY
  }
  const childEnv = readFileSync(path.join(dir, 'child-env.txt'), 'utf8')
  assert.doesNotMatch(childEnv, /VONTRA_TEST_UNRELATED_API_KEY|sk-launch-env-sentinel/, 'an ambient launch-env secret never reaches the CLI child')
  assert.match(childEnv, /^VONTRA_TEST_PROVIDER_TOKEN=provider-configured$/m, 'provider cliEnv passes through verbatim')
  assert.match(childEnv, /^PATH=/m, 'allowlisted basics (PATH) still reach the child')
}

console.log('codex-cli arg tests passed')
