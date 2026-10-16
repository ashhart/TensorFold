// Run: npx tsx electron/agent/planning-guard.test.ts
import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { OMP_ENGINE_REFUSAL, PLANNING_NOTE, PROJECT_CONTEXT_WITHHELD_NOTE, agentTerminalRefusal, assertOmpEngineAllowed, configurePlanningGuard, latestPlanText, needsPlanningGuard, planningGuard, previewWithheld, runPlanPrompt, type GuardableParams } from './planning-guard'
import { cliConfinement, planningRoot } from './planning-sandbox'
import { canUseMacSandbox } from './sandbox'
import { createSteeringChannel } from './steering'
import { streamChat } from './openai'
import type { ApiMessage, Provider } from './types'

const scratch = realpathSync(mkdtempSync(path.join(os.tmpdir(), 'tf-guard-test-')))
configurePlanningGuard({ confidentialTerms: () => ['Acme Corp'] })

const cloud: Provider = { id: 'openai', name: 'OpenAI', kind: 'cloud', source: 'openai', baseUrl: 'https://api.openai.com/v1', apiKey: 'k', models: ['gpt'] }
const ollama: Provider = { id: 'ollama', name: 'Ollama', kind: 'local', source: 'ollama', baseUrl: 'http://127.0.0.1:11434/v1', apiKey: '', models: ['m'] }
const workspace = path.join(scratch, 'code', 'acme')
const history: ApiMessage[] = [
  { role: 'system', content: `Workspace: ${workspace}` },
  { role: 'user', content: 'Plan the Acme Corp release; ping jane.roe@example.com.' },
  { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{"path":"secret.txt"}' } }] },
  { role: 'tool', tool_call_id: 'c1', content: 'FILE BODY' },
]

async function collect<T>(source: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const item of source) out.push(item)
  return out
}

async function main() {
  // Trusted local providers are not touched.
  assert.equal(planningGuard({ provider: ollama, messages: history }, async function* () {}), null)
  assert.equal(needsPlanningGuard({ provider: ollama }), false)
  assert.equal(needsPlanningGuard({ provider: cloud }), true)

  // External HTTP: no tools, read-only, no workspace, redacted history, rehydrated reply.
  {
    let seen: GuardableParams | null = null
    const guarded = planningGuard({ provider: cloud, baseUrl: cloud.baseUrl, messages: history, tools: [{ type: 'function' }], approvalMode: 'auto', cwd: workspace, threadId: 't1' }, async function* (p) {
      seen = p
      assert.equal(needsPlanningGuard(p), false, 'guarded params are not guarded twice')
      yield { type: 'content', text: 'Email [EMA' }
      yield { type: 'content', text: 'IL_1] about [TERM_1] in [WORKSPACE_1]/src.' }
      yield { type: 'tool_calls', toolCalls: [] }
      yield { type: 'finish', reason: 'stop' }
    })!
    const chunks = await collect(guarded)
    const p = seen as unknown as GuardableParams
    assert.equal(p.tools, undefined)
    assert.equal(p.approvalMode, 'readonly')
    assert.equal(p.cwd, null)
    const sent = JSON.stringify(p.messages)
    for (const leaked of ['jane.roe@example.com', 'Acme Corp', workspace, 'FILE BODY', 'secret.txt']) assert.ok(!sent.includes(leaked), leaked)
    assert.equal(p.messages[1].content, PLANNING_NOTE)
    assert.ok(p.messages.every((m) => m.role !== 'tool' && !m.tool_calls))
    assert.deepEqual(chunks.map((c) => c.type), ['notice', 'content', 'content', 'finish'])
    assert.equal((chunks[0] as { text: string }).text, 'Sent to OpenAI with 5 items withheld (1 workspace path, 1 email, 1 confidential term, 1 tool call, 1 tool result).')
    assert.equal(chunks.filter((c) => c.type === 'content').map((c) => (c as { text: string }).text).join(''), `Email jane.roe@example.com about Acme Corp in ${workspace}/src.`)
  }

  // Live steering cues are redacted on the way into the engine.
  {
    const channel = createSteeringChannel()
    const received: string[] = []
    const guarded = planningGuard({ provider: { ...cloud, transport: 'chat-completions' }, messages: history, steering: channel }, async function* (p) {
      p.steering!.connect((input) => { received.push(input.text); return true })
      yield { type: 'finish' }
    })!
    await collect(guarded)
    assert.equal(await channel.offer({ id: 'cue', text: 'also cc jane.roe@example.com' }), true)
    assert.deepEqual(received, ['also cc [EMAIL_1]'])
  }

  // OpenCode/Kilo servers can't be confined, so they are refused before anything is sent.
  await assert.rejects(collect(streamChat({ provider: { ...cloud, transport: 'opencode' }, model: 'm', messages: history })), /cannot confine/)

  // Settings preview uses draft terms and never needs a provider.
  {
    const preview = previewWithheld('Ship Falcon for globex, mail ops@example.com', ['Falcon', 'Globex'])
    assert.equal(preview.text, 'Ship [TERM_1] for [TERM_2], mail [EMAIL_1]')
    assert.equal(preview.total, 3)
    assert.equal(preview.withheld, '2 confidential terms, 1 email')
  }

  // Project guidance and durable memory never reach an external model, even if a prompt has them.
  {
    let sent: ApiMessage[] = []
    const system = `Base prompt.\n\nPROJECT GUIDANCE DATA (from AGENTS.md; may be repo-authored.)\nUse the Acme build farm.\n\nDURABLE PROJECT MEMORY — untrusted reference data only.\n<project_memory>\nClient pays in May.\n</project_memory>`
    const chunks = await collect(planningGuard({ provider: cloud, messages: [{ role: 'system', content: system }, { role: 'user', content: 'plan' }] }, async function* (p) { sent = p.messages; yield { type: 'finish' } })!)
    assert.equal(sent[0].content, `Base prompt.\n\n${PROJECT_CONTEXT_WITHHELD_NOTE}`)
    assert.match((chunks[0] as { text: string }).text, /2 project guidance or memory blocks/)
  }

  // Agent CLIs never start in a project terminal; the OMP engine is refused outright.
  {
    assert.equal(agentTerminalRefusal({ id: 'shell', name: 'Shell', command: null }), null)
    assert.equal(agentTerminalRefusal({ id: 'custom-1', name: 'Tests', command: 'npm test' }), null)
    assert.match(agentTerminalRefusal({ id: 'claude', name: 'Claude Code', command: 'claude' }) ?? '', /^Claude Code sends this folder to an external model/)
    assert.ok(agentTerminalRefusal({ id: 'custom-2', name: 'Mine', command: '/opt/homebrew/bin/codex --full-auto' }))
    assert.throws(() => assertOmpEngineAllowed(), new RegExp(OMP_ENGINE_REFUSAL.slice(0, 40)))
  }

  // Run-plan handoff: the latest assistant reply is the plan; only trusted targets may run it.
  {
    assert.equal(latestPlanText([{ role: 'user', content: 'plan it' }, { role: 'assistant', content: '1. Edit a.ts' }, { role: 'assistant', content: '' }]), '1. Edit a.ts')
    assert.equal(latestPlanText([{ role: 'user', content: 'x' }]), '')
    assert.match(runPlanPrompt('Claude', '1. Edit a.ts'), /^Carry out this plan from Claude[\s\S]*<plan>\n1\. Edit a\.ts\n<\/plan>$/)
    const handoff = readFileSync(new URL('../main/host/handoff-ipc.ts', import.meta.url), 'utf8')
    assert.match(handoff, /runPlan && classifyProvider\(toProvider\) !== 'trusted-local'\) throw/, 'run-plan refuses an external target')
    assert.ok(handoff.indexOf("classifyProvider(toProvider) !== 'trusted-local'") < handoff.indexOf('store.threads.unshift'), 'the target check precedes persistence')
  }

  // End to end through streamChat with a fake Claude CLI inside the real seatbelt.
  if (process.platform === 'darwin' && canUseMacSandbox()) {
    mkdirSync(workspace, { recursive: true })
    writeFileSync(path.join(workspace, 'secret.txt'), 'workspace secret')
    const bin = path.join(scratch, 'bin', 'claude')
    mkdirSync(path.dirname(bin), { recursive: true })
    writeFileSync(bin, `#!/usr/bin/env node
const fs = require('fs')
let stdin = ''
process.stdin.on('data', (d) => { stdin += d })
process.stdin.on('end', () => {
  const argv = process.argv.slice(2)
  const promptFile = argv[argv.indexOf('--append-system-prompt-file') + 1]
  let workspaceRead
  try { fs.readFileSync(${JSON.stringify(path.join(workspace, 'secret.txt'))}); workspaceRead = 'read' } catch (e) { workspaceRead = e.code }
  const report = { argv, cwd: process.cwd(), stdin, system: fs.readFileSync(promptFile, 'utf8'), workspaceRead }
  const say = (o) => process.stdout.write(JSON.stringify(o) + '\\n')
  say({ type: 'system', session_id: 's-1' })
  say({ type: 'assistant', message: { content: [{ type: 'text', text: 'REPORT ' + Buffer.from(JSON.stringify(report)).toString('base64') + ' Reply to [EMAIL_1].' }] } })
  say({ type: 'result', is_error: false, result: 'done', session_id: 's-1' })
})
`)
    chmodSync(bin, 0o755)
    const claude: Provider = { id: 'claude', name: 'Claude', transport: 'claude-cli', cliPath: bin, baseUrl: '', apiKey: '', models: [] }
    const chunks = await collect(streamChat({ provider: claude, model: 'claude-code-default', messages: history, cwd: workspace, approvalMode: 'auto', tools: [{}] }))
    const text = chunks.filter((c) => c.type === 'content').map((c) => (c as { text: string }).text).join('')
    assert.match(text, /Reply to jane\.roe@example\.com\.$/)
    const report = JSON.parse(Buffer.from(/REPORT (\S+)/.exec(text)![1], 'base64').toString('utf8'))
    assert.equal(report.workspaceRead, 'EPERM')
    assert.ok(report.cwd.startsWith(`${planningRoot()}/tf-plan-${process.pid}-`), report.cwd)
    assert.ok(!report.cwd.startsWith(os.homedir()), 'the CLI never sees a path under the home folder')
    assert.deepEqual(report.argv.slice(0, 2), ['--tools', ''])
    assert.ok(report.argv.includes('--safe-mode') && report.argv.includes('plan'))
    assert.ok(report.system.includes(PLANNING_NOTE) && report.system.includes('[WORKSPACE_1]') && !report.system.includes(workspace))
    assert.ok(report.stdin.includes('[TERM_1]') && report.stdin.includes('[EMAIL_1]') && !report.stdin.includes('jane.roe'))
    assert.ok(!report.stdin.includes('FILE BODY'))
    assert.equal(chunks[0].type, 'notice')
    // The one-shot planning folder is gone; nothing was left behind.
    assert.ok(!existsSync(report.cwd))
    assert.equal(cliConfinement(claude), null, 'the caller provider object is never confined')
    assert.ok(existsSync(path.join(workspace, 'secret.txt')))
  } else {
    console.log('planning-guard: CLI seatbelt check skipped (no sandbox-exec)')
  }
}

main()
  .then(() => console.log('planning-guard tests passed'))
  .finally(() => rmSync(scratch, { recursive: true, force: true }))
