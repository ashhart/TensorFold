// Run: jiti electron/agent/external-agent-runtime.test.ts
import assert from 'node:assert/strict'
import path from 'node:path'
import {
  buildExternalAgentSpawnSpec,
  createExternalAgentSession,
  createExternalAgentControlCommand,
  externalAgentAuditRecord,
  externalAgentCommandCentreItem,
  externalAgentEventRequiresApproval,
  externalAgentStatusFromEvent,
  externalAgentEventToTimelineItem,
  externalAgentProfileScope,
} from './external-agent-runtime'

const session = createExternalAgentSession({
  id: 'ext_test',
  protocol: 'acp',
  lane: 'thread',
  command: 'agent',
  args: ['--stdio'],
  cwd: '.',
  title: 'ACP Test Agent',
  threadId: 'thread_1',
  providerId: 'codex',
  providerAccountId: 'work',
  model: 'gpt-5.5',
}, 100)

assert.equal(session.id, 'ext_test')
assert.equal(session.protocol, 'acp')
assert.equal(session.lane, 'thread')
assert.equal(session.cwd, path.resolve('.'))
assert.equal(session.providerAccountId, 'work')

const spawnSpec = buildExternalAgentSpawnSpec({
  protocol: 'acp',
  lane: 'room',
  command: 'agent',
  cwd: '.',
  env: { CODEX_HOME: '/tmp/codex-a', SECRET: 'nope' },
  envAllowlist: ['CODEX_HOME'],
})
assert.deepEqual(spawnSpec.env, { CODEX_HOME: '/tmp/codex-a' })

const codexScope = externalAgentProfileScope({
  kind: 'codex',
  profileHome: '/profiles/codex-work',
  env: { OPENAI_API_KEY: 'do-not-forward' },
  extraAllowlist: ['VONTRA_TRACE_ID'],
})
assert.equal(codexScope.env.CODEX_HOME, '/profiles/codex-work')
assert.ok(codexScope.envAllowlist.includes('CODEX_HOME'))
assert.ok(codexScope.envAllowlist.includes('VONTRA_TRACE_ID'))
assert.ok(!codexScope.envAllowlist.includes('OPENAI_API_KEY'))

const claudeScope = externalAgentProfileScope({
  kind: 'claude',
  profileHome: '/profiles/claude-personal',
  profilePath: '/profiles/claude-personal/settings.json',
})
assert.equal(claudeScope.env.CLAUDE_CONFIG_DIR, '/profiles/claude-personal')
assert.equal(claudeScope.env.CLAUDE_HOME, '/profiles/claude-personal')
assert.equal(claudeScope.env.CLAUDE_CONFIG_FILE, '/profiles/claude-personal/settings.json')
assert.ok(claudeScope.profileKeys.includes('CLAUDE_CONFIG_DIR'))
assert.ok(claudeScope.envAllowlist.includes('CLAUDE_CONFIG_FILE'))

const message = externalAgentEventToTimelineItem(session, {
  id: 'msg_1',
  kind: 'message',
  text: 'I can take this task.',
  createdAt: 101,
})
assert.equal(message.kind, 'assistant')
assert.equal(message.text, 'I can take this task.')
assert.match(message.meta ?? '', /acp/)

const edit = externalAgentEventToTimelineItem(session, {
  id: 'edit_1',
  kind: 'edit',
  path: '/repo/app.ts',
  diff: [{ type: 'add', text: '+console.log(1)', newNo: 1 }],
  createdAt: 102,
})
assert.equal(edit.kind, 'tool')
assert.equal(edit.tool, 'external_agent_edit')
assert.equal(edit.state, 'awaiting')
assert.match(edit.approval ?? '', /TensorFold approval/)
assert.equal(externalAgentEventRequiresApproval({ kind: 'edit' }), true)

const mutatingTool = externalAgentEventToTimelineItem(session, {
  id: 'tool_1',
  kind: 'tool_call',
  toolName: 'write_file',
  args: { path: 'app.ts' },
  mutating: true,
  createdAt: 103,
})
assert.equal(mutatingTool.state, 'awaiting')
assert.equal(externalAgentEventRequiresApproval({ kind: 'tool_call', toolName: 'search', mutating: false }), false)
assert.equal(externalAgentStatusFromEvent({ kind: 'tool_call', toolName: 'write_file', mutating: true }), 'waiting_approval')

const error = externalAgentEventToTimelineItem(session, {
  id: 'err_1',
  kind: 'error',
  output: 'Agent exited with code 1.',
  createdAt: 104,
})
assert.equal(error.kind, 'error')
assert.match(error.text ?? '', /code 1/)

const commandCentreItem = externalAgentCommandCentreItem({
  session,
  status: 'running',
  eventCount: 4,
  pendingApprovals: 1,
  lastEventAt: 150,
})
assert.equal(commandCentreItem.id, 'external:ext_test')
assert.equal(commandCentreItem.canCancel, true)
assert.equal(commandCentreItem.canGuide, true)
assert.equal(commandCentreItem.canAudit, true)
assert.match(commandCentreItem.detail, /codex/)
assert.match(commandCentreItem.footnote ?? '', /1 approval/)

const completedItem = externalAgentCommandCentreItem({ session, status: 'completed' })
assert.equal(completedItem.canCancel, false)
assert.equal(completedItem.canGuide, false)
assert.equal(completedItem.canAudit, true)

const guide = createExternalAgentControlCommand(session, { action: 'guide', text: '  keep the patch focused  ' }, 200)
assert.equal(guide.action, 'guide')
assert.equal(guide.text, 'keep the patch focused')
assert.throws(() => createExternalAgentControlCommand(session, { action: 'guide', text: '   ' }, 201), /Guidance text/)

const cancel = createExternalAgentControlCommand(session, { action: 'cancel', reason: 'User stopped it' }, 202)
const guideAudit = externalAgentAuditRecord(session, { action: 'guide', control: guide }, 203)
assert.equal(guideAudit.kind, 'external_agent.guide')
assert.equal(guideAudit.controlAction, 'guide')
assert.equal(guideAudit.text, 'keep the patch focused')
assert.equal(guideAudit.providerAccountId, 'work')

const cancelAudit = externalAgentAuditRecord(session, { action: 'cancel', control: cancel, status: 'cancelled' }, 204)
assert.equal(cancelAudit.kind, 'external_agent.cancel')
assert.equal(cancelAudit.status, 'cancelled')
assert.equal(cancelAudit.text, 'User stopped it')

const eventAudit = externalAgentAuditRecord(session, {
  action: 'event',
  event: { kind: 'edit', text: 'please write the file', mutating: true },
}, 205)
assert.equal(eventAudit.requiresApproval, true)
assert.equal(eventAudit.status, 'waiting_approval')
assert.equal(eventAudit.eventKind, 'edit')

console.log('external-agent-runtime tests passed')
