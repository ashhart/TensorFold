import assert from 'node:assert/strict'
import { buildToolSchemas } from './tool-schema-builder'
import { executeTool } from './tools'
import { restrictTicketToolContext } from './ticket-tool-policy'
import { ticketDisallowedToolNames, ticketToolCapability } from './tickets'

const mcpTool = {
  serverId: 'private', serverName: 'Private', name: 'search', callName: 'mcp__private__search',
  inputSchema: { type: 'object', properties: {} },
}
const base = {
  mcpTools: [mcpTool],
  callMcpTool: async () => ({ ok: true, output: 'secret' }),
  runBrowserAutomation: async () => ({ ok: true }),
  runControllerAction: async () => ({ ok: true }),
  recordEmailDraft: async () => ({}),
  connectorEvents: [{ id: 'mail-1' }],
  emailSettings: { sources: [] },
  registerArtifact: () => ({}),
  fileSuggestion: () => {},
  launchSubAgent: async () => ({}),
  allowSandboxNetwork: true,
  sensitiveRetrievalAllowed: true,
}

const readonly = restrictTicketToolContext(base as any, ['read'])
const readonlyNames = buildToolSchemas(readonly.mcpTools, {
  disabledCapabilityIds: readonly.disabledCapabilityIds,
  disabledToolNames: readonly.disabledToolNames,
}).map((schema: any) => schema.function?.name)
for (const forbidden of ['write_file', 'run_command', 'web_search', 'browser_automation', 'email_read', 'tool_search', 'launch_subagent', 'delegate_task']) {
  assert.ok(!readonlyNames.includes(forbidden), `${forbidden} must be absent from a read-only ticket schema`)
}
for (const allowed of ['read_file', 'search', 'explore']) {
  assert.ok(readonlyNames.includes(allowed), `${allowed} remains available to a read-only ticket`)
}
assert.equal(readonly.callMcpTool, undefined)
assert.equal(readonly.runBrowserAutomation, undefined)
assert.equal(readonly.recordEmailDraft, undefined)
assert.equal(readonly.registerArtifact, undefined)
assert.equal(readonly.launchSubAgent, undefined)
assert.equal(readonly.allowSandboxNetwork, false)
assert.equal(readonly.sensitiveRetrievalAllowed, false)

const builder = restrictTicketToolContext(base as any, ['read', 'write', 'shell'])
const builderNames = buildToolSchemas(builder.mcpTools, { disabledToolNames: builder.disabledToolNames }).map((schema: any) => schema.function?.name)
assert.ok(builderNames.includes('write_file'))
assert.ok(builderNames.includes('run_command'))
assert.ok(!builderNames.includes('web_fetch'))
assert.ok(!builderNames.includes('email_read'))
assert.ok(!builderNames.includes('browser_automation'))
assert.equal(builder.allowSandboxNetwork, false, 'shell alone never gains task-sandbox network')
assert.equal(builder.networkCapabilityAllowed, false)
let remoteExecCalled = false
const blockedNetwork = await executeTool('run_command', { command: 'curl https://example.com' }, {
  cwd: '/remote/project',
  ...builder,
  backend: {
    kind: 'ssh', rootCwd: '/remote/project',
    exec: async () => { remoteExecCalled = true; return { code: 0, stdout: 'unexpected', stderr: '' } },
  } as any,
})
assert.equal(blockedNetwork.ok, false)
assert.match(blockedNetwork.output, /ticket capability policy/)
assert.equal(remoteExecCalled, false, 'SSH execution cannot bypass the ticket network lane')

assert.equal(ticketToolCapability('mcp__server__delete'), 'mcp')
assert.equal(ticketToolCapability('calendar_create'), 'connector')
assert.equal(ticketToolCapability('launch_subagent'), null)
assert.deepEqual(ticketDisallowedToolNames(['read_file', 'run_command', 'web_fetch'], ['read']), ['run_command', 'web_fetch'])

console.log('ticket tool capability policy tests passed')
