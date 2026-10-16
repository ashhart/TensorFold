// Run: jiti electron/agent/acp-interop.test.ts
import assert from 'node:assert/strict'
import { acpBoundaryIds, acpFirstImplementation, buildAcpExplorationSpike } from './acp-interop'

const spike = buildAcpExplorationSpike()

assert.equal(spike.sourceUrl, 'https://github.com/agentclientprotocol/agent-client-protocol')
assert.equal(spike.protocolVersion, 1)
assert.equal(spike.transport, 'json-rpc-stdio')
assert.equal(spike.firstRole, 'client')
assert.deepEqual(spike.laterRoles, ['server'])
assert.match(acpFirstImplementation(spike), /ACP client first/)

const boundaries = acpBoundaryIds(spike)
for (const id of ['auth', 'model_selection', 'tools', 'policy', 'filesystem', 'timeline']) {
  assert.ok(boundaries.includes(id), `missing ACP boundary: ${id}`)
}

assert.deepEqual(spike.prototypes.slice(0, 4), [
  'host_external_agent',
  'timeline_mapping',
  'approval_bridge',
  'cancel_guide_audit',
])
assert.ok(spike.prototypes.includes('server_export'))
assert.ok(spike.boundaries.find((boundary) => boundary.id === 'policy')?.rule.includes('TensorFold policy'))
assert.ok(spike.boundaries.find((boundary) => boundary.id === 'timeline')?.rule.includes('timeline'))

console.log('acp-interop tests passed')
