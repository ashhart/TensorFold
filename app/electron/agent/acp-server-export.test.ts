// Run: jiti electron/agent/acp-server-export.test.ts
import assert from 'node:assert/strict'
import {
  acpError,
  acpResult,
  acpSessionIdForTarget,
  buildAcpInitializeResult,
  buildAcpNewSessionResult,
  exportedAcpMethods,
} from './acp-server-export'

const target = {
  kind: 'mission' as const,
  id: 'mission_123',
  title: 'Ship review loop',
  cwd: '/repo',
  description: 'A TensorFold mission exported to an ACP client.',
}

assert.equal(acpSessionIdForTarget(target), 'vontra:mission:mission_123')

const initialized = buildAcpInitializeResult(target)
assert.equal(initialized.protocolVersion, 1)
assert.equal(initialized.agentInfo.name, 'TensorFold mission')
assert.equal(initialized.agentCapabilities.loadSession, true)
assert.equal(initialized.agentCapabilities.promptCapabilities.embeddedContext, true)
assert.equal(initialized._meta.vontra.targetId, target.id)

const session = buildAcpNewSessionResult(target)
assert.equal(session.sessionId, 'vontra:mission:mission_123')
assert.equal(session.sessionMode.currentModeId, 'default')
assert.equal(session.sessionMode.availableModes[0].name, 'TensorFold')

assert.deepEqual(acpResult('1', initialized), { jsonrpc: '2.0', id: '1', result: initialized })
assert.deepEqual(acpError('2', -32601, 'Method not found'), {
  jsonrpc: '2.0',
  id: '2',
  error: { code: -32601, message: 'Method not found' },
})

for (const method of ['initialize', 'session/new', 'session/load', 'session/prompt', 'session/cancel']) {
  assert.ok(exportedAcpMethods().includes(method))
}

console.log('acp-server-export tests passed')
