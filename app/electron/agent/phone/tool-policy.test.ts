import assert from 'node:assert/strict'
import { approvalPolicyScope, externalEffectApprovalReason, requiresFreshConnectorApproval } from '../approval-policy'
import { buildChatToolSchemas, isReadOnlyTool } from '../tool-schema-builder'
import { executeTool } from '../tools'

const names = (selected: string[]) => buildChatToolSchemas({ selectedCapabilityIds: selected }).map((schema) => schema.function?.name)
assert.ok(!names([]).includes('phone_control'), 'phone control is opt-in instead of available to every run')
assert.ok(names(['phone']).includes('phone_control'))
assert.ok(names(['phone']).includes('phone_observe'), 'plain chat receives inspection and mutation tools after its grant')
assert.equal(isReadOnlyTool('phone_observe'), true)
assert.equal(isReadOnlyTool('phone_control'), false)
assert.match(externalEffectApprovalReason('phone_control') ?? '', /requires approval/)
const policy = approvalPolicyScope('tool', 'phone_control', { action: 'tap', sessionId: 'one', observationId: 'screen-one', x: 20, y: 20 })
assert.equal(requiresFreshConnectorApproval(policy), true)
assert.notEqual(policy.key, approvalPolicyScope('tool', 'phone_control', { action: 'tap', sessionId: 'one', observationId: 'screen-two', x: 20, y: 20 }).key, 'approval cannot follow coordinates onto another phone screen')
let calls = 0
const base = { cwd: '/example/workspace', approvalMode: 'suggest', runPhoneControl: async () => { calls++; return { ok: true, output: 'Current phone labels', title: 'Phone observe' } } } as any
assert.equal((await executeTool('phone_observe', {}, base)).ok, false)
assert.equal(calls, 0)
assert.equal((await executeTool('phone_observe', {}, { ...base, phoneControlAvailable: true })).ok, true)
assert.equal(calls, 1)
assert.equal((await executeTool('phone_control', { action: 'shell', command: 'unexpected' }, { ...base, phoneControlAvailable: true })).ok, false)
assert.equal(calls, 1, 'an unknown operation never reaches a device provider')
console.log('Phone tools stay opt-in, local, typed and approval-gated.')
