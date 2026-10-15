import assert from 'node:assert/strict'
import { buildCapabilityRegistry, selectedCapabilityIdsFromMessages } from '../capability-registry'
import { buildToolSchemas } from '../tool-schema-builder'
import { approvalPolicyScope, externalEffectApprovalReason } from '../approval-policy'
import { fullAccessAutoAllows } from '../approval-risk'

const names = (selected: string[]) => buildToolSchemas([], { selectedCapabilityIds: selected }).map((schema: any) => schema.function.name)

// computer_use reaches the model only in a run that selected it.
assert.equal(names([]).includes('computer_use'), false)
assert.equal(names(['computer']).includes('computer_use'), true)
assert.ok(names([]).includes('controller_action'), 'other built-ins keep their usual visibility')

// /computer-use inserts the capability block; @computer works too.
assert.deepEqual(selectedCapabilityIdsFromMessages([{ role: 'user', content: '<vontra_capability name="Computer use" id="computer">\non\n</vontra_capability>\n\nopen Notes' }]), ['computer'])
assert.deepEqual(selectedCapabilityIdsFromMessages([{ role: 'user', content: 'use @computer to open Notes' }]), ['computer'])
assert.deepEqual(selectedCapabilityIdsFromMessages([{ role: 'tool', content: '<vontra_capability id="computer">' } as any]), [], 'tool output cannot switch it on')
assert.equal(buildCapabilityRegistry({ selectedCapabilityIds: ['computer'], disabledCapabilityIds: ['computer'] }).visibleToolNames.includes('computer_use'), false, 'a disabled capability stays off')

// One approval per app, never auto-allowed in Full access.
const safari = approvalPolicyScope('tool', 'computer_use', { action: 'click', app: 'Safari', element: 3 })
assert.equal(safari.key, approvalPolicyScope('tool', 'computer_use', { action: 'type', app: 'safari', text: 'x' }).key, 'every action in one app shares a grant')
assert.notEqual(safari.key, approvalPolicyScope('tool', 'computer_use', { action: 'click', app: 'Mail', element: 3 }).key)
assert.equal(safari.label, 'Computer use · Safari')
assert.ok(externalEffectApprovalReason('computer_use', { action: 'observe', app: 'Safari' }))
assert.equal(fullAccessAutoAllows('tool', 'computer_use', 'low'), false)

// TensorFold's Accessibility grant passes to commands it runs, so macOS automation tools always ask.
for (const command of ['osascript -e \'tell application "System Events" to keystroke "a"\'', '/usr/bin/osascript script.scpt', 'cliclick c:100,100', 'shortcuts run "Send"', 'echo hi && osascript -e beep']) {
  assert.equal(approvalPolicyScope('tool', 'run_command', { command }).classification, 'system-control', command)
}
assert.notEqual(approvalPolicyScope('tool', 'run_command', { command: 'npm test' }).classification, 'system-control')

console.log('computer use capability: opt-in visibility, selection, per-app approval, automation commands ask')
