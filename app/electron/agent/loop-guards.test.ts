import assert from 'node:assert/strict'
import { toolEvidenceVerdict, roleBoundaryViolation } from './loop-guards'

// tool-evidence: zero useful tool events -> not ok
assert.equal(toolEvidenceVerdict([]).ok, false)
assert.equal(toolEvidenceVerdict(['read:foo.ts']).ok, true)
// .pi-loop / .vontra-loop bookkeeping reads do NOT count as useful evidence
assert.equal(toolEvidenceVerdict(['read:.vontra-loop/state.json']).ok, false)

// role-boundary: read-only role (architect) cannot edit source
assert.ok(roleBoundaryViolation('architect', 'edit', { path: 'src/app.ts' }, '/repo'))
assert.equal(roleBoundaryViolation('architect', 'read', { path: 'src/app.ts' }, '/repo'), undefined)
// dev role can edit
assert.equal(roleBoundaryViolation('dev', 'edit', { path: 'src/app.ts' }, '/repo'), undefined)
// validator role (qa) may write to a temp path but not source
assert.equal(roleBoundaryViolation('qa', 'bash', { command: 'mktemp -d && echo hi > /tmp/x' }, '/repo'), undefined)
assert.ok(roleBoundaryViolation('qa', 'bash', { command: 'rm -rf src' }, '/repo'))

console.log('loop-guards tests passed')
