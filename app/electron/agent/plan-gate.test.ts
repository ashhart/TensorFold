// Run: npx jiti electron/agent/plan-gate.test.ts
import assert from 'node:assert/strict'
import {
  initPlanGate,
  proposePlan,
  decidePlan,
  editPlan,
  gateToolCall,
  recordExecution,
  canMutate,
  auditBindings,
} from './plan-gate'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ok ${name}`) }

console.log('plan-gate tests:')

// Phase 1: no plan → mutations blocked, reads allowed.
{
  const s = initPlanGate()
  assert.equal(s.phase, 'drafting')
  assert.equal(canMutate(s), false)
  assert.deepEqual(gateToolCall(s, { isMutation: false }), { allowed: true, reason: 'ok_read_only' })
  assert.deepEqual(gateToolCall(s, { isMutation: true }), { allowed: false, reason: 'blocked_no_plan' })
  assert.deepEqual(gateToolCall(s, { isMutation: true, isMeta: true }), { allowed: true, reason: 'ok_meta' })
  ok('drafting blocks mutations, allows reads and meta')
}

// Proposing a plan moves to awaiting_approval; mutations still blocked.
{
  let s = initPlanGate()
  s = proposePlan(s, [{ text: 'edit file' }, { text: 'run tests' }])
  assert.equal(s.phase, 'awaiting_approval')
  assert.equal(s.revision, 1)
  assert.equal(s.draft.length, 2)
  assert.equal(canMutate(s), false)
  assert.equal(gateToolCall(s, { isMutation: true }).reason, 'blocked_awaiting_approval')
  ok('proposePlan → awaiting_approval, mutations still blocked')
}

// Empty proposal is a no-op that stays in drafting (can't approve nothing).
{
  let s = proposePlan(initPlanGate(), [{ text: '   ' }, {}])
  assert.equal(s.phase, 'drafting')
  assert.equal(s.draft.length, 0)
  ok('empty proposal stays in drafting')
}

// Approve freezes intent → mutations unlock, verdict carries a binding.
{
  let s = proposePlan(initPlanGate(), [{ text: 'edit file' }])
  s = decidePlan(s, 'approve')
  assert.equal(s.phase, 'approved')
  assert.ok(s.frozen)
  assert.equal(s.frozen!.revision, 1)
  assert.match(s.frozen!.intentHash, /^[0-9a-f]{8}$/)
  assert.equal(canMutate(s), true)
  const v = gateToolCall(s, { isMutation: true })
  assert.equal(v.allowed, true)
  assert.equal(v.reason, 'ok_approved')
  assert.deepEqual(v.binding, { revision: 1, intentHash: s.frozen!.intentHash })
  ok('approve freezes intent and unlocks mutations with a binding')
}

// Reject blocks mutations; a fresh plan is required.
{
  let s = proposePlan(initPlanGate(), [{ text: 'edit file' }])
  s = decidePlan(s, 'reject')
  assert.equal(s.phase, 'rejected')
  assert.equal(canMutate(s), false)
  assert.equal(gateToolCall(s, { isMutation: true }).reason, 'blocked_rejected')
  ok('reject blocks mutations')
}

// Deciding with no pending draft is a no-op.
{
  const s = initPlanGate()
  assert.deepEqual(decidePlan(s, 'approve'), s)
  assert.deepEqual(decidePlan(s, 'reject'), s)
  ok('decide with no draft is a no-op')
}

// Editing an approved plan invalidates the freeze — intent must be re-approved.
{
  let s = proposePlan(initPlanGate(), [{ text: 'edit file' }])
  s = decidePlan(s, 'approve')
  const firstHash = s.frozen!.intentHash
  s = editPlan(s, [{ text: 'edit file' }, { text: 'and deploy' }])
  assert.equal(s.phase, 'awaiting_approval')
  assert.equal(s.frozen, null)
  assert.equal(s.revision, 2)
  assert.equal(canMutate(s), false)
  s = decidePlan(s, 'approve')
  assert.equal(s.phase, 'approved')
  assert.equal(s.frozen!.revision, 2)
  assert.notEqual(s.frozen!.intentHash, firstHash)
  ok('edit invalidates freeze until re-approved, bumps revision + hash')
}

// recordExecution binds a mutation to the frozen plan and appends to the audit trail.
{
  let s = proposePlan(initPlanGate(), [{ text: 'edit file' }])
  s = decidePlan(s, 'approve')
  const v = gateToolCall(s, { isMutation: true })
  s = recordExecution(s, { tool: 'write_file', binding: v.binding!, stepId: 'p0' })
  assert.equal(s.executions.length, 1)
  assert.equal(s.executions[0].tool, 'write_file')
  assert.equal(s.executions[0].boundRevision, 1)
  assert.equal(s.executions[0].stepId, 'p0')
  assert.deepEqual(auditBindings(s), [])
  ok('recordExecution binds mutation to frozen plan; audit clean')
}

// recordExecution throws when not approved.
{
  let s = proposePlan(initPlanGate(), [{ text: 'x' }])
  assert.throws(
    () => recordExecution(s, { tool: 'write_file', binding: { revision: 1, intentHash: 'deadbeef' } }),
    /no approved plan is frozen/,
  )
  ok('recordExecution throws with no approved plan')
}

// recordExecution rejects a stale binding (replay after an edit).
{
  let s = proposePlan(initPlanGate(), [{ text: 'edit file' }])
  s = decidePlan(s, 'approve')
  const stale = gateToolCall(s, { isMutation: true }).binding!
  // Edit + re-approve → binding now points at revision 1, but live is revision 2.
  s = editPlan(s, [{ text: 'edit file v2' }])
  s = decidePlan(s, 'approve')
  assert.throws(
    () => recordExecution(s, { tool: 'write_file', binding: stale }),
    /intent drift/,
  )
  ok('recordExecution rejects a stale (pre-edit) binding')
}

// Full two-phase happy path end to end.
{
  let s = initPlanGate()
  // Phase 1: draft while reading.
  assert.equal(gateToolCall(s, { isMutation: false }).allowed, true)
  s = proposePlan(s, [{ id: 'a', text: 'refactor' }, { id: 'b', text: 'test' }])
  // Gate holds mutations until decided.
  assert.equal(gateToolCall(s, { isMutation: true }).allowed, false)
  s = decidePlan(s, 'approve')
  // Phase 2: bind and run mutations.
  const v1 = gateToolCall(s, { isMutation: true })
  s = recordExecution(s, { tool: 'apply_patch', binding: v1.binding!, stepId: 'a' })
  const v2 = gateToolCall(s, { isMutation: true })
  s = recordExecution(s, { tool: 'run_command', binding: v2.binding!, stepId: 'b' })
  assert.equal(s.executions.length, 2)
  assert.ok(s.executions.every((e) => e.boundRevision === 1))
  assert.deepEqual(auditBindings(s), [])
  ok('full two-phase happy path: draft → approve → bound execution')
}

console.log(`\n${passed} plan-gate tests passed`)
