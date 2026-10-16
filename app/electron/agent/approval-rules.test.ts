import assert from 'node:assert/strict'
import { approvalRuleIsActive, boundedApprovalRuleFields, consumeApprovalRule } from './approval-rules'
import type { ApprovalRule } from './types'

const now = 1_000_000
const base: ApprovalRule = {
  id: 'rule-1', scope: 'tool', key: 'run:test', decision: 'allow', duration: 'always',
  projectId: 'project-1', workspace: '/repo', expiresAt: now + 10_000,
  maxUses: 2, useCount: 0, createdAt: now, updatedAt: now, lastUsedAt: null,
}

assert.equal(approvalRuleIsActive(base, { projectId: 'project-1', workspace: '/repo' }, now), true)
assert.equal(approvalRuleIsActive(base, { projectId: 'project-2', workspace: '/repo' }, now), false, 'project scope cannot cross projects')
assert.equal(approvalRuleIsActive(base, { projectId: 'project-1', workspace: '/other' }, now), false, 'workspace scope cannot cross workspaces')
assert.equal(approvalRuleIsActive({ ...base, expiresAt: now }, { projectId: 'project-1', workspace: '/repo' }, now), false, 'expired rule is inactive')
assert.equal(approvalRuleIsActive({ ...base, useCount: 2 }, { projectId: 'project-1', workspace: '/repo' }, now), false, 'exhausted rule is inactive')
assert.equal(approvalRuleIsActive({ ...base, projectId: null, workspace: null }, {}, now), false, 'legacy unscoped rule is inactive')
assert.equal(approvalRuleIsActive({ ...base, expiresAt: undefined, maxUses: undefined, useCount: undefined }, { projectId: 'project-1', workspace: '/repo' }, now), false, 'legacy unbounded rule is inactive')

const first = consumeApprovalRule([base], { scope: 'tool', key: 'run:test', projectId: 'project-1', workspace: '/repo', now: now + 1 })
assert.equal(first.rule?.useCount, 1)
assert.equal(base.useCount, 0, 'consumption is immutable and can be atomically swapped into the store')
const second = consumeApprovalRule(first.rules, { scope: 'tool', key: 'run:test', projectId: 'project-1', workspace: '/repo', now: now + 2 })
assert.equal(second.rule?.useCount, 2, 'last permitted use is consumed')
const exhausted = consumeApprovalRule(second.rules, { scope: 'tool', key: 'run:test', projectId: 'project-1', workspace: '/repo', now: now + 3 })
assert.equal(exhausted.rule, null, 'cap is enforced after atomic consumption')

const fields = boundedApprovalRuleFields({ projectId: 'p', workspace: '/repo' }, now)
assert.equal(fields.useCount, 0)
assert.ok(Number(fields.maxUses) > 0 && Number(fields.maxUses) <= 100)
assert.ok(Number(fields.expiresAt) > now && Number(fields.expiresAt) <= now + 24 * 60 * 60_000)
assert.throws(() => boundedApprovalRuleFields({}, now), /project or workspace scope/)

console.log('bounded scoped approval rule tests passed')
