import assert from 'node:assert/strict'
import { createApprovalRuntime } from './approval-runtime'
import { approvalPolicyScope } from '../../agent/approval-policy'

const pendingApprovals = new Map<string, (decision: 'allow' | 'allow_always' | 'deny') => void>()
const pendingApprovalMeta = new Map<string, any>()
let lookups = 0
const runtime = createApprovalRuntime({
  pendingApprovals, pendingApprovalMeta, emitJob() {}, emitToRenderer() {}, patchJob: (job: any) => job, persistJobs() {}, persistThreads() {},
  findApprovalRule: async () => { lookups++; return { decision: 'allow' as const } },
})
pendingApprovalMeta.set('phone-call', { threadId: 'synthetic-local-chat', itemId: 'synthetic-phone-item', tool: 'phone_control', policy: approvalPolicyScope('tool', 'phone_control', { action: 'home', sessionId: 'synthetic-session', observationId: 'synthetic-screen' }) })
const controller = new AbortController()
let resolved = false
const decision = runtime.approvalResolver(controller)('phone-call').then((value) => { resolved = true; return value })
await Promise.resolve()
assert.equal(lookups, 0, 'saved rules never auto-resolve a new phone input')
assert.equal(resolved, false)
assert.equal(pendingApprovals.has('phone-call'), true, 'each phone mutation reaches the live approval UI')
pendingApprovals.get('phone-call')!('allow')
assert.equal(await decision, 'allow')
console.log('Phone mutations always require a fresh interactive approval, even with saved allow rules.')
