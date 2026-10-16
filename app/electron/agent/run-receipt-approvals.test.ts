import assert from 'node:assert/strict'
import { buildAgentRunContract, buildAgentRunReceipt } from './run-contract'
import type { Thread, TimelineItem } from './types'

function receipt(items: TimelineItem[]) {
  const thread: Thread = {
    id: 'receipt-approval', title: 'Check approved work', kind: 'task', projectId: null,
    cwd: '/example/project', providerId: null, model: null, status: 'idle',
    createdAt: 1000, updatedAt: 2000, apiMessages: [], timeline: items,
  }
  const contract = buildAgentRunContract({ thread, runId: 'receipt-run', objective: 'Check approved work', provider: null, model: null, approvalMode: 'suggest', createdAt: 1000 })
  return buildAgentRunReceipt({ thread, contract, timelineStart: 0, finishedAt: 2000 })
}

const approved = receipt([
  { id: 'edit', kind: 'tool', tool: 'edit_file', path: 'example.ts', state: 'ok', approval: 'Approved once.', createdAt: 1100 },
  { id: 'test', kind: 'tool', tool: 'run_command', command: 'npm test', state: 'ok', approval: 'Approved once.', createdAt: 1200 },
])
assert.equal(approved.verdict, 'verified', 'completed approvals must not leave a successful checked run blocked')
assert.equal(approved.approvals, 2, 'retain the count of approval decisions for the audit trail')
assert.deepEqual(approved.gaps, [])

const waiting = receipt([{ id: 'wait', kind: 'tool', tool: 'write_file', state: 'awaiting', path: 'example.ts', createdAt: 1100 }])
assert.equal(waiting.verdict, 'blocked')
assert.ok(waiting.gaps.includes('approval pending'))

const denied = receipt([{ id: 'deny', kind: 'tool', tool: 'write_file', state: 'denied', path: 'example.ts', approval: 'Denied by user.', createdAt: 1100 }])
assert.equal(denied.verdict, 'blocked', 'denied work is a completed decision, but still not successful work')
assert.ok(denied.gaps.includes('action denied'))
assert.ok(!denied.gaps.includes('approval pending'), 'a denial must not be described as waiting for another approval')
assert.deepEqual(denied.changedFiles, [], 'denied writes must not claim any files were changed')
assert.deepEqual(waiting.changedFiles, [], 'pending writes must not claim any files were changed')

const failed = receipt([{ id: 'fail', kind: 'tool', tool: 'run_command', command: 'npm test', state: 'error', approval: 'Approved once.', createdAt: 1100 }])
assert.equal(failed.verdict, 'failed', 'approval history must not hide the actual tool failure')
console.log('run receipt approval settlement: ok')
