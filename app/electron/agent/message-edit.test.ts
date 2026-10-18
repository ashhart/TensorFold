import assert from 'node:assert/strict'
import type { Thread } from './types'
import { apiUserMessageRef, createEditedTaskBranch, prepareThreadMessageEdit, stageThreadMessageEdit } from './message-edit'
import { buildUserContent } from './content'
import { captureBotConfig } from '../../shared/bot-policy'

function fixture(): Thread {
  return { id: 'source-fixture', title: 'Synthetic chat', kind: 'chat', cwd: '/synthetic-workspace', providerId: 'local', model: 'fixture', status: 'idle', createdAt: 1, updatedAt: 1,
    memoryPolicy: { useExisting: false, contribute: false }, memoryPolicyRevision: 7,
    apiMessages: [{ role: 'system', content: 'system' }, { role: 'user', content: 'Original question' }, { role: 'assistant', content: null, tool_calls: [{ id: 't1', type: 'function', function: { name: 'read_file', arguments: '{}' } }] }, { role: 'tool', tool_call_id: 't1', content: 'original tool result' }, { role: 'assistant', content: 'Original answer' }, { role: 'user', content: 'Later question' }, { role: 'assistant', content: 'Later answer' }],
    timeline: [{ id: 'u1', kind: 'user', text: 'Original question', memorySource: 'user' }, { id: 't1', kind: 'tool', tool: 'read_file', output: 'original tool result', state: 'ok' }, { id: 'a1', kind: 'assistant', text: 'Original answer' }, { id: 'u2', kind: 'user', text: 'Later question' }, { id: 'a2', kind: 'assistant', text: 'Later answer' }],
  }
}
let thread = fixture()
const before = structuredClone(thread)
const prepare = prepareThreadMessageEdit(thread, 'u1')
assert.equal(prepare.affectedItems, 4)
assert.equal(prepare.mode, 'replace-chat')
assert.deepEqual(thread, before, 'preparing/cancelling an edit mutates neither history')
const input = { ...prepare, threadId: thread.id, text: 'Revised question' }
const patch = stageThreadMessageEdit(thread, input, 2)
assert.deepEqual(thread, before, 'staging is detached')
assert.deepEqual(patch.apiMessages?.map((message) => message.content), ['system', 'Revised question'])
assert.deepEqual(patch.timeline?.filter((item) => item.kind !== 'notice').map((item) => item.text), ['Revised question'])
assert.equal(patch.conversationRevision, 1)
assert.deepEqual(patch.messageEditRevisions?.[0].timeline, before.timeline)
assert.deepEqual(patch.messageEditRevisions?.[0].apiMessages, before.apiMessages.slice(1))
assert.match(patch.timeline!.at(-1)!.text!, /not undone/)
assert.equal(patch.lastRunReceipt, null)
assert.equal(patch.diffReviewedAt, null)
assert.deepEqual(stageThreadMessageEdit(thread, { ...input, text: prepare.expectedMessageText }), {})
for (const stale of [{ ...input, expectedConversationRevision: 2 }, { ...input, expectedHistoryToken: 'wrong' }, { ...input, expectedMessageText: 'wrong' }]) assert.throws(() => stageThreadMessageEdit(thread, stale), /conversation changed/)
thread.timeline.push({ id: 'late', kind: 'notice', text: 'New state' })
assert.throws(() => stageThreadMessageEdit(thread, input), /conversation changed/, 'same-epoch later appends invalidate preparation')
thread = fixture()
assert.throws(() => stageThreadMessageEdit(thread, { ...input, text: '' }), /Enter a message/)
assert.throws(() => prepareThreadMessageEdit({ ...thread, status: 'running' }, 'u1'), /Stop the active run/)
assert.throws(() => prepareThreadMessageEdit({ ...thread, orchestratorHostId: 'remote' }, 'u1'), /owning host/)
assert.throws(() => prepareThreadMessageEdit({ ...thread, claudeResumeCursor: { sessionId: 'owned-by-cli' } }, 'u1'), /separate session history/)
assert.throws(() => prepareThreadMessageEdit({ ...thread, compactions: [{}] as any }, 'u1'), /cannot be matched safely/)
assert.throws(() => prepareThreadMessageEdit({ ...thread, apiMessages: thread.apiMessages.slice(2) }, 'u1'), /cannot be matched safely/)
assert.throws(() => prepareThreadMessageEdit(thread, 'a1'), /your own sent messages/)
thread.timeline[0].memorySource = 'external'
assert.throws(() => prepareThreadMessageEdit(thread, 'u1'), /your own sent messages/)
thread = fixture()
const attachment = { id: 'doc', kind: 'document' as const, name: 'sample.txt', mime: 'text/plain', text: 'Synthetic attached text' }
thread.timeline[0].attachments = [attachment]
thread.apiMessages[1].content = buildUserContent('Original question', [attachment])
thread.timeline[0].apiMessageRef = apiUserMessageRef(thread.apiMessages.slice(0, 1), thread.apiMessages[1].content, 0)
let attachmentPatch = stageThreadMessageEdit(thread, { threadId: thread.id, ...prepareThreadMessageEdit(thread, 'u1'), text: 'Changed with attachment' })
assert.match(String(attachmentPatch.apiMessages![1].content), /Changed with attachment[\s\S]*sample.txt[\s\S]*Synthetic attached text/)
assert.deepEqual(attachmentPatch.timeline![0].attachments, [attachment])
assert.throws(() => prepareThreadMessageEdit({ ...thread, compactions: [{}] as any }, 'u1'), /predates a history compaction/)
// A leading system prompt is replaceable without invalidating the user ordinal.
thread.apiMessages.unshift({ role: 'system', content: 'Current host instructions' })
assert.equal(prepareThreadMessageEdit(thread, 'u1').expectedMessageText, 'Original question')
thread = fixture(); thread.kind = 'task'; thread.incognito = true; thread.phoneLocalOnly = true
thread.goal = 'Keep files'; thread.goalStatus = 'active'; thread.plan = [{ id: 'p', text: 'Check current state', done: true }]
thread.workspace = { kind: 'local', path: '/synthetic-workspace' }
thread.bot = captureBotConfig({ id: 'fixture-bot', name: 'Limited fixture', role: 'Inspect synthetic data', lane: 'review', instructions: 'Synthetic', providerId: 'local', model: 'fixture', toolScope: { mode: 'selected', names: [] }, approvalScope: 'readonly', executionTarget: 'workspace', createdAt: 1, updatedAt: 1 } as any, [{ id: 'local', name: 'Synthetic', baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }], 1)
const taskBefore = structuredClone(thread)
const taskPrepare = prepareThreadMessageEdit(thread, 'u1')
assert.equal(taskPrepare.mode, 'branch-task')
const taskPatch = stageThreadMessageEdit(thread, { threadId: thread.id, ...taskPrepare, text: 'Edited task request' }, 3)
const branch = createEditedTaskBranch(thread, taskPatch, 'u1', 3)
assert.notEqual(branch.id, thread.id, 'new task checkpoint identity')
assert.equal(branch.messageEditSource?.threadId, thread.id)
assert.equal(branch.parentId, thread.id)
assert.equal(branch.incognito, true); assert.equal(branch.phoneLocalOnly, true)
assert.deepEqual(branch.memoryPolicy, thread.memoryPolicy)
assert.deepEqual(branch.workspace, thread.workspace)
assert.deepEqual(branch.plan, thread.plan)
assert.deepEqual(branch.bot, thread.bot)
assert.notEqual(branch.bot, thread.bot)
assert.ok(Object.isFrozen(branch.bot) && Object.isFrozen(branch.bot!.toolScope) && Object.isFrozen(branch.bot!.toolScope!.names))
assert.equal(branch.goal, thread.goal)
assert.equal(branch.apiMessages.at(-1)?.content, 'Edited task request')
assert.deepEqual(thread, taskBefore, 'original task, later messages and checkpoint identity remain intact')
assert.match(branch.timeline.at(-1)!.text!, /current workspace files/)
console.log('Sent edits: exact API/timeline cut, preserved later branches, stale/compacted mapping refusal, unchanged cancel, attachments and independent task checkpoint identity passed')
