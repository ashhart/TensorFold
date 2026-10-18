import type { Thread } from '../agent/types'
import assert from 'node:assert/strict'
import { buildReadableExport, READABLE_EXPORT_LIMITS } from './format'
import { createReadableExportService } from './service'
import type { ReadableExportRequest } from '../../shared/readable-export'

const secret = 'sk-test-' + '0Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St'
const email = ['reader', 'example.invalid'].join('@')
const thread: any = { id: 'synthetic-thread', title: 'Synthetic conversation', updatedAt: 1, conversationRevision: 0, timeline: [
  { id: 'u1', kind: 'user', text: `Question for SyntheticProject ${email}.`, attachments: [{ path: '/synthetic/private.png' }] },
  { id: 'r1', kind: 'reasoning', text: 'Never share private reasoning.' },
  { id: 't1', kind: 'tool', args: { token: secret }, output: 'Never share tool output.' },
  { id: 'a1', kind: 'assistant', text: `Answer with [Source label](https://example.com/evidence) and password=${secret}.` },
  { id: 'u2', kind: 'user', text: 'A later independent question.' },
  { id: 'a2', kind: 'assistant', text: '</pre><script>throw 1</script><img src="https://example.com/leak">' },
] }
Object.defineProperty(thread, 'apiMessages', { get() { throw new Error('Raw model history must never be read.') } })
const request: ReadableExportRequest = { threadId: thread.id, format: 'markdown', scope: 'conversation', branch: 'active' }
const context = { terms: ['SyntheticProject'] }
for (const format of ['markdown', 'text', 'html'] as const) {
  const result = buildReadableExport(thread, { ...request, format }, context)
  assert.equal(result.messageCount, 4); assert.equal(result.redacted, true); assert.equal(result.anonymized, true)
  assert.doesNotMatch(result.body, /Never share|SyntheticProject|private\.png/)
  assert.ok(!result.body.includes(secret)); assert.ok(!result.body.includes(email))
  assert.match(result.body, /Source label/); assert.match(result.body, /not a restorable backup/)
  assert.ok(result.omissions.some((item) => item.kind === 'attachments' && item.count === 1))
  assert.ok(result.omissions.some((item) => item.kind === 'reasoning'))
  assert.ok(result.redactions.some((item) => item.kind === 'term'))
  assert.equal(result.bytes, Buffer.byteLength(result.body))
  if (format === 'html') {
    assert.match(result.body, /^<!doctype html>/); assert.match(result.body, /default-src &#39;none&#39;|default-src 'none'/)
    assert.match(result.body, /&lt;script&gt;/); assert.doesNotMatch(result.body, /<(?:script|img|iframe|a|link)\b/i)
  }
}
const selected = buildReadableExport(thread, { ...request, scope: 'selected', itemIds: ['a1', 'u1'] }, context)
assert.equal(selected.messages[0].role, 'user', 'Selection always uses preserved conversation order')
assert.equal(selected.messages[1].role, 'assistant'); assert.equal(selected.messageCount, 2)
assert.doesNotMatch(selected.body, /later independent/); assert.ok(selected.omissions.some((item) => item.kind === 'unselected' && item.count === 2))
for (const bad of [[], ['u1', 'u1'], ['missing'], ['r1'], ['t1']]) assert.throws(() => buildReadableExport(thread, { ...request, scope: 'selected', itemIds: bad }), /Select|selected message/)
assert.throws(() => buildReadableExport(thread, { ...request, branch: 'invented' as any }), /active conversation/)
assert.throws(() => buildReadableExport({ ...thread, timeline: [] }, request), /no user or assistant/)
assert.throws(() => buildReadableExport({ ...thread, timeline: [{ id: 'big', kind: 'user', text: 'x'.repeat(READABLE_EXPORT_LIMITS.sourceCharacters + 1) }] }, request), /limit/)

let clock = 100, scope = 'synthetic-profile', serial = 0, live: any = thread
const service = createReadableExportService({ readThread: () => live, scopeKey: () => scope, redactionContext: () => context, now: () => clock, newId: () => `synthetic-approval-${++serial}` })
const owner = 'synthetic-owner'
const preview = service.preview(request, owner)
const savedBody = preview.body; preview.body = 'Renderer-forged replacement.'
const claim = service.claim({ approvalId: preview.approvalId }, owner)
assert.throws(() => service.claim({ approvalId: preview.approvalId }, owner), /already used/)
const packet = service.finalize(claim.claimId, owner)
assert.equal(packet.body, savedBody, 'Only the main-owned reviewed bytes enter a native save')
assert.throws(() => service.finalize(claim.claimId, owner), /already completed/)
packet.assertCurrent(); thread.timeline[0].text += ' Changed during temp-file write.'
assert.throws(() => packet.assertCurrent(), /changed/, 'Atomic-save adapter must recheck before rename')
const stale = service.preview(request, owner); thread.timeline[0].text += ' Revision changed without timestamp.'
assert.throws(() => service.claim({ approvalId: stale.approvalId }, owner), /changed/)
const chooser = service.claim({ approvalId: service.preview(request, owner).approvalId }, owner)
thread.conversationRevision++
assert.throws(() => service.finalize(chooser.claimId, owner), /changed/, 'A native chooser cannot save an obsolete preview')
const canceledPreview = service.preview(request, owner), canceledClaim = service.claim({ approvalId: canceledPreview.approvalId }, owner)
service.cancel(canceledClaim.claimId)
assert.throws(() => service.finalize(canceledClaim.claimId, owner), /already completed/)
assert.throws(() => service.claim({ approvalId: canceledPreview.approvalId }, owner), /already used/)
const foreign = service.preview(request, owner)
assert.throws(() => service.claim({ approvalId: foreign.approvalId }, 'other-owner'), /another session/)
const otherProfile = service.preview(request, owner); scope = 'other-profile'
assert.throws(() => service.claim({ approvalId: otherProfile.approvalId }, owner), /another session/)
scope = 'synthetic-profile'
const expired = service.preview(request, owner); clock += 300_000
assert.throws(() => service.claim({ approvalId: expired.approvalId }, owner), /expired/)
const missing = service.preview(request, owner); live = null
assert.throws(() => service.claim({ approvalId: missing.approvalId }, owner), /changed/)
console.log('Readable formatter and one-use approval service pass real redaction, active/selected scope, passive HTML escaping, omissions, order, size bounds, owner/profile/TTL binding, duplicate/cancel, chooser and pre-rename stale races without reading raw model history.')

const selectedWorkspace = '/synthetic/customer-project'
const workspaceThread = { id: 'selected-workspace', title: 'Synthetic workspace example', createdAt: 1, updatedAt: 1, kind: 'chat', projectId: null, providerId: null, model: null, status: 'idle', apiMessages: [], cwd: selectedWorkspace, timeline: [{ id: 'workspace-line', kind: 'user', text: `Inspect ${selectedWorkspace}/private.txt` }] } as Thread
const workspaceService = createReadableExportService({ readThread: () => workspaceThread, scopeKey: () => 'synthetic-workspace-profile', redactionContext: thread => ({ workspace: thread.cwd }) })
const workspacePreview = workspaceService.preview({ threadId: workspaceThread.id, format: 'text', scope: 'conversation', branch: 'active' }, 'owner')
assert.ok(!workspacePreview.body.includes(selectedWorkspace), 'the actual selected conversation workspace is redacted even outside a home folder')
console.log('Selected-conversation workspace context is bound into readable-export redaction and revision checks')
