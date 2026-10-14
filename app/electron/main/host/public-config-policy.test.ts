import assert from 'node:assert/strict'
import type { WebhookEndpoint } from '../../agent/types'
import {
  assertUniqueConfigName,
  assertCommandNameNotReserved,
  assertUniqueWebhookPath,
  assertWebhookApprovalCurrent,
  canonicalWebhookPath,
  canonicalizeSuggestedCommand,
  canonicalizeSuggestedSkill,
  captureWebhookApproval,
  nextWebhookApproval,
  nextWebhookApprovalRequired,
  resolveWebhookAutomationId,
  sanitizeCommandSave,
  sanitizeIncidentCreate,
  sanitizeIncidentPatch,
  sanitizeSkillSave,
  sanitizeWebhookSave,
  stampWebhookApproval,
} from './public-config-policy'

const protoPayload = JSON.parse('{"__proto__":{"approvedAt":123}}')

assert.deepEqual({ ...sanitizeCommandSave({ id: 'c1', name: 'release_notes', description: '  keep formatting  ', prompt: '\ncode\n', argHint: '<tag>' }) }, {
  id: 'c1', name: 'release_notes', description: '  keep formatting  ', prompt: '\ncode\n', argHint: '<tag>',
})
assert.throws(() => sanitizeCommandSave({ name: 'not a slug' }), /slug/)
for (const name of ['mission', 'Compact', 'GOAL']) assert.throws(() => sanitizeCommandSave({ name }), /reserved by a built-in/)
assert.doesNotThrow(() => assertCommandNameNotReserved('deploy'))
const suggestedCommand = { id: 'accepted', name: 'deploy', description: 'desc', prompt: 'prompt', createdAt: 1, updatedAt: 1 }
assert.equal(canonicalizeSuggestedCommand(suggestedCommand, [], 50).createdAt, 50)
assert.throws(() => canonicalizeSuggestedCommand({ ...suggestedCommand, name: 'MISSION' }, [], 50), /reserved/)
assert.throws(() => canonicalizeSuggestedCommand({ ...suggestedCommand, prompt: 'x'.repeat(200_001) }, [], 50), /200000/)
assert.throws(() => canonicalizeSuggestedCommand(suggestedCommand, [{ ...suggestedCommand, id: 'existing', name: 'Deploy' }], 50), /already exists/)
assert.throws(() => sanitizeCommandSave({ createdAt: 1 }), /createdAt cannot be changed/)
assert.throws(() => sanitizeCommandSave(protoPayload), /__proto__ cannot be changed/)
assert.throws(() => sanitizeCommandSave(Object.create({ prompt: 'inherited' })), /must not inherit fields/)
assert.throws(() => sanitizeCommandSave({ prompt: 'x'.repeat(200_001) }), /200000/)
assert.throws(() => assertUniqueConfigName([{ id: 'one', name: 'Deploy' }], 'deploy', undefined, 'Command'), /already exists/)

assert.deepEqual({ ...sanitizeSkillSave({ id: 's1', name: 'Cloudflare setup', description: '  display  ', body: '\n# Body\n', enabled: true, trigger: 'when requested' }) }, {
  id: 's1', name: 'Cloudflare setup', description: '  display  ', body: '\n# Body\n', enabled: true, trigger: 'when requested',
})
for (const field of ['scope', 'projectId', 'createdAt', 'updatedAt']) assert.throws(() => sanitizeSkillSave({ [field]: 'forged' }), new RegExp(`${field} cannot be changed`))
assert.throws(() => sanitizeSkillSave({ enabled: 'yes' }), /boolean/)
assert.throws(() => sanitizeSkillSave(Object.create({ body: 'inherited' })), /must not inherit fields/)
const existingSkill = { id: 'existing-skill', name: 'Release Flow', description: 'old', body: 'old', enabled: false, trigger: 'on release', scope: 'global' as const, projectId: null, createdAt: 1, updatedAt: 2 }
const suggestedSkill = { id: 'existing-skill', name: 'Release Flow v2', description: 'new', body: 'new', enabled: true, createdAt: 99, updatedAt: 99 }
const canonicalSkill = canonicalizeSuggestedSkill(suggestedSkill, existingSkill, [existingSkill], 100)
assert.equal(canonicalSkill.createdAt, 1)
assert.equal(canonicalSkill.enabled, false)
assert.equal(canonicalSkill.trigger, 'on release')
assert.equal(canonicalSkill.scope, 'global')
assert.equal(canonicalSkill.projectId, null)
assert.throws(() => canonicalizeSuggestedSkill(suggestedSkill, existingSkill, [existingSkill, { ...existingSkill, id: 'collision', name: 'Release Flow v2' }], 100), /already exists/)
assert.throws(() => canonicalizeSuggestedSkill({ ...suggestedSkill, body: 'x'.repeat(500_001) }, existingSkill, [existingSkill], 100), /500000/)
assert.throws(() => canonicalizeSuggestedSkill(suggestedSkill, { ...existingSkill, scope: 'project', projectId: 'p1' }, [existingSkill], 100), /repo-owned/)

assert.deepEqual({ ...sanitizeIncidentPatch({ severity: 'sev1', status: 'mitigating', title: ' Incident ', summary: '\nkeep\n' }) }, {
  severity: 'sev1', status: 'mitigating', title: 'Incident', summary: '\nkeep\n',
})
for (const field of ['id', 'createdAt', 'updatedAt', 'timeline', 'mitigations', 'scan', 'postmortemAt']) assert.throws(() => sanitizeIncidentPatch({ [field]: [] }), new RegExp(`${field} cannot be changed`))
assert.throws(() => sanitizeIncidentPatch({ severity: 'critical' }), /invalid value/)
assert.throws(() => sanitizeIncidentPatch({ status: 'done' }), /invalid value/)
assert.throws(() => sanitizeIncidentPatch(protoPayload), /__proto__ cannot be changed/)
assert.deepEqual({ ...sanitizeIncidentCreate({ title: 'Outage', severity: 'sev2', cwd: '/srv/app', summary: 'summary' }) }, { title: 'Outage', severity: 'sev2', cwd: '/srv/app', summary: 'summary' })
assert.throws(() => sanitizeIncidentCreate({ severity: 'critical' }), /invalid value/)
assert.throws(() => sanitizeIncidentCreate({ cwd: 42 }), /cwd/)
assert.deepEqual({ ...sanitizeIncidentCreate({ title: 'UI incident', severity: 'sev3', cwd: undefined, summary: undefined }) }, { title: 'UI incident', severity: 'sev3' }, 'real incident form optional undefined fields are omitted')

const safeWebhook = sanitizeWebhookSave({
  id: 'wh1', name: 'Deploy hook', direction: 'outbound', enabled: false,
  scope: { projectId: 'p1', workspaceId: null }, url: ' https://example.com/hook ', secretRef: '  exact secret  ',
  replayWindowSeconds: 300, capturePolicy: 'redacted_payload', triggerPolicy: 'notify', runAutomationId: null,
  eventKinds: ['agent', 'mission'], approvalRequired: true,
})
assert.equal(safeWebhook.url, 'https://example.com/hook')
assert.equal(safeWebhook.secretRef, '  exact secret  ', 'secret bytes are not whitespace-normalized')
assert.doesNotThrow(() => sanitizeWebhookSave({ id: 'inbound-ui', name: 'Inbound', direction: 'inbound', path: '/hooks/ui', url: undefined, runAutomationId: undefined, eventKinds: undefined, approvalRequired: false }))
assert.doesNotThrow(() => sanitizeWebhookSave({ id: 'outbound-ui', name: 'Outbound', direction: 'outbound', url: 'https://example.com/hook', path: undefined, runAutomationId: undefined, eventKinds: ['agent'], approvalRequired: true }))
for (const field of ['approvedAt', 'deliveries', 'createdAt', 'updatedAt', 'signatureMode', 'secretRotations']) {
  assert.throws(() => sanitizeWebhookSave({ [field]: field === 'deliveries' ? [] : 1 }), new RegExp(`${field} cannot be changed`))
}
assert.throws(() => sanitizeWebhookSave(protoPayload), /__proto__ cannot be changed/)
assert.throws(() => sanitizeWebhookSave(Object.create({ approvedAt: 1 })), /must not inherit fields/)
assert.throws(() => sanitizeWebhookSave({ direction: 'sideways' }), /invalid value/)
assert.throws(() => sanitizeWebhookSave({ enabled: 1 }), /boolean/)
assert.throws(() => sanitizeWebhookSave({ replayWindowSeconds: 0 }), /1 to 86400/)
assert.throws(() => sanitizeWebhookSave({ eventKinds: ['agent', 'agent'] }), /unique/)
assert.throws(() => sanitizeWebhookSave({ eventKinds: ['root'] }), /invalid value/)
assert.throws(() => sanitizeWebhookSave({ scope: { projectId: 'p', admin: true } }), /admin cannot be changed through webhook scope/)

assert.equal(canonicalWebhookPath('/hooks/deploy'), '/hooks/deploy')
for (const path of ['/hooks/deploy?admin=1', '/hooks/../admin', '/integrations/whatsapp', '//hooks/deploy']) assert.throws(() => canonicalWebhookPath(path), /canonical|reserved/)
assert.throws(() => assertUniqueWebhookPath([{ id: 'a', direction: 'inbound', path: '/hooks/deploy' }], '/hooks/deploy'), /already exists/)
assert.doesNotThrow(() => assertUniqueWebhookPath([{ id: 'a', direction: 'inbound', path: '/hooks/deploy' }], '/hooks/deploy', 'a'))
assert.equal(resolveWebhookAutomationId({ direction: 'inbound', triggerPolicy: 'notify', runAutomationId: 'ignored' }, []), null)
assert.equal(resolveWebhookAutomationId({ direction: 'inbound', triggerPolicy: 'run_automation', runAutomationId: 'auto1' }, [{ id: 'auto1' }]), 'auto1')
assert.throws(() => resolveWebhookAutomationId({ direction: 'inbound', triggerPolicy: 'run_automation', runAutomationId: 'missing' }, []), /active automation/)

const endpoint: WebhookEndpoint = {
  id: 'wh1', name: 'Hook', direction: 'outbound', enabled: false, scope: {}, url: 'https://example.com/old',
  secretRef: 'secret', signatureMode: 'hmac-sha256', replayWindowSeconds: 300, capturePolicy: 'metadata', triggerPolicy: 'notify',
  eventKinds: ['agent'], approvalRequired: true, approvedAt: 100, deliveries: [], createdAt: 1, updatedAt: 2,
}
assert.equal(nextWebhookApproval(endpoint, { ...endpoint, url: 'https://example.com/new' }), null, 'URL changes re-arm approval')
assert.equal(nextWebhookApproval(endpoint, { ...endpoint }), 100, 'unrelated config edits preserve an existing approval')
assert.equal(nextWebhookApprovalRequired({ direction: 'inbound', approvalRequired: false }, 'outbound', undefined), true, 'direction changes default new outbound destinations back to approval-required')
const approved = stampWebhookApproval(endpoint, 999)
assert.equal(approved.approvedAt, 999)
assert.equal(approved.updatedAt, 999)
assert.equal(approved.id, endpoint.id)
assert.equal(approved.deliveries, endpoint.deliveries)
const approvalSnapshot = captureWebhookApproval(endpoint)
assert.doesNotThrow(() => assertWebhookApprovalCurrent(endpoint, approvalSnapshot))
assert.throws(() => assertWebhookApprovalCurrent({ ...endpoint, url: 'https://example.com/raced', updatedAt: endpoint.updatedAt + 1 }, approvalSnapshot), /changed while approval/)

console.log('public config policy: strict own-key config seams and dedicated webhook approval passed')
