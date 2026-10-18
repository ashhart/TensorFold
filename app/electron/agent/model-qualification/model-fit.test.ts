import assert from 'node:assert/strict'
import { MODEL_PROBE_REVISION, currentModelQualification, eligibleHelperRoute, emptyModelChecks, helperRouterRole, modelQualificationKey, modelTaskFit } from '../../../shared/model-fit'
import { modelPickerChoices } from '../../../src/lib/model-picker'
import type { ModelQualificationReport } from '../../../shared/model-fit'
import type { Provider } from '../types'

const local: Provider = { id: 'local', name: 'Local fixture', kind: 'local', baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
const external: Provider = { id: 'external', name: 'External fixture', kind: 'cloud', baseUrl: 'https://example.invalid/v1', apiKey: '', models: ['fixture'], accounts: [{ id: 'other', name: 'Other account', kind: 'api-key', apiKey: '', models: ['fixture'], enabled: true, createdAt: 1, updatedAt: 1 }] }
const route = { providerId: local.id, accountId: null, model: 'fixture' }
const checks = emptyModelChecks()
checks.text = { state: 'observed', message: 'Synthetic text observed.' }
checks.tools = { state: 'observed', message: 'Synthetic tools observed.' }
checks.vision = { state: 'failed', message: 'Synthetic image was not understood.' }
const report: ModelQualificationReport = { ...route, key: modelQualificationKey(route), identity: 'a'.repeat(64), requestId: 'synthetic', requestedChecks: ['text', 'tools', 'vision'], probeRevision: MODEL_PROBE_REVISION, revision: 1, checkedAt: 100, state: 'complete', checks }
const unknown = modelTaskFit({ local: true, requirements: { images: true, tools: true } })
assert.deepEqual(unknown.requested, ['vision', 'tools'])
assert.deepEqual(unknown.unknown, ['vision', 'tools'])
assert.match(unknown.labels.join(' '), /Images: unknown/)
assert.equal(unknown.disabledReason, null, 'Unknown is labelled without a fabricated failure')
assert.match(modelTaskFit({ local: true, requirements: { images: true }, report }).disabledReason || '', /vision/)
assert.match(modelTaskFit({ local: false, requirements: { localOnly: true }, report }).disabledReason || '', /external fallback is unavailable/)
assert.match(modelTaskFit({ local: true, requirements: { tools: true, toolsDisabled: true }, report }).disabledReason || '', /permission policy/)
assert.equal(currentModelQualification([report], { ...route, accountId: 'other' }), null)
assert.equal(currentModelQualification([{ ...report, probeRevision: 0 }], route), null)
const choices = modelPickerChoices([local, external], null, { requirements: { images: true }, qualifications: [report] })
assert.match(choices.find(choice => choice.provider.id === local.id)?.disabledReason || '', /vision/)
assert.equal(choices.find(choice => choice.provider.id === external.id && choice.accountId === null)?.capabilityCheckedAt, undefined, 'Another provider/account cannot borrow a qualification')
const tools = modelPickerChoices([local, external], null, { requirements: { tools: true }, qualifications: [report] })
assert.equal(tools.find(choice => choice.provider.id === local.id)?.disabledReason, null)
const planner = tools.find(choice => choice.provider.id === external.id)!
assert.equal(planner.disabledReason, null, 'External planning stays selectable without acquiring local tools')
assert.match(planner.detail, /Planning only/)
assert.ok(!planner.capabilityLabels?.some(label => label.startsWith('Tools:') && label.includes('required')))
assert.match(modelTaskFit({ local: false, requirements: { tools: true }, toolPolicyAllowed: false }).disabledReason || '', /permission policy/, 'Actual tool execution remains forbidden for a planning route')
const candidates = [{ route, local: true, available: false, report }, { route: { providerId: external.id, accountId: null, model: 'fixture' }, local: false, available: true, report }]
assert.equal(eligibleHelperRoute({ role: 'document_extraction', preferred: route, candidates, requirements: { localOnly: true } }).route, null, 'Missing local helper never substitutes a cloud route')
assert.equal(eligibleHelperRoute({ role: 'search', candidates, requirements: {} }).route, null, 'Deterministic search needs no model default')
assert.equal(helperRouterRole('naming'), 'summarizer'); assert.equal(helperRouterRole('document_extraction'), 'summarizer'); assert.equal(helperRouterRole('search'), 'scout')
candidates[0].available = true
assert.deepEqual(eligibleHelperRoute({ role: 'naming', preferred: route, candidates, requirements: { localOnly: true } }).route, route)
assert.equal(eligibleHelperRoute({ role: 'document_extraction', preferred: route, candidates, requirements: { images: true, localOnly: true } }).route, null)
for (const privateContext of ['bot', 'phone', 'google'] as const) assert.equal(eligibleHelperRoute({ role: 'naming', preferred: candidates[1].route, candidates, requirements: { privateContext } }).route, null)
console.log('PASS model fit: exact account and probe revision, honest unknowns, observed capability requirements, permission/privacy filters, helper job roles and no external substitution')
