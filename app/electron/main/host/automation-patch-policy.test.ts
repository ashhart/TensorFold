import assert from 'node:assert/strict'
import { sanitizeAutomationCreateInput, sanitizeAutomationUpdatePatch, validateAutomationProviderRoute } from './automation-patch-policy'

const current = {
  id: 'a1', name: 'Daily', prompt: 'work', scheduleKind: 'daily', everyMinutes: 60, atHour: 9, atMinute: 0,
  providerId: 'provider', providerAccountId: 'account', model: 'model', mode: 'agent', cwd: '/repo', deliverTo: 'inbox', enabled: true,
  lastRunAt: null, nextRunAt: null, lastStatus: null, runs: [], completionPolicy: { type: 'max_iterations', version: 4, max: 3 },
  createdAt: 1, updatedAt: 1,
} as any

assert.deepEqual(sanitizeAutomationCreateInput({ name: 'Safe', enabled: false }), { name: 'Safe', enabled: false })
for (const forbidden of ['id', 'runs', 'lastRunAt', 'nextRunAt', 'lastStatus', 'createdAt', 'updatedAt', 'archivedAt']) {
  assert.throws(() => sanitizeAutomationCreateInput({ [forbidden]: 'forged' }), /cannot be changed/)
}
assert.throws(() => sanitizeAutomationUpdatePatch(current, JSON.parse('{"__proto__":{"runs":[]}}')), /__proto__ cannot be changed/)
assert.throws(() => sanitizeAutomationUpdatePatch(current, { scheduleKind: 'whenever' }), /scheduleKind is invalid/)
assert.throws(() => sanitizeAutomationUpdatePatch(current, { atHour: 24 }), /atHour/)
assert.throws(() => sanitizeAutomationUpdatePatch(current, { requiredRisks: ['ok', 42] }), /risk identifiers/)

const unchanged = sanitizeAutomationUpdatePatch(current, { completionPolicy: { type: 'max_iterations', version: 999, max: 3 } })
assert.equal(unchanged.completionPolicy?.version, 4, 'caller cannot forge the epoch and an unchanged contract preserves it')
const changed = sanitizeAutomationUpdatePatch(current, { completionPolicy: { type: 'max_iterations', version: 4, max: 5 } })
assert.equal(changed.completionPolicy?.version, 5, 'host increments the completion epoch when the contract changes')
const cleared = sanitizeAutomationUpdatePatch(current, { completionPolicy: undefined })
assert.deepEqual(cleared.completionPolicy, { type: 'none', version: 5 }, 'clearing a policy remains a host-versioned contract change')
assert.throws(() => sanitizeAutomationUpdatePatch(current, { completionPolicy: { type: 'none', version: 4, prompt: 'smuggled' } }), /prompt is not supported/)

const providers = [{ id: 'provider', models: ['model'], accounts: [{ id: 'account', models: ['account-model'] }] }] as any
assert.doesNotThrow(() => validateAutomationProviderRoute({ providerId: 'provider', providerAccountId: null, model: 'model' }, providers))
assert.doesNotThrow(() => validateAutomationProviderRoute({ providerId: 'provider', providerAccountId: 'account', model: 'account-model' }, providers))
assert.throws(() => validateAutomationProviderRoute({ providerId: 'missing', providerAccountId: null, model: null }, providers), /configured automation provider/)
assert.throws(() => validateAutomationProviderRoute({ providerId: 'provider', providerAccountId: 'wrong', model: null }, providers), /belongs to the automation provider/)
assert.throws(() => validateAutomationProviderRoute({ providerId: 'provider', providerAccountId: 'account', model: 'model' }, providers), /offered by/)

console.log('automation patch policy: config-only allowlist, host-owned policy epochs, and provider routes validated')
