import assert from 'node:assert/strict'
import { pickStrictOwnPatch, sanitizeProjectCreateInput, sanitizeProjectUpdatePatch, validateProjectExecutionTargetInput, validateProjectProviderDefaults } from './project-patch-policy'

const safe = sanitizeProjectUpdatePatch({ name: 'Renamed', memory: 'context', tags: ['work'] })
assert.deepEqual(safe, { name: 'Renamed', memory: 'context', tags: ['work'] })
assert.throws(() => sanitizeProjectUpdatePatch({ id: 'replacement' }), /id cannot be changed/)
assert.throws(() => sanitizeProjectUpdatePatch({ cwd: '/tmp/other' }), /cwd cannot be changed/)
assert.throws(() => sanitizeProjectUpdatePatch({ executionTarget: 'orchestrator' }), /executionTarget cannot be changed/)
assert.throws(() => sanitizeProjectUpdatePatch({ localOnly: 'yes' }), /must be a boolean/)
assert.throws(() => sanitizeProjectUpdatePatch({ kanban: { modelMode: 'anything' } }), /single or phased/)
assert.throws(() => sanitizeProjectUpdatePatch({ kanban: { single: { providerId: 'x'.repeat(1_001) } } }), /no longer than 1,000/)
assert.throws(() => sanitizeProjectUpdatePatch({ kanban: JSON.parse('{"__proto__":{"modelMode":"phased"}}') }), /__proto__ cannot be changed/)
assert.deepEqual(sanitizeProjectCreateInput({ name: 'New', cwd: '/repo', mode: 'solo' }), { name: 'New', cwd: '/repo', mode: 'solo', workspace: undefined })
assert.throws(() => sanitizeProjectCreateInput({ id: 'chosen', name: 'New' }), /cannot choose this id/)
assert.equal(sanitizeProjectCreateInput({ id: 'proxy', name: 'Remote', createdAt: 1 }, { allowId: true, allowCreatedAt: true }).id, 'proxy')
assert.throws(() => sanitizeProjectCreateInput({ updatedAt: 1 }), /updatedAt cannot be changed/)
assert.throws(() => sanitizeProjectCreateInput(JSON.parse('{"__proto__":{"archivedAt":1}}')), /__proto__ cannot be changed/)
assert.throws(() => sanitizeProjectCreateInput({ workspace: { kind: 'ssh', path: '/repo' } }), /require a short host id/)

const injected = JSON.parse('{"__proto__":{"archivedAt":1,"orchestratorId":"evil"}}')
assert.deepEqual(Object.keys(injected), ['__proto__'])
assert.throws(() => sanitizeProjectUpdatePatch(injected), /__proto__ cannot be changed/)
const target: Record<string, unknown> = {}
assert.throws(() => {
  const patch = sanitizeProjectUpdatePatch(injected)
  Object.assign(target, patch)
})
assert.equal(Object.getPrototypeOf(target), Object.prototype)
assert.equal((target as any).archivedAt, undefined)

const picked = pickStrictOwnPatch<{ title: string; status: string }, 'title'>({ title: 'Safe' }, ['title'], 'demo')
assert.equal(Object.getPrototypeOf(picked), Object.prototype)
assert.deepEqual(picked, { title: 'Safe' })

const project = { id: 'p', defaultProviderId: null, defaultProviderAccountId: null, defaultModel: null, localOnly: false } as any
const providers = [{ id: 'local', kind: 'local', models: ['m'], accounts: [{ id: 'a', models: ['m'] }] }, { id: 'cloud', kind: 'cloud', models: ['c'] }] as any
assert.doesNotThrow(() => validateProjectProviderDefaults(project, { defaultProviderId: 'local', defaultProviderAccountId: 'a', defaultModel: 'm', localOnly: true }, providers))
assert.throws(() => validateProjectProviderDefaults(project, { defaultProviderId: 'missing' }, providers), /configured project provider/)
assert.throws(() => validateProjectProviderDefaults(project, { defaultProviderId: 'cloud', defaultModel: 'c', localOnly: true }, providers), /local-only project/)
assert.doesNotThrow(() => validateProjectProviderDefaults(project, { kanban: { modelMode: 'single', single: { providerId: 'local', providerAccountId: 'a', model: 'm' } } }, providers))
assert.throws(() => validateProjectProviderDefaults(project, { kanban: { modelMode: 'single', single: { providerId: 'missing', model: 'm' } } }, providers), /configured provider for project kanban single/)
assert.throws(() => validateProjectProviderDefaults(project, { kanban: { modelMode: 'single', single: { providerId: 'local' } } }, providers), /both a provider and model/)
assert.throws(() => validateProjectProviderDefaults(project, { kanban: { modelMode: 'single', single: { providerId: 'local', providerAccountId: 'missing', model: 'm' } } }, providers), /account that belongs/)
assert.throws(() => validateProjectProviderDefaults(project, { kanban: { modelMode: 'single', single: { providerId: 'local', providerAccountId: 'a', model: 'wrong' } } }, providers), /model offered/)
assert.throws(() => validateProjectProviderDefaults(project, { localOnly: true, kanban: { modelMode: 'single', single: { providerId: 'cloud', model: 'c' } } }, providers), /local provider for project kanban/)
assert.deepEqual(validateProjectExecutionTargetInput('local', 'ignored', []), { executionTarget: 'local', orchestratorId: null })
assert.deepEqual(validateProjectExecutionTargetInput('orchestrator', ' host ', [{ id: 'host', enabled: true }]), { executionTarget: 'orchestrator', orchestratorId: 'host' })
assert.throws(() => validateProjectExecutionTargetInput('orchestrator', 'disabled', [{ id: 'disabled', enabled: false }]), /enabled Orchestrator/)
assert.throws(() => validateProjectExecutionTargetInput('remote', null, []), /Local or Orchestrator/)

console.log('project-patch-policy: strict own-key allowlist blocks identity, routing, path, and prototype injection')
