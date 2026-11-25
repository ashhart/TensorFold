import assert from 'node:assert/strict'
import { resolveDelegationRoute } from './delegation-route'
import type { Provider, Seat } from './types'

const provider: Provider = { id: 'fixture', name: 'Fixture', kind: 'local', apiKey: '',
  baseUrl: 'http://127.0.0.1:1/v1', models: ['first', 'second'] }
const other: Provider = { ...provider, id: 'other' }
const base = { providerId: provider.id, model: 'first', reasoningEffort: 'low' as const }
const seats: Seat[] = [
  { id: 'reviewer', name: 'Reviewer', role: 'Review input', providerId: provider.id, model: 'second', reasoningEffort: 'high' },
  { id: 'legacy', name: 'Legacy', role: 'Review input', providerId: provider.id, model: 'first' },
]
assert.deepEqual(resolveDelegationRoute([provider, other], base, {}).route, base)
assert.deepEqual(resolveDelegationRoute([provider, other], base, { targetSeat: 'Reviewer' }, seats).route,
  { providerId: provider.id, providerAccountId: undefined, model: 'second', reasoningEffort: 'high' })
assert.deepEqual(resolveDelegationRoute([provider, other], base, { targetSeat: 'Legacy' }, seats).route,
  { providerId: provider.id, providerAccountId: undefined, model: 'first' }, 'an inherited target does not adopt the origin agent override')
assert.deepEqual(resolveDelegationRoute([provider, other], base, { providerId: other.id, model: 'second' }).route,
  { providerId: other.id, model: 'second', reasoningEffort: 'low' }, 'explicit provider changes retain the originating effort')
console.log('Room delegation effort routing checks passed: origin, target agent, legacy target and explicit provider/model route.')
