import assert from 'node:assert/strict'
import { webcrypto } from 'node:crypto'
import { favoriteModelChoices, favoriteModelKey, favoriteModelsScope, FavoriteModelsPreferences, MAX_FAVORITE_MODELS, orderFavoritesFirst, parseFavoriteModels } from './favorite-models'

const a = { providerId: 'local-provider', accountId: null, model: 'model-a' }
const b = { ...a, model: 'model-b' }
const account = { ...a, accountId: 'work-account' }
const env = { platform: 'darwin', origin: 'file://', pairedCredential: null, ownerId: null }
const desktop = (await favoriteModelsScope(env))!
assert.notEqual(desktop, await favoriteModelsScope({ ...env, ownerId: 'other-host' }))
assert.notEqual(desktop, await favoriteModelsScope({ ...env, platform: 'web' }))
const digest = webcrypto.subtle.digest.bind(webcrypto.subtle) as SubtleCrypto['digest']
const paired = { ...env, platform: 'web-remote', origin: 'https://first.example.test', pairedCredential: 'synthetic-pairing-credential-one' }
const pairedKey = (await favoriteModelsScope(paired, digest))!
assert.match(pairedKey, /paired:[a-f0-9]{64}$/)
assert.doesNotMatch(pairedKey, /synthetic|credential|example/)
assert.equal(pairedKey, await favoriteModelsScope(paired, digest), 'the same host/session survives a renderer restart')
for (const next of [{ origin: 'https://other.example.test' }, { pairedCredential: 'synthetic-pairing-credential-two' }, { ownerId: 'other-host' }]) {
  assert.notEqual(pairedKey, await favoriteModelsScope({ ...paired, ...next }, digest), 'host, authenticated pairing and owning orchestrator each isolate favourites')
}
assert.equal(await favoriteModelsScope({ ...paired, pairedCredential: null }, digest), null)
assert.equal(await favoriteModelsScope(paired, null), null, 'without supported cryptography do not fall back to host-only scope')

const data = new Map<string, string>()
let unavailable = false
const storage = { getItem: (key: string) => { if (unavailable) throw new Error('Synthetic storage denied'); return data.get(key) ?? null }, setItem: (key: string, value: string) => { if (unavailable) throw new Error('Synthetic quota'); data.set(key, value) } }
const events = new EventTarget()
const first = new FavoriteModelsPreferences(desktop, () => storage, events as Window)
const otherHost = new FavoriteModelsPreferences((await favoriteModelsScope({ ...env, ownerId: 'other-host' }))!, () => storage)
let notices = 0
const remove = first.subscribe(() => { notices++ })
assert.deepEqual(first.get(), [])
const initial = first.get()
assert.equal(first.get(), initial, 'snapshot identity remains stable between changes')
first.toggle(a); assert.equal(notices, 1)
assert.deepEqual(first.get(), [a])
const restarted = new FavoriteModelsPreferences(desktop, () => storage)
assert.deepEqual(restarted.get(), [a], 'saved routes survive restart')
assert.deepEqual(otherHost.get(), [])
restarted.toggle(account)
first.toggle(b)
assert.deepEqual(first.get(), [b, account, a], 'every click merges the current storage state instead of a stale UI snapshot')
first.toggle(a)
assert.deepEqual(first.get(), [b, account], 'removing one account/model route keeps another account untouched')
assert.deepEqual(favoriteModelChoices([a, b], first.get(), (item) => item), [b], 'saved accounts not in policy-allowed choices are never resurrected')
assert.deepEqual(orderFavoritesFirst([a, account, b], [b], (item) => item), [b, a, account], 'favourites lead while all other ordering remains stable')
assert.notEqual(favoriteModelKey(a), favoriteModelKey(account))
assert.notEqual(favoriteModelKey({ ...a, providerId: 'a:b', model: 'c' }), favoriteModelKey({ ...a, providerId: 'a', model: 'b:c' }))
data.set(desktop, JSON.stringify([a]))
const event = new Event('storage'); Object.defineProperty(event, 'key', { value: desktop }); events.dispatchEvent(event)
assert.equal(notices, 4)
assert.deepEqual(first.get(), [a], 'a different window storage event refreshes the current view')
const unrelated = new Event('storage'); Object.defineProperty(unrelated, 'key', { value: 'unrelated' }); events.dispatchEvent(unrelated)
assert.equal(notices, 4)
remove(); first.toggle(b); assert.equal(notices, 4)
unavailable = true
assert.deepEqual(first.get(), [b, a], 'transient read failure retains the last known view instead of pretending saved favourites are empty')
assert.throws(() => first.toggle(a), /could not be read/)
unavailable = false
assert.deepEqual(first.get(), [b, a], 'failed writes do not pretend a favourite was saved or remove the earlier durable routes')
let readFails = false, writes = 0
const readOnlyFailure = new FavoriteModelsPreferences(desktop, () => ({
  getItem: (key) => { if (readFails) throw new Error('Synthetic read-only failure'); return data.get(key) ?? null },
  setItem: (key, value) => { writes++; data.set(key, value) },
}))
assert.deepEqual(readOnlyFailure.get(), [b, a])
const beforeUnreadableWrite = data.get(desktop)
readFails = true
assert.deepEqual(readOnlyFailure.get(), [b, a])
assert.throws(() => readOnlyFailure.toggle(account), /could not be read.*unchanged/)
assert.equal(writes, 0, 'successful writes are never attempted after the current saved list could not be read')
assert.equal(data.get(desktop), beforeUnreadableWrite, 'a readable-write/unreadable-read store cannot lose unknown earlier favourites')
const unknownRead = new FavoriteModelsPreferences(desktop, () => ({ getItem: () => { throw new Error('First read denied') }, setItem: () => { writes++ } }))
assert.deepEqual(unknownRead.get(), [])
assert.throws(() => unknownRead.toggle(account), /could not be read/)
assert.equal(writes, 0, 'first-load read failure also fails closed')
readFails = false; readOnlyFailure.toggle(account)
assert.equal(writes, 1)
assert.deepEqual(readOnlyFailure.get(), [account, b, a], 'retry after read recovery merges the durable saved routes')
data.set(desktop, '{corrupt')
assert.deepEqual(first.get(), [])
first.toggle(a); assert.deepEqual(first.get(), [a], 'the next explicit click can repair corrupt preferences')
for (const invalid of [null, {}, 'not json', JSON.stringify([null, 1, [], { ...a, accountId: 10 }, { ...a, model: '\u0000bad' }, { ...a, model: 'x'.repeat(1_025) }])]) {
  assert.deepEqual(parseFavoriteModels(typeof invalid === 'string' ? invalid : JSON.stringify(invalid)), [])
}
assert.deepEqual(parseFavoriteModels(JSON.stringify([a, a, account])), [a, account])
data.set(desktop, JSON.stringify(Array.from({ length: MAX_FAVORITE_MODELS }, (_, index) => ({ ...a, model: `model-${index}` }))))
assert.throws(() => first.toggle(a), /Remove a favourite/)
assert.equal(first.get().length, MAX_FAVORITE_MODELS)
first.toggle({ ...a, model: 'model-0' }); assert.equal(first.get().length, MAX_FAVORITE_MODELS - 1)
console.log('Favourites persist exact account/host routes, reload without credential storage, merge views, preserve stable ordering and fail safely for corruption or denied storage.')
