import assert from 'node:assert/strict'
import { createWatchBackgroundController } from '../mobile/src/lib/watch-background-controller'
import type { BackgroundWatchPort } from '../mobile/src/lib/watch-background-controller'

const device = { baseUrl: 'https://host.example.invalid:7443', token: 'synthetic-phone-session', sessionId: 'fixture-session-a', label: 'Fixture', certificatePin: 'a'.repeat(64) }
function deferred() { let resolve!: () => void; const promise = new Promise<void>(yes => { resolve = yes }); return {promise, resolve} }
let configured: string | null = null, clears = 0
let grant: ReturnType<typeof deferred> | null = null
const published: string[] = []
const port: BackgroundWatchPort = {
  async configureBackground(input) { if (grant) await grant.promise; configured = input.sessionId },
  async backgroundStatus() { return {configured: !!configured, ...(configured ? {sessionId: configured} : {})} },
  async disable() { clears++; configured = null },
  async publish(payload) { published.push(payload) },
}
const control = createWatchBackgroundController(port)
assert.equal(await control.set(device, true), true)
assert.equal((await control.restore(device.sessionId)).enabled, true)
await assert.rejects(control.set({...device, sessionId: 'fixture-session-b', certificatePin: undefined}, true), /private HTTPS QR/)
assert.equal(configured, null, 'refused configuration must not retain a different previously authorized account')

grant = deferred()
const enabling = control.set(device, true)
await Promise.resolve(); await Promise.resolve()
const disabling = control.set(device, false)
grant.resolve(); assert.equal(await enabling, false); assert.equal(await disabling, false)
assert.equal(configured, null, 'late native enable must not undo an explicit disable')
grant = null

await control.set(device, true)
let revoked = 0
const unsubscribe = control.onRevoke(() => { revoked++ })
await control.disable()
assert.equal(revoked, 1, 'explicit account forget invalidates a mounted watch settings view immediately')
assert.equal(configured, null)
unsubscribe()

await control.set(device, true)
await control.restore('fixture-session-other')
assert.equal(configured, null, 'restoring a different phone account clears saved native watch consent')
await control.publish(device.sessionId, '{"session":"fixture-session-a"}')
assert.equal(published.length, 0, 'disabled native access cannot receive stale UI snapshots')
assert(clears > 0)
const unavailable = createWatchBackgroundController(null)
assert.equal((await unavailable.restore(device.sessionId)).enabled, false)
await unavailable.disable()
await assert.rejects(unavailable.set(device, true), /native iPhone build/)
console.log('Background watch consent: invalid configuration, late grant/disable, explicit account revoke, scoped restore and stale publication passed')
