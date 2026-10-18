import assert from 'node:assert/strict'
import { createPairingRegistry } from './pairing-sessions'
const registry = createPairingRegistry()
const session = registry.redeemPairingLink(registry.issuePairingLink('Example phone').credential).session!
assert.equal(registry.isWatchSpeechAllowed(session.id), false)
assert.equal(registry.listSessions()[0].watchSpeech, false)
registry.setWatchSpeechAllowed(session.id, true)
assert.equal(registry.isWatchSpeechAllowed(session.id), true)
assert.equal(createPairingRegistry({}, registry.snapshot()).isWatchSpeechAllowed(session.id), true)
registry.setWatchSpeechAllowed(session.id, false)
assert.equal(registry.isWatchSpeechAllowed(session.id), false)
registry.setWatchSpeechAllowed(session.id, true); registry.revokeSession(session.id)
assert.equal(registry.isWatchSpeechAllowed(session.id), false)
assert.throws(() => registry.setWatchSpeechAllowed(session.id, true), /revoked/)
assert.equal(JSON.stringify(registry.listSessions()).includes(session.token), false)
console.log('Per-device watch audio defaults, durable snapshots, revocation and token-free permission views verified')
