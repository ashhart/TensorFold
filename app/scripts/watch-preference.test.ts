import assert from 'node:assert/strict'
import { createWatchPreferenceQueue } from '../mobile/src/lib/watch-preference'
let identity = 'one', enabled = false, stored: string | null = null
const calls: boolean[] = [], pending: (() => void)[] = []
const queue = createWatchPreferenceQueue({identity: () => identity, disable: () => {enabled = false}, write: async (id, next) => {calls.push(next); await new Promise<void>(resolve => pending.push(resolve)); stored = next ? id : null}, apply: (_id, next) => {enabled = next}, error: () => {throw new Error('unexpected write failure')}})
const first = queue.set('one', true)
await new Promise(resolve => setTimeout(resolve, 0))
const second = queue.set('one', false)
assert.equal(enabled, false); assert.deepEqual(calls, [true], 'Keychain writes are serialized')
pending.shift()!(); await first
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal(enabled, false, 'late enable cannot undo a newer disable')
pending.shift()!(); await second
assert.equal(stored, null); assert.equal(enabled, false)
const changed = queue.set('one', true)
await new Promise(resolve => setTimeout(resolve, 0)); identity = 'two'; queue.invalidate(); pending.shift()!(); await changed
assert.equal(enabled, false, 'late old-account consent cannot enable the new account')
console.log('Watch consent write serialization, last-intent disable and account-switch fencing verified')
