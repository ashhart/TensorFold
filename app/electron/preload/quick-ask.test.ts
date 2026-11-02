import assert from 'node:assert/strict'
import { quickAskBridge, quickAskLauncher } from './quick-ask'

const calls: unknown[][] = [], listeners = new Map<string, any>()
const ipc = { invoke: async (...args: unknown[]) => { calls.push(args) }, on: (channel: string, callback: any) => { listeners.set(channel, callback) }, removeListener: (channel: string) => { listeners.delete(channel) } } as any
const api = quickAskBridge(ipc)
assert.equal('saveSettings' in api, false); assert.equal('startRun' in api, false); assert.equal('invoke' in api, false)
await api.clipboard(); await api.capture('display'); await api.copy('Example')
assert.deepEqual(calls, [['quick-ask:clipboard'], ['quick-ask:capture', 'display'], ['quick-ask:copy', 'Example']])
calls.length = 0
await api.resize(104); await api.pickWindow(); await api.useWindow('opaque-token')
assert.deepEqual(calls, [['quick-ask:resize', 104], ['quick-ask:pick-window'], ['quick-ask:use-window', 'opaque-token']])
let received: unknown
const unsubscribe = api.onEvent(value => { received = value })
listeners.get('quick-ask:event')({ token: 'host-private' }, { kind: 'focus' })
assert.deepEqual(received, { kind: 'focus' }, 'Electron event objects never cross the bridge')
unsubscribe(); assert.equal(listeners.size, 0)
const launcher = quickAskLauncher(ipc)
assert.deepEqual(Object.keys(launcher).sort(), ['consumeHandoff', 'onHandoff', 'open'])
console.log('Quick Ask preload allowlist and listener cleanup checks passed')
