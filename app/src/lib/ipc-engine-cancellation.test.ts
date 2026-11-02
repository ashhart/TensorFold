import assert from 'node:assert/strict'
import path from 'node:path'
import { load } from '../components/reading/reading-test-harness'

const notices: string[] = []
const calls: string[] = []
const status = { state: 'ready', model: 'fixture/model' }
let failure: Error | null = null
let synchronous = false
const bridge = {
  platform: 'darwin',
  engineLoad(repo: string) {
    assert.equal(this, bridge, 'the wrapper preserves the native bridge receiver')
    calls.push(repo)
    if (failure) { if (synchronous) throw failure; return Promise.reject(failure) }
    return Promise.resolve(status)
  },
  engineUnload() { return Promise.reject(new Error('Engine start was cancelled.')) },
  engineRefresh() { return Promise.reject(new Error('Synthetic snapshot failure')) },
  googleWorkspaceStatus() { return Promise.reject(new Error('Synthetic Google store unavailable')) },
}
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
Object.defineProperty(globalThis, 'window', { configurable: true, value: {
  tensorfoldStudio: bridge,
  dispatchEvent(event: CustomEvent<string>) { assert.equal(event.type, 'vontra:ipc-error'); notices.push(event.detail) },
} })
try {
  const unexpected = () => { throw new Error('A native bridge must not initialize a browser transport') }
  const { api } = load(path.join(__dirname, 'ipc.ts'), {
    './browser-demo-bootstrap': { preparedBrowserDemo: unexpected },
    './remote-transport': { createRemoteBrowserApi: unexpected },
    './remote-pairing': { pairedToken: unexpected },
  })
  assert.equal(api.platform, 'darwin')
  for (const message of [
    'Engine start was cancelled.',
    "Error invoking remote method 'engine:load': Error: Engine start was cancelled.",
  ]) {
    failure = new Error(message)
    await assert.rejects(api.engineLoad('fixture/model'), (error: unknown) => error === failure)
    assert.deepEqual(notices, [], 'authoritative cancellation is still rejected but is not toasted as an error')
  }
  synchronous = true
  assert.throws(() => api.engineLoad('fixture/model'), (error: unknown) => error === failure)
  assert.deepEqual(notices, [])
  synchronous = false
  failure = new Error("Error invoking remote method 'engine:load': Error: Synthetic model failed to load.")
  await assert.rejects(api.engineLoad('fixture/model'), (error: unknown) => error === failure)
  assert.deepEqual(notices, ['Synthetic model failed to load.'])
  failure = new Error('Engine start was cancelled. Additional failure detail.')
  await assert.rejects(api.engineLoad('fixture/model'), (error: unknown) => error === failure)
  assert.equal(notices.at(-1), failure.message, 'similar-looking failures are not silently reclassified as cancellation')
  const before = notices.length
  await assert.rejects(api.engineUnload(), /Engine start was cancelled/)
  assert.equal(notices.length, before + 1, 'cancellation suppression is limited to the engineLoad channel')
  await assert.rejects(api.engineRefresh(), /Synthetic snapshot failure/)
  assert.equal(notices.length, before + 1, 'background read errors keep their existing inline reporting')
  await assert.rejects(api.googleWorkspaceStatus(), /Synthetic Google store unavailable/)
  assert.equal(notices.length, before + 1, 'periodic Google status failures are handled inline without repeated global toasts')
  failure = null
  assert.equal(await api.engineLoad('fixture/model'), status)
  assert.deepEqual(calls, Array(6).fill('fixture/model'))
} finally {
  if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow)
  else delete (globalThis as any).window
}
console.log('Native IPC wrapper keeps engine-load cancellation rejected without an error toast; real failures still report.')
