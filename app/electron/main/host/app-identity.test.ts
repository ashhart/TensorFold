import assert from 'node:assert/strict'
import path from 'node:path'
import { APP_NAME, PROFILE_APP_NAME, configureAppIdentity } from './app-identity'

for (const isolated of [false, true]) {
  const events: Array<[string, string]> = []
  let ready: (() => void) | undefined
  const app = {
    setName: (name: string) => { events.push(['name', name]) },
    getPath: (_name: 'appData') => { assert.equal(isolated, false); return '/example/app-data' },
    setPath: (_name: 'userData', value: string) => { assert.equal(isolated, false); events.push(['userData', value]) },
    once: (_event: 'ready', listener: () => void) => { ready = listener },
  }
  configureAppIdentity(app, isolated)
  assert.equal(APP_NAME, 'TensorFold')
  assert.equal(PROFILE_APP_NAME, 'TensorFold Studio')
  assert.deepEqual(events, isolated ? [['name', PROFILE_APP_NAME]] : [['name', PROFILE_APP_NAME], ['userData', path.join('/example/app-data', PROFILE_APP_NAME)]])
  assert.ok(ready, 'Public runtime branding changes only after the existing crypto/profile identity is established')
  ready()
  assert.deepEqual(events.at(-1), ['name', APP_NAME])
}
console.log('Public app rename preserves the existing profile lock and pre-ready cookie encryption namespace')
