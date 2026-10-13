import assert from 'node:assert/strict'
import {
  HEADLESS_BOOTSTRAP_ENV,
  headlessBootstrapMessages,
  isStrongRemoteControlToken,
  prepareHeadlessRemoteControl,
  terminateFailedHeadlessBoot,
} from './headless-bootstrap'
import type { RemoteControlSettings } from '../../agent/types'

const strongA = 'A'.repeat(43)
const strongB = 'B'.repeat(43)
const base = (patch: Partial<RemoteControlSettings> = {}): RemoteControlSettings => ({
  enabled: false,
  serverName: 'TensorFold Studio Orchestrator',
  bindHost: '0.0.0.0',
  port: 8765,
  authToken: '',
  allowLan: true,
  headlessBootstrappedAt: null,
  lastStartedAt: null,
  lastClientAt: null,
  lastError: null,
  ...patch,
})

assert.equal(isStrongRemoteControlToken('short'), false)
assert.equal(isStrongRemoteControlToken(strongA), true)

{
  const prior = process.exitCode
  let exitedWith: number | null = null
  terminateFailedHeadlessBoot((code) => { exitedWith = code })
  assert.equal(exitedWith, 1, 'headless bootstrap failure exits Electron explicitly with status 1')
  assert.equal(process.exitCode, 1, 'the Node fallback exit status is also non-zero')
  process.exitCode = prior
}

{
  const result = prepareHeadlessRemoteControl(base(), { env: {}, now: () => 123, newToken: () => strongA })
  assert.equal(result.remoteControl.enabled, true)
  assert.equal(result.remoteControl.bindHost, '127.0.0.1', 'headless defaults to loopback even if stale settings allowed LAN')
  assert.equal(result.remoteControl.allowLan, false)
  assert.equal(result.remoteControl.authToken, strongA)
  assert.equal(result.remoteControl.headlessBootstrappedAt, null, 'marker remains pending until the listener and credential persistence succeed')
  assert.equal(result.completionMarker, 123)
  assert.equal(result.revealCredentials, true)
  assert.equal(result.resetPairingSessions, false)
  assert.equal(result.changed, true)
}

{
  const result = prepareHeadlessRemoteControl(base({ authToken: strongA }), { env: {}, now: () => 124, newToken: () => strongB })
  assert.equal(result.remoteControl.authToken, strongB, 'first headless bootstrap always uses freshly generated entropy, not an inherited long string')
}

{
  const current = base({ enabled: true, bindHost: '127.0.0.1', allowLan: false, authToken: strongA, headlessBootstrappedAt: 100 })
  const result = prepareHeadlessRemoteControl(current, { env: {}, now: () => 200, newToken: () => strongB })
  assert.equal(result.remoteControl.authToken, strongA, 'normal restart reuses the persisted strong token')
  assert.equal(result.remoteControl.headlessBootstrappedAt, 100)
  assert.equal(result.revealCredentials, false, 'normal restart never logs the token again')
  assert.equal(result.completionMarker, null)
  assert.equal(result.changed, false)
}

assert.throws(
  () => prepareHeadlessRemoteControl(base(), { env: { VONTRA_HEADLESS_BIND_HOST: '0.0.0.0' }, newToken: () => strongA }),
  /loopback-only|127\.0\.0\.1/,
)

assert.throws(
  () => prepareHeadlessRemoteControl(base(), { env: { VONTRA_HEADLESS_BIND_HOST: '192.168.1.2' }, newToken: () => strongA }),
  /must be 127\.0\.0\.1/,
)
assert.throws(
  () => prepareHeadlessRemoteControl(base(), { env: { VONTRA_HEADLESS_PORT: '0' }, newToken: () => strongA }),
  /integer from 1 to 65535/,
)

{
  const first = headlessBootstrapMessages({ url: 'http://127.0.0.1:8765', token: strongA, pairingCredential: 'pair_secret', pairingExpiresAt: 1_000 })
  assert.match(first.join('\n'), new RegExp(strongA))
  assert.match(first.join('\n'), /#pair=pair_secret/)
  assert.doesNotMatch(first.join('\n'), /\?pair=/, 'one-time credentials never enter HTTP query strings')
  assert.match(first.join('\n'), /ssh -N -L 8765:127\.0\.0\.1:8765/)
  const later = headlessBootstrapMessages({ url: 'http://127.0.0.1:8765' })
  assert.doesNotMatch(later.join('\n'), new RegExp(strongA))
  assert.match(later.join('\n'), /not being reprinted/)
  assert.match(later.join('\n'), new RegExp(`${HEADLESS_BOOTSTRAP_ENV}=1`))
}

console.log('headless bootstrap tests passed')
