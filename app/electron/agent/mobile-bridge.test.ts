import assert from 'node:assert/strict'
import { mkdtemp, lstat, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { authorizeMobileBridgeRequest, createMobileBridge, isMobilePrivateAddress, validateMobileBridgeConfig } from './mobile-bridge'
import { loadMobileBridgeCertificate } from './mobile-bridge-certificate'
import { createPairingRegistry } from './pairing-sessions'

const interfaces = [{ name: 'fixture', address: '192.168.50.2' }]
for (const host of ['10.1.2.3', '172.16.0.2', '172.31.255.254', '192.168.50.2']) assert.equal(isMobilePrivateAddress(host), true)
for (const host of ['127.0.0.1', '0.0.0.0', '::1', '8.8.8.8', '100.64.0.1', '169.254.2.3', '172.15.1.2', '172.32.1.2', '192.168.300.2', 'localhost', '192.168.1.2.example.test']) assert.equal(isMobilePrivateAddress(host), false)
validateMobileBridgeConfig({ version: 1, enabled: true, host: interfaces[0].address, port: 8766 }, interfaces)
for (const host of ['0.0.0.0', '127.0.0.1', '8.8.8.8', '192.168.50.3']) assert.throws(() => validateMobileBridgeConfig({ version: 1, enabled: true, host, port: 8766 }, interfaces))
for (const port of [0, 443, 65536, NaN, 8766.5]) assert.throws(() => validateMobileBridgeConfig({ version: 1, enabled: true, host: interfaces[0].address, port }, interfaces))

const ownerToken = 'owner_' + 'a'.repeat(40), registry = createPairingRegistry()
const credential = registry.issuePairingLink('Fixture phone').credential, paired = registry.redeemPairingLink(credential).session!
const metadata = { target: '/v1/snapshot', method: 'GET', expectedOrigin: 'https://192.168.50.2:8766', hostHeader: '192.168.50.2:8766', remoteAddress: '192.168.50.3', authorization: `Bearer ${paired.token}` }
const authorize = (token: string) => registry.authorizeSessionToken(token)
assert.equal(authorizeMobileBridgeRequest(metadata, ownerToken, authorize).token, paired.token)
assert.equal(authorizeMobileBridgeRequest({ ...metadata, target: '/v1/device-status' }, ownerToken, authorize).claim, false)
assert.equal(authorizeMobileBridgeRequest({ ...metadata, method: 'POST', target: '/v1/auth/pair', authorization: undefined }, ownerToken, authorize).claim, true)
for (const target of ['/health', '/', '/v1/auth/sessions', '/v1/files', '/v1/terminals', '/v1/engine/start', '/v1/approve', '/v1/auth/issue', '/v1/threads/one/archive']) assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, target }, ownerToken, authorize), /route_not_available/)
for (const target of ['//other.example.test/v1/snapshot', 'https://other.example.test/v1/snapshot', '/v1/snapshot?token=fixture', '/v1/snapshot#pair=fixture']) assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, target }, ownerToken, authorize))
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, origin: 'https://other.example.test' }, ownerToken, authorize), /origin_denied/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, hostHeader: 'other.example.test:8766' }, ownerToken, authorize), /origin_denied/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, remoteAddress: '203.0.113.20' }, ownerToken, authorize), /private_client_required/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, authorization: `Bearer ${ownerToken}` }, ownerToken, authorize), /owner_token_not_allowed/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, hasCookie: true }, ownerToken, authorize), /bearer_header_required/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, hasLegacyToken: true }, ownerToken, authorize), /bearer_header_required/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, authorization: 'Bearer invalid' }, ownerToken, authorize), /paired_session_required/)
assert.throws(() => authorizeMobileBridgeRequest(metadata, ownerToken, () => ({ ok: true, sessionId: 'fixture', scope: 'owner' })), /paired_session_required/)
assert.throws(() => authorizeMobileBridgeRequest({ ...metadata, method: 'POST', target: '/v1/auth/pair' }, ownerToken, authorize), /pairing_body_required/)
registry.revokeSession(paired.id)
assert.throws(() => authorizeMobileBridgeRequest(metadata, ownerToken, authorize), /paired_session_required/)

const profile = await mkdtemp(path.join(os.tmpdir(), 'studio-mobile-policy-'))
try {
  const cert = await loadMobileBridgeCertificate(profile, interfaces[0].address)
  assert.match(cert.sha256, /^[a-f0-9]{64}$/)
  assert.equal((await lstat(path.join(profile, 'mobile-bridge', 'private-key.pem'))).mode & 0o777, 0o600)
  assert.equal((await loadMobileBridgeCertificate(profile, interfaces[0].address)).sha256, cert.sha256)
  let owner = false, certificateLoads = 0
  const bridge = createMobileBridge({ interfaces: () => interfaces, requireOwner: () => { if (!owner) throw new Error('owner_required') }, ownerToken: () => ownerToken, authorizeSessionToken: authorize, loopbackUrl: () => 'http://127.0.0.1:8765', issuePairingLink: async () => registry.issuePairingLink('Fixture'), loadCertificate: async () => { certificateLoads++; return cert } })
  assert.equal(bridge.status().enabled, false)
  assert.equal(bridge.status().listening, false)
  await assert.rejects(() => bridge.configure({ version: 1, enabled: true, host: interfaces[0].address, port: 8766 }), /owner_required/)
  await assert.rejects(() => bridge.issuePairing('Fixture'), /owner_required/)
  owner = true
  await assert.rejects(() => bridge.configure({ version: 1, enabled: true, host: '0.0.0.0', port: 8766 }), /Select an available/)
  assert.equal(certificateLoads, 0, 'Public bind rejection precedes certificate or listener creation')
  await assert.rejects(() => bridge.issuePairing('Fixture'), /Enable the private mobile/)
  await bridge.configure({ version: 1, enabled: false, host: interfaces[0].address, port: 8766 })
  assert.equal(bridge.status().listening, false)
  await bridge.stop()

  // A superseded certificate operation must not overwrite the latest owner's successful choice.
  let releaseCertificate!: () => void, certificateEntered!: () => void
  const delayedCertificate = new Promise<void>(resolve => { releaseCertificate = resolve })
  const enteredCertificate = new Promise<void>(resolve => { certificateEntered = resolve })
  const overlapping = createMobileBridge({ interfaces: () => interfaces, requireOwner: () => {}, ownerToken: () => ownerToken, authorizeSessionToken: authorize, loopbackUrl: () => 'http://127.0.0.1:8765', issuePairingLink: async () => registry.issuePairingLink('Fixture'), loadCertificate: async () => { certificateEntered(); await delayedCertificate; return cert } })
  const oldEnable = overlapping.configure({ version: 1, enabled: true, host: interfaces[0].address, port: 8766 })
  await enteredCertificate
  await overlapping.configure({ version: 1, enabled: false, host: interfaces[0].address, port: 8767 })
  releaseCertificate()
  await assert.rejects(oldEnable, /disabled while preparing|newer owner choice/)
  assert.equal(overlapping.status().enabled, false)
  assert.equal(overlapping.status().lastError, null, 'A superseded enable must preserve the latest successful owner choice')
  await overlapping.stop()

  let simultaneousCertificateLoads = 0
  const simultaneous = createMobileBridge({ interfaces: () => interfaces, requireOwner: () => {}, ownerToken: () => ownerToken, authorizeSessionToken: authorize, loopbackUrl: () => 'http://127.0.0.1:8765', issuePairingLink: async () => registry.issuePairingLink('Fixture'), loadCertificate: async () => { simultaneousCertificateLoads++; return cert } })
  const [earlierChoice, latestChoice] = await Promise.allSettled([
    simultaneous.configure({ version: 1, enabled: true, host: interfaces[0].address, port: 8766 }),
    simultaneous.configure({ version: 1, enabled: false, host: interfaces[0].address, port: 8767 }),
  ])
  assert.equal(earlierChoice.status, 'rejected')
  assert.equal(latestChoice.status, 'fulfilled')
  assert.equal(simultaneousCertificateLoads, 0, 'A same-tick superseded enable must not begin certificate or listener setup')
  assert.equal(simultaneous.status().lastError, null)
  await simultaneous.stop()
} finally { await rm(profile, { recursive: true, force: true }) }
console.log('mobile-bridge.test.ts passed deterministic private-interface, origin/route/header/owner/paired-scope/revocation and private certificate policy checks; live LAN TLS is a separate bounded integration check.')
