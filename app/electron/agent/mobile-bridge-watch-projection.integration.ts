import assert from 'node:assert/strict'
import {createServer as httpServer} from 'node:http'
import {request as httpsRequest} from 'node:https'
import {createServer as portProbe} from 'node:net'
import {mkdtemp, rm} from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {createMobileBridge, mobileBridgeInterfaces} from './mobile-bridge'
import {loadMobileBridgeCertificate} from './mobile-bridge-certificate'
import {createWatchCommandProjection} from '../main/host/watch-command-projection'
import {store} from '../main/state'
import type {Thread} from './types'

const address = mobileBridgeInterfaces()[0]?.address
assert.ok(address, 'Actual private-interface TLS regression requires a private IPv4 address')
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-watch-projection-fixture-'))
const original = {...store}, previousRole = process.env.VONTRA_SECURITY_ROLE
const token = 'fixture_' + 'p'.repeat(40), session = 'paired-session'
const caps = {taskStart: true, taskGuide: true, taskStop: true, approvalResolve: false, transcriptReview: true, changesReview: false, engineControl: false}
let mode = 'private', statusReads = 0, authorized = true, projectionEnabled = true
const fixture = (): Thread => ({id: 'shared-task', title: 'SYNTHETIC_LATE_PRIVATE_TITLE', kind: 'task', cwd: '', providerId: 'example', model: 'example', createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [], timeline: [{id: 'assistant-test', kind: 'assistant', text: 'SYNTHETIC_LATE_PRIVATE_REPLY'}]})
const core = httpServer((req, res) => {
  assert.equal(req.headers.authorization, `Bearer ${token}`)
  let value: unknown
  if (req.url === '/v1/device-status') {
    statusReads++
    if (statusReads === 2) {
      if (mode === 'private') store.threads[0].incognito = true
      if (mode === 'role') process.env.VONTRA_SECURITY_ROLE = 'viewer'
      if (mode === 'revoked') authorized = false
    }
    value = {scope: 'paired', capabilities: caps, engine: null}
  } else {
    value = {serverName: 'Synthetic host', activeThreadIds: [], threads: structuredClone(store.threads), work: {jobs: {running: 0, queued: 0}, approvals: {pending: 0}}}
  }
  res.writeHead(200, {'content-type': 'application/json'}); res.end(JSON.stringify(value))
})
await new Promise<void>(resolve => core.listen(0, '127.0.0.1', resolve))
const corePort = (core.address() as {port: number}).port
const port = await new Promise<number>(resolve => { const server = portProbe(); server.listen(0, address, () => {const value = (server.address() as {port: number}).port; server.close(() => resolve(value))}) })
const project = createWatchCommandProjection({activeThreadIds: () => []})
const bridge = createMobileBridge({requireOwner() {}, loopbackUrl: () => `http://127.0.0.1:${corePort}`, ownerToken: () => 'fixture-owner-token', authorizeSessionToken: value => ({ok: authorized && value === token, sessionId: session, scope: 'paired'}), issuePairingLink: async () => ({credential: 'unused', expiresAt: 0}), loadCertificate: host => loadMobileBridgeCertificate(profile, host), projectWatchSnapshot: (value, id) => projectionEnabled ? project(value, id) : value})
try {
  store.settings = {remoteControl: {serverName: 'Synthetic host'}} as any
  store.jobs = []; store.connectorEvents = []; store.missions = []
  process.env.VONTRA_SECURITY_ROLE = 'owner'
  await bridge.configure({version: 1, enabled: true, host: address, port})
  const cert = await loadMobileBridgeCertificate(profile, address)
  async function refresh(): Promise<{status: number; value: any}> {
    const body = Buffer.from(JSON.stringify({version: 1, id: `request-${mode}-${projectionEnabled}`, session, action: 'refresh', generatedAt: Date.now()}))
    return await new Promise((resolve, reject) => {
      const req = httpsRequest({hostname: address, port, path: '/v1/watch/command', method: 'POST', ca: cert.cert, agent: false, headers: {authorization: `Bearer ${token}`, 'content-type': 'application/json', 'content-length': body.length}}, res => {
        const chunks: Buffer[] = []; res.on('data', part => chunks.push(part)); res.on('end', () => resolve({status: res.statusCode!, value: JSON.parse(Buffer.concat(chunks).toString())})); res.on('error', reject)
      }); req.on('error', reject); req.setTimeout(5_000, () => req.destroy(new Error('Synthetic route deadline'))); req.end(body)
    })
  }
  store.threads = [fixture()]
  projectionEnabled = false
  const before = await refresh()
  assert.match(JSON.stringify(before.value), /SYNTHETIC_LATE_PRIVATE_TITLE/, 'Red-capable fixture confirms raw relay snapshot is stale after second permissions read')
  projectionEnabled = true; statusReads = 0; store.threads = [fixture()]
  const after = await refresh()
  assert.equal(after.status, 200); assert.equal(after.value.ok, true); assert.equal(after.value.snapshot.jobs.length, 0)
  assert.doesNotMatch(JSON.stringify(after.value), /SYNTHETIC_LATE_PRIVATE_(TITLE|REPLY)/)
  mode = 'role'; statusReads = 0; store.threads = [fixture()]
  const downgraded = await refresh()
  assert.equal(downgraded.value.snapshot.jobs[0].canReply, false)
  assert.equal(downgraded.value.snapshot.jobs[0].latestReply, null)
  assert.equal(downgraded.value.snapshot.transcriptReview, false)
  process.env.VONTRA_SECURITY_ROLE = 'owner'; mode = 'revoked'; statusReads = 0
  assert.equal((await refresh()).status, 401, 'Identity is checked again after all awaited work')
  await bridge.stop()
  authorized = true; mode = 'unguarded'; statusReads = 0; store.threads = [fixture()]
  const missingGuard = createMobileBridge({requireOwner() {}, loopbackUrl: () => `http://127.0.0.1:${corePort}`, ownerToken: () => 'fixture-owner-token', authorizeSessionToken: value => ({ok: value === token, sessionId: session, scope: 'paired'}), issuePairingLink: async () => ({credential: 'unused', expiresAt: 0}), loadCertificate: host => loadMobileBridgeCertificate(profile, host)})
  try {
    await missingGuard.configure({version: 1, enabled: true, host: address, port})
    const unguarded = await refresh()
    assert.equal(unguarded.status, 503); assert.equal(unguarded.value.error, 'watch_projection_unavailable')
    assert.doesNotMatch(JSON.stringify(unguarded.value), /SYNTHETIC_LATE_PRIVATE_/)
  } finally { await missingGuard.stop() }
  console.log('Actual private HTTPS watch route reproduced stale relay projection with guard bypass, then current-store guard excluded late private task, clamped late role downgrade and refused late revoked session')
} finally {
  await bridge.stop(); await new Promise<void>(resolve => core.close(() => resolve())); await rm(profile, {recursive: true, force: true})
  Object.assign(store, original)
  if (previousRole === undefined) delete process.env.VONTRA_SECURITY_ROLE; else process.env.VONTRA_SECURITY_ROLE = previousRole
}
