import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-connector-decisions-'))
const oldHome = process.env.VONTRA_HOME
const oldActor = process.env.VONTRA_AUDIT_ACTOR
process.env.VONTRA_HOME = path.join(tmp, 'profile')
process.env.VONTRA_AUDIT_ACTOR = 'synthetic-connector-test'
const actualAudit = await import('./audit')
const oldRole = process.env.VONTRA_SECURITY_ROLE
process.env.VONTRA_SECURITY_ROLE = 'owner'
const fixtureKey = '__studioConnectorDecisionFixture'
const state = { handlers: new Map<string, (...args: any[]) => Promise<any>>(), store: { connectorEvents: [] as any[], projects: [], settings: {} }, requireSecurityRole: actualAudit.requireSecurityRole }
;(globalThis as any)[fixtureKey] = state
try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, `export const store=globalThis.${fixtureKey}.store;export const handle=(c,cb)=>globalThis.${fixtureKey}.handlers.set(c,cb);export const persistConnectorEvents=()=>{};export const saveConnectorEvents=async()=>{};export const persistProjectMemoryCandidates=()=>{};export const requireSecurityRole=(...args)=>globalThis.${fixtureKey}.requireSecurityRole(...args);export const auditActor=()=>'synthetic-owner';export const auditEvent=async()=>{};`, { mode: 0o600 })
  const noops = path.join(tmp, 'registrations.mjs')
  await fs.writeFile(noops, ['Email', 'IMessage', 'WhatsApp', 'Discord', 'Mattermost', 'Matrix', 'GoogleChat', 'Slack', 'Calendar', 'Document', 'Browser'].map((name) => `export const register${name}Ipc=()=>{};`).join('\n'), { mode: 0o600 })
  const alias: Record<string, string> = { './ipc': fixture, './persistence': fixture, './audit': fixture, '../state': fixture, '../../store': fixture }
  for (const name of ['email', 'imessage', 'whatsapp', 'discord', 'mattermost', 'matrix', 'googlechat', 'slack', 'calendar', 'documents', 'browser']) alias[`./connector-ipc/${name}`] = noops
  const jiti = createJiti(__filename, { alias, moduleCache: false, fsCache: false })
  const { registerAllConnectorIpc } = await jiti.import<any>(path.join(__dirname, 'connector-ipc.ts'))
  const { sealConnectorOutboundEvent } = await jiti.import<any>(path.join(__dirname, 'connector-outbound-delivery.ts'))
  let sends = 0
  let sendFailure = false
  let deliveryHook: (() => Promise<void>) | null = null
  const guard = new WorkspaceMutationGuard()
  const callbacks = { reserveOperation: () => guard.reserveStart({ workspaceKeys: ['connector:operation'] }), rememberApprovalRule: async (): Promise<any> => ({ id: 'synthetic-rule' }) }
  registerAllConnectorIpc({
    spine: { rememberConnectorEvents: async () => {} },
    emailTransport: { isEmailSendRequest: () => true, deliverApprovedEmailEvent: async () => { sends++; await deliveryHook?.(); if (sendFailure) throw new Error('synthetic delivery failed'); return {transport:'synthetic',status:'delivered'} } },
    channels: { resolveChannelApprovalByThread: () => {}, channelAdapters: [] }, callbacks,
  })
  const resolve = state.handlers.get('assistant:resolveApproval')!
  const fresh = () => {
    const event = sealConnectorOutboundEvent({ id: 'draft-1', connectorId: 'email', action: 'send_requested', requiresApproval: true, approvalStatus: 'pending', threadKey: 'email:synthetic', payloadKeys: ['to'], payload: { to: 'recipient@example.test', subject:'Synthetic subject',body:'Synthetic reviewed body' }, title: 'synthetic', actor: 'owner', summary: 'synthetic' })
    state.store.connectorEvents = [event]
    sends = 0
    return event
  }
  for (const [first, second, expectedSends, expectedStatus] of [
    ['approve', 'approve', 1, 'approved'], ['approve', 'deny', 1, 'approved'], ['deny', 'approve', 0, 'denied'],
  ] as const) {
    const event = fresh()
    let resume!: () => void
    const gate = new Promise<void>((done) => { resume = done })
    callbacks.rememberApprovalRule = async () => { await gate; return { id: 'synthetic-rule' } }
    const a = resolve({}, { eventId: event.id, decision: first, reviewDigest:event.reviewDigest })
    const b = resolve({}, { eventId: event.id, decision: second, reviewDigest:event.reviewDigest })
    await Promise.resolve()
    resume()
    await Promise.all([a, b])
    assert.equal(sends, expectedSends, 'one pending draft receives at most one external delivery')
    assert.equal(state.store.connectorEvents[0].approvalStatus, expectedStatus, 'the first durable decision wins without a later conflicting override')
  }
  const pendingDelivery = fresh()
  let finishDelivery!: () => void
  let enteredDelivery!: () => void
  const deliveryStarted = new Promise<void>((done) => { enteredDelivery = done })
  const deliveryGate = new Promise<void>((done) => { finishDelivery = done })
  deliveryHook = async () => { enteredDelivery(); await deliveryGate }
  callbacks.rememberApprovalRule = async () => ({ id: 'synthetic-rule' })
  const delivered = resolve({}, { eventId: pendingDelivery.id, decision: 'approve', reviewDigest:pendingDelivery.reviewDigest })
  await deliveryStarted
  let acquired = false
  const updating = guard.acquireGlobal('App update installation').then((lease) => { acquired = true; return lease })
  await Promise.resolve()
  const acquiredBeforeDelivery = acquired
  finishDelivery()
  await delivered
  const updateLease = await updating
  assert.equal(acquiredBeforeDelivery, false, 'updater waits for the complete admitted connector delivery')
  await assert.rejects(() => resolve({}, { eventId: pendingDelivery.id, decision: 'approve', reviewDigest:pendingDelivery.reviewDigest }), /update installation/i, 'new mutable connector IPC is refused after update admission closes')
  updateLease.release()
  deliveryHook = null

  // A failed policy write leaves the draft pending and releases the queue for retry.
  const failedPolicy = fresh()
  Object.assign(failedPolicy, {connectorId:'calendar',action:'access_requested'})
  callbacks.rememberApprovalRule = async () => { throw new Error('synthetic policy write failed') }
  await assert.rejects(() => resolve({}, { eventId: failedPolicy.id, decision: 'approve', reviewDigest:failedPolicy.reviewDigest }), /policy write failed/)
  assert.equal(failedPolicy.approvalStatus, 'pending')
  assert.equal(sends, 0)
  callbacks.rememberApprovalRule = async () => ({ id: 'synthetic-rule' })
  await resolve({}, { eventId: failedPolicy.id, decision: 'approve', reviewDigest:failedPolicy.reviewDigest })
  assert.equal(sends, 0, 'non-transmitting decisions never dispatch')

  const removed = fresh()
  Object.assign(removed, {connectorId:'calendar',action:'access_requested'})
  let finishPolicy!: () => void
  const evictionGate = new Promise<void>((done) => { finishPolicy = done })
  callbacks.rememberApprovalRule = async () => { await evictionGate; return { id: 'synthetic-rule' } }
  const evicted = resolve({}, { eventId: removed.id, decision: 'approve', reviewDigest:removed.reviewDigest })
  await Promise.resolve()
  state.store.connectorEvents = []
  finishPolicy()
  await assert.rejects(() => evicted, /approval changed|approval not found/i)
  assert.equal(sends, 0, 'a removed/replaced draft is never transmitted after the policy await')
  callbacks.rememberApprovalRule = async () => ({ id: 'synthetic-rule' })

  // Delivery failure has an uncertain external outcome; repeated clicks cannot resend it.
  const failedSend = fresh()
  sendFailure = true
  const uncertain = await resolve({}, { eventId: failedSend.id, decision: 'approve', reviewDigest:failedSend.reviewDigest })
  assert.equal(uncertain.delivery.status, 'uncertain', 'unknown external completion is never called delivered or safely retryable')
  await resolve({}, { eventId: failedSend.id, decision: 'approve', reviewDigest:failedSend.reviewDigest })
  assert.equal(sends, 1, 'failed transport is not retried by repeated approval')
  sendFailure = false
  const reentrant = fresh()
  deliveryHook = async () => {
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      await Promise.race([
        resolve({}, { eventId: reentrant.id, decision: 'approve', reviewDigest:reentrant.reviewDigest }),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('reentrant approval deadlock')), 750) }),
      ])
    } finally { if (timer) clearTimeout(timer) }
  }
  await resolve({}, { eventId: reentrant.id, decision: 'approve', reviewDigest:reentrant.reviewDigest })
  assert.equal(sends, 1, 'reentrant resolved approval returns without duplicate delivery or waiting on itself')
  deliveryHook = null

  process.env.VONTRA_SECURITY_ROLE = 'operator'
  const operatorDraft = fresh()
  await assert.rejects(() => resolve({}, { eventId: operatorDraft.id, decision: 'approve', reviewDigest:operatorDraft.reviewDigest }), /requires owner/)
  assert.equal(sends, 0)
  await resolve({}, { eventId: operatorDraft.id, decision: 'deny' })
  assert.equal(state.store.connectorEvents[0].approvalStatus, 'denied', 'operator may deny a draft without sending it')
  process.env.VONTRA_SECURITY_ROLE = 'viewer'
  const viewerDraft = fresh()
  await assert.rejects(() => resolve({}, { eventId: viewerDraft.id, decision: 'deny' }), /requires operator/)
  assert.equal(viewerDraft.approvalStatus, 'pending')
  await actualAudit.flushAuditEvents()

  console.log('registered connector approval concurrency, conflicts, policy failure recovery and delivery failure controls passed')
} finally {
  if (oldRole === undefined) delete process.env.VONTRA_SECURITY_ROLE
  else process.env.VONTRA_SECURITY_ROLE = oldRole
  await actualAudit.flushAuditEvents()
  if (oldHome === undefined) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = oldHome
  if (oldActor === undefined) delete process.env.VONTRA_AUDIT_ACTOR
  else process.env.VONTRA_AUDIT_ACTOR = oldActor
  delete (globalThis as any)[fixtureKey]
  await fs.rm(tmp, { recursive: true, force: true })
}
