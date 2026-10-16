// Run: jiti electron/agent/connector-action-safety.test.ts
//
// Connector action safety (VST-013): per-mode capability proof across every connector
// family the spec names, safe outbound previews (bounded + redacted), and the sync check
// that keeps the renderer doctor's write-capable set honest.
import assert from 'node:assert/strict'
import {
  connectorConfiguredVerdict,
  connectorModeCapabilities,
  outboundActionPreview,
} from './connector-action-safety'
import { WRITE_CAPABLE_CONNECTORS } from '../../src/lib/readiness-doctor'
import type { ConnectorId } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('connector-action-safety tests:')

{
  // The spec's connector matrix: every family gets the right mode claims.
  const readWrite: ConnectorId[] = ['email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'calendar', 'webhook']
  for (const id of readWrite) assert.deepEqual(connectorModeCapabilities(id), ['read', 'write'], `${id} claims read+write`)
  assert.deepEqual(connectorModeCapabilities('browser'), ['read'], 'browser is read-only')
  assert.deepEqual(connectorModeCapabilities('docs'), ['read'], 'documents are read-only')
  ok('modes: email/Slack/Discord/Mattermost/Matrix/GChat/calendar/webhooks read+write; browser/documents read-only')
}

{
  // No connector is "configured" without proving what its mode claims.
  for (const id of ['email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'calendar', 'webhook'] as ConnectorId[]) {
    const bare = connectorConfiguredVerdict(id, {})
    assert.equal(bare.configured, false)
    assert.deepEqual(bare.missing, ['read', 'write'], `${id}: nothing proven → both named`)

    const readOnlyProof = connectorConfiguredVerdict(id, { syncedAt: 1000 })
    assert.equal(readOnlyProof.configured, false, `${id}: read proof alone does not prove write`)
    assert.deepEqual(readOnlyProof.missing, ['write'])
    assert.match(readOnlyProof.detail, /write unproven — no successful outbound/)

    const full = connectorConfiguredVerdict(id, { syncedAt: 1000, lastOutboundOkAt: 2000 })
    assert.equal(full.configured, true, `${id}: both proofs → configured`)
  }
  // Read-only families need only read proof.
  assert.equal(connectorConfiguredVerdict('browser', { sampleItemCount: 3 }).configured, true)
  assert.equal(connectorConfiguredVerdict('docs', {}).configured, false)
  assert.deepEqual(connectorConfiguredVerdict('docs', {}).missing, ['read'])
  ok('verdicts: configured is earned per capability, missing ones are named')
}

{
  // Outbound previews: target + bounded redacted content, per family.
  const email = outboundActionPreview('email_compose', {
    to: 'boss@example.com',
    subject: 'Quarterly numbers',
    body: `Here are the numbers. My token is sk-live-a1b2c3d4e5f6g7h8i9j0k1l2. ${'x'.repeat(500)}`,
  })
  assert.equal(email?.connectorId, 'email')
  assert.equal(email?.target, 'boss@example.com', 'the recipient is visible — approvals decide about real targets')
  assert.match(email!.text, /compose email/)
  assert.match(email!.text, /Quarterly numbers/)
  assert.ok(!email!.excerpt.includes('sk-live-a1b2c3d4e5f6g7h8i9j0k1l2'), 'secrets never appear in a preview')
  assert.ok(email!.excerpt.length <= 281, 'content is bounded')

  const message = outboundActionPreview('message_compose', { channel: '#releases', text: 'Shipping v2 now' })
  assert.equal(message?.connectorId, 'slack')
  assert.equal(message?.target, '#releases')

  const event = outboundActionPreview('calendar_create', { title: 'Launch review', start: '2026-07-11T10:00', attendees: ['maya@example.com'] })
  assert.equal(event?.connectorId, 'calendar')
  assert.match(event!.text, /create event/)
  assert.ok(event!.text.includes('maya@example.com'), 'the approver SEES who gets invited — targets are decision-relevant, like the email recipient')

  const browser = outboundActionPreview('browser_automation', { url: 'https://example.com/checkout', goal: 'fill the address form' })
  assert.equal(browser?.connectorId, 'browser')
  assert.match(browser!.text, /example\.com\/checkout/)

  assert.equal(outboundActionPreview('read_file', { path: 'x' }), null, 'read tools have no outbound preview')
  assert.equal(outboundActionPreview('email_list', {}), null)
  ok('previews: email/message/calendar/browser with redaction + bounds; read tools none')
}

{
  // The renderer doctor's write-capable set must match the runtime source of truth.
  const runtimeWriteCapable = new Set(
    (['email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp', 'imessage', 'telegram', 'docs', 'calendar', 'browser', 'controller', 'webhook'] as ConnectorId[])
      .filter((id) => connectorModeCapabilities(id).includes('write')),
  )
  assert.deepEqual([...runtimeWriteCapable].sort(), [...WRITE_CAPABLE_CONNECTORS].sort(), 'doctor set drifted from connector-action-safety modes')
  ok('sync: the doctor write-capable set matches the runtime mode map')
}

console.log(`  ${passed} passed`)
