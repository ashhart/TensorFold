// Run: jiti electron/agent/comms-routing-fixes.test.ts
// Regression tests for the A-comms-routing cluster bug fixes:
//   1. slackThreadKey carries the real channel id in the conversation segment (routing + bursts)
//   2. parseApprovalReply does NOT prefix-match hard decision keywords (no accidental approvals)
//   3. mattermostEventFromMessage stamps payload.messageId from its own message (dedup safety)
import assert from 'node:assert/strict'
import { slackThreadKey } from './slack-connector.ts'
import { parseApprovalReply } from './channel-adapter.ts'
import { parseThreadAddress, coalesceThreadBursts } from './thread-address.ts'
import { mattermostEventFromMessage } from './mattermost-connector.ts'
import type { MattermostReadResult } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('comms-routing fixes tests:')

// ---- 1. slack threadKey shape: account=sourceId, conversation=channelId, message=ts ----
{
  const key = slackThreadKey({ sourceId: 'slack_aaa', channelId: 'C123', ts: '1699.1' })
  assert.equal(key, 'slack:slack_aaa:C123:1699.1')
  const addr = parseThreadAddress(key)
  assert.ok(addr)
  assert.equal(addr!.account, 'slack_aaa')
  assert.equal(addr!.conversation, 'C123', 'conversation is the CHANNEL id (what the adapter posts to), not the source id')
  assert.equal(addr!.message, '1699.1')
  ok('slackThreadKey routes the conversation segment to the channel id')
}

// ---- 1b. burst coalescing keys per channel, not per workspace ----
{
  const ev = (channelId: string, ts: string, createdAt: number) => ({
    threadKey: slackThreadKey({ sourceId: 'slack_aaa', channelId, ts }),
    createdAt,
  })
  const bursts = coalesceThreadBursts([
    ev('C_general', '1.1', 100),
    ev('C_random', '2.1', 200),
    ev('C_alerts', '3.1', 300),
  ])
  assert.equal(bursts.length, 3, 'three distinct channels in one workspace stay three conversations')
  const convos = new Set(bursts.map((b) => b.address.conversation))
  assert.deepEqual([...convos].sort(), ['C_alerts', 'C_general', 'C_random'])
  ok('coalesceThreadBursts groups per channel, not per workspace')
}

// ---- 2. parseApprovalReply: hard keywords as a sentence prefix are NOT decisions ----
{
  // The exact failure scenario: ordinary messages whose first word is a decision keyword.
  for (const notADecision of ['approve the budget tomorrow', 'cancel that', 'reject it', 'deny the rumor please', 'allow me to explain']) {
    assert.equal(parseApprovalReply(notADecision), null, `"${notADecision}" must NOT resolve an approval`)
  }
  // But the legitimate decision payloads still resolve.
  assert.equal(parseApprovalReply('approve'), 'approve')
  assert.equal(parseApprovalReply('/approve'), 'approve')
  assert.equal(parseApprovalReply('APPROVE'), 'approve')
  assert.equal(parseApprovalReply('approve:ca_ab12'), 'approve', 'token-suffixed button payload still works')
  assert.equal(parseApprovalReply('/deny:ca_ab12'), 'deny')
  assert.equal(parseApprovalReply('deny'), 'deny')
  assert.equal(parseApprovalReply('cancel'), 'deny')
  assert.equal(parseApprovalReply('👍'), 'approve')
  assert.equal(parseApprovalReply('❌'), 'deny')
  // Soft affirmations remain whole-reply-only.
  assert.equal(parseApprovalReply('yes'), 'approve')
  assert.equal(parseApprovalReply('okay so then'), null)
  ok('parseApprovalReply requires hard keywords to be the whole reply / command / button payload')
}

// ---- 3. mattermostEventFromMessage stamps messageId from ITS OWN message ----
{
  const mk = (id: string, text: string): MattermostReadResult => ({
    id,
    sourceId: 'mm1',
    sourceLabel: 'MM',
    channelId: 'chan1',
    channelName: 'general',
    teamName: 'team',
    senderId: 'u1',
    senderName: 'Alice',
    isBot: false,
    date: 1000,
    text,
    snippet: text,
    rootId: null,
    replyTo: null,
    replyToAuthor: null,
    replyToText: null,
    threadKey: 'mattermost:mm1:chan1',
    raw: {},
  })
  // Two DISTINCT posts with identical text must keep distinct messageIds (no content collision).
  const a = mattermostEventFromMessage(mk('post_a', 'ok'))
  const b = mattermostEventFromMessage(mk('post_b', 'ok'))
  assert.equal(a.payload?.messageId, 'post_a')
  assert.equal(b.payload?.messageId, 'post_b')
  assert.notEqual(a.payload?.messageId, b.payload?.messageId, 'identical-text posts keep distinct messageIds')
  assert.ok(a.payloadKeys.includes('messageId'))
  ok('mattermostEventFromMessage stamps messageId from its own source post')
}

console.log(`\n${passed} checks passed.`)
