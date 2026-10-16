// Run: jiti electron/agent/thread-address.test.ts
import assert from 'node:assert/strict'
import { parseThreadAddress, formatThreadAddress, sameConversation, coalesceThreadBursts } from './thread-address.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-address tests:')

{
  // Shape by part count, matching how each connector builds its key.
  assert.deepEqual(parseThreadAddress('imessage:+447700900500'), { platform: 'imessage', account: null, conversation: '+447700900500', message: null, raw: 'imessage:+447700900500' })
  assert.deepEqual(parseThreadAddress('discord:src1:chan9'), { platform: 'discord', account: 'src1', conversation: 'chan9', message: null, raw: 'discord:src1:chan9' })
  assert.deepEqual(parseThreadAddress('slack:T1:C2:1699999999.001'), { platform: 'slack', account: 'T1', conversation: 'C2', message: '1699999999.001', raw: 'slack:T1:C2:1699999999.001' })
  assert.equal(parseThreadAddress('')!, null as any)
  assert.equal(parseThreadAddress('plainword')?.platform, 'plainword')
  ok('parseThreadAddress: 1/2/3-part shapes + edge cases')
}

{
  // Round-trips.
  for (const k of ['imessage:+447', 'discord:s:c', 'slack:T1:C2:ts', 'email:root123']) {
    assert.equal(formatThreadAddress(parseThreadAddress(k)!), k, `round-trip ${k}`)
  }
  ok('formatThreadAddress: round-trips parseThreadAddress')
}

{
  // Same conversation ignores the message id; different channel/account differ.
  assert.equal(sameConversation(parseThreadAddress('slack:T1:C2:ts1'), parseThreadAddress('slack:T1:C2:ts2')), true)
  assert.equal(sameConversation(parseThreadAddress('slack:T1:C2:ts1'), parseThreadAddress('slack:T1:C3:ts1')), false)
  assert.equal(sameConversation(parseThreadAddress('whatsapp:a:jid'), parseThreadAddress('whatsapp:b:jid')), false)
  ok('sameConversation: same platform+account+conversation, message ignored')
}

{
  // Coalesce: one signal per conversation, burst when several land in the window.
  const ev = (threadKey: string, createdAt: number) => ({ threadKey, createdAt })
  const bursts = coalesceThreadBursts([
    ev('slack:T1:C2:a', 1000),
    ev('slack:T1:C2:b', 1500), // same conversation, different message
    ev('slack:T1:C2:c', 2000), // 3 within the window -> burst
    ev('imessage:+447', 5000),
  ], 60_000)
  assert.equal(bursts.length, 2, 'two conversations')
  const slack = bursts.find((b) => b.address.conversation === 'C2')!
  assert.equal(slack.count, 3)
  assert.equal(slack.burst, true, 'rapid back-and-forth is one burst signal')
  assert.equal(slack.latest.createdAt, 2000, 'latest message represents the thread')
  const im = bursts.find((b) => b.address.platform === 'imessage')!
  assert.equal(im.count, 1)
  assert.equal(im.burst, false)
  // newest-first
  assert.equal(bursts[0].address.platform, 'imessage')
  ok('coalesceThreadBursts: one signal/conversation, burst flag, newest-first')
}

console.log(`\n${passed} passed`)
