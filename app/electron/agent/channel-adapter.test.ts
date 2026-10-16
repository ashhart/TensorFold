// Run: jiti electron/agent/channel-adapter.test.ts
import assert from 'node:assert/strict'
import { sendViaChannel, resolveAdapter, sendablePlatforms, draftStreamingPlatforms, streamDraftViaChannel, type ChannelAdapter, type DraftHandle } from './channel-adapter.ts'
import { parseThreadAddress } from './thread-address.ts'
import { makeDraftEditThrottle, draftEditIntervalForPlatform } from './draft-throttle.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function fakeAdapter(platform: string, sink: any[]): ChannelAdapter {
  return { platform, send: async (address, reply) => { sink.push({ platform, address, reply }); return { ok: true, sentId: `${platform}-1` } } }
}

console.log('channel-adapter tests:')

{
  // Routes to the adapter matching the threadKey's platform, with the parsed address.
  const sink: any[] = []
  const adapters = [fakeAdapter('imessage', sink), fakeAdapter('slack', sink)]
  const r = await sendViaChannel(adapters, 'slack:T1:C2:169.1', { text: 'hi', inReplyTo: '169.1' })
  assert.equal(r.ok, true)
  assert.equal(r.sentId, 'slack-1')
  assert.equal(sink.length, 1)
  assert.equal(sink[0].platform, 'slack')
  assert.equal(sink[0].address.conversation, 'C2')
  assert.equal(sink[0].reply.text, 'hi')
  assert.equal(sink[0].reply.inReplyTo, '169.1')
  ok('sendViaChannel: routes by platform with the parsed address + reply')
}

{
  // Outbound silence guard: empty/whitespace reply is never sent.
  const sink: any[] = []
  const adapters = [fakeAdapter('imessage', sink)]
  assert.equal((await sendViaChannel(adapters, 'imessage:+447', { text: '   ' })).ok, false)
  assert.equal((await sendViaChannel(adapters, 'imessage:+447', { text: '' })).ok, false)
  assert.equal(sink.length, 0, 'no send attempted for an empty reply')
  ok('sendViaChannel: silence guard blocks empty replies')
}

{
  // Unknown platform / bad key → graceful error, no throw.
  const adapters = [fakeAdapter('imessage', [])]
  assert.match((await sendViaChannel(adapters, 'slack:T1:C2', { text: 'hi' })).error || '', /No channel adapter/)
  assert.match((await sendViaChannel(adapters, '', { text: 'hi' })).error || '', /Unrecognized/)
  ok('sendViaChannel: unknown platform + empty key handled')
}

{
  // An adapter that throws is caught into a normalized error result.
  const boom: ChannelAdapter = { platform: 'discord', send: async () => { throw new Error('rate limited') } }
  const r = await sendViaChannel([boom], 'discord:s:c', { text: 'hi' })
  assert.equal(r.ok, false)
  assert.match(r.error || '', /rate limited/)
  ok('sendViaChannel: adapter throw → normalized error, never rejects')
}

{
  assert.equal(resolveAdapter([fakeAdapter('slack', [])], parseThreadAddress('slack:a:b'))?.platform, 'slack')
  assert.equal(resolveAdapter([fakeAdapter('slack', [])], parseThreadAddress('imessage:x')), null)
  assert.deepEqual(sendablePlatforms([fakeAdapter('slack', []), fakeAdapter('imessage', [])]).sort(), ['imessage', 'slack'])
  ok('resolveAdapter / sendablePlatforms')
}

// ---- per-platform draft streaming: capability + per-channel edit-limit selection ----

// A fake edit-in-place adapter modelled on the real Slack/Discord draft adapters: sendDraft posts
// and returns the new message id as the handle, updateDraft records an edit frame, finalizeDraft
// commits the (split-aware) final text. Records every send/edit so a test can assert the frames.
function fakeDraftAdapter(platform: string, log: { sends: string[]; edits: string[] }): ChannelAdapter {
  let n = 0
  return {
    platform,
    supportsDraftStreaming: true,
    send: async (_a, reply) => { log.sends.push(reply.text); return { ok: true, sentId: `${platform}-${++n}` } },
    sendDraft: async (_a, text) => { log.sends.push(text); return { ok: true, handle: { messageId: `${platform}-${++n}`, platform } } },
    updateDraft: async (_a, handle: DraftHandle, text) => { log.edits.push(text); return { ok: true, handle } },
    finalizeDraft: async (_a, _handle, reply) => { log.edits.push(`final:${reply.text}`); return { ok: true, sentId: `${platform}-final`, chunkCount: 1 } },
  }
}

{
  // Capability report: only adapters implementing sendDraft + finalizeDraft are draft-streaming.
  const plain = fakeAdapter('imessage', [])
  const slackD = fakeDraftAdapter('slack', { sends: [], edits: [] })
  const discordD = fakeDraftAdapter('discord', { sends: [], edits: [] })
  assert.deepEqual(draftStreamingPlatforms([plain, slackD, discordD]).sort(), ['discord', 'slack'])
  assert.deepEqual(draftStreamingPlatforms([plain]), [], 'a plain adapter is not draft-capable')
  ok('draftStreamingPlatforms: reports only Slack + Discord as draft-capable')
}

{
  // Streaming path drives sendDraft -> (throttled) updateDraft -> finalizeDraft, landing the final.
  const log = { sends: [] as string[], edits: [] as string[] }
  const adapters = [fakeDraftAdapter('slack', log)]
  // A controllable clock + the SLACK per-platform throttle (1.2s). The opening sendDraft markFinal's
  // the throttle at t=0, so an intermediate frame is allowed only once 1.2s has elapsed.
  let now = 0
  const throttle = (() => {
    const inner = makeDraftEditThrottle(draftEditIntervalForPlatform('slack'))
    return { shouldEmit: () => inner.shouldEmit(now), markFinal: () => inner.markFinal(now) }
  })()
  // The clock advances WHEN the next partial is produced (after the prior yield resumes), so the
  // throttle is evaluated against `now` as it stands when each partial reaches the consumer.
  const partials = (function* () {
    yield 'Hel'                 // opens the draft at now=0 (markFinal(0))
    now = 100; yield 'Hello wor'   // reaches the consumer at now=100 -> throttled (slack needs 1.2s)
    now = 1300; yield 'Hello world' // reaches the consumer at now=1300 -> allowed, edits
  })()
  const r = await streamDraftViaChannel(adapters, 'slack:T1:C2:169.1', { partials, final: { text: 'Hello world!' } }, throttle)
  assert.equal(r.ok, true)
  assert.equal(r.sentId, 'slack-final')
  assert.equal(log.sends.length, 1, 'exactly one draft message opened (sendDraft)')
  assert.equal(log.sends[0], 'Hel', 'the first partial opened the draft')
  // The 100ms frame was dropped by the 1.2s throttle; only the 1.3s frame was emitted as an edit.
  assert.deepEqual(log.edits.filter((e) => !e.startsWith('final:')), ['Hello world'], 'only the post-interval frame edits')
  assert.ok(log.edits.includes('final:Hello world!'), 'finalize always commits the authoritative text')
  ok('streamDraftViaChannel: Slack draft opens, throttles intermediate edits, commits the final')
}

{
  // Graceful fallback: an adapter WITHOUT draft methods never sees a partial — the stream is drained
  // and the final text is sent once via the normal send path.
  const sink: any[] = []
  const adapters = [fakeAdapter('imessage', sink)]
  const partials = (function* () { yield 'partial-1'; yield 'partial-2' })()
  const r = await streamDraftViaChannel(adapters, 'imessage:+447', { partials, final: { text: 'done' } })
  assert.equal(r.ok, true)
  assert.equal(sink.length, 1, 'non-draft adapter sends exactly once')
  assert.equal(sink[0].reply.text, 'done', 'and sends the FINAL text, not a partial')
  ok('streamDraftViaChannel: non-capable adapter degrades to a single final send')
}

console.log(`\n${passed} passed`)
