// Run: jiti electron/agent/channel-split.test.ts
import assert from 'node:assert/strict'
import {
  splitMessage,
  splitForPlatform,
  limitForPlatform,
  PLATFORM_MESSAGE_LIMITS,
} from './channel-split.ts'
import { sendViaChannel, streamDraftViaChannel, draftStreamingPlatforms, composingDraftStream, COMPOSING_INDICATOR, type ChannelAdapter, type DraftHandle } from './channel-adapter.ts'
import { makeDraftEditThrottle } from './draft-throttle.ts'

// A throttle that never drops a frame — for the tests that assert the full per-frame animation
// contract (the production default spaces edits 1.5s apart, which these synchronous streams would
// otherwise collapse). The throttle's own behaviour is covered in draft-throttle.test.ts.
const noThrottle = () => makeDraftEditThrottle(0)

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// UTF-16 code-unit length (what the platforms cap on). `[...s].length` would count code
// POINTS — we deliberately use .length here to assert the surrogate-aware cap.
const units = (s: string) => s.length

console.log('channel-split tests:')

// ---- no-op under the limit ----
{
  assert.deepEqual(splitMessage('hello world', { maxUnits: 2000 }), ['hello world'])
  // Exactly at the limit is still one chunk.
  const exact = 'x'.repeat(100)
  assert.deepEqual(splitMessage(exact, { maxUnits: 100 }), [exact])
  // No cap (null / 0 / undefined) -> single trimmed chunk.
  assert.deepEqual(splitMessage('  spaced  ', { maxUnits: null }), ['spaced'])
  assert.deepEqual(splitMessage('a b c', {}), ['a b c'])
  // Empty / whitespace -> no chunks at all (the silence guard owns the empty case downstream).
  assert.deepEqual(splitMessage('', { maxUnits: 10 }), [])
  assert.deepEqual(splitMessage('   \n  ', { maxUnits: 10 }), [])
  ok('splitMessage: under/at the limit and unlimited inputs pass through unsplit')
}

// ---- correct chunking over the limit; every chunk obeys the cap ----
{
  const para = 'A'.repeat(1500) + '\n\n' + 'B'.repeat(1500) + '\n\n' + 'C'.repeat(1500)
  const chunks = splitMessage(para, { maxUnits: 2000 })
  assert.ok(chunks.length >= 2, 'a long reply splits into multiple chunks')
  for (const c of chunks) assert.ok(units(c) <= 2000, `chunk within cap (${units(c)})`)
  // Reassembling the chunks recovers all the letters (no content dropped, only break whitespace).
  const joined = chunks.join('')
  assert.equal((joined.match(/A/g) || []).length, 1500)
  assert.equal((joined.match(/B/g) || []).length, 1500)
  assert.equal((joined.match(/C/g) || []).length, 1500)
  ok('splitMessage: over the limit splits into capped chunks that preserve all content')
}

// ---- boundary preference: paragraph > line > sentence > word ----
{
  // A paragraph break well within the window is preferred over cutting mid-text.
  const text = 'First paragraph here.\n\nSecond paragraph here.'
  const chunks = splitMessage(text, { maxUnits: 30 })
  assert.equal(chunks[0], 'First paragraph here.', 'splits on the blank line, not mid-word')
  assert.equal(chunks[1], 'Second paragraph here.')
  ok('splitMessage: prefers the paragraph break')
}
{
  // No paragraph break -> fall back to a single newline.
  const text = 'line one is here\nline two is here\nline three'
  const chunks = splitMessage(text, { maxUnits: 20 })
  assert.equal(chunks[0], 'line one is here', 'breaks at the newline')
  for (const c of chunks) assert.ok(units(c) <= 20)
  ok('splitMessage: falls back to a line break')
}
{
  // No newline at all -> break at a sentence end, then a word boundary.
  const text = 'Sentence one is complete. Sentence two keeps going and going and going.'
  const chunks = splitMessage(text, { maxUnits: 40 })
  assert.equal(chunks[0], 'Sentence one is complete.', 'breaks after the sentence period+space')
  for (const c of chunks) assert.ok(units(c) <= 40)
  ok('splitMessage: falls back to a sentence boundary')
}
{
  // One long run with only spaces -> word breaks, never mid-word (each chunk under cap).
  const text = 'alpha bravo charlie delta echo foxtrot golf hotel india juliet'
  const chunks = splitMessage(text, { maxUnits: 18 })
  for (const c of chunks) {
    assert.ok(units(c) <= 18, `chunk under cap (${units(c)})`)
    assert.ok(!/^\s|\s$/.test(c), 'chunk is trimmed at the break')
  }
  // No word was severed: every original word appears whole somewhere.
  for (const word of text.split(' ')) assert.ok(chunks.some((c) => c.split(/\s/).includes(word)), `"${word}" kept whole`)
  ok('splitMessage: word-breaks without severing a word')
}

// ---- emoji / surrogate-pair safety: a chunk boundary never splits an astral code point ----
{
  // Rocket "🚀" is a surrogate pair (2 UTF-16 units). Pack many so the cap lands inside a pair.
  const rockets = '🚀'.repeat(50) // 100 UTF-16 units, no break chars
  const chunks = splitMessage(rockets, { maxUnits: 7 }) // 7 is odd: a naive cut lands mid-pair
  for (const c of chunks) {
    assert.ok(units(c) <= 7, `chunk within cap (${units(c)})`)
    // No chunk may start with a lone low-surrogate or end with a lone high-surrogate.
    const first = c.charCodeAt(0)
    const last = c.charCodeAt(c.length - 1)
    assert.ok(!(first >= 0xdc00 && first <= 0xdfff), 'no leading lone low-surrogate')
    assert.ok(!(last >= 0xd800 && last <= 0xdbff), 'no trailing lone high-surrogate')
    // Every rocket in the chunk is intact: code-point count * 2 == unit count (all pairs).
    assert.equal([...c].length * 2, units(c), 'only whole surrogate pairs in the chunk')
  }
  // All 50 rockets survive reassembly.
  assert.equal([...chunks.join('')].filter((ch) => ch === '🚀').length, 50)
  ok('splitMessage: never splits a surrogate pair (emoji stay intact)')
}
{
  // Mixed text + emoji with a space boundary just before the cap.
  const text = 'wave 👋 and rocket 🚀 then star ⭐ and more text to overflow the small limit'
  const chunks = splitMessage(text, { maxUnits: 16 })
  for (const c of chunks) {
    assert.ok(units(c) <= 16)
    const last = c.charCodeAt(c.length - 1)
    const first = c.charCodeAt(0)
    assert.ok(!(last >= 0xd800 && last <= 0xdbff), 'no dangling high-surrogate at chunk end')
    assert.ok(!(first >= 0xdc00 && first <= 0xdfff), 'no dangling low-surrogate at chunk start')
  }
  ok('splitMessage: mixed emoji/text chunks keep pairs intact and stay under cap')
}

// ---- the per-platform limit table ----
{
  assert.equal(PLATFORM_MESSAGE_LIMITS.discord.maxUnits, 2000)
  assert.equal(PLATFORM_MESSAGE_LIMITS.telegram.maxUnits, 4096)
  assert.equal(PLATFORM_MESSAGE_LIMITS.whatsapp.maxUnits, 1024)
  assert.ok((PLATFORM_MESSAGE_LIMITS.slack.maxUnits ?? 0) <= 4000, 'slack stays under the ~4000 text cap')
  assert.equal(PLATFORM_MESSAGE_LIMITS.email.maxUnits, null, 'email is uncapped')
  // limitForPlatform: known platforms resolve; unknown / missing fall back to unlimited.
  assert.equal(limitForPlatform('discord').maxUnits, 2000)
  assert.equal(limitForPlatform('does-not-exist').maxUnits, null)
  assert.equal(limitForPlatform(null).maxUnits, null)
  assert.equal(limitForPlatform(undefined).maxUnits, null)
  ok('limit table: per-platform caps and the unlimited fallback')
}

// ---- splitForPlatform applies the right cap ----
{
  const long = 'word '.repeat(600).trim() // 2999 units, over Discord's 2000
  assert.ok(splitForPlatform('discord', long).length >= 2, 'discord splits a 3000-char reply')
  assert.equal(splitForPlatform('email', long).length, 1, 'email never splits')
  assert.equal(splitForPlatform('email', long)[0], long)
  // Under Telegram's big cap it stays whole.
  assert.equal(splitForPlatform('telegram', long).length, 1, 'under telegram cap -> one message')
  ok('splitForPlatform: discord splits, email/telegram pass a 3k reply whole')
}

// ---- sendViaChannel multi-send: continuation ids + chunkCount, first chunk anchored ----
{
  const sent: Array<{ text: string; inReplyTo: string | null }> = []
  let n = 0
  const adapter: ChannelAdapter = {
    platform: 'discord',
    send: async (_addr, reply) => { sent.push({ text: reply.text, inReplyTo: reply.inReplyTo ?? null }); n++; return { ok: true, sentId: `m${n}` } },
  }
  const long = 'word '.repeat(900).trim() // ~4499 units, > 2 Discord chunks
  const r = await sendViaChannel([adapter], 'discord:src:chan', { text: long, inReplyTo: 'origin-1' })
  assert.equal(r.ok, true)
  assert.ok(sent.length >= 2, 'sent as multiple messages')
  assert.equal(r.sentId, 'm1', 'sentId is the first message id')
  assert.deepEqual(r.continuationIds, sent.slice(1).map((_v, i) => `m${i + 2}`), 'continuation ids in order')
  assert.equal(r.chunkCount, sent.length, 'chunkCount matches messages sent')
  assert.equal(sent[0].inReplyTo, 'origin-1', 'only the first chunk anchors to inReplyTo')
  assert.equal(sent[1].inReplyTo, null, 'continuations are not re-anchored')
  for (const m of sent) assert.ok(m.text.length <= 2000, 'each sent chunk under the Discord cap')
  ok('sendViaChannel: long reply -> multiple messages with continuationIds + chunkCount')
}

// ---- sendViaChannel single-send is unchanged (no continuation fields) ----
{
  const adapter: ChannelAdapter = { platform: 'slack', send: async () => ({ ok: true, sentId: 'only' }) }
  const r = await sendViaChannel([adapter], 'slack:T:C:1', { text: 'short reply' })
  assert.equal(r.ok, true)
  assert.equal(r.sentId, 'only')
  assert.equal(r.continuationIds, undefined, 'no continuationIds for a single message')
  assert.equal(r.chunkCount, undefined, 'no chunkCount for a single message')
  ok('sendViaChannel: short reply path is byte-for-byte the old single send')
}

// ---- sendViaChannel stops on a mid-stream failure and reports what got through ----
{
  let n = 0
  const adapter: ChannelAdapter = {
    platform: 'discord',
    send: async () => { n++; return n === 1 ? { ok: true, sentId: 'm1' } : { ok: false, error: 'rate limited' } },
  }
  const long = 'word '.repeat(900).trim()
  const r = await sendViaChannel([adapter], 'discord:src:chan', { text: long })
  assert.equal(r.ok, false)
  assert.match(r.error || '', /rate limited/)
  assert.equal(r.sentId, 'm1', 'reports the first message that did go out')
  assert.equal(r.chunkCount, 1, 'chunkCount reflects only what was actually sent')
  ok('sendViaChannel: aborts after a failed continuation and reports partial progress')
}

// ---- draft streaming: a capable adapter animates a preview then commits ----
{
  const frames: string[] = []
  let opened = ''
  let finalText = ''
  const adapter: ChannelAdapter = {
    platform: 'telegram',
    supportsDraftStreaming: true,
    send: async () => ({ ok: true, sentId: 'send' }),
    sendDraft: async (_addr, text) => { opened = text; return { ok: true, handle: { messageId: 'd1', platform: 'telegram' } } },
    updateDraft: async (_addr, handle, text) => { frames.push(text); return { ok: true, handle } },
    finalizeDraft: async (_addr, _handle, reply) => { finalText = reply.text; return { ok: true, sentId: 'd1' } },
  }
  async function* partials() { yield 'Hel'; yield 'Hello'; yield 'Hello wor'; yield 'Hello world' }
  const r = await streamDraftViaChannel([adapter], 'telegram:123', { partials: partials(), final: { text: 'Hello world!' } }, noThrottle())
  assert.equal(r.ok, true)
  assert.equal(r.sentId, 'd1')
  assert.equal(opened, 'Hel', 'first partial opened the draft')
  assert.deepEqual(frames, ['Hello', 'Hello wor', 'Hello world'], 'each later partial edited in place')
  assert.equal(finalText, 'Hello world!', 'finalize committed the final reply text')
  ok('streamDraftViaChannel: capable adapter opens, edits in place, then finalizes')
}

// ---- draft streaming: final text defaults to the last partial when none is given ----
{
  let finalText = ''
  const adapter: ChannelAdapter = {
    platform: 'telegram',
    supportsDraftStreaming: true,
    send: async () => ({ ok: true }),
    sendDraft: async () => ({ ok: true, handle: { messageId: 'd1', platform: 'telegram' } }),
    updateDraft: async (_addr, handle) => ({ ok: true, handle }),
    finalizeDraft: async (_addr, _handle, reply) => { finalText = reply.text; return { ok: true, sentId: 'd1' } },
  }
  const r = await streamDraftViaChannel([adapter], 'telegram:123', { partials: ['draft a', 'draft b final'] }, noThrottle())
  assert.equal(r.ok, true)
  assert.equal(finalText, 'draft b final', 'last partial is the committed text when no final given')
  ok('streamDraftViaChannel: defaults the final text to the last partial')
}

// ---- draft streaming: the throttle drops rapid intermediate edits but the final still commits ----
{
  const frames: string[] = []
  let finalText = ''
  const adapter: ChannelAdapter = {
    platform: 'telegram',
    supportsDraftStreaming: true,
    send: async () => ({ ok: true }),
    sendDraft: async () => ({ ok: true, handle: { messageId: 'd1', platform: 'telegram' } }),
    updateDraft: async (_addr, handle, text) => { frames.push(text); return { ok: true, handle } },
    finalizeDraft: async (_addr, _handle, reply) => { finalText = reply.text; return { ok: true, sentId: 'd1' } },
  }
  // A burst of token frames arriving back-to-back (no real time elapses between synchronous yields).
  // With the production default (1.5s spacing) every intermediate edit is dropped — the draft opens,
  // nothing is edited mid-stream, and finalize lands the complete text.
  async function* burst() { yield 'a'; yield 'ab'; yield 'abc'; yield 'abcd'; yield 'abcde' }
  const r = await streamDraftViaChannel([adapter], 'telegram:123', { partials: burst(), final: { text: 'abcde done' } })
  assert.equal(r.ok, true)
  assert.equal(frames.length, 0, 'rapid intermediate edits are all throttled away (no editMessageText burst)')
  assert.equal(finalText, 'abcde done', 'the final commit is never throttled — full text still lands')
  ok('streamDraftViaChannel: throttle suppresses a rapid edit burst yet finalizes the full reply')
}

// ---- composingDraftStream: a finished-string reply animates a placeholder then commits the text ----
{
  let openedWith = ''
  let finalText = ''
  let finalInReplyTo: string | null | undefined = 'unset'
  const edits: string[] = []
  const adapter: ChannelAdapter = {
    platform: 'telegram',
    supportsDraftStreaming: true,
    send: async () => ({ ok: true, sentId: 'send' }),
    sendDraft: async (_addr, text) => { openedWith = text; return { ok: true, handle: { messageId: 'd1', platform: 'telegram' } } },
    updateDraft: async (_addr, handle, text) => { edits.push(text); return { ok: true, handle } },
    finalizeDraft: async (_addr, _handle, reply) => { finalText = reply.text; finalInReplyTo = reply.inReplyTo; return { ok: true, sentId: 'd1' } },
  }
  // Default throttle on purpose: opening the placeholder and finalizing happen in the same tick, so
  // no intermediate editMessageText fires — exactly the connector:reply shape.
  const r = await streamDraftViaChannel([adapter], 'telegram:123', composingDraftStream('The real reply', 'origin-7'))
  assert.equal(r.ok, true)
  assert.equal(r.sentId, 'd1')
  assert.equal(openedWith, COMPOSING_INDICATOR, 'the draft opens on the composing placeholder, not the real text')
  assert.equal(edits.length, 0, 'no intermediate edit — placeholder goes straight to finalize')
  assert.equal(finalText, 'The real reply', 'finalize commits the true reply text')
  assert.equal(finalInReplyTo, 'origin-7', 'the inReplyTo anchor is carried into finalize')
  ok('composingDraftStream: opens a placeholder then finalizes the real reply on a capable channel')
}

// ---- composingDraftStream: non-capable channel sends the real text once (no placeholder leaks) ----
{
  const sent: string[] = []
  const plain: ChannelAdapter = { platform: 'imessage', send: async (_a, reply) => { sent.push(reply.text); return { ok: true, sentId: 'm1' } } }
  const r = await streamDraftViaChannel([plain], 'imessage:+447', composingDraftStream('Just send it'))
  assert.equal(r.ok, true)
  assert.deepEqual(sent, ['Just send it'], 'exactly one message, the real text — the "…" placeholder never goes out')
  ok('composingDraftStream: degrades to a single real-text send on a non-streaming channel')
}

// ---- composingDraftStream: an empty reply animates nothing and hits the silence guard ----
{
  const sent: string[] = []
  let opened = false
  const adapter: ChannelAdapter = {
    platform: 'telegram',
    supportsDraftStreaming: true,
    send: async (_a, r) => { sent.push(r.text); return { ok: true } },
    sendDraft: async () => { opened = true; return { ok: true, handle: { messageId: 'd1', platform: 'telegram' } } },
    finalizeDraft: async (_a, _h, r) => { sent.push(r.text); return { ok: true } },
  }
  const r = await streamDraftViaChannel([adapter], 'telegram:123', composingDraftStream('   '))
  assert.equal(r.ok, false, 'an all-whitespace reply is refused')
  assert.match(r.error || '', /empty/i)
  assert.equal(opened, false, 'no placeholder draft opened for an empty reply')
  assert.equal(sent.length, 0, 'nothing posted')
  ok('composingDraftStream: empty reply yields no placeholder and hits the silence guard')
}

// ---- draft streaming: graceful fallback for an adapter without the capability ----
{
  const sent: string[] = []
  let drafted = false
  const plain: ChannelAdapter = {
    platform: 'imessage',
    // No supportsDraftStreaming / sendDraft -> must fall back to a single send.
    send: async (_addr, reply) => { sent.push(reply.text); return { ok: true, sentId: 'm1' } },
    sendDraft: async () => { drafted = true; return { ok: false } }, // present but capability flag off
  }
  const r = await streamDraftViaChannel([plain], 'imessage:+447', { partials: ['typ', 'typing', 'typing…done'], final: { text: 'Final answer' } })
  assert.equal(r.ok, true)
  assert.equal(drafted, false, 'no draft was opened on a non-streaming adapter')
  assert.deepEqual(sent, ['Final answer'], 'exactly one message sent, the final text')
  ok('streamDraftViaChannel: falls back to a single send when streaming is unsupported')
}

// ---- draft streaming: an empty stream + empty final is the silence guard, not a send ----
{
  const sent: string[] = []
  const adapter: ChannelAdapter = { platform: 'imessage', send: async (_a, r) => { sent.push(r.text); return { ok: true } } }
  const r = await streamDraftViaChannel([adapter], 'imessage:+447', { partials: ['  ', ''] })
  assert.equal(r.ok, false, 'nothing to send -> refused')
  assert.match(r.error || '', /empty/i)
  assert.equal(sent.length, 0, 'no blank message posted')
  ok('streamDraftViaChannel: empty stream with no final hits the silence guard')
}

// ---- draftStreamingPlatforms lists only fully-capable adapters ----
{
  const handle: DraftHandle = { messageId: 'x', platform: 'telegram' }
  const adapters: ChannelAdapter[] = [
    { platform: 'imessage', send: async () => ({ ok: true }) },
    { platform: 'telegram', supportsDraftStreaming: true, send: async () => ({ ok: true }), sendDraft: async () => ({ ok: true, handle }), finalizeDraft: async () => ({ ok: true }) },
    // Flag on but missing finalizeDraft -> not listed (incomplete capability).
    { platform: 'slack', supportsDraftStreaming: true, send: async () => ({ ok: true }), sendDraft: async () => ({ ok: true, handle }) },
  ]
  assert.deepEqual(draftStreamingPlatforms(adapters), ['telegram'])
  ok('draftStreamingPlatforms: only adapters with the flag + sendDraft + finalizeDraft')
}

console.log(`\n${passed} passed`)
