// Run: jiti electron/agent/googlechat-connector.test.ts
import assert from 'node:assert/strict'
import {
  makeGoogleChatSource,
  validateGoogleChatSource,
  verifyGoogleChatConnection,
  isGoogleChatChannelAllowed,
  fetchGoogleChatMessages,
  postGoogleChatMessage,
  extractGoogleChatMediaUrls,
  googlechatEventFromMessage,
} from './googlechat-connector.ts'
import type { GoogleChatReadResult, GoogleChatSource } from './googlechat-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// A valid-shaped incoming-webhook URL (host + /v1/spaces/<id>/messages + key & token).
const WEBHOOK = 'https://chat.googleapis.com/v1/spaces/AAAAspace1/messages?key=abc&token=xyz'

// ---- fetch stub (save/restore) -------------------------------------------------
const realFetch = globalThis.fetch
type StubFn = (url: string, init?: any) => { status: number; body: any }
function stubFetch(handler: StubFn) {
  globalThis.fetch = (async (input: any, init?: any) => {
    const url = typeof input === 'string' ? input : String(input)
    const { status, body } = handler(url, init)
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as any
  }) as any
}
function restoreFetch() { globalThis.fetch = realFetch }

console.log('googlechat-connector tests:')

{
  // A fresh source is enabled, send-first (empty webhook), with a stable id + empty allowlist.
  const s = makeGoogleChatSource('Eng space', 1000)
  assert.match(s.id, /^googlechat_/)
  assert.equal(s.enabled, true)
  assert.equal(s.webhookUrl, '')
  assert.equal(s.label, 'Eng space')
  assert.equal(s.allowedChannels, '')
  ok('makeGoogleChatSource creates an enabled, send-first source with a stable id')
}

{
  // Validation needs at least one credential, and rejects a malformed webhook URL.
  assert.match(validateGoogleChatSource(makeGoogleChatSource('x'))!, /webhook URL.*required|required/i)
  assert.match(
    validateGoogleChatSource({ ...makeGoogleChatSource('x'), webhookUrl: 'https://evil.example.com/hook' })!,
    /chat\.googleapis\.com/i,
  )
  assert.match(
    validateGoogleChatSource({ ...makeGoogleChatSource('x'), webhookUrl: 'https://chat.googleapis.com/v1/spaces/AAAA/messages' })!,
    /key\/token/i,
  )
  assert.equal(validateGoogleChatSource({ ...makeGoogleChatSource('x'), webhookUrl: WEBHOOK }), null)
  // A bearer token alone (Option B) is also a valid credential, no webhook needed.
  assert.equal(validateGoogleChatSource({ ...makeGoogleChatSource('x'), accessToken: 'ya29.fake' }), null)
  ok('validateGoogleChatSource requires a credential and checks webhook URL shape')
}

{
  // verify never posts: it parses the space id out of the webhook URL (no network).
  const res = await verifyGoogleChatConnection({ ...makeGoogleChatSource('x'), webhookUrl: WEBHOOK })
  assert.equal(res.ok, true)
  assert.equal(res.mode, 'webhook')
  assert.equal(res.space, 'spaces/AAAAspace1')
  ok('verifyGoogleChatConnection validates URL shape + extracts the space without posting')
}

{
  // Allowlist matches the full `spaces/ID` name OR a bare id, case-insensitively; a
  // non-empty allowlist that omits the space denies.
  assert.equal(isGoogleChatChannelAllowed('', 'spaces/AAAA'), true, 'no allowlist => unrestricted')
  assert.equal(isGoogleChatChannelAllowed('spaces/AAAA, spaces/BBBB', 'spaces/aaaa'), true, 'full name in allowlist (case-insensitive)')
  assert.equal(isGoogleChatChannelAllowed('AAAA', 'spaces/AAAA'), true, 'bare id allows the full name')
  assert.equal(isGoogleChatChannelAllowed('spaces/AAAA', 'spaces/ZZZZ'), false, 'space NOT in allowlist rejected')
  ok('isGoogleChatChannelAllowed gates on space name/id, case-insensitive, denies by default')
}

{
  // READ allowlist (Option B): with a token, fetch only returns messages for ALLOWED spaces.
  // The stub serves /v1/spaces and per-space /messages; the disallowed space is never queried.
  let queriedDisallowed = false
  stubFetch((url) => {
    if (url.includes('/v1/spaces?') || url.endsWith('/v1/spaces')) {
      return { status: 200, body: { spaces: [
        { name: 'spaces/AAAA', displayName: 'Allowed', spaceType: 'SPACE' },
        { name: 'spaces/ZZZZ', displayName: 'Forbidden', spaceType: 'SPACE' },
      ] } }
    }
    if (url.includes('/v1/spaces/ZZZZ/messages')) { queriedDisallowed = true; return { status: 200, body: { messages: [] } } }
    if (url.includes('/v1/spaces/AAAA/messages')) {
      return { status: 200, body: { messages: [
        { name: 'spaces/AAAA/messages/m1', sender: { displayName: 'Maya', email: 'maya@example.com', type: 'HUMAN' }, text: 'hi there', createTime: '2026-01-01T00:00:00Z', thread: { name: 'spaces/AAAA/threads/t1' } },
        { name: 'spaces/AAAA/messages/m2', sender: { displayName: 'Bot', type: 'BOT' }, text: 'i am a bot', createTime: '2026-01-01T00:01:00Z' },
      ] } }
    }
    return { status: 404, body: { error: { message: 'not found' } } }
  })
  try {
    const source: GoogleChatSource = { ...makeGoogleChatSource('x'), accessToken: 'ya29.fake', allowedChannels: 'spaces/AAAA' }
    const msgs = await fetchGoogleChatMessages(source)
    assert.equal(queriedDisallowed, false, 'disallowed space is never fetched (allowlist on read)')
    assert.equal(msgs.length, 1, 'bot-authored message skipped, only the human message survives')
    const m = msgs[0]
    assert.equal(m.id, 'spaces/AAAA/messages/m1')
    assert.equal(m.senderName, 'Maya')
    assert.equal(m.isBot, false)
    assert.equal(m.text, 'hi there')
    assert.equal(m.threadName, 'spaces/AAAA/threads/t1')
    assert.equal(m.threadKey, `googlechat:${source.id}:spaces/AAAA/threads/t1`)
    ok('fetchGoogleChatMessages normalizes messages, skips bots, and honors the read allowlist')
  } finally {
    restoreFetch()
  }
}

{
  // WRITE allowlist (fail closed): posting to a space NOT in the allowlist throws before any
  // fetch; posting to an allowed space hits the webhook and returns ok.
  let posted = false
  stubFetch((url, init) => {
    posted = true
    assert.equal(init?.method, 'POST')
    assert.equal(typeof init?.dispatcher?.dispatch, 'function', 'webhook delivery uses a DNS-pinned dispatcher')
    return { status: 200, body: { name: 'spaces/AAAAspace1/messages/created1', thread: { name: 'spaces/AAAAspace1/threads/tk' } } }
  })
  try {
    const source: GoogleChatSource = { ...makeGoogleChatSource('x'), webhookUrl: WEBHOOK, allowedChannels: 'spaces/AAAAspace1' }
    await assert.rejects(
      () => postGoogleChatMessage(source, 'spaces/ZZZZ', 'nope'),
      /not in this source's allowed channels/i,
      'send to a disallowed space is rejected (fail closed)',
    )
    assert.equal(posted, false, 'no network call happened for the rejected send')
    const res = await postGoogleChatMessage(source, 'spaces/AAAAspace1', 'hello team')
    assert.equal(res.ok, true)
    assert.equal(res.id, 'spaces/AAAAspace1/messages/created1')
    assert.equal(posted, true, 'allowed send posts the webhook')
    ok('postGoogleChatMessage enforces the write allowlist (fail closed) then posts')
  } finally {
    restoreFetch()
  }
}

{
  // Webhook-only sources can't read: fetch rejects with the token-required notice.
  const source: GoogleChatSource = { ...makeGoogleChatSource('x'), webhookUrl: WEBHOOK }
  await assert.rejects(() => fetchGoogleChatMessages(source), /requires a service-account|access token/i)
  ok('fetchGoogleChatMessages on a webhook-only source signals that inbound needs a token')
}

{
  // Media extraction pulls attachment download URIs (prefers downloadUri over thumbnailUri),
  // de-duplicated; a message with no attachments yields nothing.
  const raw = {
    attachment: [
      { downloadUri: 'https://chat.google.com/dl/a/1.png', thumbnailUri: 'https://chat.google.com/th/a/1.png' },
      { thumbnailUri: 'https://chat.google.com/th/a/2.png' }, // only thumbnail present
    ],
  }
  assert.deepEqual(extractGoogleChatMediaUrls(raw), [
    'https://chat.google.com/dl/a/1.png',
    'https://chat.google.com/th/a/2.png',
  ])
  assert.deepEqual(extractGoogleChatMediaUrls({}), [])
  assert.deepEqual(extractGoogleChatMediaUrls(null), [])
  ok('extractGoogleChatMediaUrls pulls attachment uris, prefers download, dedupes')
}

{
  // A message becomes an inbound connector event keyed by its thread, with the googlechat id.
  const msg: GoogleChatReadResult = {
    id: 'spaces/AAAA/messages/m9',
    sourceId: 'googlechat_1',
    sourceLabel: 'Eng',
    spaceName: 'spaces/AAAA',
    spaceDisplayName: 'Engineering',
    threadName: 'spaces/AAAA/threads/t9',
    senderId: 'users/42',
    senderName: 'Sam',
    senderEmail: 'sam@example.com',
    isBot: false,
    date: 2000,
    text: 'deploy ready?',
    snippet: 'deploy ready?',
    replyTo: null,
    threadKey: 'googlechat:googlechat_1:spaces/AAAA/threads/t9',
    raw: {},
  }
  const event = googlechatEventFromMessage(msg, 2000)
  assert.equal(event.connectorId, 'googlechat')
  assert.equal(event.direction, 'inbound')
  assert.equal(event.classification, 'communication')
  assert.equal(event.action, 'message_received')
  assert.equal(event.threadKey, 'googlechat:googlechat_1:spaces/AAAA/threads/t9')
  assert.equal(event.actor, 'Sam')
  assert.equal(event.payload?.space, 'spaces/AAAA')
  assert.deepEqual(event.payloadKeys, ['text', 'space', 'thread'])
  ok('googlechatEventFromMessage maps a message to an inbound thread-keyed event')
}

console.log(`\n${passed} checks passed.`)
