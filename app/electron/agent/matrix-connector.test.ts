// Run: jiti electron/agent/matrix-connector.test.ts
import assert from 'node:assert/strict'
import {
  makeMatrixSource,
  validateMatrixSource,
  matrixEventFromMessage,
  isMatrixRoomAllowed,
  extractMatrixMediaUrls,
  fetchMatrixMessages,
  postMatrixMessage,
  verifyMatrixConnection,
  listMatrixChannels,
} from './matrix-connector.ts'
import type { MatrixReadResult, MatrixSource } from './matrix-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('matrix-connector tests:')

// --- canned-fetch harness (no live network) -----------------------------------
// Route requests by URL substring so each test wires only the endpoints it exercises.
type Handler = (url: string, init?: any) => { status?: number; body: any }
const realFetch = globalThis.fetch
function installFetch(routes: Array<{ match: string; handle: Handler }>, log?: Array<{ url: string; init?: any }>) {
  globalThis.fetch = (async (input: any, init?: any) => {
    const url = String(input)
    log?.push({ url, init })
    const route = routes.find((r) => url.includes(r.match))
    if (!route) return new Response(JSON.stringify({ errcode: 'M_NOT_FOUND', error: 'no route' }), { status: 404, headers: { 'content-type': 'application/json' } })
    const { status = 200, body } = route.handle(url, init)
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
}
function restoreFetch() { globalThis.fetch = realFetch }

const baseSource = (over: Partial<MatrixSource> = {}): MatrixSource => ({
  ...makeMatrixSource('Account', 1000),
  baseUrl: 'https://example.com',
  token: 'syt_fake',
  ...over,
})

{
  // Synchronous fast-path: unrestricted with no allowlist; explicit id OR alias passes;
  // an unknown id/alias falls through (resolved on the write path). It must NOT be
  // permissive — verifier bypass: agent harvests a restricted room id and posts to it.
  assert.equal(isMatrixRoomAllowed('', '!room:hs.example'), true, 'no allowlist => unrestricted')
  assert.equal(isMatrixRoomAllowed('!room:hs.example', '!room:hs.example'), true, 'id in allowlist')
  assert.equal(isMatrixRoomAllowed('#general:hs.example', '#general:hs.example'), true, 'alias in allowlist')
  assert.equal(isMatrixRoomAllowed('!room:hs.example, #general:hs.example', '!other:hs.example'), false, 'id NOT in allowlist rejected')
  assert.equal(isMatrixRoomAllowed('#general:hs.example', '!unknown:hs.example'), false, 'unknown id falls through (not permissive)')
  ok('isMatrixRoomAllowed gates by id/alias; unknown ids fall through to resolution')
}

{
  // A fresh source is enabled with a stable id, default homeserver, and an empty token.
  const s = makeMatrixSource('My matrix', 1000)
  assert.match(s.id, /^matrix_/)
  assert.equal(s.enabled, true)
  assert.equal(s.token, '')
  assert.equal(s.label, 'My matrix')
  assert.equal(s.allowedRooms, '')
  assert.equal(s.baseUrl, 'https://matrix.org')
  ok('makeMatrixSource creates an enabled source with a stable id + default homeserver')
}

{
  // Validation requires a homeserver URL (http(s)) and an access token before any network.
  assert.match(validateMatrixSource(makeMatrixSource('x'))!, /access token is required/i)
  assert.match(validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'ftp://nope', token: 't' })!, /start with http/i)
  assert.equal(validateMatrixSource({ ...makeMatrixSource('x'), token: 'syt_fake' }), null)
  ok('validateMatrixSource requires an http(s) homeserver + access token')
}

{
  // SSRF guard: a loopback/link-local/metadata baseUrl must be rejected at validate time —
  // an attacker-chosen host like this would otherwise receive the account's Bearer token.
  assert.match(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'http://169.254.169.254', token: 'syt_fake' })!,
    /local\/private/i,
    'cloud metadata endpoint rejected',
  )
  assert.match(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'http://127.0.0.1:6379', token: 'syt_fake' })!,
    /local\/private/i,
    'loopback rejected',
  )
  assert.match(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'http://10.0.0.5', token: 'syt_fake' })!,
    /local\/private/i,
    'private RFC1918 host rejected',
  )
  assert.equal(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'https://matrix.example.com', token: 'syt_fake' }),
    null,
    'a normal public homeserver still validates',
  )
  assert.equal(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'http://10.0.0.5:8008', token: 'syt_fake', allowPrivateNetwork: true }),
    null,
    'an explicitly opted-in self-hosted LAN homeserver validates',
  )
  assert.match(
    validateMatrixSource({ ...makeMatrixSource('x'), baseUrl: 'http://169.254.169.254', token: 'syt_fake', allowPrivateNetwork: true })!,
    /metadata|link-local/i,
    'private-network opt-in never permits cloud metadata/link-local targets',
  )
  ok('validateMatrixSource rejects loopback/private/link-local/metadata homeserver URLs (SSRF)')
}

{
  // Defense in depth: even if a private/loopback baseUrl bypassed validate() (e.g. set
  // directly via updateSource patch, or set valid-at-save-time then DNS-rebound), the live
  // call sites must refuse to fetch — the Bearer token must never reach the bad host.
  const log: Array<{ url: string }> = []
  installFetch([{ match: '/account/whoami', handle: () => ({ body: { user_id: '@me:hs.example' } }) }], log)
  try {
    const evil = baseSource({ baseUrl: 'http://169.254.169.254' })
    await assert.rejects(() => verifyMatrixConnection(evil.baseUrl, evil.token), /local\/private|private address/i)
    assert.equal(log.length, 0, 'no fetch issued for verifyMatrixConnection against a metadata host')

    await assert.rejects(() => listMatrixChannels(baseSource({ baseUrl: 'http://127.0.0.1:6379' })), /local\/private|private address/i)
    assert.equal(log.length, 0, 'no fetch issued for listMatrixChannels against a loopback host')

    await assert.rejects(
      () => postMatrixMessage(baseSource({ baseUrl: 'http://127.0.0.1:6379', allowedRooms: '' }), '!room:hs.example', 'leak me'),
      /local\/private|private address/i,
    )
    assert.equal(log.length, 0, 'no fetch issued for postMatrixMessage against a loopback host')
    ok('live Matrix call sites refuse to fetch a private/loopback/metadata baseUrl (SSRF defense in depth)')
  } finally {
    restoreFetch()
  }
}

await (async () => {
  const log: Array<{ url: string }> = []
  installFetch([{ match: '/account/whoami', handle: () => ({ body: { user_id: '@local:lan' } }) }], log)
  try {
    const verified = await verifyMatrixConnection('http://127.0.0.1:8008', 'syt_fake', true)
    assert.equal(verified.ok, true)
    assert.equal(verified.userId, '@local:lan')
    assert.equal(log.length, 1)
    ok('explicit private-network opt-in reaches a self-hosted Matrix homeserver')
  } finally {
    restoreFetch()
  }
})()

{
  // A message becomes an inbound connector event keyed by its thread root; reply context
  // is folded into the summary.
  const msg: MatrixReadResult = {
    id: '$evt1', sourceId: 'matrix_1', sourceLabel: 'Account',
    roomId: '!room:hs.example', roomName: 'Ops', roomAlias: '#ops:hs.example',
    senderId: '@sam:hs.example', senderName: 'Sam', isSelf: false,
    date: 2000, text: 'deploy ready?', snippet: 'deploy ready?',
    threadRootId: '$root9', replyTo: '$parent5', replyToAuthor: 'Maya', replyToText: 'pricing?',
    threadKey: 'matrix:matrix_1:$root9', raw: {},
  }
  const event = matrixEventFromMessage(msg, 2000)
  assert.equal(event.connectorId, 'matrix')
  assert.equal(event.direction, 'inbound')
  assert.equal(event.classification, 'communication')
  assert.equal(event.action, 'message_received')
  assert.equal(event.threadKey, 'matrix:matrix_1:$root9')
  assert.equal(event.actor, 'Sam')
  assert.ok(event.payloadKeys.includes('thread') && event.payloadKeys.includes('reply'), 'thread + reply payload keys present')
  assert.match(event.summary, /In reply to Maya: "pricing\?"/, 'reply context folded into summary')
  ok('matrixEventFromMessage maps a message to an inbound thread-keyed event with reply context')
}

{
  // Media extraction converts mxc:// (content.url + thumbnail) to homeserver download
  // URLs, de-duplicated; a plain text event yields nothing.
  const imageEvent = {
    content: {
      msgtype: 'm.image',
      body: 'photo.png',
      url: 'mxc://hs.example/abc123',
      info: { thumbnail_url: 'mxc://hs.example/thumb9' },
    },
  }
  const urls = extractMatrixMediaUrls(imageEvent, 'https://hs.example')
  assert.deepEqual(urls, [
    'https://hs.example/_matrix/client/v1/media/download/hs.example/abc123',
    'https://hs.example/_matrix/client/v1/media/download/hs.example/thumb9',
  ])
  assert.deepEqual(extractMatrixMediaUrls({ content: { msgtype: 'm.text', body: 'hi' } }, 'https://hs.example'), [])
  assert.deepEqual(extractMatrixMediaUrls(null, 'https://hs.example'), [])
  ok('extractMatrixMediaUrls converts mxc url + thumbnail to download urls, dedupes')
}

{
  // Full normalization + allowlist on READ: a one-shot sync returns two rooms' timelines;
  // only the allowed room's text message (not the m.image, not our own event) is ingested.
  installFetch([
    { match: '/account/whoami', handle: () => ({ body: { user_id: '@me:hs.example' } }) },
    { match: '/joined_rooms', handle: () => ({ body: { joined_rooms: ['!ops:hs.example', '!secret:hs.example'] } }) },
    { match: '/state/m.room.name/', handle: (u) => ({ body: { name: u.includes('!ops') ? 'Ops' : 'Secret' } }) },
    { match: '/state/m.room.canonical_alias/', handle: () => ({ status: 404, body: {} }) },
    { match: '/profile/', handle: (u) => ({ body: { displayname: u.includes('sam') ? 'Sam' : 'Other' } }) },
    {
      match: '/sync',
      handle: () => ({
        body: {
          rooms: {
            join: {
              '!ops:hs.example': {
                timeline: {
                  events: [
                    { type: 'm.room.message', event_id: '$a', sender: '@sam:hs.example', origin_server_ts: 1500, content: { msgtype: 'm.text', body: 'hello from ops' } },
                    { type: 'm.room.message', event_id: '$self', sender: '@me:hs.example', origin_server_ts: 1600, content: { msgtype: 'm.text', body: 'my own reply' } },
                    { type: 'm.room.message', event_id: '$img', sender: '@sam:hs.example', origin_server_ts: 1700, content: { msgtype: 'm.image', body: 'x.png', url: 'mxc://hs.example/x' } },
                  ],
                },
              },
              '!secret:hs.example': {
                timeline: {
                  events: [
                    { type: 'm.room.message', event_id: '$s1', sender: '@other:hs.example', origin_server_ts: 1800, content: { msgtype: 'm.text', body: 'secret traffic' } },
                  ],
                },
              },
            },
          },
        },
      }),
    },
  ])
  try {
    const msgs = await fetchMatrixMessages(baseSource({ allowedRooms: '!ops:hs.example' }))
    assert.equal(msgs.length, 1, 'only one allowed, non-self, text message ingested')
    assert.equal(msgs[0].id, '$a')
    assert.equal(msgs[0].roomId, '!ops:hs.example')
    assert.equal(msgs[0].roomName, 'Ops')
    assert.equal(msgs[0].senderName, 'Sam')
    assert.equal(msgs[0].isSelf, false)
    assert.equal(msgs[0].text, 'hello from ops')
    assert.ok(!msgs.some((m) => m.roomId === '!secret:hs.example'), 'disallowed room excluded on read')
    assert.ok(!msgs.some((m) => m.id === '$self'), 'own message skipped (no reply-to-self loop)')
    ok('fetchMatrixMessages normalizes, enforces allowlist on read, skips self + non-text')
  } finally {
    restoreFetch()
  }
}

{
  // Allowlist on the WRITE path: a send to a disallowed room is rejected (fail closed)
  // BEFORE any send PUT fires; a send to the allowed room issues the PUT and returns its
  // event_id.
  const log: Array<{ url: string; init?: any }> = []
  installFetch([
    { match: '/joined_rooms', handle: () => ({ body: { joined_rooms: ['!ops:hs.example'] } }) },
    { match: '/state/m.room.name/', handle: () => ({ body: { name: 'Ops' } }) },
    { match: '/state/m.room.canonical_alias/', handle: () => ({ status: 404, body: {} }) },
    { match: '/send/m.room.message/', handle: () => ({ body: { event_id: '$sent42' } }) },
  ], log)
  try {
    await assert.rejects(
      () => postMatrixMessage(baseSource({ allowedRooms: '!ops:hs.example' }), '!secret:hs.example', 'leak'),
      /not in this source's allowed rooms/i,
      'send to disallowed room rejected',
    )
    assert.ok(!log.some((l) => l.url.includes('/send/')), 'no send PUT fired for the rejected room')

    const sent = await postMatrixMessage(baseSource({ allowedRooms: '!ops:hs.example' }), '!ops:hs.example', 'ok to ops', { threadRootId: '$root1' })
    assert.equal(sent.ok, true)
    assert.equal(sent.id, '$sent42')
    const putCall = log.find((l) => l.url.includes('/send/m.room.message/'))
    assert.ok(putCall, 'send PUT issued for allowed room')
    assert.equal(putCall!.init?.method, 'PUT')
    const body = JSON.parse(putCall!.init.body)
    assert.equal(body.msgtype, 'm.text')
    assert.equal(body['m.relates_to']?.rel_type, 'm.thread', 'threaded reply relation attached')
    assert.equal(body['m.relates_to']?.event_id, '$root1')
    ok('postMatrixMessage fails closed on disallowed room; sends + threads to an allowed one')
  } finally {
    restoreFetch()
  }
}

console.log(`\n${passed} checks passed.`)
