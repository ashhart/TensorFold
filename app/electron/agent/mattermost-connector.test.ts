// Run: jiti electron/agent/mattermost-connector.test.ts
import assert from 'node:assert/strict'
import {
  makeMattermostSource,
  validateMattermostSource,
  verifyMattermostConnection,
  fetchMattermostMessages,
  postMattermostMessage,
  isMattermostChannelAllowed,
  extractMattermostMediaUrls,
  mattermostEventFromMessage,
  listMattermostChannels,
} from './mattermost-connector.ts'
import type { MattermostReadResult, MattermostSource } from './mattermost-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('mattermost-connector tests:')

// --- a tiny fetch stub (save/restore) returning canned Mattermost JSON by URL ---
const realFetch = globalThis.fetch
function stubFetch(routes: Record<string, unknown>) {
  globalThis.fetch = (async (input: any) => {
    const url = typeof input === 'string' ? input : input?.url ?? String(input)
    const path = url.replace(/^https?:\/\/[^/]+/, '') // strip origin
    for (const [needle, body] of Object.entries(routes)) {
      if (path.startsWith(needle)) {
        return { ok: true, status: 200, json: async () => body } as any
      }
    }
    return { ok: false, status: 404, json: async () => ({ message: `no stub for ${path}` }) } as any
  }) as typeof fetch
}
const restoreFetch = () => { globalThis.fetch = realFetch }

{
  // A fresh source is enabled with a stable id, empty token + base URL, no allowlist.
  const s = makeMattermostSource('Acme MM', 1000)
  assert.match(s.id, /^mattermost_/)
  assert.equal(s.enabled, true)
  assert.equal(s.token, '')
  assert.equal(s.baseUrl, '')
  assert.equal(s.label, 'Acme MM')
  assert.equal(s.allowedChannels, '')
  ok('makeMattermostSource creates an enabled source with a stable id')
}

{
  // Validation requires a base URL (http[s]) AND a token before any network attempt.
  assert.match(validateMattermostSource(makeMattermostSource('x'))!, /server URL is required/i)
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'mm.example.com' })!,
    /must start with http/i,
  )
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'https://mm.example.com' })!,
    /token is required/i,
  )
  assert.equal(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'https://mm.example.com', token: 'tok' }),
    null,
  )
  ok('validateMattermostSource requires an http(s) base URL and a token')
}

{
  // SSRF guard: a loopback/link-local/metadata baseUrl must be rejected at validate time —
  // an attacker-chosen host like this would otherwise receive the PAT as a Bearer token.
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'http://169.254.169.254', token: 'tok' })!,
    /local\/private/i,
    'cloud metadata endpoint rejected',
  )
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'http://127.0.0.1:6379', token: 'tok' })!,
    /local\/private/i,
    'loopback rejected',
  )
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'http://192.168.1.5', token: 'tok' })!,
    /local\/private/i,
    'private RFC1918 host rejected',
  )
  assert.equal(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'https://mm.example.com', token: 'tok' }),
    null,
    'a normal public server still validates',
  )
  assert.equal(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'http://192.168.1.5:8065', token: 'tok', allowPrivateNetwork: true }),
    null,
    'an explicitly opted-in self-hosted LAN server validates',
  )
  assert.match(
    validateMattermostSource({ ...makeMattermostSource('x'), baseUrl: 'http://169.254.169.254', token: 'tok', allowPrivateNetwork: true })!,
    /metadata|link-local/i,
    'private-network opt-in never permits cloud metadata/link-local targets',
  )
  ok('validateMattermostSource rejects loopback/private/link-local/metadata server URLs (SSRF)')
}

{
  // The synchronous fast-path: allowed only with no allowlist or an explicit id. A
  // name/slug allowlist is NOT permissive at send time (caller must resolve names->ids).
  assert.equal(isMattermostChannelAllowed('', 'ch_abc123'), true, 'no allowlist => unrestricted')
  assert.equal(isMattermostChannelAllowed('ch_abc123', 'ch_abc123'), true, 'id in allowlist')
  assert.equal(isMattermostChannelAllowed('ch_abc123, ch_zzz', 'ch_other'), false, 'id NOT in id-allowlist rejected')
  assert.equal(isMattermostChannelAllowed('~town-square, general', 'ch_other'), false, 'slug-only allowlist not permissive at send time')
  ok('isMattermostChannelAllowed gates sends; slug allowlists fall through to resolution')
}

{
  // Media URLs: each file_id becomes an authenticated download URL off the server base.
  const raw = { file_ids: ['file1', 'file2', 'file1'] } // dup file1 collapses
  assert.deepEqual(extractMattermostMediaUrls(raw, 'https://mm.example.com/'), [
    'https://mm.example.com/api/v4/files/file1',
    'https://mm.example.com/api/v4/files/file2',
  ])
  assert.deepEqual(extractMattermostMediaUrls({}, 'https://mm.example.com'), [])
  assert.deepEqual(extractMattermostMediaUrls(null, 'https://mm.example.com'), [])
  assert.deepEqual(extractMattermostMediaUrls(raw, 'not-a-url'), [], 'bad base URL => no urls')
  ok('extractMattermostMediaUrls builds authed file download urls, dedupes, fails closed')
}

{
  // A post becomes an inbound connector event keyed by its channel; a threaded reply folds
  // the quoted parent into the summary and keys to the thread root.
  const msg: MattermostReadResult = {
    id: 'post1', sourceId: 'mattermost_1', sourceLabel: 'Acme',
    channelId: 'ch1', channelName: 'Engineering', teamName: 'Acme',
    senderId: 'u1', senderName: 'Sam', isBot: false,
    date: 2000, text: 'deploy ready?', snippet: 'deploy ready?',
    rootId: 'root9', replyTo: 'root9', replyToAuthor: 'Maya', replyToText: 'is the build green?',
    threadKey: 'mattermost:mattermost_1:ch1:root9',
    raw: {},
  }
  const event = mattermostEventFromMessage(msg, 2000)
  assert.equal(event.connectorId, 'mattermost')
  assert.equal(event.direction, 'inbound')
  assert.equal(event.classification, 'communication')
  assert.equal(event.action, 'message_received')
  assert.equal(event.threadKey, 'mattermost:mattermost_1:ch1:root9')
  assert.equal(event.actor, 'Sam')
  assert.match(event.summary, /Maya/, 'reply-to author folded into summary')
  assert.match(event.summary, /deploy ready\?/, 'new message present in summary')
  assert.equal(event.payload?.thread, 'root9')
  ok('mattermostEventFromMessage maps a post to an inbound thread-keyed event with reply context')
}

await (async () => {
  // Defense in depth: even if a private/loopback baseUrl bypassed validate() (e.g. set
  // directly via updateSource patch, or set valid-at-save-time then DNS-rebound), the live
  // call sites must refuse to fetch — the PAT must never reach the bad host.
  let calls = 0
  globalThis.fetch = (async () => { calls++; throw new Error('fetch must not be called') }) as typeof fetch
  try {
    await assert.rejects(() => verifyMattermostConnection('http://169.254.169.254', 'tok'), /local\/private|private address/i)
    assert.equal(calls, 0, 'no fetch issued for verifyMattermostConnection against a metadata host')

    await assert.rejects(() => listMattermostChannels('http://127.0.0.1:6379', 'tok'), /local\/private|private address/i)
    assert.equal(calls, 0, 'no fetch issued for listMattermostChannels against a loopback host')

    const source: MattermostSource = { ...makeMattermostSource('Acme', 1000), baseUrl: 'http://127.0.0.1:6379', token: 'tok', allowedChannels: '' }
    await assert.rejects(() => postMattermostMessage(source, 'ch_allowed', 'leak me'), /local\/private|private address/i)
    assert.equal(calls, 0, 'no fetch issued for postMattermostMessage against a loopback host')
    ok('live Mattermost call sites refuse to fetch a private/loopback/metadata baseUrl (SSRF defense in depth)')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  // Self-hosted Mattermost is commonly bound to loopback/LAN. The explicit source opt-in
  // must reach the live DNS guard too, while the default remains fail-closed above.
  stubFetch({ '/api/v4/users/me': { id: 'local_bot', username: 'lan-bot' } })
  try {
    const v = await verifyMattermostConnection('http://127.0.0.1:8065', 'tok', true)
    assert.equal(v.ok, true)
    assert.equal(v.user, 'lan-bot')
    ok('explicit private-network opt-in reaches a self-hosted Mattermost server')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  // verifyMattermostConnection hits GET /users/me and returns the bot identity.
  stubFetch({ '/api/v4/users/me': { id: 'bot_self', username: 'vontra-bot' } })
  try {
    const v = await verifyMattermostConnection('https://example.com', 'tok')
    assert.equal(v.ok, true)
    assert.equal(v.userId, 'bot_self')
    assert.equal(v.user, 'vontra-bot')
    ok('verifyMattermostConnection returns the bot identity from /users/me')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  // READ-path allowlist: with an id allowlist, only the listed channel is synced; the bot's
  // own posts and system posts are skipped. Channels are resolved via teams->channels.
  const source: MattermostSource = {
    ...makeMattermostSource('Acme', 1000),
    baseUrl: 'https://example.com',
    token: 'tok',
    allowedChannels: 'ch_allowed', // only this channel id
  }
  stubFetch({
    '/api/v4/users/me/teams/team1/channels': [
      { id: 'ch_allowed', name: 'eng', display_name: 'Engineering', type: 'O' },
      { id: 'ch_denied', name: 'secret', display_name: 'Secret', type: 'P' },
    ],
    '/api/v4/users/me/teams': [{ id: 'team1', display_name: 'Acme' }],
    '/api/v4/users/me': { id: 'bot_self', username: 'vontra-bot' },
    '/api/v4/channels/ch_allowed/posts': {
      order: ['p3', 'p2', 'p1'],
      posts: {
        p1: { id: 'p1', channel_id: 'ch_allowed', user_id: 'human', message: 'hello', create_at: 100, type: '' },
        p2: { id: 'p2', channel_id: 'ch_allowed', user_id: 'bot_self', message: 'I am the bot', create_at: 200, type: '' },
        p3: { id: 'p3', channel_id: 'ch_allowed', user_id: 'sys', message: 'joined', create_at: 300, type: 'system_join_channel' },
      },
    },
  })
  try {
    const msgs = await fetchMattermostMessages(source)
    // Only p1 survives: p2 is the bot's own post, p3 is a system post, ch_denied is filtered out.
    assert.equal(msgs.length, 1, 'only the human, non-system post in the allowed channel')
    assert.equal(msgs[0]!.id, 'p1')
    assert.equal(msgs[0]!.channelName, 'Engineering')
    assert.equal(msgs[0]!.text, 'hello')
    assert.equal(msgs[0]!.senderId, 'human')
    ok('fetchMattermostMessages honours channel allowlist + skips bot/system posts')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  // WRITE-path allowlist (fail closed): posting to a non-allowed channel throws BEFORE any
  // POST, even when the channel is technically visible to the bot.
  const source: MattermostSource = {
    ...makeMattermostSource('Acme', 1000),
    baseUrl: 'https://example.com',
    token: 'tok',
    allowedChannels: 'ch_allowed',
  }
  stubFetch({
    '/api/v4/users/me/teams/team1/channels': [
      { id: 'ch_allowed', name: 'eng', display_name: 'Engineering', type: 'O' },
      { id: 'ch_denied', name: 'secret', display_name: 'Secret', type: 'P' },
    ],
    '/api/v4/users/me/teams': [{ id: 'team1', display_name: 'Acme' }],
    '/api/v4/posts': { id: 'new_post' },
  })
  try {
    await assert.rejects(
      () => postMattermostMessage(source, 'ch_denied', 'should not send'),
      /not in this source's allowed channels/i,
      'send to denied channel rejected',
    )
    // And a send to the allowed channel succeeds.
    const sent = await postMattermostMessage(source, 'ch_allowed', 'shipping it')
    assert.equal(sent.ok, true)
    assert.equal(sent.id, 'new_post')
    ok('postMattermostMessage enforces the write allowlist (fail closed) and posts when allowed')
  } finally {
    restoreFetch()
  }
})()

console.log(`\n${passed} checks passed.`)
