// Run: jiti electron/agent/discord-connector.test.ts
import assert from 'node:assert/strict'
import { makeDiscordSource, validateDiscordSource, discordEventFromMessage, isDiscordChannelAllowed, extractDiscordMediaUrls, fetchDiscordMessages } from './discord-connector.ts'
import type { DiscordReadResult } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('discord-connector tests:')

{
  // The synchronous fast-path: allowed only when there's no allowlist or the id is explicitly
  // listed. A name-based allowlist returns false here so the caller resolves names->ids — it
  // must NOT be permissive (verifier bypass: agent harvests a restricted id and posts to it).
  assert.equal(isDiscordChannelAllowed('', '123456789012345678'), false, 'empty allowlist grants no access')
  assert.equal(isDiscordChannelAllowed('*', '123456789012345678'), true, 'standalone wildcard explicitly grants all')
  assert.equal(isDiscordChannelAllowed('123456789012345678', '123456789012345678'), true, 'id in allowlist')
  assert.equal(isDiscordChannelAllowed('123456789012345678, 999', '424242424242424242'), false, 'id NOT in id-allowlist rejected')
  assert.equal(isDiscordChannelAllowed('#general, #random', '424242424242424242'), false, 'name-only allowlist is NOT permissive at send time')
  ok('isDiscordChannelAllowed gates sends; name allowlists fall through to resolution')
}

{
  // A fresh source stays disconnected with a stable id and an empty token.
  const s = makeDiscordSource('My server', 1000)
  assert.match(s.id, /^discord_/)
  assert.equal(s.enabled, false)
  assert.equal(s.token, '')
  assert.equal(s.label, 'My server')
  assert.equal(s.allowedChannels, '')
  ok('makeDiscordSource creates a disconnected source with a stable id')
}

{
  // Validation requires a token before any network attempt.
  assert.match(validateDiscordSource(makeDiscordSource('x'))!, /token is required/i)
  assert.equal(validateDiscordSource({ ...makeDiscordSource('x'), token: 'MTk4NjIy...fake' }), null)
  ok('validateDiscordSource requires a bot token')
}

{
  // A message becomes an inbound connector event keyed by its thread/channel.
  const msg: DiscordReadResult = {
    id: '111', sourceId: 'discord_1', sourceLabel: 'Server',
    channelId: '999', channelName: 'general', threadId: null, threadName: null,
    guildName: 'Server', senderId: '222', senderName: 'Sam', isBot: false,
    date: 2000, text: 'deploy ready?', snippet: 'deploy ready?',
    replyTo: null, threadKey: 'discord:discord_1:999',
    raw: {},
  }
  const event = discordEventFromMessage(msg, 2000)
  assert.equal(event.connectorId, 'discord')
  assert.equal(event.direction, 'inbound')
  assert.equal(event.action, 'message_received')
  assert.equal(event.threadKey, 'discord:discord_1:999')
  assert.equal(event.actor, 'Sam')
  ok('discordEventFromMessage maps a message to an inbound thread-keyed event')
}

{
  // Media URL extraction pulls attachment urls (prefers canonical `url` over proxy) and
  // image-embed urls, de-duplicated. A message with no media yields nothing.
  const raw = {
    attachments: [
      { url: 'https://cdn.discordapp.com/a/1/photo.png', proxy_url: 'https://media.discordapp.net/a/1/photo.png' },
      { proxy_url: 'https://media.discordapp.net/a/2/doc.pdf' }, // only proxy_url present
    ],
    embeds: [
      { image: { url: 'https://cdn.example.com/embed.jpg' } },
      { thumbnail: { url: 'https://cdn.example.com/thumb.jpg' } },
    ],
  }
  assert.deepEqual(extractDiscordMediaUrls(raw), [
    'https://cdn.discordapp.com/a/1/photo.png',
    'https://media.discordapp.net/a/2/doc.pdf',
    'https://cdn.example.com/embed.jpg',
    'https://cdn.example.com/thumb.jpg',
  ])
  assert.deepEqual(extractDiscordMediaUrls({}), [])
  assert.deepEqual(extractDiscordMediaUrls(null), [])
  ok('extractDiscordMediaUrls pulls attachment + embed image urls, prefers url, dedupes')
}

// --- a query-string-aware fetch stub (save/restore) for the messages pagination tests ---
const realFetch = globalThis.fetch
const TEST_HTTP_DEPS = {
  lookup: async () => [{ address: '8.8.8.8', family: 4 }],
  fetchImpl: ((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => globalThis.fetch(input, init)) as typeof fetch,
}
function stubFetch(handler: (path: string, query: URLSearchParams) => unknown) {
  globalThis.fetch = (async (input: any) => {
    const url = typeof input === 'string' ? input : input?.url ?? String(input)
    const u = new URL(url)
    const path = u.pathname.replace(/^\/api\/v\d+/, '') // strip the /api/v10 DISCORD_API prefix
    const body = handler(path, u.searchParams)
    if (body === undefined) return { ok: false, status: 404, json: async () => ({ message: `no stub for ${path}${u.search}` }) } as any
    return { ok: true, status: 200, json: async () => body } as any
  }) as typeof fetch
}
const restoreFetch = () => { globalThis.fetch = realFetch }

// Build a fake Discord message with a numeric-string snowflake id (so ids sort the same
// lexically and numerically across this test's small ranges) and a timestamp derived from it.
function fakeMessage(id: number) {
  return {
    id: String(id),
    author: { id: 'human1', username: 'sam', bot: false },
    content: `message ${id}`,
    timestamp: new Date(id * 1000).toISOString(),
  }
}

await (async () => {
  // Without a cursor, fetchDiscordMessages does a single bounded fetch of the newest 50 —
  // unchanged full-resync behaviour (no after= param sent).
  const all = Array.from({ length: 80 }, (_, i) => fakeMessage(i + 1)) // ids 1..80
  stubFetch((path, query) => {
    if (path === '/users/@me/guilds') return [{ id: 'guild1', name: 'Acme' }]
    if (path === '/guilds/guild1/channels') return [{ id: 'chan1', name: 'general', type: 0 }]
    if (path === '/guilds/guild1/threads/active') return { threads: [] }
    if (path === '/channels/chan1/messages') {
      assert.equal(query.get('after'), null, 'no cursor => no after= param sent')
      // Discord returns the newest `limit` messages, descending, when there is no after=.
      return [...all].slice(-50).reverse()
    }
    return undefined
  })
  try {
    const source = { ...makeDiscordSource('Acme', 1000), enabled: true, allowedChannels: '*', token: 'tok' }
    const messages = await fetchDiscordMessages(source, TEST_HTTP_DEPS)
    assert.equal(messages.length, 50, 'no cursor => bounded to the newest 50 (unchanged behaviour)')
    ok('fetchDiscordMessages with no cursor stays bounded to the newest 50, no after= sent')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  // THE BUG: a burst of 80 messages arrives while offline. With a prior cursor (snowflake
  // id 10, i.e. messages 11..80 are new), fetchDiscordMessages must paginate forward with
  // after=<cursor id>, then after=<last id of each page>, until the backlog (70 messages)
  // is fully retrieved — not truncated to the newest 50.
  const all = Array.from({ length: 80 }, (_, i) => fakeMessage(i + 1)) // ids 1..80
  const seenAfter: (string | null)[] = []
  stubFetch((path, query) => {
    if (path === '/users/@me/guilds') return [{ id: 'guild1', name: 'Acme' }]
    if (path === '/guilds/guild1/channels') return [{ id: 'chan1', name: 'general', type: 0 }]
    if (path === '/guilds/guild1/threads/active') return { threads: [] }
    if (path === '/channels/chan1/messages') {
      const after = query.get('after')
      seenAfter.push(after)
      const floor = after ? Number(after) : 0
      // The requested after-window is returned newest-to-oldest, as documented.
      const page = all.filter((m) => Number(m.id) > floor).slice(0, 50).reverse()
      return page
    }
    return undefined
  })
  try {
    const source = {
      ...makeDiscordSource('Acme', 1000),
      enabled: true, allowedChannels: '*', token: 'tok',
      cursor: { position: 10_000, lastId: '10', updatedAt: 1000 },
    }
    const messages = await fetchDiscordMessages(source, TEST_HTTP_DEPS)
    assert.equal(messages.length, 70, 'all 70 backlog messages (ids 11..80) retrieved, not just the newest 50')
    const ids = messages.map((m) => Number(m.id)).sort((a, b) => a - b)
    assert.deepEqual(ids, Array.from({ length: 70 }, (_, i) => i + 11), 'every backlog id 11..80 present, none dropped')
    assert.ok(seenAfter.length > 1, 'paginated across more than one request')
    assert.equal(seenAfter[0], '10', 'first page requested after= the prior cursor id')
    assert.equal(seenAfter[1], '60', 'second page advances to the greatest snowflake rather than the oldest returned message')
    assert.equal(seenAfter.length, 2, 'the descending response order does not cause overlapping requests')
    ok('fetchDiscordMessages paginates with after= when a cursor is present, recovering a full backlog burst')
  } finally {
    restoreFetch()
  }
})()

await (async () => {
  const base = 90_071_992_547_409_930n
  const all = Array.from({ length: 80 }, (_, i) => ({ ...fakeMessage(i + 1), id: String(base + BigInt(i + 1)) }))
  const seenAfter: string[] = []
  stubFetch((path, query) => {
    if (path === '/users/@me/guilds') return [{ id: 'guild1', name: 'Example server' }]
    if (path === '/guilds/guild1/channels') return [{ id: 'chan1', name: 'general', type: 0 }]
    if (path === '/guilds/guild1/threads/active') return { threads: [] }
    if (path === '/channels/chan1/messages') {
      const after = query.get('after')!
      seenAfter.push(after)
      return all.filter((message) => BigInt(message.id) > BigInt(after)).slice(0, 50).reverse()
    }
  })
  try {
    const messages = await fetchDiscordMessages({ ...makeDiscordSource('Example server'), enabled: true, allowedChannels: '*', token: 'fixture-token', cursor: { position: 0, lastId: String(base + 10n), updatedAt: 0 } }, TEST_HTTP_DEPS)
    assert.equal(messages.length, 70)
    assert.equal(new Set(messages.map((message) => message.id)).size, 70)
    assert.equal(seenAfter[1], String(base + 60n), 'pagination preserves neighboring snowflakes beyond Number precision')
    assert.equal(seenAfter.length, 2)
    ok('descending Discord pages advance exactly with high-precision snowflakes')
  } finally { restoreFetch() }
})()

console.log(`\n${passed} checks passed.`)
