import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'vontra-bot-poll-'))

const { defaultSettings } = await import('../../store')
const { store } = await import('../state')
const { createBotPollingRuntime, createTelegramApiFetch, readTelegramJson } = await import('./bot-polling-runtime')

{
  let lookups = 0
  const secure = createTelegramApiFetch({
    lookup: async () => {
      lookups++
      return lookups === 1 ? [{ address: '93.184.216.34', family: 4 }] : [{ address: '169.254.169.254', family: 4 }]
    },
    fetchImpl: (async (_input: any, init: any) => {
      assert.ok(init.dispatcher, 'request carries the DNS-pinned dispatcher')
      return new Response('{}', { headers: { 'content-type': 'application/json' } })
    }) as typeof fetch,
  })
  await secure('https://api.telegram.org/botredacted/getUpdates')
  assert.equal(lookups, 1, 'the request resolves once; a rebinding second answer is never consulted')
  await assert.rejects(() => createTelegramApiFetch({ lookup: async () => [{ address: '10.0.0.7', family: 4 }] })('https://api.telegram.org/botredacted/getUpdates'), /private or non-public/)
  await assert.rejects(() => secure('https://example.com/steal'), /exact configured HTTPS origin/)
}

await assert.rejects(
  () => readTelegramJson(new Response('<html>bad</html>', { headers: { 'content-type': 'text/html' } })),
  /non-JSON/,
)
await assert.rejects(
  () => readTelegramJson(new Response('{"ok":true}', { headers: { 'content-type': 'application/json', 'content-length': '9999999' } })),
  /exceeded/,
)
await assert.rejects(
  () => readTelegramJson(new Response('x'.repeat(128), { headers: { 'content-type': 'application/json' } }), 32),
  /exceeded/,
)

function configure() {
  store.settings = defaultSettings()
  store.settings.botControl.telegram = {
    ...store.settings.botControl.telegram,
    enabled: true,
    polling: true,
    botToken: 'sensitive-bot-token',
    allowedChatIds: '42',
    allowedUserIds: '7',
  }
  store.projects = []
  store.rooms = []
  store.botSessions = []
}

function runtime(fetchImpl: typeof fetch) {
  return createBotPollingRuntime({
    telegramFetch: fetchImpl,
    botControlRuntime: () => ({ abortRoom: () => {}, hasActiveRoom: () => false, emitRoom: () => {}, rememberConnectorEvents: () => {} }),
    resolveChannelApproval: () => false,
    resolveChannelApprovalByThread: () => false,
  })
}

configure()
{
  const fetchImpl = (async () => new Response('<html>proxy error</html>', { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch
  const result = await runtime(fetchImpl).pollTelegramUpdates()
  assert.equal(result.ok, false)
  assert.match(result.error ?? '', /non-JSON/)
}

configure()
{
  const huge = (async () => new Response('{"ok":true}', { status: 200, headers: { 'content-type': 'application/json', 'content-length': '99999999' } })) as typeof fetch
  const result = await runtime(huge).pollTelegramUpdates()
  assert.equal(result.ok, false)
  assert.match(result.error ?? '', /exceeded/)
}

configure()
{
  let call = 0
  const update = { ok: true, result: [{ update_id: 1, message: { chat: { id: 42 }, from: { id: 7 }, text: 'status' } }] }
  const fetchImpl = (async () => {
    call++
    return call === 1
      ? new Response(JSON.stringify(update), { headers: { 'content-type': 'application/json' } })
      : new Response('{}', { status: 503, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  const result = await runtime(fetchImpl).pollTelegramUpdates()
  assert.equal(result.ok, false, 'outbound non-2xx Telegram replies are reported honestly')
  assert.match(result.error ?? '', /sendMessage failed \(503\)/)
  assert.equal(result.processed, 1)
  assert.equal(result.lastUpdateId, 1, 'a command whose reply failed is still checkpointed, never replayed')
}

configure()
{
  let call = 0
  const update = { ok: true, result: [
    { update_id: 1, message: { chat: { id: 42 }, from: { id: 7 }, text: 'status' } },
    { update_id: 2, message: { chat: { id: 42 }, from: { id: 7 }, text: 'status' } },
  ] }
  const fetchImpl = (async () => {
    call++
    if (call === 1) return new Response(JSON.stringify(update), { headers: { 'content-type': 'application/json' } })
    if (call === 2) return new Response('{}', { status: 503, headers: { 'content-type': 'application/json' } })
    return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  const result = await runtime(fetchImpl).pollTelegramUpdates()
  assert.equal(result.ok, false, 'one delivery failure remains visible')
  assert.equal(result.processed, 2, 'a failed first reply does not abort later updates')
  assert.equal(result.lastUpdateId, 2, 'the whole consumed batch advances its checkpoint')
  assert.equal(call, 3, 'both replies were attempted')
}

configure()
{
  const token = store.settings.botControl.telegram.botToken
  const result = await runtime((async () => { throw new Error(`transport exposed ${token} and ${token}`) }) as typeof fetch).pollTelegramUpdates()
  assert.equal(result.ok, false)
  assert.ok(!result.error?.includes(token), 'every occurrence of the bot token is removed from errors')
}

console.log('bot polling runtime security tests passed')
