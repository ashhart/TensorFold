import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { blueBubblesAccess, blueBubblesChats, blueBubblesIdentity, scanBlueBubbles, sendBlueBubbles } from './bluebubbles-connector'
import { syncIMessageConnector, readIMessageMessage } from './imessage-connector'
import type { IMessageConnectorSettings } from './types'

const chat = 'iMessage;+;chat123456789'
const requests: { url: URL; body: any }[] = []
let mode = 'ok'
const server = createServer(async (req, res) => {
  let raw = ''; for await (const chunk of req) raw += chunk
  const url = new URL(req.url!, 'http://127.0.0.1')
  const body = raw ? JSON.parse(raw) : null
  requests.push({ url, body })
  res.setHeader('Content-Type', 'application/json')
  if (mode === 'auth') { res.writeHead(401); res.end(JSON.stringify({ error: 'secret-server-diagnostic' })); return }
  if (mode === 'timeout') { req.socket.destroy(); return }
  if (mode === 'redirect') { res.writeHead(302, { Location: 'https://outside.example.com/' }); res.end('{}'); return }
  if (url.pathname.endsWith('/info')) { res.end(JSON.stringify({ status: 200, data: { serverVersion: 'synthetic' } })); return }
  if (url.pathname.endsWith('/query')) {
    res.end(JSON.stringify({ status: 200, data: [
      { guid: 'synthetic-guid', text: 'Synthetic message', dateCreated: 1_798_000_000_000, isFromMe: false, handle: { address: 'sender@example.com' }, chats: [{ guid: chat, displayName: 'Selected group' }] },
      { guid: 'wrong-chat', text: 'OUTSIDE_SCOPE_CANARY', dateCreated: 1_798_000_000_100, chats: [{ guid: 'iMessage;-;other@example.com' }] },
    ] })); return
  }
  res.end(JSON.stringify({ status: 200, data: { guid: 'synthetic-sent-id' } }))
})
await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
try {
  const settings: IMessageConnectorSettings = { enabled: true, transport: 'bluebubbles', blueBubblesUrl: `http://127.0.0.1:${(server.address() as any).port}`,
    blueBubblesPassword: 'synthetic-secret', allowedChatIds: chat, maxMessages: 25, lastSyncAt: null, lastError: null }
  assert.equal((await blueBubblesAccess(settings)).state, 'ok')
  assert.equal(requests[0].url.searchParams.get('password'), 'synthetic-secret')
  assert.deepEqual(blueBubblesChats(settings), [chat], 'semicolons belong to a chat GUID, not the list delimiter')
  assert.throws(() => blueBubblesChats({ ...settings, allowedChatIds: '' }), /exact BlueBubbles/)
  const messages = await scanBlueBubbles(settings)
  assert.equal(messages.length, 1)
  assert.equal(requests.at(-1)!.body.chatGuid, chat, 'unselected chats are never requested')
  assert.equal((await readIMessageMessage(messages[0].id, settings))?.text, 'Synthetic message', 'real iMessage read dispatch supports the bridge on every OS')
  const sync = await syncIMessageConnector(settings, new Set())
  assert.equal(sync.events[0].threadKey, `imessage:bb-${blueBubblesIdentity(settings)}:${chat}`)
  assert.equal(sync.events[0].payload?.messageId, messages[0].id)
  const again = await syncIMessageConnector(settings, new Set(sync.events.map(event => `imessage:msg:${event.payload?.messageId}`)))
  assert.equal(again.events.length, 0, 'repeated sync never duplicates a message')
  assert.notEqual(blueBubblesIdentity({ ...settings, blueBubblesUrl: 'https://other.example.com' }), blueBubblesIdentity(settings))
  const exact = '  Exact reviewed text\nwith whitespace.  '
  assert.equal((await sendBlueBubbles(settings, chat, exact)).sentId, 'synthetic-sent-id')
  assert.equal(requests.at(-1)!.body.message, exact)
  assert.equal(requests.at(-1)!.body.method, 'apple-script', 'basic sends never enable a private API')
  const count = requests.length
  await assert.rejects(() => sendBlueBubbles(settings, chat, exact, { assertCurrent: () => { throw new Error('Revoked during DNS lookup') } }), /could not be reached/)
  assert.equal(requests.length, count, 'scope revocation is checked at the actual socket admission after DNS resolution')
  await assert.rejects(() => sendBlueBubbles(settings, 'iMessage;-;blocked@example.com', exact), /scope/)
  assert.equal(requests.length, count)
  for (const next of ['auth', 'timeout', 'redirect']) {
    mode = next
    const before = requests.length
    await assert.rejects(() => sendBlueBubbles(settings, chat, exact), error => !String(error).includes('synthetic-secret') && !String(error).includes('secret-server-diagnostic'))
    assert.equal(requests.length, before + 1, 'uncertain sends never retry or follow redirects')
  }
  console.log('BlueBubbles loopback protocol, scope, group identity, dispatch, deduplication, exact text and non-retry checks passed')
} finally { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())) }
