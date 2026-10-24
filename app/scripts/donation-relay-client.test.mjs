// Donor-side relay client: frames in, local donation node out.
// Run: node node_modules/jiti/lib/jiti-cli.mjs scripts/donation-relay-client.test.mjs
import assert from 'node:assert/strict'
import http from 'node:http'
import { attachDonationRelay } from '../electron/donation-relay-client.ts'

const b64 = (value) => Buffer.from(value).toString('base64')
const text = (bytes) => Buffer.from(bytes, 'base64').toString('utf8')

class FakeSocket {
  readyState = 0
  listeners = new Map()
  sent = []
  constructor(url, protocols) {
    this.url = url
    this.protocols = protocols
  }
  addEventListener(type, listener) {
    const list = this.listeners.get(type) ?? []
    list.push(listener)
    this.listeners.set(type, list)
  }
  emit(type, event) {
    for (const listener of this.listeners.get(type) ?? []) listener(event)
  }
  open() {
    this.readyState = 1
    this.emit('open', {})
  }
  send(data) {
    this.sent.push(JSON.parse(data))
  }
  deliver(frame) {
    this.emit('message', { data: JSON.stringify(frame) })
  }
  close() {
    this.readyState = 3
    this.emit('close', {})
  }
}

const waitFor = async (predicate, label, timeoutMs = 5_000) => {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    const value = predicate()
    if (value) return value
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
  throw new Error(`Timed out waiting for ${label}`)
}

const server = http.createServer((req, res) => {
  const chunks = []
  req.on('data', (chunk) => chunks.push(chunk))
  req.on('end', () => {
    const body = Buffer.concat(chunks).toString('utf8')
    if (req.url === '/v1/models') {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ object: 'list', data: [{ id: 'poc-model' }] }))
      return
    }
    if (req.headers.authorization !== 'Bearer card-token') {
      res.writeHead(401, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: { message: 'The card token was refused.' } }))
      return
    }
    assert.equal(body, JSON.stringify({ model: 'poc-model' }), 'the request body reaches the node unchanged')
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    res.write('data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n')
    setTimeout(() => {
      res.write('data: {"choices":[{"delta":{"content":"relay"}}]}\n\n')
      res.end('data: [DONE]\n\n')
    }, 30)
  })
})

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const origin = `http://127.0.0.1:${server.address().port}`

let socket
const attachment = attachDonationRelay({
  relayUrl: 'https://relay.example',
  cardId: 'donate-test01',
  donorKey: 'k'.repeat(32),
  localOrigin: origin,
  connect: (url, protocols) => (socket = new FakeSocket(url, protocols)),
  autoReconnect: false,
})
socket.open()

assert.equal(socket.url, 'wss://relay.example/donor/donate-test01', 'the attach URL is derived from the relay address')
assert.deepEqual(socket.protocols, ['tf-donation-v1', 'k'.repeat(32)], 'the donor key travels as the second subprotocol')
assert.equal(attachment.attached(), true)
console.log('PASS 1: attaches outbound with the card id in the path and the donor key as a subprotocol')

socket.deliver({ t: 'req', rid: 'r1', method: 'POST', path: '/v1/chat/completions', headers: { authorization: 'Bearer card-token', 'content-type': 'application/json' }, body: b64(JSON.stringify({ model: 'poc-model' })) })
await waitFor(() => socket.sent.find((frame) => frame.t === 'end' && frame.rid === 'r1'), 'the streamed response to finish')
const head = socket.sent.find((frame) => frame.t === 'head' && frame.rid === 'r1')
const streamed = socket.sent.filter((frame) => frame.t === 'chunk' && frame.rid === 'r1').map((frame) => text(frame.data)).join('')
assert.equal(head.status, 200)
assert.equal(head.headers['content-type'], 'text/event-stream')
assert.match(streamed, /Hello /)
assert.match(streamed, /relay/)
assert.match(streamed, /\[DONE\]/)
console.log('PASS 2: streams the node response back as head, chunk and end frames')

socket.deliver({ t: 'req', rid: 'r2', method: 'GET', path: '/v1/models', headers: {}, body: '' })
await waitFor(() => socket.sent.find((frame) => frame.t === 'end' && frame.rid === 'r2'), 'the models response')
const models = socket.sent.filter((frame) => frame.t === 'chunk' && frame.rid === 'r2').map((frame) => text(frame.data)).join('')
assert.match(models, /poc-model/)
console.log('PASS 3: GET requests are forwarded without a body')

socket.deliver({ t: 'req', rid: 'r3', method: 'POST', path: '/v1/chat/completions', headers: { authorization: 'Bearer wrong' }, body: b64('{}') })
await waitFor(() => socket.sent.find((frame) => frame.t === 'head' && frame.rid === 'r3'), 'the refusal to come back')
assert.equal(socket.sent.find((frame) => frame.t === 'head' && frame.rid === 'r3').status, 401)
console.log('PASS 4: the node refusal is passed through unchanged (the relay never re-decides)')

const offline = attachDonationRelay({ relayUrl: 'https://relay.example', cardId: 'donate-test02', donorKey: 'k'.repeat(32), localOrigin: 'http://127.0.0.1:1', connect: () => new FakeSocket(), autoReconnect: false })
offline.close()
console.log('PASS 5: close() is safe before the socket opens')

attachment.close()
await new Promise((resolve) => server.close(resolve))
console.log('donation relay client: ok')
