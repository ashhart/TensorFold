import { SELF } from 'cloudflare:test'
import { describe, expect, it } from 'vitest'

/**
 * These run inside the Workers runtime (workerd via the vitest plugin): the
 * Worker, the Durable Object and its WebSocket plumbing are the deployed code,
 * only the donor is simulated.
 */

const KEY = 'k'.repeat(32)
/**
 * The Durable Object instance survives between tests in a file (only storage is
 * reset), so every test uses its own card id rather than assuming a clean object.
 */
let sequence = 0
const nextCard = () => `donate-t${(sequence += 1).toString().padStart(4, '0')}`
const ORIGIN = 'https://relay.test'

type Frame = { t: string; rid?: string; [key: string]: unknown }

async function attachDonor(card: string, key = KEY) {
  const response = await SELF.fetch(`${ORIGIN}/donor/${card}`, {
    headers: { Upgrade: 'websocket', 'Sec-WebSocket-Protocol': `tf-donation-v1, ${key}` },
  })
  const socket = response.webSocket
  return { response, socket }
}

const replyWith = (socket: WebSocket, chunks: string[]) => {
  socket.addEventListener('message', (event) => {
    const frame = JSON.parse(String((event as MessageEvent).data)) as Frame
    if (frame.t !== 'req') return
    const send = (payload: Frame) => socket.send(JSON.stringify(payload))
    const rid = String(frame.rid)
    send({ t: 'head', rid, status: 200, headers: { 'content-type': 'text/event-stream' } })
    for (const chunk of chunks) send({ t: 'chunk', rid, data: btoa(chunk) })
    send({ t: 'end', rid })
  })
}

describe('donation relay', () => {
  it('answers its health route without a donor attached', async () => {
    const response = await SELF.fetch(`${ORIGIN}/health`)
    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({ status: 'ok' })
  })

  it('reports 503 when no donor is attached to the card', async () => {
    const response = await SELF.fetch(`${ORIGIN}/${nextCard()}/v1/models`, { headers: { authorization: 'Bearer t' } })
    expect(response.status).toBe(503)
    expect(await response.json()).toMatchObject({ error: { message: expect.stringContaining('No donor') } })
  })

  it('refuses routes it does not lend', async () => {
    const response = await SELF.fetch(`${ORIGIN}/${nextCard()}/v1/embeddings`, { method: 'POST', body: '{}' })
    expect(response.status).toBe(404)
  })

  it('refuses a request body larger than it forwards', async () => {
    const response = await SELF.fetch(`${ORIGIN}/${nextCard()}/v1/chat/completions`, { method: 'POST', body: new Uint8Array(1_200_000) })
    expect(response.status).toBe(413)
  })

  it('refuses a donor that does not offer a key', async () => {
    const response = await SELF.fetch(`${ORIGIN}/donor/${nextCard()}`, { headers: { Upgrade: 'websocket' } })
    expect(response.status).toBe(400)
  })

  it('streams a donor response back to the renter', async () => {
    const card = nextCard()
    const { response: attach, socket } = await attachDonor(card)
    expect(attach.status).toBe(101)
    expect(socket).toBeDefined()
    socket!.accept()
    replyWith(socket as WebSocket, ['data: one\n\n', 'data: two\n\n'])

    const response = await SELF.fetch(`${ORIGIN}/${card}/v1/chat/completions`, {
      method: 'POST',
      headers: { authorization: 'Bearer card-token', 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'm', messages: [] }),
    })
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe('text/event-stream')
    expect(await response.text()).toBe('data: one\n\ndata: two\n\n')
  })

  it('passes the donor refusals through untouched', async () => {
    const card = nextCard()
    const { response: attach, socket } = await attachDonor(card)
    expect(attach.status).toBe(101)
    socket!.accept()
    socket!.addEventListener('message', (event) => {
      const frame = JSON.parse(String((event as MessageEvent).data)) as Frame
      socket!.send(JSON.stringify({ t: 'err', rid: String(frame.rid), status: 401, message: 'The donation token is missing or wrong.' }))
    })

    const response = await SELF.fetch(`${ORIGIN}/${card}/v1/models`, { headers: { authorization: 'Bearer nope' } })
    expect(response.status).toBe(401)
    expect(await response.json()).toMatchObject({ error: { message: 'The donation token is missing or wrong.' } })
  })

  it('keeps one donor per card id', async () => {
    const card = nextCard()
    const first = await attachDonor(card)
    expect(first.response.status).toBe(101)
    first.socket!.accept()

    const second = await attachDonor(card)
    expect(second.response.status).toBe(409)
  })

  it('refuses a different donor key for a claimed card id', async () => {
    const card = nextCard()
    const first = await attachDonor(card)
    expect(first.response.status).toBe(101)
    first.socket!.accept()
    first.socket!.close()

    const other = await attachDonor(card, 'z'.repeat(32))
    expect(other.response.status).toBe(403)
  })
})
