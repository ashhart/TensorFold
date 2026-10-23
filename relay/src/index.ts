import { CARD_ID, GET_ROUTES, LENT_ROUTE } from './protocol'

export { DonorRelay } from './donor'

export interface Env {
  DONORS: DurableObjectNamespace
}

const reply = (status: number, message: string): Response => Response.json({ error: { message } }, { status })

/**
 * `https://<relay>/<card-id>/v1/...` is an OpenAI-compatible base URL, so a card
 * only needs to point its `endpoint` here: no app change on the renter side.
 *
 * `wss://<relay>/donor/<card-id>` is the donor's outbound attachment.
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === '/health') return Response.json({ status: 'ok', schema: 'tensorfold-donation-relay/1' })

    const donor = url.pathname.match(/^\/donor\/([^/]+)$/)
    if (donor) {
      if (request.headers.get('upgrade')?.toLowerCase() !== 'websocket') return reply(426, 'A donor attaches with a WebSocket upgrade.')
      if (!CARD_ID.test(donor[1])) return reply(400, 'That is not a valid card id.')
      return env.DONORS.get(env.DONORS.idFromName(donor[1])).fetch(request)
    }

    const card = url.pathname.match(/^\/([^/]+)(\/.*)$/)
    if (card) {
      const [, id, rest] = card
      if (!CARD_ID.test(id)) return reply(404, 'Unknown card id.')
      if (!LENT_ROUTE.test(rest)) return reply(404, 'Only the health probe, models, chat completions and responses are relayed.')
      const expected = GET_ROUTES.has(rest) ? 'GET' : 'POST'
      if (request.method !== expected) return reply(405, `Use ${expected} for ${rest}.`)
      return env.DONORS.get(env.DONORS.idFromName(id)).fetch(request)
    }

    return reply(404, 'Nothing is served here.')
  },
}
