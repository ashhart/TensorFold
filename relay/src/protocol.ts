/**
 * Wire protocol between the relay Worker and a donor's dialer.
 *
 * The relay never parses OpenAI payloads: a renter's HTTP request becomes one
 * `req` frame, the donor answers with `head`, any number of `chunk` frames and
 * one terminal `end` or `err`. Bodies are base64 so frames stay JSON and can be
 * logged or inspected without a binary decoder.
 */

/** Largest renter request body the relay will forward (matches the donor's cap). */
export const MAX_REQUEST_BYTES = 1 << 20
/** In-flight renter requests per card id. */
export const MAX_INFLIGHT = 8
/** How long a renter waits for the donor's response headers. */
export const HEAD_TIMEOUT_MS = 20_000
/** A stream that delivers nothing for this long is aborted. */
export const IDLE_TIMEOUT_MS = 120_000
/** Total bytes buffered for one renter before the relay gives up. */
export const MAX_STREAM_BYTES = 16 << 20

export type RelayFrame =
  | { t: 'req'; rid: string; method: string; path: string; headers: Record<string, string>; body: string }
  | { t: 'head'; rid: string; status: number; headers: Record<string, string> }
  | { t: 'chunk'; rid: string; data: string }
  | { t: 'end'; rid: string }
  | { t: 'err'; rid: string; status?: number; message: string }
  | { t: 'ping' }
  | { t: 'pong' }

/** Card ids are `donate-<hex>` today; keep the alphabet narrow so paths are simple. */
export const CARD_ID = /^[a-z0-9][a-z0-9-]{5,63}$/
/**
 * Only these routes are forwarded; everything else is a 404 from the relay.
 * `/health` is included because the app probes it before it accepts a card.
 */
export const LENT_ROUTE = /^\/(health|v1\/(models|chat\/completions|responses))$/
/** The two routes that take GET; everything else lent takes POST. */
export const GET_ROUTES = new Set(['/health', '/v1/models'])
