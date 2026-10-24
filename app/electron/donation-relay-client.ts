/**
 * Donor side of the donation relay.
 *
 * The donor never accepts inbound traffic: it dials the relay, then plays the
 * frames it receives against its own local donation node (`127.0.0.1:<port>`),
 * which keeps every guarantee where it already lives — token check, window,
 * caps, receipts.
 *
 * Only the transport is new here. The relay sees prompts in plaintext, so a
 * deployment that cares about that should encrypt the payload; that is not in
 * this POC.
 */
import { Buffer } from 'node:buffer'

/** Mirrors `relay/src/protocol.ts`; the relay is a separate deployable, so the types are duplicated rather than imported across package boundaries. */
type RelayFrame =
  | { t: 'req'; rid: string; method: string; path: string; headers: Record<string, string>; body: string }
  | { t: 'head'; rid: string; status: number; headers: Record<string, string> }
  | { t: 'chunk'; rid: string; data: string }
  | { t: 'end'; rid: string }
  | { t: 'err'; rid: string; status?: number; message: string }
  | { t: 'ping' }
  | { t: 'pong' }

interface SocketLike {
  readyState: number
  send(data: string): void
  close(): void
  addEventListener(type: string, listener: (event: unknown) => void): void
}

type Connect = (url: string, protocols: string[]) => SocketLike

export interface AttachDonationRelayOptions {
  /** `https://relay.example` or `wss://relay.example`; the `/donor/<id>` path is added here. */
  relayUrl: string
  cardId: string
  /** Random secret the donor keeps; the relay binds it to the card id on first attach. */
  donorKey: string
  /** Origin of the local donation node, e.g. `http://127.0.0.1:8020`. */
  localOrigin: string
  log?: (line: string) => void
  /** Test seam: replaces the global WebSocket. */
  connect?: Connect
  /** Test seam: disables the reconnect timer. */
  autoReconnect?: boolean
}

export interface DonationRelayAttachment {
  readonly cardId: string
  attached(): boolean
  close(): void
}

const FORWARDED_RESPONSE_HEADERS = ['content-type', 'cache-control']

const relaySocketUrl = (relayUrl: string, cardId: string): string => {
  const url = new URL(relayUrl)
  if (url.protocol === 'https:') url.protocol = 'wss:'
  else if (url.protocol === 'http:') url.protocol = 'ws:'
  if (url.protocol !== 'wss:' && url.protocol !== 'ws:') throw new Error('The relay address must be https, http, wss or ws.')
  url.pathname = `/donor/${assertCardId(cardId)}`
  url.search = ''
  url.hash = ''
  return url.toString()
}

const assertCardId = (value: string): string => {
  if (!/^[a-z0-9][a-z0-9-]{5,63}$/.test(value)) throw new Error('The card id is not one the relay accepts.')
  return value
}

/** The donor key travels as the second subprotocol because neither browsers nor Node can set WebSocket headers. */
export function attachDonationRelay(options: AttachDonationRelayOptions): DonationRelayAttachment {
  const log = options.log ?? (() => {})
  const autoReconnect = options.autoReconnect !== false
  const connect: Connect =
    options.connect ??
    ((url, protocols) => {
      const Ctor = (globalThis as unknown as { WebSocket?: new (url: string, protocols?: string[]) => SocketLike }).WebSocket
      if (!Ctor) throw new Error('This Node build has no WebSocket client.')
      return new Ctor(url, protocols)
    })

  let socket: SocketLike | null = null
  let stopped = false
  let attempt = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  const inFlight = new Set<AbortController>()

  const send = (frame: RelayFrame): void => {
    if (!socket || socket.readyState !== 1) return
    socket.send(JSON.stringify(frame))
  }

  const settle = async (frame: Extract<RelayFrame, { t: 'req' }>): Promise<void> => {
    const controller = new AbortController()
    inFlight.add(controller)
    try {
      const body = frame.method === 'GET' || frame.method === 'HEAD' ? undefined : Buffer.from(frame.body, 'base64')
      const response = await fetch(`${options.localOrigin.replace(/\/+$/, '')}${frame.path}`, {
        method: frame.method,
        headers: frame.headers,
        body,
        signal: controller.signal,
      })
      const headers: Record<string, string> = {}
      for (const name of FORWARDED_RESPONSE_HEADERS) {
        const value = response.headers.get(name)
        if (value) headers[name] = value
      }
      send({ t: 'head', rid: frame.rid, status: response.status, headers })
      if (response.body) {
        for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
          send({ t: 'chunk', rid: frame.rid, data: Buffer.from(chunk).toString('base64') })
        }
      }
      send({ t: 'end', rid: frame.rid })
    } catch (error) {
      send({ t: 'err', rid: frame.rid, status: 502, message: `The donor could not reach its local engine proxy: ${(error as Error).message}` })
    } finally {
      inFlight.delete(controller)
    }
  }

  const open = (): void => {
    const url = relaySocketUrl(options.relayUrl, options.cardId)
    log(`attaching ${options.cardId} to ${url}`)
    const next = connect(url, ['tf-donation-v1', options.donorKey])
    socket = next
    next.addEventListener('open', () => {
      attempt = 0
      log(`attached ${options.cardId}; renters can use the relay URL`)
    })
    next.addEventListener('message', (event) => {
      const raw = (event as { data?: unknown }).data
      if (typeof raw !== 'string') return
      let frame: RelayFrame
      try {
        frame = JSON.parse(raw) as RelayFrame
      } catch {
        return
      }
      if (frame.t === 'ping') return send({ t: 'pong' })
      if (frame.t === 'req') void settle(frame)
    })
    next.addEventListener('close', () => {
      if (socket === next) socket = null
      for (const controller of inFlight) controller.abort()
      inFlight.clear()
      if (stopped || !autoReconnect) return log(`detached ${options.cardId}`)
      attempt += 1
      const delay = Math.min(30_000, 1_000 * 2 ** Math.min(attempt, 5))
      log(`relay connection closed; retrying in ${Math.round(delay / 1000)}s`)
      timer = setTimeout(open, delay)
    })
    next.addEventListener('error', () => {
      log('relay connection reported an error')
    })
  }

  open()

  return {
    cardId: options.cardId,
    attached: () => socket !== null && socket.readyState === 1,
    close: () => {
      stopped = true
      if (timer) clearTimeout(timer)
      for (const controller of inFlight) controller.abort()
      inFlight.clear()
      socket?.close()
      socket = null
    },
  }
}
