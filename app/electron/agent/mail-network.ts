// DNS-pinned raw-socket helpers for credential-bearing mail protocols. HTTP callers use
// undici dispatchers; IMAP/SMTP open net/tls sockets and need the same resolve-once,
// validate-every-answer, connect-only-to-that-set invariant.

import { connect as netConnect, isIP, type Socket } from 'node:net'
import {
  createPinnedLookup,
  resolvePinnedHost,
  type PinnedHost,
  type PinnedHostResolver,
  type PinnedLookup,
} from './pinned-fetch'
import { isGloballyRoutableIp } from './security'

const MAIL_CONNECT_TIMEOUT_MS = 30_000

function mailHostUrl(rawHost: string, label: string): URL {
  const raw = String(rawHost || '').trim()
  if (!raw) throw new Error(`${label} host is required.`)
  const unbracketed = raw.replace(/^\[|\]$/g, '')
  const host = isIP(unbracketed) === 6 ? `[${unbracketed}]` : raw
  let url: URL
  try { url = new URL(`https://${host}/`) } catch { throw new Error(`${label} host is invalid.`) }
  if (url.username || url.password || url.port || url.pathname !== '/' || url.search || url.hash) {
    throw new Error(`${label} host must be a hostname or IP address without credentials, a port, or a path.`)
  }
  return url
}

export async function resolvePublicMailHost(
  rawHost: string,
  label: 'IMAP' | 'SMTP',
  lookup?: PinnedHostResolver,
): Promise<PinnedHost> {
  const url = mailHostUrl(rawHost, label)
  return resolvePinnedHost(url, (address) => {
    if (!isGloballyRoutableIp(address)) {
      throw new Error(`${label} host ${url.hostname} resolves to a private or non-public address (${address}); refusing to send credentials.`)
    }
  }, { lookup })
}

export function mailPinnedLookup(target: PinnedHost): PinnedLookup {
  return createPinnedLookup(target.hostname, target.addresses)
}

export type PinnedSocketFactory = (
  options: unknown,
  callback: (error: Error | null, socketOptions?: { connection: Socket }) => void,
) => void

/** Nodemailer getSocket factory: preconnects a raw socket through the pinned lookup. */
export function createPinnedSocketFactory(
  target: PinnedHost,
  port: number,
  timeoutMs = MAIL_CONNECT_TIMEOUT_MS,
): PinnedSocketFactory {
  const lookup = mailPinnedLookup(target)
  return (_options, callback) => {
    let handedOff = false
    const socket = netConnect({ host: target.hostname, port, lookup: lookup as any })
    const timer = setTimeout(() => {
      const error = Object.assign(new Error(`Timed out connecting to pinned mail host ${target.hostname}:${port}.`), { code: 'ETIMEDOUT' })
      socket.destroy(error)
    }, Math.max(1, timeoutMs))
    timer.unref?.()

    const fail = (error: Error) => {
      if (handedOff) return
      handedOff = true
      clearTimeout(timer)
      callback(error)
    }
    socket.once('error', fail)
    socket.once('connect', () => {
      if (handedOff) return
      handedOff = true
      clearTimeout(timer)
      // Hand the connected socket to Nodemailer before removing our bootstrap handler;
      // its callback synchronously attaches the transport's long-lived error handlers.
      callback(null, { connection: socket })
      socket.removeListener('error', fail)
    })
  }
}
