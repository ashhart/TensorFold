// DNS-to-connect pinning for outbound Node/Electron fetches.
//
// A DNS safety check followed by an ordinary fetch is vulnerable to rebinding: the
// validator can see an allowed address and the socket's later DNS lookup can see a
// private/metadata address. This module resolves once, lets the caller apply its own
// endpoint policy to every result, then installs those exact results as undici's socket
// lookup. There is deliberately no unpinned fallback.

import { lookup as dnsLookup } from 'node:dns/promises'
import { isIP } from 'node:net'
import type { Dispatcher } from 'undici'

export type PinnedAddressFamily = 4 | 6

export interface PinnedAddress {
  readonly address: string
  readonly family: PinnedAddressFamily
}

const PINNED_HOST = Symbol('vontra.pinned-host')

/** Opaque result of resolving and vetting one URL host. */
export interface PinnedHost {
  readonly url: URL
  readonly hostname: string
  readonly addresses: readonly PinnedAddress[]
  readonly [PINNED_HOST]: true
}

export type AssertPinnedAddress = (
  address: string,
  family: PinnedAddressFamily,
  url: URL,
) => void | boolean | Promise<void | boolean>

export type PinnedHostResolver = (hostname: string) => Promise<readonly {
  address: string
  family: number
}[]>

export interface ResolvePinnedHostOptions {
  /** Test seam. Production callers should use the default node:dns lookup. */
  lookup?: PinnedHostResolver
}

type LookupOptions = {
  all?: boolean
  family?: number | string
}

type LookupCallback = (
  error: NodeJS.ErrnoException | null,
  addressOrAddresses?: string | readonly PinnedAddress[],
  family?: number,
) => void

export type PinnedLookup = (
  hostname: string,
  options: LookupOptions,
  callback: LookupCallback,
) => void

type UndiciAgentConstructor = new (options: Record<string, unknown>) => Dispatcher

export interface PinnedDispatcherLease {
  readonly dispatcher: Dispatcher
  readonly hostname: string
  readonly addresses: readonly PinnedAddress[]
  /** Gracefully stop accepting work and close sockets after active response bodies finish. */
  close(): Promise<void>
  /** Immediately abort active work. */
  destroy(error?: Error): Promise<void>
}

export type PinnedFetch = typeof fetch

/** Release a response whose body the caller intentionally does not consume. */
export async function discardResponseBody(response: Pick<Response, 'body'> | null | undefined): Promise<void> {
  try { await response?.body?.cancel() } catch { /* best-effort cleanup; preserve the caller's result */ }
}

function normalizedHostname(raw: string): string {
  let hostname = String(raw ?? '').trim().toLowerCase()
  if (hostname.startsWith('[') && hostname.endsWith(']')) hostname = hostname.slice(1, -1)
  while (hostname.endsWith('.')) hostname = hostname.slice(0, -1)
  if (!hostname || hostname.includes('\0') || /[\s/\\]/.test(hostname)) {
    throw new Error('Cannot pin an empty or invalid hostname.')
  }
  return hostname
}

function addressFamily(raw: number): PinnedAddressFamily {
  if (raw === 4 || raw === 6) return raw
  throw new Error(`Cannot pin an address with unsupported family ${String(raw)}.`)
}

function normalizedAddresses(input: readonly { address: string; family: number }[]): readonly PinnedAddress[] {
  if (!input.length) throw new Error('Cannot create a pinned dispatcher without vetted addresses.')
  const seen = new Set<string>()
  const result: PinnedAddress[] = []
  for (const raw of input) {
    const address = String(raw?.address ?? '').trim()
    const family = addressFamily(Number(raw?.family))
    const actualFamily = isIP(address)
    if (!actualFamily) throw new Error(`Cannot pin invalid IP address ${JSON.stringify(address)}.`)
    if (actualFamily !== family) {
      throw new Error(`Cannot pin ${address} as IPv${family}; it is IPv${actualFamily}.`)
    }
    const key = `${family}:${address.toLowerCase()}`
    if (seen.has(key)) continue
    seen.add(key)
    result.push(Object.freeze({ address, family }))
  }
  if (!result.length) throw new Error('Cannot create a pinned dispatcher without vetted addresses.')
  return Object.freeze(result)
}

function lookupError(message: string, code: string): NodeJS.ErrnoException {
  return Object.assign(new Error(message), { code })
}

/**
 * Create the socket lookup installed in undici. It can only return the vetted addresses
 * for the exact expected host; an unexpected host or family fails closed without touching
 * system DNS. Exported because it is also useful for other Node socket clients.
 */
export function createPinnedLookup(
  hostname: string,
  addresses: readonly { address: string; family: number }[],
): PinnedLookup {
  const expected = normalizedHostname(hostname)
  const pinned = normalizedAddresses(addresses)

  return (requestedHost, options, callback) => {
    let requested: string
    try {
      requested = normalizedHostname(requestedHost)
    } catch (error) {
      callback(error as NodeJS.ErrnoException)
      return
    }
    if (requested !== expected) {
      callback(lookupError(
        `Refusing unpinned DNS lookup for unexpected host ${requestedHost}; expected ${hostname}.`,
        'EHOSTUNREACH',
      ))
      return
    }

    const rawFamily = options?.family
    const requestedFamily = rawFamily === 'IPv4' ? 4 : rawFamily === 'IPv6' ? 6 : Number(rawFamily ?? 0)
    if (requestedFamily !== 0 && requestedFamily !== 4 && requestedFamily !== 6) {
      callback(lookupError(
        `No vetted IPv${String(rawFamily)} address is available for ${hostname}.`,
        'EAI_ADDRFAMILY',
      ))
      return
    }
    const candidates = requestedFamily === 4 || requestedFamily === 6
      ? pinned.filter((entry) => entry.family === requestedFamily)
      : pinned
    if (!candidates.length) {
      callback(lookupError(
        `No vetted IPv${requestedFamily} address is available for ${hostname}.`,
        'EAI_ADDRFAMILY',
      ))
      return
    }

    // Keep callback timing consistent with node:dns.lookup and avoid surprising re-entrancy
    // in net/tls. Modern Node/undici requests `all: true`; the scalar branch keeps this
    // compatible with clients that use the traditional lookup callback.
    queueMicrotask(() => {
      if (options?.all) {
        callback(null, candidates)
      } else {
        callback(null, candidates[0].address, candidates[0].family)
      }
    })
  }
}

function requireUndiciAgent(): UndiciAgentConstructor {
  try {
    // Keep the runtime dependency explicit and fail closed. The caller must never fall back
    // to global fetch without a dispatcher if packaging omitted `undici`.
    const runtime = require('undici') as { Agent?: UndiciAgentConstructor }
    if (typeof runtime?.Agent !== 'function') throw new Error('undici.Agent is unavailable')
    return runtime.Agent
  } catch (cause) {
    throw new Error(
      'DNS-pinned fetch requires the direct runtime dependency "undici"; refusing an unpinned request.',
      { cause },
    )
  }
}

/** Create an undici Agent whose socket lookup is limited to already-vetted addresses. */
export function createPinnedDispatcher(
  hostname: string,
  addresses: readonly { address: string; family: number }[],
): PinnedDispatcherLease {
  const expected = normalizedHostname(hostname)
  const pinned = normalizedAddresses(addresses)
  const Agent = requireUndiciAgent()
  const agent = new Agent({ connect: { lookup: createPinnedLookup(expected, pinned) } })
  let cleanup: Promise<void> | null = null

  return {
    dispatcher: agent,
    hostname: expected,
    addresses: pinned,
    close() {
      cleanup ??= agent.close()
      return cleanup
    },
    destroy(error?: Error) {
      cleanup ??= error ? agent.destroy(error) : agent.destroy()
      return cleanup
    },
  }
}

/**
 * Resolve a URL once and require the caller's policy to approve every address before a
 * dispatcher can be created. Resolver failures and empty answers are fatal (fail closed).
 */
export async function resolvePinnedHost(
  rawUrl: string | URL,
  assertAddress: AssertPinnedAddress,
  options: ResolvePinnedHostOptions = {},
): Promise<PinnedHost> {
  const url = rawUrl instanceof URL ? new URL(rawUrl.toString()) : new URL(String(rawUrl))
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('DNS-pinned fetch only supports http and https URLs.')
  }
  const hostname = normalizedHostname(url.hostname)
  const resolve = options.lookup ?? (async (host: string) => dnsLookup(host, { all: true }))

  let addresses: readonly PinnedAddress[]
  try {
    addresses = normalizedAddresses(await resolve(hostname))
  } catch (cause) {
    throw new Error(`Refusing to fetch ${hostname}: DNS resolution could not be safely pinned.`, { cause })
  }

  for (const entry of addresses) {
    const accepted = await assertAddress(entry.address, entry.family, url)
    if (accepted === false) {
      throw new Error(`Refusing to fetch ${hostname}: address ${entry.address} was rejected by endpoint policy.`)
    }
  }

  return Object.freeze({
    url,
    hostname,
    addresses,
    [PINNED_HOST]: true as const,
  })
}

function assertPinnedHost(target: PinnedHost): void {
  if (!target || target[PINNED_HOST] !== true) {
    throw new Error('pinnedFetch requires a host returned by resolvePinnedHost.')
  }
  if (normalizedHostname(target.url.hostname) !== target.hostname) {
    throw new Error('Refusing request because the pinned URL hostname changed after DNS validation.')
  }
}

function inputUrl(input: Parameters<typeof fetch>[0] | URL): URL {
  if (input instanceof URL) return new URL(input.toString())
  if (typeof input === 'string') return new URL(input)
  return new URL(input.url)
}

async function fetchAtPinnedHost(
  target: PinnedHost,
  input: Parameters<typeof fetch>[0] | URL,
  init: Parameters<typeof fetch>[1] = {},
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  assertPinnedHost(target)
  const url = inputUrl(input)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('DNS-pinned fetch only supports http and https URLs.')
  }
  if (normalizedHostname(url.hostname) !== target.hostname || url.origin !== target.url.origin) {
    throw new Error(`Refusing request to ${url.origin}: dispatcher is pinned to origin ${target.url.origin}.`)
  }
  const redirect = init?.redirect ?? 'error'
  if (redirect === 'follow') {
    throw new Error('DNS-pinned fetch cannot auto-follow redirects; resolve and pin each hop, or use redirect: "manual".')
  }

  const lease = createPinnedDispatcher(target.hostname, target.addresses)
  const requestInit = {
    ...(init ?? {}),
    redirect,
    // Never let a caller replace the pinned dispatcher through an extended RequestInit.
    dispatcher: lease.dispatcher,
  } as RequestInit & { dispatcher: Dispatcher }

  try {
    // Preserve a caller-supplied Request object (method/body/headers); `url` above is only
    // the immutable parsed view used for host validation.
    const response = await fetchImpl(input as Parameters<typeof fetch>[0], requestInit)
    // close() is graceful: it refuses new work immediately, then resolves after the caller
    // consumes/cancels the returned body. Do not await here or a streaming body would deadlock.
    void lease.close().catch(() => {})
    return response
  } catch (error) {
    await lease.destroy(error instanceof Error ? error : undefined).catch(() => {})
    throw error
  }
}

/** Fetch the exact URL that was resolved, using only its vetted addresses. */
export function pinnedFetch(
  target: PinnedHost,
  init: Parameters<typeof fetch>[1] = {},
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  return fetchAtPinnedHost(target, target.url, init, fetchImpl)
}

/**
 * Return a fetch-compatible function for retry/manual-redirect helpers. Every call must
 * retain the pinned hostname; a caller must resolvePinnedHost again for a new redirect host.
 */
export function createPinnedFetch(target: PinnedHost, fetchImpl: typeof fetch = fetch): PinnedFetch {
  assertPinnedHost(target)
  return ((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => (
    fetchAtPinnedHost(target, input, init, fetchImpl)
  )) as typeof fetch
}
