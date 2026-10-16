// OAuth 2.1 + PKCE request builder and MCP registry search/install descriptor parser —
// the pure pieces that let TensorFold connect to REMOTE MCP servers (Streamable HTTP transport)
// that gate their endpoints behind an OAuth authorization server, and to discover/install
// servers from an MCP registry. `mcp-client.ts` is stdio/local oriented; this is the remote
// half's PURE core.
//
// Everything here is side-effect-free and I/O is dependency-injected (a `fetch` impl, a
// `random` source for the PKCE verifier, a `now` clock) so the whole flow is unit-testable
// without a real browser, a real authorization server, or a real registry — the same idiom as
// connector-retry.ts (fetchImpl), ssh-connection.ts (clientFactory), pr-publish.ts (runner).
//
// Scope (deliberately contained):
//   - PKCE (RFC 7636) verifier + S256 challenge generation.
//   - Authorization-Server metadata discovery per RFC 8414 (.well-known/oauth-authorization-server)
//     — pure URL builder + response parser.
//   - Authorization-request URL construction (RFC 6749 §4.1 + PKCE, `resource` per RFC 8707).
//   - Authorization-code → token exchange request builder + token-response parser (RFC 6749 §4.1.3,
//     §5.1/§5.2) and refresh-token grant (§6).
//   - MCP registry search request builder + search-result / install-descriptor parser that maps a
//     registry entry into a normalized `RegistryInstallDescriptor` (remote HTTP or stdio package).
//
// Deferred (NOT here — needs process/browser/persistence, so it's a wire, not a pure core):
//   opening the system browser, running the loopback redirect listener, persisting tokens to the
//   keychain, and wiring the resulting bearer token into the Streamable-HTTP MCP transport.

import { readBoundedJsonResponse } from './bounded-response'
import { discardResponseBody, type PinnedHostResolver } from './pinned-fetch'
import { createConnectorFetch } from './secure-fetch'

// ----------------------------------------------------------------------------------------------
// PKCE (RFC 7636)
// ----------------------------------------------------------------------------------------------

const UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'

/** base64url without padding (RFC 7636 Appendix A). Accepts bytes. */
export function base64UrlEncode(bytes: Uint8Array): string {
  let bin = ''
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
  const b64 = typeof btoa === 'function'
    ? btoa(bin)
    : Buffer.from(bytes).toString('base64')
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/**
 * Generate a PKCE `code_verifier`: a high-entropy string of 43–128 chars from the unreserved
 * set (RFC 7636 §4.1). `random` is injectable (default `Math.random`) so tests are deterministic;
 * `length` defaults to 64 and is clamped to the legal [43,128] range.
 */
export function generateCodeVerifier(random: () => number = Math.random, length = 64): string {
  const n = Math.min(128, Math.max(43, Math.floor(length)))
  let out = ''
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(random() * UNRESERVED.length) % UNRESERVED.length
    out += UNRESERVED[idx]
  }
  return out
}

/** SHA-256 of the ASCII verifier as raw bytes. Uses WebCrypto if present, else node:crypto. */
async function sha256(input: string): Promise<Uint8Array> {
  const data = new TextEncoder().encode(input)
  const wc: any = (globalThis as any).crypto
  if (wc?.subtle?.digest) {
    const buf = await wc.subtle.digest('SHA-256', data)
    return new Uint8Array(buf)
  }
  // node:crypto fallback (main process / jiti test runner).
  const { createHash } = await import('node:crypto')
  return new Uint8Array(createHash('sha256').update(Buffer.from(data)).digest())
}

export interface PkcePair {
  verifier: string
  challenge: string
  method: 'S256'
}

/**
 * Derive the S256 `code_challenge` for a verifier (RFC 7636 §4.2). Async because the digest is.
 * OAuth 2.1 mandates S256 (the `plain` method is removed), so this only ever emits S256.
 */
export async function deriveCodeChallenge(verifier: string): Promise<string> {
  return base64UrlEncode(await sha256(verifier))
}

/** Convenience: generate a fresh verifier + its S256 challenge in one call. */
export async function createPkcePair(random: () => number = Math.random, length = 64): Promise<PkcePair> {
  const verifier = generateCodeVerifier(random, length)
  const challenge = await deriveCodeChallenge(verifier)
  return { verifier, challenge, method: 'S256' }
}

// ----------------------------------------------------------------------------------------------
// Authorization-Server metadata discovery (RFC 8414)
// ----------------------------------------------------------------------------------------------

export interface AuthServerMetadata {
  issuer: string
  authorizationEndpoint: string
  tokenEndpoint: string
  registrationEndpoint: string | null
  // Advertised PKCE methods (RFC 8414). Empty when the server omits the field.
  codeChallengeMethodsSupported: string[]
  scopesSupported: string[]
  raw: Record<string, unknown>
}

/**
 * Build the RFC 8414 metadata URL for an issuer. The well-known segment is inserted between the
 * host and the issuer's path component (per RFC 8414 §3), not naively appended — so
 * `https://as.example.com/tenant1` → `https://as.example.com/.well-known/oauth-authorization-server/tenant1`.
 * A bare-origin issuer yields `https://as.example.com/.well-known/oauth-authorization-server`.
 * Pure; throws on an unparseable issuer.
 */
export function authServerMetadataUrl(issuer: string): string {
  const u = new URL(issuer)
  const path = u.pathname.replace(/\/+$/, '')
  const wellKnown = '/.well-known/oauth-authorization-server'
  u.pathname = path && path !== '/' ? `${wellKnown}${path}` : wellKnown
  u.search = ''
  u.hash = ''
  return u.toString()
}

/** Pure parser for an RFC 8414 metadata document. Throws when required endpoints are missing. */
export function parseAuthServerMetadata(doc: unknown): AuthServerMetadata {
  if (!doc || typeof doc !== 'object') throw new Error('OAuth metadata is not an object')
  const o = doc as Record<string, unknown>
  const str = (k: string): string | null => (typeof o[k] === 'string' && (o[k] as string).trim() ? (o[k] as string) : null)
  const authorizationEndpoint = str('authorization_endpoint')
  const tokenEndpoint = str('token_endpoint')
  if (!authorizationEndpoint) throw new Error('OAuth metadata missing authorization_endpoint')
  if (!tokenEndpoint) throw new Error('OAuth metadata missing token_endpoint')
  const arr = (k: string): string[] => (Array.isArray(o[k]) ? (o[k] as unknown[]).filter((x): x is string => typeof x === 'string') : [])
  return {
    issuer: str('issuer') ?? '',
    authorizationEndpoint,
    tokenEndpoint,
    registrationEndpoint: str('registration_endpoint'),
    codeChallengeMethodsSupported: arr('code_challenge_methods_supported'),
    scopesSupported: arr('scopes_supported'),
    raw: o,
  }
}

/**
 * Fetch + parse RFC 8414 metadata. `fetchImpl` is injected. Throws on a non-2xx response or a
 * malformed document. This is the one async network helper — kept thin over the pure parser.
 */
export async function discoverAuthServerMetadata(
  issuer: string,
  opts: { fetchImpl?: typeof fetch; lookup?: PinnedHostResolver } = {},
): Promise<AuthServerMetadata> {
  const url = authServerMetadataUrl(issuer)
  const fetchImpl = createConnectorFetch(issuer, 'OAuth authorization server', { lookup: opts.lookup }, opts.fetchImpl ?? fetch)
  const res = await fetchImpl(url, { headers: { accept: 'application/json' }, redirect: 'error' })
  if (!res.ok) {
    await discardResponseBody(res)
    throw new Error(`OAuth metadata discovery failed: HTTP ${res.status}`)
  }
  return parseAuthServerMetadata(await readBoundedJsonResponse(res, 'OAuth metadata'))
}

// ----------------------------------------------------------------------------------------------
// Authorization request (RFC 6749 §4.1 + PKCE + RFC 8707 resource indicator)
// ----------------------------------------------------------------------------------------------

export interface AuthorizeUrlParams {
  authorizationEndpoint: string
  clientId: string
  redirectUri: string
  codeChallenge: string
  // OAuth 2.1 keeps `state` as CSRF protection; caller supplies it and matches on callback.
  state: string
  scopes?: string[]
  // RFC 8707 resource indicator — for MCP this is the server's canonical resource URI so the
  // issued token is audience-bound to this MCP server.
  resource?: string
  // Extra vendor params merged last (e.g. `audience`, `prompt`). Never overrides the core set.
  extraParams?: Record<string, string>
}

/**
 * Build the full authorization-request URL (RFC 6749 §4.1.1) with PKCE (`code_challenge` +
 * `code_challenge_method=S256`) and an optional RFC 8707 `resource`. Pure; preserves any query
 * already present on the endpoint. Throws when a required field is empty.
 */
export function buildAuthorizeUrl(p: AuthorizeUrlParams): string {
  for (const [k, v] of [['authorizationEndpoint', p.authorizationEndpoint], ['clientId', p.clientId], ['redirectUri', p.redirectUri], ['codeChallenge', p.codeChallenge], ['state', p.state]] as const) {
    if (!v || !String(v).trim()) throw new Error(`buildAuthorizeUrl: missing ${k}`)
  }
  const u = new URL(p.authorizationEndpoint)
  const q = u.searchParams
  q.set('response_type', 'code')
  q.set('client_id', p.clientId)
  q.set('redirect_uri', p.redirectUri)
  q.set('code_challenge', p.codeChallenge)
  q.set('code_challenge_method', 'S256')
  q.set('state', p.state)
  if (p.scopes && p.scopes.length) q.set('scope', p.scopes.join(' '))
  if (p.resource) q.set('resource', p.resource)
  for (const [k, v] of Object.entries(p.extraParams ?? {})) {
    if (!q.has(k)) q.set(k, v)
  }
  return u.toString()
}

// ----------------------------------------------------------------------------------------------
// Token exchange (RFC 6749 §4.1.3 authorization_code, §6 refresh_token)
// ----------------------------------------------------------------------------------------------

export interface TokenRequest {
  url: string
  method: 'POST'
  headers: Record<string, string>
  // application/x-www-form-urlencoded body, ready to pass as fetch `body`.
  body: string
}

export interface AuthCodeExchangeParams {
  tokenEndpoint: string
  clientId: string
  code: string
  redirectUri: string
  codeVerifier: string
  // Confidential clients only; public clients (the MCP desktop case) omit it.
  clientSecret?: string
  resource?: string
}

function formEncode(pairs: Record<string, string | undefined>): string {
  const b = new URLSearchParams()
  for (const [k, v] of Object.entries(pairs)) if (v != null && v !== '') b.set(k, v)
  return b.toString()
}

/**
 * Build the authorization-code → token exchange request (RFC 6749 §4.1.3) carrying the PKCE
 * `code_verifier`. A `clientSecret`, when present, is sent via HTTP Basic auth (RFC 6749 §2.3.1);
 * otherwise the client is public (PKCE-only). Pure — returns a descriptor; the caller does the
 * network with its injected fetch. Throws on missing required fields.
 */
export function buildAuthCodeTokenRequest(p: AuthCodeExchangeParams): TokenRequest {
  for (const [k, v] of [['tokenEndpoint', p.tokenEndpoint], ['clientId', p.clientId], ['code', p.code], ['redirectUri', p.redirectUri], ['codeVerifier', p.codeVerifier]] as const) {
    if (!v || !String(v).trim()) throw new Error(`buildAuthCodeTokenRequest: missing ${k}`)
  }
  const headers: Record<string, string> = {
    'content-type': 'application/x-www-form-urlencoded',
    accept: 'application/json',
  }
  const bodyPairs: Record<string, string | undefined> = {
    grant_type: 'authorization_code',
    code: p.code,
    redirect_uri: p.redirectUri,
    code_verifier: p.codeVerifier,
    client_id: p.clientId,
    resource: p.resource,
  }
  if (p.clientSecret) {
    headers.authorization = 'Basic ' + base64Basic(p.clientId, p.clientSecret)
    // With Basic auth the client_id is still allowed in the body but not required; keep it for
    // servers that read it from the body regardless.
  }
  return { url: p.tokenEndpoint, method: 'POST', headers, body: formEncode(bodyPairs) }
}

export interface RefreshTokenParams {
  tokenEndpoint: string
  clientId: string
  refreshToken: string
  clientSecret?: string
  scopes?: string[]
  resource?: string
}

/** Build a refresh-token grant request (RFC 6749 §6). Pure. Throws on missing required fields. */
export function buildRefreshTokenRequest(p: RefreshTokenParams): TokenRequest {
  for (const [k, v] of [['tokenEndpoint', p.tokenEndpoint], ['clientId', p.clientId], ['refreshToken', p.refreshToken]] as const) {
    if (!v || !String(v).trim()) throw new Error(`buildRefreshTokenRequest: missing ${k}`)
  }
  const headers: Record<string, string> = {
    'content-type': 'application/x-www-form-urlencoded',
    accept: 'application/json',
  }
  const bodyPairs: Record<string, string | undefined> = {
    grant_type: 'refresh_token',
    refresh_token: p.refreshToken,
    client_id: p.clientId,
    scope: p.scopes && p.scopes.length ? p.scopes.join(' ') : undefined,
    resource: p.resource,
  }
  if (p.clientSecret) headers.authorization = 'Basic ' + base64Basic(p.clientId, p.clientSecret)
  return { url: p.tokenEndpoint, method: 'POST', headers, body: formEncode(bodyPairs) }
}

function base64Basic(id: string, secret: string): string {
  const raw = `${encodeURIComponent(id)}:${encodeURIComponent(secret)}`
  const bytes = new TextEncoder().encode(raw)
  let bin = ''
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
  return typeof btoa === 'function' ? btoa(bin) : Buffer.from(bytes).toString('base64')
}

export interface TokenSet {
  accessToken: string
  tokenType: string
  refreshToken: string | null
  // Absolute expiry (ms epoch) computed from `expires_in` + `now`, or null when not provided.
  expiresAt: number | null
  scopes: string[]
  raw: Record<string, unknown>
}

/**
 * Parse an RFC 6749 §5.1 success token response into a normalized `TokenSet`, or throw a rich
 * error for an §5.2 error response. `now` is injectable so `expiresAt` is deterministic. Pure.
 */
export function parseTokenResponse(doc: unknown, now: number = Date.now()): TokenSet {
  if (!doc || typeof doc !== 'object') throw new Error('token response is not an object')
  const o = doc as Record<string, unknown>
  if (typeof o.error === 'string') {
    const desc = typeof o.error_description === 'string' ? `: ${o.error_description}` : ''
    throw new Error(`OAuth token error [${o.error}]${desc}`)
  }
  const accessToken = typeof o.access_token === 'string' ? o.access_token : ''
  if (!accessToken) throw new Error('token response missing access_token')
  const expiresIn = typeof o.expires_in === 'number' ? o.expires_in
    : typeof o.expires_in === 'string' && /^\d+$/.test(o.expires_in) ? Number(o.expires_in)
      : null
  const scopeStr = typeof o.scope === 'string' ? o.scope : ''
  return {
    accessToken,
    tokenType: typeof o.token_type === 'string' ? o.token_type : 'Bearer',
    refreshToken: typeof o.refresh_token === 'string' ? o.refresh_token : null,
    expiresAt: expiresIn != null ? now + expiresIn * 1000 : null,
    scopes: scopeStr ? scopeStr.split(/\s+/).filter(Boolean) : [],
    raw: o,
  }
}

/** True when a TokenSet is expired (or within `skewMs`, default 30s) at `now`. Pure. */
export function isTokenExpired(token: Pick<TokenSet, 'expiresAt'>, now: number = Date.now(), skewMs = 30_000): boolean {
  if (token.expiresAt == null) return false
  return now >= token.expiresAt - skewMs
}

// ----------------------------------------------------------------------------------------------
// MCP registry: search request + result / install-descriptor parsing
// ----------------------------------------------------------------------------------------------

export interface RegistrySearchParams {
  // Registry base URL, e.g. https://registry.modelcontextprotocol.io
  registryUrl: string
  query?: string
  limit?: number
  // Opaque pagination cursor the registry returned previously.
  cursor?: string
}

/**
 * Build the registry search URL. Follows the MCP registry convention of a `/v0/servers` listing
 * endpoint with `search`, `limit`, and `cursor` query params. Pure. Preserves an existing path on
 * the base URL (so a registry mounted under a sub-path still works). Throws on a bad base URL.
 */
export function buildRegistrySearchUrl(p: RegistrySearchParams): string {
  const base = new URL(p.registryUrl)
  const basePath = base.pathname.replace(/\/+$/, '')
  const u = new URL(base.toString())
  u.pathname = `${basePath}/v0/servers`
  u.search = ''
  if (p.query && p.query.trim()) u.searchParams.set('search', p.query.trim())
  if (p.limit != null && Number.isFinite(p.limit)) u.searchParams.set('limit', String(Math.max(1, Math.floor(p.limit))))
  if (p.cursor) u.searchParams.set('cursor', p.cursor)
  return u.toString()
}

export type RegistryTransportKind = 'stdio' | 'streamable-http' | 'sse'

export interface RegistryRemoteEndpoint {
  transport: RegistryTransportKind // 'streamable-http' | 'sse'
  url: string
}

export interface RegistryStdioPackage {
  // e.g. 'npm' | 'pypi' | 'oci' — the registry name (registry_name / registryType).
  registryType: string
  identifier: string
  version: string | null
  runtimeHint: string | null // e.g. 'npx' | 'uvx' | 'docker'
}

export interface RegistryInstallDescriptor {
  id: string
  name: string
  description: string
  version: string | null
  // A remote server exposes HTTP/SSE `remotes`; a local server exposes installable `packages`.
  remotes: RegistryRemoteEndpoint[]
  packages: RegistryStdioPackage[]
  // Convenience: does this entry offer a remote (OAuth-capable) HTTP endpoint?
  isRemote: boolean
  raw: Record<string, unknown>
}

export interface RegistrySearchResult {
  servers: RegistryInstallDescriptor[]
  nextCursor: string | null
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null
}
function asString(v: unknown): string | null {
  return typeof v === 'string' && v.trim() ? v : null
}

/** Normalize one registry server entry into an install descriptor. Tolerant of field-name drift
 *  across registry versions (`registry_name`/`registryType`, `runtime_hint`/`runtimeHint`). */
export function parseRegistryServer(entry: unknown): RegistryInstallDescriptor | null {
  const o = asRecord(entry)
  if (!o) return null
  // Some registries nest the server under `server` and metadata under `_meta`; unwrap.
  const s = asRecord(o.server) ?? o
  const name = asString(s.name) ?? asString(o.name)
  if (!name) return null
  const id = asString(s.id) ?? asString(o.id) ?? asString((asRecord(o._meta) ?? {}).id) ?? name

  const remotes: RegistryRemoteEndpoint[] = []
  const remoteList = Array.isArray(s.remotes) ? s.remotes : []
  for (const r of remoteList) {
    const ro = asRecord(r)
    if (!ro) continue
    const url = asString(ro.url)
    if (!url) continue
    const t = asString(ro.transport) ?? asString(ro.type)
    const transport: RegistryTransportKind = t === 'sse' ? 'sse' : 'streamable-http'
    remotes.push({ transport, url })
  }

  const packages: RegistryStdioPackage[] = []
  const pkgList = Array.isArray(s.packages) ? s.packages : []
  for (const pk of pkgList) {
    const po = asRecord(pk)
    if (!po) continue
    const identifier = asString(po.identifier) ?? asString(po.name)
    if (!identifier) continue
    packages.push({
      registryType: asString(po.registry_type) ?? asString(po.registryType) ?? asString(po.registry_name) ?? 'npm',
      identifier,
      version: asString(po.version),
      runtimeHint: asString(po.runtime_hint) ?? asString(po.runtimeHint),
    })
  }

  return {
    id,
    name,
    description: asString(s.description) ?? asString(o.description) ?? '',
    version: asString(s.version) ?? asString((asRecord(s.version_detail) ?? {}).version),
    remotes,
    packages,
    isRemote: remotes.length > 0,
    raw: o,
  }
}

/** Parse a registry `/v0/servers` search response into descriptors + a pagination cursor. Pure. */
export function parseRegistrySearchResponse(doc: unknown): RegistrySearchResult {
  const o = asRecord(doc)
  if (!o) return { servers: [], nextCursor: null }
  const list = Array.isArray(o.servers) ? o.servers : Array.isArray(o.data) ? o.data : []
  const servers: RegistryInstallDescriptor[] = []
  for (const e of list) {
    const d = parseRegistryServer(e)
    if (d) servers.push(d)
  }
  // Cursor lives under metadata.next_cursor (MCP registry) or a top-level next_cursor.
  const meta = asRecord(o.metadata)
  const nextCursor = asString(meta?.next_cursor ?? '') ?? asString(o.next_cursor ?? '') ?? asString(o.nextCursor ?? '')
  return { servers, nextCursor }
}

/**
 * Fetch + parse a registry search in one call. `fetchImpl` injected. Throws on non-2xx.
 * Thin wrapper over the pure builder/parser above.
 */
export async function searchMcpRegistry(
  p: RegistrySearchParams,
  opts: { fetchImpl?: typeof fetch; lookup?: PinnedHostResolver } = {},
): Promise<RegistrySearchResult> {
  const url = buildRegistrySearchUrl(p)
  const fetchImpl = createConnectorFetch(p.registryUrl, 'MCP registry', { lookup: opts.lookup }, opts.fetchImpl ?? fetch)
  const res = await fetchImpl(url, { headers: { accept: 'application/json' }, redirect: 'error' })
  if (!res.ok) {
    await discardResponseBody(res)
    throw new Error(`MCP registry search failed: HTTP ${res.status}`)
  }
  return parseRegistrySearchResponse(await readBoundedJsonResponse(res, 'MCP registry'))
}

/**
 * Pick the best remote endpoint from a descriptor for connecting a Streamable-HTTP MCP client:
 * prefer `streamable-http` over the legacy `sse` transport. Returns null for a local-only entry.
 * Pure.
 */
export function preferredRemoteEndpoint(d: RegistryInstallDescriptor): RegistryRemoteEndpoint | null {
  return d.remotes.find(r => r.transport === 'streamable-http') ?? d.remotes[0] ?? null
}
