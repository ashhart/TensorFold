// Run: jiti electron/agent/mcp-oauth-registry.test.ts
import assert from 'node:assert/strict'
import {
  base64UrlEncode,
  generateCodeVerifier,
  deriveCodeChallenge,
  createPkcePair,
  authServerMetadataUrl,
  parseAuthServerMetadata,
  discoverAuthServerMetadata,
  buildAuthorizeUrl,
  buildAuthCodeTokenRequest,
  buildRefreshTokenRequest,
  parseTokenResponse,
  isTokenExpired,
  buildRegistrySearchUrl,
  parseRegistryServer,
  parseRegistrySearchResponse,
  searchMcpRegistry,
  preferredRemoteEndpoint,
} from './mcp-oauth-registry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('mcp-oauth-registry tests:')

const publicLookup = async () => [{ address: '8.8.8.8', family: 4 as const }]

// A deterministic pseudo-random sequence in [0,1) for reproducible verifiers.
function seq(values: number[]): () => number {
  let i = 0
  return () => values[(i++) % values.length]
}

// ---- base64url ----
{
  assert.equal(base64UrlEncode(new Uint8Array([0xff, 0xef, 0xff])), '_-__')
  assert.equal(base64UrlEncode(new Uint8Array([1])), 'AQ') // no '=' padding
  ok('base64UrlEncode: URL-safe alphabet, no padding')
}

// ---- PKCE verifier ----
{
  const v = generateCodeVerifier(seq([0]))
  assert.equal(v.length, 64)
  assert.match(v, /^[A-Za-z0-9\-._~]+$/) // unreserved set only
  // length clamps into [43,128]
  assert.equal(generateCodeVerifier(seq([0]), 10).length, 43)
  assert.equal(generateCodeVerifier(seq([0]), 1000).length, 128)
  // deterministic random → deterministic verifier
  const a = generateCodeVerifier(seq([0.1, 0.5, 0.9]))
  const b = generateCodeVerifier(seq([0.1, 0.5, 0.9]))
  assert.equal(a, b)
  ok('generateCodeVerifier: 43–128 unreserved chars, deterministic under injected random')
}

// ---- PKCE S256 challenge (known RFC 7636 Appendix B test vector) ----
{
  const verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
  const challenge = await deriveCodeChallenge(verifier)
  assert.equal(challenge, 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM') // RFC 7636 Appendix B
  const pair = await createPkcePair(seq([0.2, 0.4, 0.6]))
  assert.equal(pair.method, 'S256')
  assert.equal(pair.challenge, await deriveCodeChallenge(pair.verifier))
  ok('deriveCodeChallenge: matches RFC 7636 Appendix B S256 vector')
}

// ---- RFC 8414 metadata URL insertion ----
{
  assert.equal(authServerMetadataUrl('https://as.example.com'), 'https://as.example.com/.well-known/oauth-authorization-server')
  assert.equal(authServerMetadataUrl('https://as.example.com/'), 'https://as.example.com/.well-known/oauth-authorization-server')
  assert.equal(authServerMetadataUrl('https://as.example.com/tenant1'), 'https://as.example.com/.well-known/oauth-authorization-server/tenant1')
  ok('authServerMetadataUrl: well-known inserted before issuer path (RFC 8414 §3)')
}

// ---- metadata parse ----
{
  const md = parseAuthServerMetadata({
    issuer: 'https://as.example.com',
    authorization_endpoint: 'https://as.example.com/authorize',
    token_endpoint: 'https://as.example.com/token',
    registration_endpoint: 'https://as.example.com/register',
    code_challenge_methods_supported: ['S256', 123],
    scopes_supported: ['a', 'b'],
  })
  assert.equal(md.authorizationEndpoint, 'https://as.example.com/authorize')
  assert.equal(md.tokenEndpoint, 'https://as.example.com/token')
  assert.equal(md.registrationEndpoint, 'https://as.example.com/register')
  assert.deepEqual(md.codeChallengeMethodsSupported, ['S256']) // non-strings filtered
  assert.throws(() => parseAuthServerMetadata({ token_endpoint: 'x' }), /authorization_endpoint/)
  assert.throws(() => parseAuthServerMetadata({ authorization_endpoint: 'x' }), /token_endpoint/)
  assert.throws(() => parseAuthServerMetadata(null), /not an object/)
  ok('parseAuthServerMetadata: extracts endpoints, filters junk, throws on missing required')
}

// ---- metadata discovery with injected fetch ----
{
  const calls: string[] = []
  const fetchImpl = (async (url: string) => {
    calls.push(url)
    return { ok: true, status: 200, json: async () => ({ authorization_endpoint: 'https://a/az', token_endpoint: 'https://a/tk' }) }
  }) as unknown as typeof fetch
  const md = await discoverAuthServerMetadata('https://a/tenant', { fetchImpl, lookup: publicLookup })
  assert.equal(calls[0], 'https://a/.well-known/oauth-authorization-server/tenant')
  assert.equal(md.tokenEndpoint, 'https://a/tk')
  // non-2xx throws
  const badFetch = (async () => ({ ok: false, status: 404, json: async () => ({}) })) as unknown as typeof fetch
  await assert.rejects(discoverAuthServerMetadata('https://a', { fetchImpl: badFetch, lookup: publicLookup }), /HTTP 404/)
  ok('discoverAuthServerMetadata: builds well-known URL, parses, throws on non-2xx')
}

// ---- authorize URL ----
{
  const url = buildAuthorizeUrl({
    authorizationEndpoint: 'https://as.example.com/authorize?ui=dark',
    clientId: 'client-123',
    redirectUri: 'http://127.0.0.1:5599/callback',
    codeChallenge: 'CHALLENGE',
    state: 'state-xyz',
    scopes: ['mcp', 'offline_access'],
    resource: 'https://mcp.example.com/',
    extraParams: { prompt: 'consent', response_type: 'should-not-override' },
  })
  const u = new URL(url)
  assert.equal(u.searchParams.get('response_type'), 'code') // core param not overridden by extra
  assert.equal(u.searchParams.get('client_id'), 'client-123')
  assert.equal(u.searchParams.get('redirect_uri'), 'http://127.0.0.1:5599/callback')
  assert.equal(u.searchParams.get('code_challenge'), 'CHALLENGE')
  assert.equal(u.searchParams.get('code_challenge_method'), 'S256')
  assert.equal(u.searchParams.get('state'), 'state-xyz')
  assert.equal(u.searchParams.get('scope'), 'mcp offline_access')
  assert.equal(u.searchParams.get('resource'), 'https://mcp.example.com/')
  assert.equal(u.searchParams.get('ui'), 'dark') // preserved existing query
  assert.equal(u.searchParams.get('prompt'), 'consent') // extra merged
  assert.throws(() => buildAuthorizeUrl({ authorizationEndpoint: 'https://a', clientId: '', redirectUri: 'r', codeChallenge: 'c', state: 's' }), /missing clientId/)
  ok('buildAuthorizeUrl: PKCE S256 + resource + state, preserves query, extras never override core')
}

// ---- auth-code token request (public client, PKCE-only) ----
{
  const req = buildAuthCodeTokenRequest({
    tokenEndpoint: 'https://as.example.com/token',
    clientId: 'client-123',
    code: 'authcode',
    redirectUri: 'http://127.0.0.1:5599/callback',
    codeVerifier: 'VERIFIER',
    resource: 'https://mcp.example.com/',
  })
  assert.equal(req.method, 'POST')
  assert.equal(req.url, 'https://as.example.com/token')
  assert.equal(req.headers['content-type'], 'application/x-www-form-urlencoded')
  assert.equal(req.headers.authorization, undefined) // public client → no Basic
  const body = new URLSearchParams(req.body)
  assert.equal(body.get('grant_type'), 'authorization_code')
  assert.equal(body.get('code'), 'authcode')
  assert.equal(body.get('code_verifier'), 'VERIFIER')
  assert.equal(body.get('client_id'), 'client-123')
  assert.equal(body.get('resource'), 'https://mcp.example.com/')
  ok('buildAuthCodeTokenRequest: form body carries PKCE verifier + resource, no Basic for public client')
}

// ---- auth-code token request (confidential client → Basic auth) ----
{
  const req = buildAuthCodeTokenRequest({
    tokenEndpoint: 'https://as/token', clientId: 'id', code: 'c', redirectUri: 'r', codeVerifier: 'v', clientSecret: 'secret',
  })
  assert.ok(req.headers.authorization?.startsWith('Basic '))
  const decoded = Buffer.from(req.headers.authorization.slice(6), 'base64').toString('utf8')
  assert.equal(decoded, 'id:secret')
  assert.throws(() => buildAuthCodeTokenRequest({ tokenEndpoint: '', clientId: 'i', code: 'c', redirectUri: 'r', codeVerifier: 'v' }), /missing tokenEndpoint/)
  ok('buildAuthCodeTokenRequest: confidential client → HTTP Basic (RFC 6749 §2.3.1)')
}

// ---- refresh token request ----
{
  const req = buildRefreshTokenRequest({ tokenEndpoint: 'https://as/token', clientId: 'id', refreshToken: 'rt', scopes: ['mcp'] })
  const body = new URLSearchParams(req.body)
  assert.equal(body.get('grant_type'), 'refresh_token')
  assert.equal(body.get('refresh_token'), 'rt')
  assert.equal(body.get('scope'), 'mcp')
  assert.throws(() => buildRefreshTokenRequest({ tokenEndpoint: 'x', clientId: 'i', refreshToken: '' }), /missing refreshToken/)
  ok('buildRefreshTokenRequest: refresh_token grant (RFC 6749 §6)')
}

// ---- token response parse (success + expiry from injected clock) ----
{
  const now = 1_000_000
  const t = parseTokenResponse({ access_token: 'AT', token_type: 'Bearer', refresh_token: 'RT', expires_in: 3600, scope: 'mcp offline_access' }, now)
  assert.equal(t.accessToken, 'AT')
  assert.equal(t.refreshToken, 'RT')
  assert.equal(t.expiresAt, now + 3600 * 1000)
  assert.deepEqual(t.scopes, ['mcp', 'offline_access'])
  // expires_in as string is tolerated
  assert.equal(parseTokenResponse({ access_token: 'x', expires_in: '120' }, now).expiresAt, now + 120_000)
  // no expires_in → null expiry, never expired
  const noExp = parseTokenResponse({ access_token: 'x' }, now)
  assert.equal(noExp.expiresAt, null)
  assert.equal(isTokenExpired(noExp, now + 1e12), false)
  ok('parseTokenResponse: normalizes success, computes expiresAt from injected clock')
}

// ---- token response error (RFC 6749 §5.2) ----
{
  assert.throws(() => parseTokenResponse({ error: 'invalid_grant', error_description: 'bad code' }), /invalid_grant.*bad code/)
  assert.throws(() => parseTokenResponse({ token_type: 'Bearer' }), /missing access_token/)
  ok('parseTokenResponse: throws rich error for OAuth error response / missing token')
}

// ---- token expiry with skew ----
{
  const now = 1_000_000
  assert.equal(isTokenExpired({ expiresAt: now + 60_000 }, now), false) // 60s out, 30s skew → live
  assert.equal(isTokenExpired({ expiresAt: now + 20_000 }, now), true)  // 20s out, 30s skew → treat expired
  assert.equal(isTokenExpired({ expiresAt: now - 1 }, now), true)
  ok('isTokenExpired: honors clock skew window')
}

// ---- registry search URL ----
{
  assert.equal(buildRegistrySearchUrl({ registryUrl: 'https://registry.mcp.io' }), 'https://registry.mcp.io/v0/servers')
  const u = new URL(buildRegistrySearchUrl({ registryUrl: 'https://registry.mcp.io/', query: 'github', limit: 5, cursor: 'abc' }))
  assert.equal(u.pathname, '/v0/servers')
  assert.equal(u.searchParams.get('search'), 'github')
  assert.equal(u.searchParams.get('limit'), '5')
  assert.equal(u.searchParams.get('cursor'), 'abc')
  // sub-path base preserved
  assert.equal(new URL(buildRegistrySearchUrl({ registryUrl: 'https://host/mcp-registry' })).pathname, '/mcp-registry/v0/servers')
  ok('buildRegistrySearchUrl: /v0/servers with search/limit/cursor, preserves base sub-path')
}

// ---- registry server parse: remote (OAuth-capable) entry ----
{
  const d = parseRegistryServer({
    server: {
      id: 'srv-1',
      name: 'io.github.acme/remote-tools',
      description: 'Remote tools',
      version_detail: { version: '1.2.3' },
      remotes: [
        { transport: 'sse', url: 'https://mcp.acme.com/sse' },
        { transport: 'streamable-http', url: 'https://mcp.acme.com/mcp' },
      ],
    },
  })!
  assert.equal(d.id, 'srv-1')
  assert.equal(d.name, 'io.github.acme/remote-tools')
  assert.equal(d.version, '1.2.3')
  assert.equal(d.isRemote, true)
  assert.equal(d.remotes.length, 2)
  // preferredRemoteEndpoint favors streamable-http over sse
  assert.equal(preferredRemoteEndpoint(d)!.transport, 'streamable-http')
  assert.equal(preferredRemoteEndpoint(d)!.url, 'https://mcp.acme.com/mcp')
  ok('parseRegistryServer: remote entry → isRemote, preferredRemoteEndpoint prefers streamable-http')
}

// ---- registry server parse: local stdio package entry (field-name drift tolerated) ----
{
  const d = parseRegistryServer({
    name: 'local-fs',
    description: 'Filesystem',
    packages: [
      { registry_name: 'npm', identifier: '@modelcontextprotocol/server-filesystem', version: '0.5.0', runtime_hint: 'npx' },
      { registryType: 'pypi', name: 'mcp-fs', runtimeHint: 'uvx' }, // camelCase + `name` fallback
    ],
  })!
  assert.equal(d.isRemote, false)
  assert.equal(preferredRemoteEndpoint(d), null)
  assert.equal(d.packages.length, 2)
  assert.equal(d.packages[0].registryType, 'npm')
  assert.equal(d.packages[0].identifier, '@modelcontextprotocol/server-filesystem')
  assert.equal(d.packages[0].runtimeHint, 'npx')
  assert.equal(d.packages[1].registryType, 'pypi')
  assert.equal(d.packages[1].identifier, 'mcp-fs')
  // junk / nameless entries rejected
  assert.equal(parseRegistryServer({ description: 'no name' }), null)
  assert.equal(parseRegistryServer(null), null)
  ok('parseRegistryServer: stdio packages, tolerant of snake/camel field drift, rejects nameless')
}

// ---- registry search response + pagination cursor ----
{
  const r = parseRegistrySearchResponse({
    servers: [
      { server: { name: 'a', remotes: [{ transport: 'streamable-http', url: 'https://x/mcp' }] } },
      { description: 'junk-no-name' },
      { name: 'b', packages: [{ registry_name: 'npm', identifier: 'b-pkg' }] },
    ],
    metadata: { next_cursor: 'CURSOR2' },
  })
  assert.equal(r.servers.length, 2) // junk dropped
  assert.equal(r.servers[0].name, 'a')
  assert.equal(r.nextCursor, 'CURSOR2')
  // top-level next_cursor + `data` array fallback
  const r2 = parseRegistrySearchResponse({ data: [{ name: 'c' }], next_cursor: 'C3' })
  assert.equal(r2.servers[0].name, 'c')
  assert.equal(r2.nextCursor, 'C3')
  assert.deepEqual(parseRegistrySearchResponse(null), { servers: [], nextCursor: null })
  ok('parseRegistrySearchResponse: filters junk, reads cursor from metadata/top-level, data fallback')
}

// ---- searchMcpRegistry with injected fetch ----
{
  const calls: string[] = []
  const fetchImpl = (async (url: string) => {
    calls.push(url)
    return { ok: true, status: 200, json: async () => ({ servers: [{ name: 'srv', remotes: [{ transport: 'streamable-http', url: 'https://s/mcp' }] }], metadata: { next_cursor: 'N' } }) }
  }) as unknown as typeof fetch
  const res = await searchMcpRegistry({ registryUrl: 'https://reg', query: 'q', limit: 3 }, { fetchImpl, lookup: publicLookup })
  assert.ok(calls[0].startsWith('https://reg/v0/servers?'))
  assert.equal(res.servers[0].name, 'srv')
  assert.equal(res.nextCursor, 'N')
  const badFetch = (async () => ({ ok: false, status: 500, json: async () => ({}) })) as unknown as typeof fetch
  await assert.rejects(searchMcpRegistry({ registryUrl: 'https://reg' }, { fetchImpl: badFetch, lookup: publicLookup }), /HTTP 500/)
  ok('searchMcpRegistry: builds URL, parses, throws on non-2xx')
}

console.log(`\n${passed} checks passed.`)
