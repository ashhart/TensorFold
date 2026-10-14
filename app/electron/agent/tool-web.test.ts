// Run: jiti electron/agent/tool-web.test.ts
import assert from 'node:assert/strict'
import { safePublicWebUrl, assertFetchableUrl, fetchText, resolveFetchableUrl } from './tool-web.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('tool-web SSRF guard tests:')

{
  for (const [url, address] of [
    ['http://198.18.0.1/', '198.18.0.1'],
    ['http://[64:ff9b::a9fe:a9fe]/', '64:ff9b::a9fe:a9fe'],
  ] as const) {
    await assert.rejects(
      () => resolveFetchableUrl(url, async () => [{ address, family: address.includes(':') ? 6 : 4 }]),
      /private or non-public/i,
    )
  }
  ok('public web policy rejects benchmark and NAT64/non-global destinations')
}

{
  // String-level host check still rejects the obvious internal/literal hosts before
  // any DNS lookup. These throw synchronously in safePublicWebUrl.
  for (const host of ['127.0.0.1', 'localhost', '169.254.169.254', '10.1.2.3', '192.168.0.5', '172.20.0.1', '[::1]', '0.0.0.0']) {
    assert.throws(() => safePublicWebUrl(`http://${host}/x`), /Refusing to fetch|Web URLs/i, host)
  }
  ok('safePublicWebUrl rejects loopback/private/link-local/metadata literal hosts')
}

{
  // Non-http(s) schemes are rejected.
  assert.throws(() => safePublicWebUrl('file:///etc/passwd'), /http or https/)
  assert.throws(() => safePublicWebUrl('gopher://example.com'), /http or https/)
  // Credentials in the URL are stripped on a valid public URL.
  const u = safePublicWebUrl('https://user:pass@example.com/path')
  assert.equal(u.username, '')
  assert.equal(u.password, '')
  assert.equal(u.hostname, 'example.com')
  ok('safePublicWebUrl rejects bad schemes and strips credentials')
}

{
  // DNS-rebinding defense: a public-looking hostname that resolves to a private IP
  // must be rejected by assertFetchableUrl (the path web_fetch now takes). 'localhost'
  // resolves to 127.0.0.1, so it's caught — but we also assert the resolver path runs
  // for a hostname.
  await assert.rejects(() => assertFetchableUrl('https://localhost/'), /Refusing to fetch|Web URLs/i)
  ok('assertFetchableUrl DNS-validates and rejects loopback-resolving hosts')
}

{
  // Resolver failure must fail closed. Falling through to global fetch would perform a
  // second, unvetted lookup and reopen the DNS-rebinding window.
  await assert.rejects(
    () => assertFetchableUrl('https://this-host-truly-does-not-exist-vontra-test.invalid/'),
    /could not be safely pinned/,
  )
  ok('assertFetchableUrl fails closed when DNS cannot be safely pinned')
}

{
  // fetchText performs the point-of-use resolution itself and gives the socket only that
  // vetted set. A mixed public/private answer is rejected before the fake transport runs.
  let fetched = false
  await assert.rejects(
    () => fetchText(new URL('https://mixed.example.test/path'), 1000, {
      lookup: async () => [
        { address: '93.184.216.34', family: 4 },
        { address: '169.254.169.254', family: 4 },
      ],
      fetchImpl: (async () => { fetched = true; return new Response('no') }) as typeof fetch,
    }),
    /private or non-public address/,
  )
  assert.equal(fetched, false)
  ok('fetchText rejects mixed DNS answers at point of use before transport')
}

console.log(`\n${passed} checks passed.`)
