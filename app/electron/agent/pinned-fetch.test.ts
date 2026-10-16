// Run: jiti electron/agent/pinned-fetch.test.ts
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import {
  createPinnedDispatcher,
  createPinnedFetch,
  createPinnedLookup,
  pinnedFetch,
  resolvePinnedHost,
  type PinnedAddress,
} from './pinned-fetch.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('pinned-fetch tests:')

function runLookup(
  lookup: ReturnType<typeof createPinnedLookup>,
  hostname: string,
  options: { all?: boolean; family?: number } = {},
): Promise<{ addresses?: readonly PinnedAddress[]; address?: string; family?: number }> {
  return new Promise((resolve, reject) => {
    lookup(hostname, options, (error, addressOrAddresses, family) => {
      if (error) { reject(error); return }
      if (Array.isArray(addressOrAddresses)) {
        resolve({ addresses: addressOrAddresses as readonly PinnedAddress[] })
      } else {
        resolve({ address: addressOrAddresses as string, family })
      }
    })
  })
}

{
  const lookup = createPinnedLookup('API.Example.test.', [
    { address: '192.0.2.20', family: 4 },
    { address: '2001:db8::20', family: 6 },
    { address: '192.0.2.20', family: 4 }, // duplicate is removed
  ])
  assert.deepEqual((await runLookup(lookup, 'api.example.test', { all: true })).addresses, [
    { address: '192.0.2.20', family: 4 },
    { address: '2001:db8::20', family: 6 },
  ])
  assert.deepEqual(await runLookup(lookup, 'api.example.test', { all: true, family: 6 }), {
    addresses: [{ address: '2001:db8::20', family: 6 }],
  })
  assert.deepEqual(await runLookup(lookup, 'api.example.test', { family: 4 }), {
    address: '192.0.2.20',
    family: 4,
  })
  await assert.rejects(runLookup(lookup, 'other.example.test', { all: true }), (error: any) => (
    error?.code === 'EHOSTUNREACH' && /unexpected host/i.test(error.message)
  ))
  await assert.rejects(runLookup(lookup, 'api.example.test', { all: true, family: 5 }), /No vetted IPv5/)
  ok('lookup pins IPv4/IPv6, filters requested family, de-duplicates, and fails closed for another host')
}

{
  assert.throws(() => createPinnedLookup('example.test', []), /without vetted addresses/)
  assert.throws(() => createPinnedLookup('example.test', [{ address: 'not-an-ip', family: 4 }]), /invalid IP/)
  assert.throws(() => createPinnedLookup('example.test', [{ address: '2001:db8::1', family: 4 }]), /it is IPv6/)
  assert.throws(() => createPinnedLookup('', [{ address: '192.0.2.1', family: 4 }]), /invalid hostname/)
  ok('malformed or empty pin sets are rejected before an Agent exists')
}

{
  const checked: string[] = []
  const target = await resolvePinnedHost('https://service.example.test/path', (address, family, url) => {
    checked.push(`${family}:${address}:${url.hostname}`)
    return address !== '198.51.100.99'
  }, {
    lookup: async () => [
      { address: '198.51.100.10', family: 4 },
      { address: '2001:db8::10', family: 6 },
    ],
  })
  assert.equal(target.hostname, 'service.example.test')
  assert.deepEqual(target.addresses, [
    { address: '198.51.100.10', family: 4 },
    { address: '2001:db8::10', family: 6 },
  ])
  assert.deepEqual(checked, [
    '4:198.51.100.10:service.example.test',
    '6:2001:db8::10:service.example.test',
  ])

  await assert.rejects(resolvePinnedHost('ftp://service.example.test/file', () => true), /only supports http and https/)
  await assert.rejects(resolvePinnedHost('https://service.example.test', () => false, {
    lookup: async () => [{ address: '169.254.169.254', family: 4 }],
  }), /rejected by endpoint policy/)
  await assert.rejects(resolvePinnedHost('https://service.example.test', () => true, {
    lookup: async () => { throw new Error('resolver unavailable') },
  }), /could not be safely pinned/)
  ok('resolvePinnedHost vets every answer and fails closed on policy, scheme, and resolver errors')
}

{
  const target = await resolvePinnedHost('https://api.example.test/v1', () => true, {
    lookup: async () => [{ address: '192.0.2.25', family: 4 }],
  })
  let capturedUrl = ''
  let capturedInit: any
  const fakeFetch = (async (input: any, init?: any) => {
    capturedUrl = String(input)
    capturedInit = init
    return new Response('ok')
  }) as typeof fetch

  const response = await pinnedFetch(target, {}, fakeFetch)
  assert.equal(await response.text(), 'ok')
  assert.equal(capturedUrl, 'https://api.example.test/v1')
  assert.equal(capturedInit.redirect, 'error', 'automatic redirects are denied by default')
  assert.equal(typeof capturedInit.dispatcher?.dispatch, 'function', 'an actual undici Agent is installed')

  const fetchImpl = createPinnedFetch(target, fakeFetch)
  await fetchImpl('https://api.example.test/v2', { redirect: 'manual' })
  assert.equal(capturedUrl, 'https://api.example.test/v2')
  assert.equal(capturedInit.redirect, 'manual')
  const request = new Request('https://api.example.test/v3', { method: 'POST', body: 'payload' })
  let sawOriginalRequest = false
  const requestAwareFetch = (async (input: any) => {
    sawOriginalRequest = input === request && input.method === 'POST'
    return new Response('ok')
  }) as typeof fetch
  await createPinnedFetch(target, requestAwareFetch)(request)
  assert.equal(sawOriginalRequest, true, 'fetch adapter preserves Request method/body/headers')
  await assert.rejects(fetchImpl('https://evil.example.test/steal'), /pinned to origin https:\/\/api\.example\.test/)
  await assert.rejects(fetchImpl('https://api.example.test:444/steal'), /pinned to origin/, 'a different port is a different credential boundary')
  await assert.rejects(fetchImpl('http://api.example.test/steal'), /pinned to origin/, 'a protocol downgrade is a different credential boundary')
  await assert.rejects(fetchImpl('https://api.example.test/redirect', { redirect: 'follow' }), /cannot auto-follow redirects/)
  ok('pinnedFetch installs its dispatcher, defaults redirect:error, and its fetch adapter cannot change origin')
}

{
  // End-to-end rebinding regression. The synthetic hostname does not exist in system DNS.
  // Our safety resolver returns the local test server the first time, but would return the
  // cloud metadata address on a second lookup. A vulnerable validate-then-fetch flow would
  // call it twice. The pinned Agent calls it once and connects using that vetted first answer.
  const server = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'text/plain' })
    response.end(`pinned:${request.headers.host}`)
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const port = (server.address() as AddressInfo).port
  let resolverCalls = 0
  const resolver = async () => {
    resolverCalls++
    return resolverCalls === 1
      ? [{ address: '127.0.0.1', family: 4 }]
      : [{ address: '169.254.169.254', family: 4 }]
  }

  try {
    const target = await resolvePinnedHost(`http://rebind.invalid:${port}/probe`, (address) => {
      if (address !== '127.0.0.1') throw new Error(`blocked unsafe rebound address ${address}`)
    }, { lookup: resolver })
    const response = await pinnedFetch(target)
    assert.equal(await response.text(), `pinned:rebind.invalid:${port}`)
    assert.equal(resolverCalls, 1, 'connect used the vetted address instead of resolving again')
    ok('DNS rebinding TOCTOU: real socket connects only to the one vetted resolution')
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
  }
}

{
  const lease = createPinnedDispatcher('dual-stack.example.test', [
    { address: '192.0.2.30', family: 4 },
    { address: '2001:db8::30', family: 6 },
  ])
  assert.equal(typeof lease.dispatcher.dispatch, 'function')
  assert.deepEqual(lease.addresses, [
    { address: '192.0.2.30', family: 4 },
    { address: '2001:db8::30', family: 6 },
  ])
  const firstClose = lease.close()
  assert.equal(lease.close(), firstClose, 'close is idempotent')
  await firstClose
  ok('dispatcher supports dual-stack pins and deterministic idempotent cleanup')
}

console.log(`pinned-fetch: ${passed} checks passed`)
