import assert from 'node:assert/strict'
import { isIP } from 'node:net'
import { checkServerIdentity } from 'node:tls'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { createHubFetch, HF_ORIGIN, resolveHubEndpoint } from './hub-network'
import { assertConnectorBaseUrlResolved } from '../security'
import { isBenchmarkIp, isGloballyRoutableIp } from '../security-addresses'
import { createPinnedLookup } from '../pinned-fetch'
import { decodeHubFailure } from '../../../shared/hub-failure'

const inside = ['198.18.0.78', '198.18.0.0', '198.19.255.255', '2001:2::1e',
  '2001:2:0:ffff:ffff:ffff:ffff:ffff', '2001:0002:0000:0000:0000:0000:0000:001e',
  '::ffff:198.18.0.78', '::ffff:c612:4e', '0:0:0:0:0:ffff:c612:004e']
const outside = ['198.17.255.255', '198.20.0.0', '2001:2:1::', '2001:1:ffff:ffff:ffff:ffff:ffff:ffff']
const privateAddresses = ['0.0.0.0', '10.0.0.1', '172.16.0.1', '192.168.1.1', '127.0.0.1',
  '169.254.169.254', '168.63.129.16', '100.100.100.200', '100.64.0.1', '::', '::1',
  'fe80::1', 'fd00::1', 'fd00:ec2::254', '::ffff:169.254.169.254', '0:0:0:0:0:ffff:a9fe:a9fe']
const answer = (address: string) => ({ address, family: isIP(address) })

async function main() {
  assert.equal(resolveHubEndpoint(), HF_ORIGIN)
  assert.equal(resolveHubEndpoint(' HTTPS://Hub.Example.Test:443/ '), 'https://hub.example.test')
  assert.equal(resolveHubEndpoint('https://fc-mirror.example.test'), 'https://fc-mirror.example.test')
  for (const endpoint of ['http://hub.example.test', 'https://198.18.0.78', 'https://[2001:2::1e]',
    'https://localhost', 'https://user:pass@hub.example.test', 'https://@hub.example.test',
    'https://hub.example.test/base', 'https://hub.example.test/?', 'https://hub.example.test/#',
    'https://hub.example.test\\base', 'https://bad_host.test', 'https://hub.exa\nmple.test']) {
    assert.throws(() => resolveHubEndpoint(endpoint), /TF_HUB_POLICY/)
  }
  for (const address of inside) {
    assert.equal(isBenchmarkIp(address), true, address)
    assert.equal(isGloballyRoutableIp(address), false, address)
    let calls = 0
    const fetchImpl = (async (input, init) => {
      calls++
      assert.equal(String(input), `${HF_ORIGIN}/api/models?search=x`)
      assert.equal(init?.credentials, 'omit')
      assert.equal(init?.redirect, 'manual')
      assert.equal(typeof (init as any).dispatcher?.dispatch, 'function', 'real pinned dispatcher is present')
      assert.equal(new Headers(init?.headers).has('authorization'), false)
      return new Response('[]')
    }) as typeof fetch
    const client = createHubFetch(HF_ORIGIN, { lookup: async (host) => {
      assert.equal(host, 'huggingface.co'); return [answer(address)]
    }, fetchImpl })
    await (await client(`${HF_ORIGIN}/api/models?search=x`, { redirect: 'manual' })).text()
    assert.equal(calls, 1)
    await assert.rejects(() => assertConnectorBaseUrlResolved(HF_ORIGIN, 'generic', {
      lookup: async () => [answer(address)],
    }), /non-public/)
  }
  for (const address of outside) assert.equal(isBenchmarkIp(address), false, address)
  assert.equal(isBenchmarkIp('2001:2::1%en0'), false)
  for (const address of privateAddresses) {
    for (const answers of [[answer(address)], [answer('198.18.0.78'), answer('2606:4700::1111'), answer(address)]]) {
      let calls = 0
      const client = createHubFetch(HF_ORIGIN, { lookup: async () => answers,
        fetchImpl: (async () => { calls++; return new Response('unexpected') }) as typeof fetch })
      await assert.rejects(() => client(`${HF_ORIGIN}/api/models`), (error: unknown) => {
        const failure = decodeHubFailure(error)
        assert.equal(failure.category, 'policy')
        assert.ok(failure.message.includes(address))
        assert.match(failure.message, /huggingface.co.*fake-ip-filter.*HF_ENDPOINT/)
        assert.doesNotMatch(failure.message, /reconnect|credentials/i)
        return true
      })
      assert.equal(calls, 0)
    }
  }
  let lookups = 0, transports = 0
  const scoped = createHubFetch('https://mirror.example.test', { lookup: async () => {
    lookups++; return [answer('198.18.0.78')]
  }, fetchImpl: (async () => { transports++; return new Response('[]') }) as typeof fetch })
  for (const url of ['https://huggingface.co/api/models', 'http://mirror.example.test/api/models',
    'https://mirror.example.test:444/api/models', 'https://user:pass@mirror.example.test/api/models']) {
    await assert.rejects(() => scoped(url), /TF_HUB_POLICY/)
  }
  await assert.rejects(() => scoped('https://mirror.example.test/api/models', { headers: { Cookie: 'placeholder' } }), /TF_HUB_POLICY/)
  await assert.rejects(() => scoped(new Request('https://mirror.example.test/api/models', { headers: { Authorization: 'placeholder' } })), /TF_HUB_POLICY/)
  await assert.rejects(() => scoped('https://mirror.example.test/api/models', { redirect: 'follow' }), /TF_HUB_POLICY/)
  assert.equal(lookups, 0); assert.equal(transports, 0)
  await (await scoped('https://mirror.example.test/api/models')).text()
  assert.equal(lookups, 1); assert.equal(transports, 1)

  // The socket lookup never asks DNS again and refuses a different hostname.
  const lookup = createPinnedLookup('huggingface.co', [answer('198.18.0.78'), answer('2001:2::1e')])
  const resolved = await new Promise((resolve, reject) => lookup('huggingface.co', { all: true }, (error, addresses) => error ? reject(error) : resolve(addresses)))
  assert.deepEqual(resolved, [answer('198.18.0.78'), answer('2001:2::1e')])
  await assert.rejects(() => new Promise((resolve, reject) => lookup('other.example.test', {}, (error, address) => error ? reject(error) : resolve(address))), /unexpected host/)
  const wrongName = checkServerIdentity('huggingface.co', { subjectaltname: 'DNS:other.example.test' } as any)
  assert.equal((wrongName as NodeJS.ErrnoException)?.code, 'ERR_TLS_CERT_ALTNAME_INVALID')
  const pinned = readFileSync(path.join(__dirname, '..', 'pinned-fetch.ts'), 'utf8')
  assert.match(pinned, /new Agent\(\{ connect: \{ lookup: createPinnedLookup\(expected, pinned\) \} \}\)/)
  assert.doesNotMatch(pinned, /rejectUnauthorized\s*:\s*false|checkServerIdentity\s*:/)
  for (const code of ['ERR_TLS_CERT_ALTNAME_INVALID', 'UNABLE_TO_VERIFY_LEAF_SIGNATURE']) {
    let calls = 0
    const client = createHubFetch(HF_ORIGIN, { lookup: async () => [answer('198.18.0.78')],
      fetchImpl: (async (input) => { calls++; assert.equal(new URL(String(input)).hostname, 'huggingface.co'); throw Object.assign(new Error(code), { code }) }) as typeof fetch })
    await assert.rejects(() => client(`${HF_ORIGIN}/api/models`), (error: unknown) => decodeHubFailure(error).category === 'transport')
    assert.equal(calls, 1, 'TLS errors never trigger an unpinned retry')
  }
  const unavailable = createHubFetch(HF_ORIGIN, { lookup: async () => { throw new Error('DNS unavailable') } })
  await assert.rejects(() => unavailable(`${HF_ORIGIN}/api/models`), (error: unknown) => decodeHubFailure(error).category === 'transport')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
