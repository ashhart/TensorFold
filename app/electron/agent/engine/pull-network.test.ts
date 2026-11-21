import assert from 'node:assert/strict'
import { engineChildEnv, engineLaunchEnv } from './engine-env'
import { PROXY_ENV_KEYS, pullEnvironment, pullFailure, redactPullDiagnostic, setPullProxyResolver } from './pull-network'

let passed = 0
async function check(name: string, run: () => unknown | Promise<unknown>) {
  await run()
  passed++
  console.log(`PASS ${name}`)
}
const proxy = 'http://proxy-user:needle%3Apass@example.test:8080/private?token=hidden'
const proxies = Object.fromEntries(PROXY_ENV_KEYS.map((key) => [key, key.toLowerCase() === 'no_proxy' ? 'localhost,.example.test' : proxy]))

await check('all eight proxy values survive the production allowlist and pinned launch environment', () => {
  const base = { ...proxies, OPENAI_API_KEY: 'provider-secret', NODE_OPTIONS: '--inspect', HF_TOKEN: 'hub-secret' }
  const child = engineChildEnv(base)
  const launch = engineLaunchEnv({ env: base }).env
  for (const key of PROXY_ENV_KEYS) { assert.equal(child[key], proxies[key]); assert.equal(launch[key], proxies[key]) }
  for (const key of ['OPENAI_API_KEY', 'NODE_OPTIONS', 'HF_TOKEN']) assert.equal(child[key], undefined)
})
await check('each explicit proxy or bypass variable, including an empty one, wins over system settings', async () => {
  for (const key of PROXY_ENV_KEYS) {
    for (const value of ['', proxies[key]]) {
      let called = false
      const env = { [key]: value }
      assert.deepEqual(await pullEnvironment(env, 'darwin', async () => { called = true; return 'PROXY ignored.example.test:80' }), env)
      assert.equal(called, false)
    }
  }
})
await check('other platforms retain their environment without consulting Electron', async () => {
  let called = false
  assert.deepEqual(await pullEnvironment({}, 'linux', async () => { called = true; return 'PROXY ignored.example.test:80' }), {})
  assert.equal(called, false)
})
await check('macOS resolves the pinned Hub origin and forwards the preferred system proxy', async () => {
  const source = { HF_ENDPOINT: 'https://hub.example.test' }
  const child = await pullEnvironment(source, 'darwin', async (url) => { assert.equal(url, source.HF_ENDPOINT); return 'PROXY example.test:8080; DIRECT' })
  for (const key of ['HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY']) assert.equal(child[key], 'http://example.test:8080')
  assert.deepEqual(source, { HF_ENDPOINT: 'https://hub.example.test' }, 'the caller environment is not mutated')
})
await check('system proxy parsing accepts default ports, IPv6 and SOCKS routes', async () => {
  for (const [route, expected] of [
    ['PROXY example.test:80', 'http://example.test:80'],
    ['HTTPS example.test:443', 'https://example.test:443'],
    ['PROXY [::1]:8080', 'http://[::1]:8080'],
    ['SOCKS5 example.test:1080', 'socks5://example.test:1080'],
    ['SOCKS example.test:1080', 'socks4://example.test:1080'],
  ]) assert.equal((await pullEnvironment({}, 'darwin', async () => route)).HTTPS_PROXY, expected)
})
await check('DIRECT in the first position respects the system bypass decision', async () => {
  assert.deepEqual(await pullEnvironment({}, 'darwin', async () => 'DIRECT; PROXY ignored.example.test:80'), {})
})
await check('unsupported or malformed proxy routes fail without silently bypassing the proxy', async () => {
  for (const route of ['UNSUPPORTED example.test:80; DIRECT', 'PROXY example.test', 'PROXY example.test:8080/private']) {
    await assert.rejects(() => pullEnvironment({}, 'darwin', async () => route), /system proxy.*retry/i)
  }
})
await check('system resolver errors never disclose authenticated addresses', async () => {
  await assert.rejects(() => pullEnvironment({}, 'darwin', async () => { throw new Error(proxy) }), (error: unknown) => {
    assert.ok(error instanceof Error)
    assert.match(error.message, /system proxy/)
    assert.doesNotMatch(error.message, /proxy-user|needle|private|hidden/)
    return true
  })
})
await check('a hung system proxy lookup is bounded and offers retry', async () => {
  await assert.rejects(() => pullEnvironment({}, 'darwin', () => new Promise<string>(() => {}), 10), /timed out.*retry/)
})
await check('desktop resolver injection stays at the host boundary', async () => {
  setPullProxyResolver(async () => 'PROXY example.test:3128')
  assert.equal((await pullEnvironment({}, 'darwin')).HTTPS_PROXY, 'http://example.test:3128')
})
await check('proxy diagnostics show only scheme, host and port and scrub encoded and decoded credentials', () => {
  const env = { HTTPS_PROXY: proxy }
  const safe = redactPullDiagnostic(`Proxy ${proxy}; proxy-user needle%3Apass needle:pass`, env)
  assert.match(safe, /http:\/\/example\.test:8080/)
  assert.doesNotMatch(safe, /proxy-user|needle|private|hidden|token=/)
  assert.equal(redactPullDiagnostic('https://user:password@example.test/secret', {}), 'https://example.test:443')
  assert.doesNotMatch(redactPullDiagnostic('secret-pass', { ALL_PROXY: 'user:secret-pass@example.test:80' }), /secret-pass/)
})
await check('connection, proxy and timeout errors name the Hub and offer retry without credentials', () => {
  for (const diagnostic of ['ConnectionRefused', `proxy refused ${proxy}`, 'TLS certificate failed', 'request timed out']) {
    const message = pullFailure(diagnostic, { HTTPS_PROXY: proxy })
    assert.match(message, /Hugging Face Hub/)
    assert.match(message, /proxy \(http:\/\/example\.test:8080\)/)
    assert.match(message, /retry/)
    assert.doesNotMatch(message, /proxy-user|needle|private|hidden/)
  }
})
await check('the inactivity diagnostic explains the 60-second bound and likely connection cause', () => {
  const message = pullFailure('', {}, true)
  assert.match(message, /timed out.*60 seconds.*Hugging Face Hub may be unreachable.*retry/)
})
await check('unrelated engine errors retain their exact Unicode diagnostic', () => {
  assert.equal(pullFailure('error: checkpoint café is incomplete', {}), 'error: checkpoint café is incomplete')
})
console.log(`${passed}/14 pull network checks passed`)
