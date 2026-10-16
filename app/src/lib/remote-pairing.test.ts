// Run: jiti src/lib/remote-pairing.test.ts
//
// Paired-session bootstrap (VST-035): boot-mode decisions (desktop/paired/needs-pairing/
// demo), URL credential redemption + scrubbing, token verification against the host.
import assert from 'node:assert/strict'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('remote-pairing tests:')

// --- browser shims (installed BEFORE importing the module under test) ---
const storage = new Map<string, string>()
let href = 'http://192.168.1.20:8765/'
let replaced: string[] = []
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => { storage.set(k, v) },
    removeItem: (k: string) => { storage.delete(k) },
  },
})
Object.defineProperty(globalThis, 'window', {
  configurable: true,
  value: {
    get location() { return { href } },
    history: { replaceState: (_s: unknown, _t: string, url: string) => { replaced.push(url); href = url } },
    dispatchEvent: () => true,
  },
})
try {
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: 'test-agent' } })
} catch { /* node exposes a getter-only navigator; the module only slices userAgent */ }

const { preparePairedSession, pairedToken, clearPairedToken, redeemPairingCredential } = await import('./remote-pairing')

type FetchStep = { path: RegExp; status: number; body?: unknown }
function fakeFetch(steps: FetchStep[]): typeof fetch {
  const calls: string[] = []
  const impl = (async (url: RequestInfo | URL) => {
    const path = String(url)
    calls.push(path)
    const step = steps.find((s) => s.path.test(path))
    if (!step) return { ok: false, status: 404, json: async () => ({}) }
    return { ok: step.status >= 200 && step.status < 300, status: step.status, json: async () => step.body ?? {} }
  }) as unknown as typeof fetch
  ;(impl as unknown as { calls: string[] }).calls = calls
  return impl
}

{
  // Desktop: window.tensorfoldStudio wins before anything network.
  ;(globalThis.window as { vontra?: unknown }).vontra = {}
  assert.equal(await preparePairedSession(fakeFetch([])), 'desktop')
  delete (globalThis.window as { vontra?: unknown }).vontra
  ok('desktop mode: the preload bridge short-circuits')
}

{
  // Plain browser (vite demo): no /health → demo mock.
  storage.clear()
  assert.equal(await preparePairedSession(fakeFetch([])), 'demo')
  ok('demo mode: a non-host page keeps the browser demo')
}

{
  // Host-served page, no token → needs-pairing.
  storage.clear()
  const mode = await preparePairedSession(fakeFetch([{ path: /\/health$/, status: 200, body: { ok: true, listening: true } }]))
  assert.equal(mode, 'needs-pairing')
  ok('needs-pairing: host page without a token lands on the gate')
}

{
  clearPairedToken()
  href = 'http://127.0.0.1:8765/'
  const wrongMime = (async () => new Response('<html>fake host</html>', { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch
  assert.equal(await preparePairedSession(wrongMime), 'demo', 'non-JSON health cannot impersonate a TensorFold Studio host')
  const oversized = (async () => new Response('{}', {
    status: 200,
    headers: { 'content-type': 'application/json', 'content-length': String(65 * 1024) },
  })) as typeof fetch
  assert.equal(await preparePairedSession(oversized), 'demo', 'oversized health metadata is rejected')
  ok('host probe: MIME and bounded response checks fail closed')
}

{
  // Legacy ?pair= credential: redeemed, token stored, URL scrubbed, mode paired.
  storage.clear()
  replaced = []
  href = 'http://192.168.1.20:8765/?pair=cred-123'
  const fetchImpl = fakeFetch([
    { path: /\/v1\/auth\/pair$/, status: 200, body: { sessionId: 'dev-1', token: 'tok-dev-1', label: 'iPhone' } },
  ])
  const mode = await preparePairedSession(fetchImpl)
  assert.equal(mode, 'paired')
  assert.equal(pairedToken(), 'tok-dev-1', 'the device token is stored')
  assert.ok(replaced.length > 0 && !replaced[replaced.length - 1].includes('pair='), 'the one-time credential is scrubbed from the URL')
  ok('legacy ?pair= flow: redeem → store token → scrub URL → paired')
}

{
  // New links carry the one-time credential in the fragment, which is never part of an HTTP
  // request target. The bootstrap still decodes and scrubs it before continuing.
  storage.clear()
  replaced = []
  href = 'http://192.168.1.20:8765/#pair=cred%2Ffragment'
  let submitted = ''
  const fetchImpl = (async (url: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(String(url), '/v1/auth/pair')
    submitted = String(init?.body ?? '')
    return { ok: true, status: 200, json: async () => ({ token: 'tok-fragment', label: 'Browser' }) }
  }) as unknown as typeof fetch
  assert.equal(await preparePairedSession(fetchImpl), 'paired')
  assert.match(submitted, /cred\/fragment/, 'the fragment credential is decoded into the POST body')
  assert.ok(!href.includes('pair='), 'the credential fragment is removed from browser history')
  ok('#pair= flow: never sent in the page request, then redeemed + scrubbed')
}

{
  // Stored token verified against the host: 200 keeps it, 401 clears it → needs-pairing.
  storage.clear()
  href = 'http://192.168.1.20:8765/'
  storage.set('vontra.paired.token', 'tok-live')
  const live = await preparePairedSession(fakeFetch([{ path: /\/v1\/snapshot$/, status: 200, body: {} }]))
  assert.equal(live, 'paired')

  storage.set('vontra.paired.token', 'tok-revoked')
  const revoked = await preparePairedSession(fakeFetch([
    { path: /\/v1\/snapshot$/, status: 401 },
    { path: /\/health$/, status: 200, body: { ok: true, listening: true } },
  ]))
  assert.equal(revoked, 'needs-pairing', 'a revoked token lands back on the pairing gate')
  assert.equal(pairedToken(), null, 'the dead token is cleared')
  ok('stored token: verified live, revoked → cleared + gate')
}

{
  // Redeem failures map to actionable messages.
  storage.clear()
  const consumed = await redeemPairingCredential('cred-x', fakeFetch([{ path: /\/v1\/auth\/pair$/, status: 400, body: { error: 'already_consumed' } }]))
  assert.equal(consumed.ok, false)
  assert.match(consumed.error ?? '', /already used/)
  const expired = await redeemPairingCredential('cred-y', fakeFetch([{ path: /\/v1\/auth\/pair$/, status: 400, body: { error: 'expired' } }]))
  assert.match(expired.error ?? '', /expired/)
  clearPairedToken()
  ok('redeem errors: consumed/expired credentials explain themselves')
}

console.log(`  ${passed} passed`)
