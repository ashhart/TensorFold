import assert from 'node:assert/strict'
import { EngineService, engineChildEnv } from './service'
import { engineLaunchEnv } from './engine-env'
import { preflightRepo, searchHub, type PreflightContext } from './hub'
import { HF_ORIGIN } from './hub-network'
import { makeHarness, waitFor } from './test-fixtures/harness'
import { decodeHubFailure } from '../../../shared/hub-failure'

const GIB = 1024 ** 3, SHA = 'e'.repeat(40), DRAFT_SHA = 'f'.repeat(40)
const ENDPOINT = 'https://mirror.example.test'
const NEXT_ENDPOINT = 'https://other.example.test'
const REPO = 'Org/Model', DRAFTER = 'Org/Drafter'

function mirror(endpoint: string, withDrafter = false) {
  const calls: URL[] = []
  const fetchImpl = (async (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
    calls.push(url)
    assert.equal(url.origin, endpoint)
    assert.equal(init?.redirect, 'manual')
    assert.equal(new Headers(init?.headers).has('authorization'), false)
    assert.equal(new Headers(init?.headers).has('cookie'), false)
    const json = (body: unknown, headers: Record<string, string> = {}) => new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json', ...headers } })
    if (url.pathname === '/api/models') return json([{ id: REPO }])
    if (url.pathname === `/api/models/${REPO}/revision/main`) return json({ sha: SHA, gated: false })
    if (url.pathname === `/api/models/${REPO}/tree/${SHA}`) return url.searchParams.has('cursor')
      ? json([{ type: 'file', size: 20 }])
      : json([{ type: 'file', size: 10 }], { link: `<${endpoint}/api/models/${REPO}/tree/${SHA}?cursor=2>; rel="next"` })
    if (url.pathname === `/${REPO}/resolve/${SHA}/config.json`) return new Response(null, {
      status: 307, headers: { location: `/api/resolve-cache/models/${REPO}/${SHA}/config.json` },
    })
    if (url.pathname === `/api/resolve-cache/models/${REPO}/${SHA}/config.json`) return json({ model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } })
    if (withDrafter && url.pathname === `/api/models/${DRAFTER}/revision/main`) return json({ sha: DRAFT_SHA, gated: false })
    if (withDrafter && url.pathname === `/api/models/${DRAFTER}/tree/${DRAFT_SHA}`) return json([{ type: 'file', size: 5 }])
    return new Response('missing', { status: 404 })
  }) as typeof fetch
  return { calls, fetchImpl }
}

async function main() {
  const env = { HOME: '/placeholder', PATH: '/bin', HF_ENDPOINT: `${ENDPOINT}/`, HF_TOKEN: 'placeholder', OPENAI_API_KEY: 'placeholder' }
  assert.deepEqual(engineChildEnv(env), { HOME: '/placeholder', PATH: '/bin', HF_ENDPOINT: ENDPOINT })
  const launch = engineLaunchEnv({ env })
  assert.equal(launch.endpoint, ENDPOINT)
  assert.equal(launch.env.HF_ENDPOINT, ENDPOINT)
  assert.equal(launch.env.HF_TOKEN, undefined)
  assert.equal(launch.env.OPENAI_API_KEY, undefined)
  env.HF_ENDPOINT = NEXT_ENDPOINT
  assert.equal(launch.env.HF_ENDPOINT, ENDPOINT)
  assert.equal(engineLaunchEnv({ env: {} }).endpoint, HF_ORIGIN)
  assert.throws(() => engineLaunchEnv({ env: { HF_ENDPOINT: `${ENDPOINT}/base` } }), /TF_HUB_POLICY/)

  const m = mirror(ENDPOINT, true)
  assert.equal((await searchHub('test', { fetch: m.fetchImpl, endpoint: ENDPOINT }))[0].repo, REPO)
  const ctx: PreflightContext = { endpoint: ENDPOINT, fetch: m.fetchImpl, families: { qwen3_5: ['mlx-q4g64'] },
    totalMemoryBytes: 32 * GIB, memoryLimitBytes: 24 * GIB, installed: new Set(),
    drafterFor: () => DRAFTER, cachedBytes: async () => 0, freeDiskBytes: async () => 100 * GIB }
  const pre = await preflightRepo(REPO, null, ctx)
  assert.equal(pre.verdict, 'ok')
  assert.equal(pre.totalBytes, 35)
  assert.equal(pre.resolvedRevision, SHA)
  assert.equal(pre.drafter?.resolvedRevision, DRAFT_SHA)
  for (const required of ['/api/models', `/api/models/${REPO}/revision/main`, `/api/models/${REPO}/tree/${SHA}`,
    `/${REPO}/resolve/${SHA}/config.json`, `/api/resolve-cache/models/${REPO}/${SHA}/config.json`,
    `/api/models/${DRAFTER}/revision/main`, `/api/models/${DRAFTER}/tree/${DRAFT_SHA}`]) {
    assert.ok(m.calls.some((url) => url.pathname === required), required)
  }
  assert.ok(m.calls.some((url) => url.searchParams.get('cursor') === '2'))

  const h = await makeHarness({ HF_ENDPOINT: `${ENDPOINT}/`, HF_TOKEN: 'placeholder', FAKE_PULL_FILES: '1', FAKE_PULL_SIZE: '64', FAKE_PULL_DELAY_MS: '1' })
  const network = mirror(ENDPOINT)
  const events: any[] = []
  const service = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root,
    childEnv: h.env, fetch: network.fetchImpl, totalMemoryBytes: 32 * GIB, pollMs: 20, emit: (e) => events.push(e) })
  const otherCalls: string[] = []
  const other = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root,
    childEnv: { ...h.env, HF_ENDPOINT: NEXT_ENDPOINT }, totalMemoryBytes: 32 * GIB,
    fetch: (async (input) => { otherCalls.push(String(input)); return new Response('unavailable', { status: 503 }) }) as typeof fetch,
    emit: () => {} })
  try {
    assert.equal((await service.preflight(REPO)).verdict, 'ok')
    h.env.HF_ENDPOINT = NEXT_ENDPOINT
    await service.search('model')
    assert.equal((await service.pull(REPO)).state, 'running')
    await waitFor(() => events.some((e) => e.kind === 'pull' && e.pull.state === 'done'))
    const records = (await h.records()).filter((row) => row.kind === 'pull')
    assert.equal(records.length, 1)
    assert.equal(records[0].endpoint, ENDPOINT, 'pull uses the endpoint that authorized its preflight')
    assert.equal(records[0].hasHubToken, false)
    assert.equal(records[0].revision, SHA)
    await assert.rejects(() => other.pull(REPO), (error: unknown) => decodeHubFailure(error).category === 'http')
    assert.ok(otherCalls.some((url) => url === `${NEXT_ENDPOINT}/api/models/${REPO}/revision/main`), 'another service cannot reuse the first origin verdict')
    assert.equal((await h.records()).filter((row) => row.kind === 'pull').length, 1)
    const injectedEnv = new EngineService({ binaryPath: '/missing', appHome: h.root, env: { HF_ENDPOINT: ENDPOINT }, fetch: network.fetchImpl, emit: () => {} })
    await injectedEnv.search('env-injection')
    injectedEnv.shutdown()
  } finally {
    service.shutdown(); other.shutdown(); await h.cleanup()
  }
  const offline = new EngineService({ binaryPath: '/missing', appHome: '/placeholder', env: { HF_HUB_OFFLINE: '1' },
    fetch: (async () => { throw new Error('offline must not fetch') }) as typeof fetch, emit: () => {} })
  await assert.rejects(() => offline.preflight(REPO), (error: unknown) => decodeHubFailure(error).category === 'offline')
  offline.shutdown()
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
