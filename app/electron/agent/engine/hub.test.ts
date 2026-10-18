import assert from 'node:assert/strict'
import { HF_ORIGIN, parseSearchResults, preflightRepo, searchHub, type PreflightContext } from './hub'

const GIB = 1024 ** 3

type Route = { status?: number; body?: unknown; text?: string; headers?: Record<string, string> }

// Mocked Hugging Face: tests never reach the network.
function mockHub(routes: Record<string, Route>) {
  const calls: string[] = []
  const fetchImpl = (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    calls.push(url)
    assert.equal(init?.redirect, 'manual', 'hub requests never auto-follow redirects')
    const path = url.slice(HF_ORIGIN.length)
    const route = routes[path] ?? routes[path.split('?')[0]]
    if (!route) return new Response('not found', { status: 404 })
    const body = route.text ?? JSON.stringify(route.body)
    return new Response(body, { status: route.status ?? 200, headers: { 'content-type': route.text ? 'text/plain' : 'application/json', ...(route.headers ?? {}) } })
  }) as typeof fetch
  return { fetchImpl, calls }
}

const SHA = 'c'.repeat(40)

function repoRoutes(repo: string, opts: { modelType?: string; quant?: unknown; gated?: unknown; sizes?: number[]; dflash?: boolean } = {}): Record<string, Route> {
  const config = { model_type: opts.modelType ?? 'qwen3_5', ...(opts.quant === undefined ? { quantization: { bits: 4, group_size: 64 } } : opts.quant ? { quantization: opts.quant } : {}), ...(opts.dflash ? { dflash_config: {} } : {}) }
  const sizes = opts.sizes ?? [GIB, 512]
  return {
    [`/api/models/${repo}/revision/main`]: { body: { id: repo, sha: SHA, gated: opts.gated ?? false, config: { model_type: config.model_type } } },
    [`/api/models/${repo}/tree/${SHA}`]: { body: sizes.map((size, i) => ({ type: 'file', path: `f${i}`, size, ...(size > 1000 ? { lfs: { size, oid: 'x' } } : {}) })).concat([{ type: 'directory', path: 'sub', size: 0 } as any]) },
    [`/${repo}/resolve/${SHA}/config.json`]: { text: JSON.stringify(config) },
  }
}

function context(fetchImpl: typeof fetch, overrides: Partial<PreflightContext> = {}): PreflightContext {
  return {
    fetch: fetchImpl,
    families: { qwen3_5: ['mlx-q4g64'], nemotron_h: ['mlx-q4g64'] },
    familyTitles: { qwen3_5: 'Qwen3.5-2B or Qwen3.8-27B' },
    totalMemoryBytes: 32 * GIB,
    memoryLimitBytes: Math.floor(32 * GIB * 0.7),
    installed: new Set(),
    drafterFor: (repo) => (repo === 'TensorFold/Qwen3.8-27B-MLX-4bit' ? 'z-lab/Qwen3.8-27B-DFlash2' : null),
    cachedBytes: async () => 0,
    freeDiskBytes: async () => 500 * GIB,
    now: () => 1,
    ...overrides,
  }
}

async function main() {
  // Search: any repo comes back, with the fields the badges use.
  const results = parseSearchResults([
    { id: 'mlx-community/Qwen3.5-2B-MLX-4bit', downloads: 120, likes: 3, gated: false, config: { model_type: 'qwen3_5' }, pipeline_tag: 'text-generation', library_name: 'mlx', lastModified: '2026-09-01T00:00:00.000Z' },
    { id: 'meta-llama/Llama-9', gated: 'manual' },
    { modelId: 'legacy/shape' },
    { id: 'no-slash' },
  ])
  assert.deepEqual(results.map((r) => r.repo), ['mlx-community/Qwen3.5-2B-MLX-4bit', 'meta-llama/Llama-9', 'legacy/shape'])
  assert.equal(results[0].modelType, 'qwen3_5')
  assert.equal(results[1].gated, 'manual')
  assert.throws(() => parseSearchResults({ error: 'x' }), /unexpected shape/)

  const search = mockHub({ '/api/models': { body: [{ id: 'Org/A', downloads: 1 }] } })
  assert.deepEqual((await searchHub('qwen 2b', { fetch: search.fetchImpl })).map((r) => r.repo), ['Org/A'])
  const url = new URL(search.calls[0])
  assert.equal(url.origin, HF_ORIGIN)
  assert.equal(url.searchParams.get('search'), 'qwen 2b')
  assert.deepEqual(url.searchParams.getAll('expand').includes('config'), true)
  assert.deepEqual(await searchHub('   ', { fetch: search.fetchImpl }), [])
  await assert.rejects(() => searchHub('x', { fetch: search.fetchImpl, offline: true }), /HF_HUB_OFFLINE/)
  await assert.rejects(() => searchHub('x'.repeat(201), { fetch: search.fetchImpl }), /too long/)

  // Supported, right format, fits: ok, with the floor = weights + 2 GiB margin.
  const ok = mockHub(repoRoutes('mlx-community/Qwen3.5-2B-MLX-4bit'))
  const good = await preflightRepo('mlx-community/Qwen3.5-2B-MLX-4bit', null, context(ok.fetchImpl))
  assert.equal(good.verdict, 'ok', good.reasons.join('; '))
  assert.equal(good.totalBytes, GIB + 512)
  assert.equal(good.memoryFloorBytes, GIB + 512 + 2 * GIB)
  assert.equal(good.format, 'mlx-q4g64')
  assert.equal(good.family, 'Qwen3.5-2B or Qwen3.8-27B')
  assert.equal(good.fileCount, 2)

  // Real HF resolve URLs redirect to their same-origin config cache before serving JSON.
  const redirectedRoutes = repoRoutes('Org/Redirected')
  const redirectedConfig = redirectedRoutes[`/Org/Redirected/resolve/${SHA}/config.json`]
  redirectedRoutes[`/Org/Redirected/resolve/${SHA}/config.json`] = { status: 307, headers: { location: `/api/resolve-cache/models/Org/Redirected/${SHA}/config.json?etag=example` } }
  redirectedRoutes[`/api/resolve-cache/models/Org/Redirected/${SHA}/config.json`] = redirectedConfig
  const redirected = mockHub(redirectedRoutes)
  const resolved = await preflightRepo('Org/Redirected', null, context(redirected.fetchImpl))
  assert.equal(resolved.format, 'mlx-q4g64', 'the real resolve-cache redirect must preserve compatibility evidence')
  assert.equal(resolved.formatOk, true)
  for (const location of ['https://evil.example/config.json', 'http://huggingface.co/config.json', 'https://user:secret@huggingface.co/config.json', `/Org/Blocked/resolve/${SHA}/config.json`]) {
    const routes = repoRoutes('Org/Blocked')
    routes[`/Org/Blocked/resolve/${SHA}/config.json`] = { status: 307, headers: { location } }
    const blocked = mockHub(routes)
    await assert.rejects(() => preflightRepo('Org/Blocked', null, context(blocked.fetchImpl)), /TF_HUB_(POLICY|HTTP)/)
    assert.equal(blocked.calls.some((url) => !url.startsWith(HF_ORIGIN)), false, 'redirects never leave the fixed HTTPS Hugging Face origin')
    assert.ok(blocked.calls.filter((url) => url.endsWith('/config.json')).length <= 6, 'same-origin redirect loops are bounded')
  }

  // Installed already.
  const installed = await preflightRepo('mlx-community/Qwen3.5-2B-MLX-4bit', null, context(ok.fetchImpl, { installed: new Set(['mlx-community/Qwen3.5-2B-MLX-4bit']) }))
  assert.equal(installed.verdict, 'installed')

  // Unsupported family, and a drafter on its own.
  const gemma = mockHub(repoRoutes('google/gemma-9', { modelType: 'gemma4_unified' }))
  const unsupported = await preflightRepo('google/gemma-9', null, context(gemma.fetchImpl))
  assert.equal(unsupported.verdict, 'unsupported')
  assert.match(unsupported.reasons[0], /serves no gemma4_unified family/)
  const drafterOnly = mockHub(repoRoutes('z-lab/Qwen3.8-27B-DFlash2', { dflash: true }))
  assert.equal((await preflightRepo('z-lab/Qwen3.8-27B-DFlash2', null, context(drafterOnly.fetchImpl))).verdict, 'unsupported')

  // A served family in a format the engine does not read.
  const eightBit = mockHub(repoRoutes('mlx-community/Qwen3.5-2B-8bit', { quant: { bits: 8, group_size: 64 } }))
  const format = await preflightRepo('mlx-community/Qwen3.5-2B-8bit', null, context(eightBit.fetchImpl))
  assert.equal(format.verdict, 'format')
  assert.match(format.reasons.join(' '), /reads 4-bit, group 64 for this family; this checkpoint is 8-bit, group 64/)

  // Gated: pull sends no token, so it is refused up front and config.json is not fetched.
  const gated = mockHub(repoRoutes('Org/Gated', { gated: 'auto' }))
  const gatedResult = await preflightRepo('Org/Gated', null, context(gated.fetchImpl))
  assert.equal(gatedResult.verdict, 'gated')
  assert.equal(gated.calls.some((c) => c.endsWith('/config.json')), false)

  // Too big for this Mac (16 GB: limit 11.2 GB).
  const big = mockHub(repoRoutes('Org/Big', { sizes: [10 * GIB, 2 * GIB] }))
  const tooBig = await preflightRepo('Org/Big', null, context(big.fetchImpl, { totalMemoryBytes: 16 * GIB, memoryLimitBytes: Math.floor(16 * GIB * 0.7) }))
  assert.equal(tooBig.verdict, 'too-big')
  assert.match(tooBig.reasons.join(' '), /needs about 14\.0 GB of memory; the limit on this Mac is 11\.2 GB \(70% of 16\.0 GB\)/)

  // Not enough disk: what is already cached does not count against it.
  const disk = await preflightRepo('mlx-community/Qwen3.5-2B-MLX-4bit', null, context(ok.fetchImpl, { freeDiskBytes: async () => GIB }))
  assert.equal(disk.verdict, 'disk')
  const resumable = await preflightRepo('mlx-community/Qwen3.5-2B-MLX-4bit', null, context(ok.fetchImpl, { freeDiskBytes: async () => 2 * GIB, cachedBytes: async () => GIB }))
  assert.equal(resumable.verdict, 'ok')
  assert.equal(resumable.neededDiskBytes, 512)

  // The 27B's size and floor include its drafter.
  const routes = { ...repoRoutes('TensorFold/Qwen3.8-27B-MLX-4bit', { sizes: [15 * GIB] }), '/api/models/z-lab/Qwen3.8-27B-DFlash2/revision/main': { body: { sha: SHA } }, [`/api/models/z-lab/Qwen3.8-27B-DFlash2/tree/${SHA}`]: { body: [{ type: 'file', path: 'model.safetensors', size: GIB }] } }
  const withDrafter = await preflightRepo('TensorFold/Qwen3.8-27B-MLX-4bit', null, context(mockHub(routes).fetchImpl, { totalMemoryBytes: 64 * GIB, memoryLimitBytes: Math.floor(64 * GIB * 0.7) }))
  assert.equal(withDrafter.verdict, 'ok')
  assert.deepEqual(withDrafter.drafter, { repo: 'z-lab/Qwen3.8-27B-DFlash2', totalBytes: GIB, resolvedRevision: SHA })
  assert.equal(withDrafter.totalBytes, 16 * GIB)

  // Pinned revisions resolve through the revision API; tree pagination stays on huggingface.co.
  const pinned = mockHub({
    '/api/models/Org/Paged/revision/v1': { body: { sha: SHA, gated: false } },
    [`/api/models/Org/Paged/tree/${SHA}?recursive=true`]: { body: [{ type: 'file', path: 'a', size: 10 }], headers: { link: `<${HF_ORIGIN}/api/models/Org/Paged/tree/${SHA}?recursive=true&cursor=2>; rel="next"` } },
    [`/api/models/Org/Paged/tree/${SHA}?recursive=true&cursor=2`]: { body: [{ type: 'file', path: 'b', size: 32 }] },
    [`/Org/Paged/resolve/${SHA}/config.json`]: { text: '{"model_type":"nemotron_h","quantization":{"bits":4,"group_size":64}}' },
  })
  const paged = await preflightRepo('Org/Paged', 'v1', context(pinned.fetchImpl))
  assert.equal(paged.totalBytes, 42, 'both tree pages are summed')
  assert.equal(paged.revision, 'v1')
  const evilNext = mockHub({
    '/api/models/Org/Evil/revision/main': { body: { sha: SHA, gated: false } },
    [`/api/models/Org/Evil/tree/${SHA}`]: { body: [{ type: 'file', path: 'a', size: 10 }], headers: { link: '<https://evil.example/next>; rel="next"' } },
    [`/Org/Evil/resolve/${SHA}/config.json`]: { text: '{"model_type":"nemotron_h","quantization":{"bits":4,"group_size":64}}' },
  })
  await assert.rejects(() => preflightRepo('Org/Evil', null, context(evilNext.fetchImpl)), /TF_HUB_POLICY/)
  assert.equal(evilNext.calls.some((c) => c.includes('evil.example')), false, 'a cross-origin next link is refused')

  const missing = mockHub({})
  await assert.rejects(() => preflightRepo('Org/Missing', null, context(missing.fetchImpl)), /not found on Hugging Face, or private/)
  await assert.rejects(() => preflightRepo('../x', null, context(missing.fetchImpl)), /owner\/name/)
  await assert.rejects(() => preflightRepo('Org/X', 'a/b', context(missing.fetchImpl)), /branch, tag or commit/)

  console.log('engine hub tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
