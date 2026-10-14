import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { HF_ORIGIN } from './hub'
import { makeHarness, seedModel, waitFor } from './test-fixtures/harness'

const SHA = 'd'.repeat(40)
const OTHER = 'e'.repeat(40)
const REPO = 'Org/Reviewed'
const PRIMARY = 'TensorFold/Qwen3.8-27B-MLX-4bit'
const DRAFTER = 'z-lab/Qwen3.8-27B-DFlash2'
const DRAFTER_SHA = 'f'.repeat(40)

function reviewedFetch(options: { invalidSha?: boolean; missingDrafter?: boolean } = {}): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0]) => {
    const url = String(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
    const route = url.slice(HF_ORIGIN.length).split('?')[0]
    let body: unknown
    const ownerRepo = /^(?:\/api\/models\/|\/)([^/]+\/[^/]+)\//.exec(route)?.[1]
    if ([REPO, PRIMARY, DRAFTER].includes(ownerRepo ?? '')) {
      const sha = ownerRepo === DRAFTER ? DRAFTER_SHA : SHA
      if (route === `/api/models/${ownerRepo}/revision/main`) body = options.invalidSha ? {} : { sha }
      if (route === `/api/models/${ownerRepo}/tree/${sha}`) body = [{ type: 'file', path: 'model.safetensors', size: 8192 }]
      if (route === `/${ownerRepo}/resolve/${sha}/config.json`) body = { model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } }
      if (ownerRepo === DRAFTER && options.missingDrafter) body = undefined
    }
    if (route === `/api/models/${REPO}/revision/unsupported`) body = { sha: OTHER }
    if ([`/api/models/${REPO}/tree/${SHA}`, `/api/models/${REPO}/tree/${OTHER}`].includes(route)) body = [{ type: 'file', path: 'model.safetensors', size: 8192 }]
    if (route === `/${REPO}/resolve/${SHA}/config.json`) body = { model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } }
    if (route === `/${REPO}/resolve/${OTHER}/config.json`) body = { model_type: 'gemma4_unified' }
    return body ? new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } }) : new Response('missing', { status: 404 })
  }) as typeof fetch
}

async function main() {
  const h = await makeHarness({ FAKE_PULL_FILES: '1', FAKE_PULL_SIZE: '8192' })
  const canonicalHub = await fs.realpath(h.hub)
  const service = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: reviewedFetch(), totalMemoryBytes: 32 * 1024 ** 3, pollMs: 20, emit: () => {} })
  const services = [service]
  try {
    await service.preflight(REPO)
    await service.pull(REPO)
    await waitFor(() => service.pullList().find((p) => p.repo === REPO && p.state === 'done'))
    const pull = (await h.records()).find((r) => r.kind === 'pull')
    assert.equal(pull.revision, SHA, 'the actual child must pull the immutable checkpoint whose format and memory were reviewed')
    await service.preflight(REPO) // Re-populate the main-review cache after the completed pull invalidates it.
    await assert.rejects(() => service.pull(REPO, 'unsupported'), /serves no gemma4_unified/, 'a cached main preflight cannot authorize another revision')

    const registry = path.join(h.root, 'engine', 'model-revisions.json')
    const persisted = JSON.parse(await fs.readFile(registry, 'utf8'))
    assert.equal(persisted.revisions[REPO], SHA)
    assert.equal((await fs.stat(registry)).mode & 0o777, 0o600)
    const snapshot = path.join(canonicalHub, 'models--Org--Reviewed', 'snapshots', SHA)
    const ref = path.join(h.hub, 'models--Org--Reviewed', 'refs', 'main')
    const fresh = () => {
      const next = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: reviewedFetch(), totalMemoryBytes: 32 * 1024 ** 3, pollMs: 20, emit: () => {} })
      services.push(next)
      return next
    }

    // The engine itself follows refs/main. Studio must use the reviewed snapshot after that ref
    // vanishes or another HF client points it at an unsupported checkpoint.
    await fs.unlink(ref)
    let restarted = fresh()
    assert.equal((await restarted.installed()).find((m) => m.repo === REPO)?.modelType, 'qwen3_5')
    assert.equal((await restarted.readiness(REPO)).ready, true)
    await restarted.load(REPO)
    let serve = (await h.records()).filter((r) => r.kind === 'serve').at(-1)
    assert.equal(serve.args[0], snapshot)
    assert.equal(serve.args[serve.args.indexOf('--name') + 1], REPO, 'logical model identity is preserved for the API')
    await assert.rejects(() => restarted.pull(REPO), /Unload/, 'a pull cannot change a currently serving selection')
    await restarted.unload()
    await seedModel(h.hub, REPO, { modelType: 'gemma4_unified' })
    const beforeRef = await fs.readFile(ref, 'utf8')
    restarted = fresh()
    assert.equal((await restarted.installed())[0].modelType, 'qwen3_5')
    await restarted.load(REPO)
    serve = (await h.records()).filter((r) => r.kind === 'serve').at(-1)
    assert.equal(serve.args[0], snapshot)
    assert.equal(await fs.readFile(ref, 'utf8'), beforeRef, 'info/discovery/serve never edit shared refs')
    await restarted.unload()
    assert.ok((await h.records()).some((r) => r.kind === 'info' && r.argument === snapshot))

    // A stale selection must fail closed instead of serving the other cached checkpoint.
    const hidden = `${snapshot}-held`
    await fs.rename(snapshot, hidden)
    assert.deepEqual(await fresh().installed(), [])
    assert.equal((await fresh().readiness(REPO)).ready, false)
    await assert.rejects(() => fresh().load(REPO), /selected checkpoint.*missing/)
    await fs.symlink(hidden, snapshot)
    await assert.rejects(() => fresh().load(REPO), /selected checkpoint.*missing/, 'snapshot directory symlinks cannot redirect exact-path authority')
    await fs.unlink(snapshot)
    await fs.rename(hidden, snapshot)

    const valid = await fs.readFile(registry, 'utf8')
    for (const corrupt of ['{', JSON.stringify({ version: 1, revisions: { [REPO]: '../escape' } }), JSON.stringify({ version: 1, revisions: { '../../outside': SHA } })]) {
      await fs.writeFile(registry, corrupt)
      await assert.rejects(() => fresh().load(REPO), /saved model revisions are invalid/)
      await assert.rejects(() => fresh().pull(REPO), /saved model revisions are invalid/)
      await assert.rejects(() => fresh().remove(REPO), /saved model revisions are invalid/)
      assert.equal(await fs.readFile(registry, 'utf8'), corrupt, 'corruption evidence is preserved')
      await fs.stat(snapshot)
    }
    await fs.writeFile(registry, valid)

    // Companion commits have the same immutable download and exact serve-path contract.
    await service.preflight(PRIMARY)
    await service.pull(PRIMARY)
    await waitFor(() => service.pullList().find((p) => p.repo === PRIMARY && p.state === 'done'))
    const primarySnapshot = path.join(canonicalHub, 'models--TensorFold--Qwen3.8-27B-MLX-4bit', 'snapshots', SHA)
    const drafterSnapshot = path.join(canonicalHub, 'models--z-lab--Qwen3.8-27B-DFlash2', 'snapshots', DRAFTER_SHA)
    const drafterRef = path.join(h.hub, 'models--z-lab--Qwen3.8-27B-DFlash2', 'refs', 'main')
    await fs.unlink(drafterRef)
    restarted = fresh()
    assert.equal((await restarted.installed()).find((m) => m.repo === PRIMARY)?.drafterInstalled, true)
    await restarted.load(PRIMARY)
    serve = (await h.records()).filter((r) => r.kind === 'serve').at(-1)
    assert.equal(serve.args[0], primarySnapshot)
    assert.equal(serve.args[serve.args.indexOf('--drafter') + 1], drafterSnapshot)
    assert.equal((await h.records()).find((r) => r.kind === 'pull' && r.repo === DRAFTER).revision, DRAFTER_SHA)
    await restarted.unload()
    await fs.rename(drafterSnapshot, `${drafterSnapshot}-held`)
    restarted = fresh()
    assert.equal((await restarted.readiness(PRIMARY)).ready, false)
    await assert.rejects(() => restarted.load(PRIMARY), /needs its drafter/)
    await fs.rename(`${drafterSnapshot}-held`, drafterSnapshot)

    // Removal clears only the selected model and its unshared companion, preserving other pins.
    await fresh().remove(PRIMARY)
    const afterRemove = JSON.parse(await fs.readFile(registry, 'utf8')).revisions
    assert.equal(afterRemove[REPO], SHA)
    assert.equal(afterRemove[PRIMARY], undefined)
    assert.equal(afterRemove[DRAFTER], undefined)
    await fresh().remove(REPO)
    assert.deepEqual(await fresh().installed(), [])

    // Missing registries retain ordinary pre-existing caches, with honest legacy CLI semantics.
    await fs.unlink(registry)
    await seedModel(h.hub, 'Org/Legacy')
    assert.equal((await fresh().installed())[0].repo, 'Org/Legacy')
    await fresh().info('Org/Legacy')
    assert.ok((await h.records()).some((r) => r.kind === 'info' && r.argument === 'Org/Legacy'))

    const noCommit = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: reviewedFetch({ invalidSha: true }), emit: () => {} })
    await assert.rejects(() => noCommit.pull(REPO), /immutable commit/)
    const missingDrafter = new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: reviewedFetch({ missingDrafter: true }), emit: () => {} })
    await assert.rejects(() => missingDrafter.pull(PRIMARY), /drafter.*immutable commit/)
  } finally {
    for (const service of services) service.shutdown()
    await h.cleanup()
  }
  console.log('engine reviewed model revision tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
