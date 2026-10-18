import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { EngineService } from './service'
import { LEARNING_MODEL, LearningCopies } from './learning-copy'
import { makeHarness, seedModel, waitFor } from './test-fixtures/harness'

const GIB = 1024 ** 3
function service(h: Awaited<ReturnType<typeof makeHarness>>) {
  return new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 64 * GIB, pollMs: 20, emit: () => {} })
}

async function main() {
  // The fake engine uses no GPU. Exercise both admission branches on any CI runner.
  const platform = os.platform
  const architecture = os.arch
  os.platform = () => 'darwin'
  os.arch = () => 'arm64'
  try {
  const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1', FAKE_ENGINE_CORTHEON: '1', FAKE_LEARN_DELAY_MS: '150' })
  await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
  const svc = service(h)
  try {
    const cache = path.join(h.hub, 'models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')
    const source = path.join(cache, 'snapshots', 'a'.repeat(40))
    const original = await fs.readFile(path.join(source, 'model.safetensors'))
    assert.equal((await svc.learningSnapshot()).supported, true, 'bundled slide and graph flags must be recognized')
    await svc.load(LEARNING_MODEL)
    const originalConnection = await svc.connection()
    const reply = await svc.supervisor.acquire(LEARNING_MODEL, { start: false })
    await assert.rejects(svc.setCortheon(true), /temporarily disabled in this beta/)
    assert.equal((await svc.status()).cortheon?.enabled, false, 'a busy refusal cannot persist Cortheon')
    await assert.rejects(svc.setLearning(true, LEARNING_MODEL), /active TensorFold requests/)
    assert.equal((await svc.learningSnapshot()).enabled, false, 'a busy refusal cannot persist learning')
    reply.release()
    const enabled = await svc.setLearning(true, LEARNING_MODEL)
    assert.equal(enabled.enabled, true)
    assert.equal(enabled.active, true)
    assert.equal(enabled.privateCopy, true)
    assert.notEqual(svc.supervisor.live()?.apiKey, originalConnection?.apiKey, 'enabling learning revokes the previously shared native API key')
    await assert.rejects(svc.connection(), /Local API sharing is paused/, 'the GUI must never expose a native learner key to external clients')
    const runs = (await h.records()).filter((record) => record.kind === 'serve')
    assert.equal(runs.length, 2)
    assert.ok(runs[1].args.includes('--slide'))
    assert.ok(runs[1].args.includes('--slide-graph'))
    const privateModel = runs[1].args[0]
    assert.notEqual(privateModel, source)
    assert.equal((await fs.lstat(path.join(privateModel, 'model.safetensors'))).nlink, 1)
    assert.equal((await fs.lstat(path.join(privateModel, 'model.safetensors'))).isSymbolicLink(), false)
    await fs.writeFile(path.join(privateModel, 'model.safetensors'), 'private change')
    assert.deepEqual(await fs.readFile(path.join(source, 'model.safetensors')), original)

    const job = await svc.teach('The synthetic QA beacon is amber.')
    assert.equal(job.job?.state, 'running')
    await assert.rejects(svc.unload(), /active TensorFold requests/)
    await assert.rejects(svc.setCortheon(true), /temporarily disabled in this beta/)
    await assert.rejects(svc.setLearning(false), /current lesson/)
    const during = await svc.supervisor.acquire(LEARNING_MODEL, { start: false })
    during.release() // normal same-model chat stays available while the native learner yields.
    await waitFor(async () => (await svc.learningSnapshot()).job?.state === 'done', 6000, 'saved lesson')
    const done = await svc.learningSnapshot()
    assert.equal(done.job?.learned, 1)
    assert.equal(done.graph?.facts[0].text, 'The synthetic QA beacon is amber.')
    await assert.rejects(svc.teach('The synthetic QA beacon is amber.'), /already submitted/)
    assert.deepEqual(await fs.readFile(path.join(source, 'model.safetensors')), original)

    await svc.teach('The synthetic QA kite is violet.')
    await svc.cancelTeach()
    await waitFor(async () => (await svc.learningSnapshot()).job?.state === 'cancelled', 6000, 'safe cancellation')
    await assert.rejects(svc.teach('The synthetic QA beacon is amber.'), /already submitted/, 'older lessons remain deduplicated after another lesson')
    assert.equal(svc.supervisor.activeRequestCount, 0)
    await svc.setLearning(false)
    assert.ok(await svc.connection(), 'ordinary local API sharing resumes with teaching disabled')
    const offRun = (await h.records()).filter((record) => record.kind === 'serve').at(-1)
    assert.equal(offRun.args[0], privateModel, 'turning teaching off preserves the learned model')
    assert.ok(!offRun.args.includes('--slide'))
    await svc.unload()
    const again = service(h)
    await again.load(LEARNING_MODEL)
    assert.equal((await again.learningSnapshot()).graph?.facts.length, 2, 'the graph survives app and engine restarts')
    const restored = (await h.records()).filter((record) => record.kind === 'serve').at(-1)
    assert.equal(restored.args[0], privateModel)
    await again.setLearning(true, LEARNING_MODEL)
    await assert.rejects(again.teach('The synthetic QA beacon is amber.'), /already submitted/, 'lesson idempotency survives app and engine restarts')
    await again.unload()

    const copies = new LearningCopies(h.root)
    await fs.link(path.join(privateModel, 'config.json'), path.join(h.root, 'shared-config'))
    await assert.rejects(copies.existing(LEARNING_MODEL), /linked, shared/)
    await fs.rm(path.join(h.root, 'shared-config'))
  } finally { svc.shutdown(); await svc.unload().catch(() => {}); await h.cleanup() }

  {
    const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1', FAKE_LEARN_DELAY_MS: '20', FAKE_SAVE_DELAY_MS: '250' })
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      const cache = path.join(h.hub, 'models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')
      const original = await fs.readFile(path.join(cache, 'blobs', 'weights'))
      await svc.setLearning(true, LEARNING_MODEL)
      await svc.load(LEARNING_MODEL)
      await svc.teach('The synthetic privacy test lantern is cyan.')
      await waitFor(async () => (await svc.learningSnapshot()).graph?.model?.save?.state === 'saving', 3000, 'save before erase')
      let began = false
      const beginning = svc.beginLearningPurge().then(() => { began = true })
      await assert.rejects(svc.load(LEARNING_MODEL), /learned data is erased/, 'purge admission closes synchronously')
      await assert.rejects(svc.teach('Another synthetic fact.'), /learned data is erased/)
      await assert.rejects(svc.setLearning(false), /learned data is erased/)
      await new Promise((resolve) => setTimeout(resolve, 20))
      assert.equal(began, false, 'erasure must wait for a native checkpoint save')
      await beginning
      const erased = await svc.purgeLearningData()
      assert.ok(erased.files >= 4, 'the receipt includes private checkpoint files, graph and job journal')
      assert.equal(svc.supervisor.status().state, 'stopped')
      assert.equal(await fs.stat(path.join(h.root, 'engine', 'learning')).then(() => true, () => false), false)
      assert.equal(await fs.stat(path.join(h.root, 'engine', 'learning-job.json')).then(() => true, () => false), false)
      assert.deepEqual(await fs.readFile(path.join(cache, 'blobs', 'weights')), original, 'erasure preserves the original HF model')
      svc.endLearningPurge()
      svc.endLearningPurge() // Release is idempotent, including outer finally paths.
      const empty = await svc.learningSnapshot()
      assert.equal(empty.enabled, false)
      assert.equal(empty.model, null)
      assert.equal(empty.job, null)
      assert.equal(empty.graph, null)
      await svc.load(LEARNING_MODEL)
      assert.ok(await svc.connection(), 'ordinary inference remains available after the erased private copy is removed')
      await svc.setLearning(true, LEARNING_MODEL)
      await svc.teach('The synthetic privacy test lantern is cyan.')
      await waitFor(async () => (await svc.learningSnapshot()).job?.state === 'done', 3000, 'fresh lesson after erasure')
    } finally { svc.endLearningPurge(); await svc.unload(); await h.cleanup() }
  }
  {
    const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1', FAKE_LEARN_DELAY_MS: '20', FAKE_SAVE_DELAY_MS: '250' })
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      await svc.setLearning(true, LEARNING_MODEL)
      await svc.load(LEARNING_MODEL)
      await svc.teach('The synthetic QA clock is silver.')
      await waitFor(async () => (await svc.learningSnapshot()).graph?.model?.save?.state === 'saving', 3000, 'checkpoint save begins')
      assert.equal(svc.isLearningBusy(), true)
      let settled = false
      const quitting = svc.prepareToQuit().then(() => { settled = true })
      await new Promise((resolve) => setTimeout(resolve, 30))
      assert.equal(settled, false, 'graceful quit must wait for the native checkpoint write')
      assert.ok(svc.supervisor.live(), 'the saving engine remains alive without exposing its API key')
      await assert.rejects(svc.teach('Another fact.'), /before quitting/)
      await quitting
      assert.equal(svc.isLearningBusy(), false)
      assert.equal(svc.supervisor.activeRequestCount, 0)
      assert.equal((await svc.learningSnapshot()).graph?.model?.save?.state, 'saved')
    } finally { await svc.unload(); await h.cleanup() }
  }
  {
    const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1', FAKE_ENGINE_CORTHEON: '1', FAKE_CORTHEON_FAIL: '1', FAKE_SLIDE_FAIL: '1' })
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      await svc.load(LEARNING_MODEL)
      await assert.rejects(svc.setCortheon(true), /temporarily disabled in this beta/)
      assert.equal((await svc.status()).cortheon?.enabled, false)
      assert.equal((await svc.status()).state, 'ready', 'refused research leaves the previous server running')
      await assert.rejects(svc.setLearning(true, LEARNING_MODEL), /requested mode failed/)
      assert.equal((await svc.learningSnapshot()).enabled, false)
      assert.equal((await svc.status()).state, 'ready')
      const runs = (await h.records()).filter((r) => r.kind === 'serve')
      assert.equal(runs.at(-1).args.includes('--slide'), false)
    } finally { await svc.unload(); await h.cleanup() }
  }
  {
    const h = await makeHarness({ FAKE_ENGINE_CORTHEON: '1' })
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      await svc.load(LEARNING_MODEL)
      const changes = await Promise.allSettled([svc.setCortheon(true), svc.setCortheon(false)])
      assert.equal(changes[0].status, 'rejected')
      assert.equal(changes[1].status, 'fulfilled')
      assert.equal((await svc.status()).cortheon?.enabled, false, 'a refused enable cannot race a permitted disable')
      assert.equal((await svc.status()).cortheon?.active, false)
    } finally { await svc.unload(); await h.cleanup() }
  }
  {
    const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1' })
    const svc = service(h)
    try {
      os.platform = () => 'linux'
      assert.equal((await svc.learningSnapshot()).supported, false)
      await assert.rejects(svc.setLearning(true, LEARNING_MODEL), /unavailable/, 'unqualified hardware never receives learning flags')
    } finally { os.platform = () => 'darwin'; await h.cleanup() }
  }
  {
    const h = await makeHarness({ FAKE_ENGINE_LEARNING: '1', FAKE_INFO_EXTRA_BYTES: String(100 * GIB) })
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      await assert.rejects(svc.setLearning(true, LEARNING_MODEL), /enough model memory/)
      assert.equal(await fs.stat(path.join(h.root, 'engine', 'learning')).then(() => true, () => false), false, 'insufficient RAM refuses before making a private model copy')
    } finally { await h.cleanup() }
  }
  {
    const h = await makeHarness()
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    const svc = service(h)
    try {
      await assert.rejects(svc.setLearning(true, LEARNING_MODEL), /unavailable/)
      assert.equal((await svc.learningSnapshot()).enabled, false)
    } finally { await h.cleanup() }
  }
  {
    const h = await makeHarness()
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    try {
      const cache = path.join(h.hub, 'models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')
      const source = await fs.realpath(path.join(cache, 'snapshots', 'a'.repeat(40)))
      const copies = new LearningCopies(h.root)
      const controller = new AbortController()
      await assert.rejects(copies.prepare(LEARNING_MODEL, source, cache, (copied) => { if (copied > 0) controller.abort(new Error('synthetic cancellation')) }, controller.signal), /synthetic cancellation/)
      assert.deepEqual(await fs.readdir(copies.root), [], 'cancelled preparation cannot leave a partial usable checkpoint')
      const stats = await fs.statfs(h.root)
      await fs.truncate(path.join(cache, 'blobs', 'weights'), Number(stats.bavail) * Number(stats.bsize) + 2 * GIB)
      await assert.rejects(copies.prepare(LEARNING_MODEL, source, cache, () => {}), /free disk space/)
      assert.deepEqual(await fs.readdir(copies.root), [], 'disk refusal never publishes a partial model')
    } finally { await h.cleanup() }
  }
  {
    const h = await makeHarness()
    await seedModel(h.hub, LEARNING_MODEL, { modelType: 'nemotron_h' })
    try {
      const cache = path.join(h.hub, 'models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')
      const source = path.join(cache, 'snapshots', 'a'.repeat(40))
      await fs.writeFile(path.join(h.root, 'outside'), 'must not copy')
      await fs.symlink(path.join(h.root, 'outside'), path.join(source, 'outside'))
      await assert.rejects(new LearningCopies(h.root).prepare(LEARNING_MODEL, await fs.realpath(source), cache, () => {}), /outside its Hugging Face repository/)
      assert.deepEqual(await fs.readdir(path.join(h.root, 'engine', 'learning')), [])
    } finally { await h.cleanup() }
  }
  console.log('Living Weights: private independent copies, transactional flags, explicit streamed teaching, safe cancellation, duplicate prevention and restart persistence')
  } finally { os.platform = platform; os.arch = architecture }
}

main().catch((error) => { console.error(error); process.exit(1) })
