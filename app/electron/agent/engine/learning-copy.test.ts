import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { LEARNING_MODEL, LearningCopies } from './learning-copy'

const NATIVE_BACKUPS = ['.tensorfold-original', '.tensorfold-original.partial'] as const

async function fixture() {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-private-backup-')))
  const home = path.join(root, 'profile')
  const cache = path.join(root, 'original-checkpoint')
  const source = path.join(cache, 'snapshots', 'a'.repeat(40))
  await fs.mkdir(source, { recursive: true, mode: 0o700 })
  await fs.writeFile(path.join(source, 'config.json'), JSON.stringify({ model_type: 'nemotron_h', quantization: { bits: 4, group_size: 64 } }))
  await fs.writeFile(path.join(source, 'model.safetensors'), 'synthetic original checkpoint weights')
  const originals = await sourceHashes(source)
  const copies = new LearningCopies(home)
  const model = await copies.prepare(LEARNING_MODEL, source, cache, () => {})
  assert.notEqual((await fs.stat(path.join(source, 'model.safetensors'))).ino, (await fs.stat(path.join(model, 'model.safetensors'))).ino)
  return { root, home, source, originals, copies, model, cleanup: () => fs.rm(root, { recursive: true, force: true }) }
}

async function sourceHashes(directory: string) {
  return Promise.all(['config.json', 'model.safetensors'].map(async (name) => [name, createHash('sha256').update(await fs.readFile(path.join(directory, name))).digest('hex')]))
}

async function nativeBackup(model: string, name: typeof NATIVE_BACKUPS[number], partial = false) {
  const backup = path.join(model, name)
  await fs.mkdir(backup, { mode: 0o755 })
  await fs.chmod(backup, 0o755) // Match native mkdir independently of the test runner's umask.
  await fs.copyFile(path.join(model, 'config.json'), path.join(backup, 'config.json'))
  if (!partial) {
    await fs.copyFile(path.join(model, 'model.safetensors'), path.join(backup, 'model.safetensors'))
    await fs.writeFile(path.join(backup, 'SHA256SUMS'), 'synthetic native manifest\n', { mode: 0o644 })
    await fs.writeFile(path.join(backup, 'graphs'), 'synthetic-facts.json\n', { mode: 0o644 })
  }
  return backup
}

async function refused(label: string, change: (h: Awaited<ReturnType<typeof fixture>>) => Promise<void>, message: RegExp, checkExisting = true) {
  const h = await fixture()
  try {
    await change(h)
    if (checkExisting) await assert.rejects(h.copies.existing(LEARNING_MODEL), message, `${label}: reuse must fail closed`)
    await assert.rejects(h.copies.purge(), message, `${label}: erasure must fail closed before removing files`)
    assert.ok(await fs.stat(h.model), `${label}: refused erasure must preserve the private copy`)
    assert.deepEqual(await sourceHashes(h.source), h.originals, `${label}: the original checkpoint must stay unchanged`)
  } finally { await h.cleanup() }
}

async function main() {
  for (const name of NATIVE_BACKUPS) {
    const h = await fixture()
    try {
      await nativeBackup(h.model, name, name.endsWith('.partial'))
      await fs.writeFile(path.join(h.model, 'model.safetensors'), 'synthetic taught private weights')
      assert.equal(await h.copies.existing(LEARNING_MODEL), h.model, `${name}: a normal first native lesson must remain reusable`)
      assert.equal(await h.copies.prepare(LEARNING_MODEL, h.source, path.dirname(path.dirname(h.source)), () => {}), h.model, 're-enabling preserves the existing private model')
      assert.equal(await fs.readFile(path.join(h.model, 'model.safetensors'), 'utf8'), 'synthetic taught private weights')
      for (const directory of [h.copies.root, h.copies.folder(LEARNING_MODEL), h.model]) assert.equal((await fs.lstat(directory)).mode & 0o777, 0o700, 'root, profile and model authority stays private')
      const removed = await h.copies.purge()
      assert.ok(removed.files >= 4)
      await assert.rejects(fs.stat(h.copies.root), { code: 'ENOENT' })
      assert.deepEqual(await sourceHashes(h.source), h.originals, 'erasure and reuse preserve the original checkpoint bytes')
    } finally { await h.cleanup() }
  }
  {
    const h = await fixture()
    try {
      const completed = await nativeBackup(h.model, '.tensorfold-original')
      const partial = await nativeBackup(h.model, '.tensorfold-original.partial', true)
      await fs.chmod(completed, 0o750)
      await fs.chmod(partial, 0o700)
      assert.equal(await h.copies.existing(LEARNING_MODEL), h.model, 'stricter native umasks and an interrupted partial beside a completed snapshot are supported')
      assert.equal((await h.copies.purge()).files, 8)
      assert.deepEqual(await sourceHashes(h.source), h.originals)
    } finally { await h.cleanup() }
  }
  for (const name of ['unexpected', '.tensorfold-original.part', '.tensorfold-original.partial-old']) {
    await refused(name, async (h) => {
      const directory = path.join(h.model, name)
      await fs.mkdir(directory, { mode: 0o755 })
      await fs.chmod(directory, 0o755)
    }, /private app-owned folder/)
  }
  for (const mode of [0o775, 0o757, 0o777, 0o1755]) {
    await refused(`unsafe backup mode ${mode.toString(8)}`, async (h) => {
      await fs.chmod(await nativeBackup(h.model, '.tensorfold-original'), mode)
    }, /native backup.*writable/)
  }
  for (const target of ['root', 'profile', 'model'] as const) {
    await refused(`public ${target} authority`, async (h) => {
      await nativeBackup(h.model, '.tensorfold-original')
      await fs.chmod(target === 'root' ? h.copies.root : target === 'profile' ? h.copies.folder(LEARNING_MODEL) : h.model, 0o755)
    }, /private app-owned folder/)
  }
  for (const location of ['root', 'profile', 'other-profile'] as const) {
    await refused(`backup name at ${location}`, async (h) => {
      const parent = location === 'root' ? h.copies.root : location === 'profile' ? h.copies.folder(LEARNING_MODEL) : path.join(h.copies.root, 'other-profile', 'model')
      if (location === 'other-profile') await fs.mkdir(parent, { recursive: true, mode: 0o700 })
      const directory = path.join(parent, '.tensorfold-original')
      await fs.mkdir(directory, { mode: 0o755 })
      await fs.chmod(directory, 0o755)
    }, /private app-owned folder/, false)
  }
  for (const insideBackup of [false, true]) {
    await refused(`nested public folder ${insideBackup}`, async (h) => {
      const parent = insideBackup ? await nativeBackup(h.model, '.tensorfold-original') : path.join(h.model, 'nested')
      if (!insideBackup) await fs.mkdir(parent, { mode: 0o700 })
      const directory = path.join(parent, '.tensorfold-original')
      await fs.mkdir(directory, { mode: 0o755 })
      await fs.chmod(directory, 0o755)
    }, /private app-owned folder/)
  }
  await refused('redirected backup container', async (h) => {
    await fs.symlink(h.source, path.join(h.model, '.tensorfold-original'))
  }, /linked, shared/)
  await refused('redirected file inside backup', async (h) => {
    const backup = await nativeBackup(h.model, '.tensorfold-original')
    await fs.symlink(path.join(h.source, 'model.safetensors'), path.join(backup, 'redirected.safetensors'))
  }, /linked, shared/)
  await refused('redirected directory inside backup', async (h) => {
    const backup = await nativeBackup(h.model, '.tensorfold-original')
    await fs.symlink(h.source, path.join(backup, 'redirected'))
  }, /linked, shared/)
  for (const fromOriginal of [true, false]) {
    await refused(`hardlinked backup file ${fromOriginal}`, async (h) => {
      const backup = await nativeBackup(h.model, '.tensorfold-original')
      await fs.link(path.join(fromOriginal ? h.source : h.model, 'model.safetensors'), path.join(backup, 'shared.safetensors'))
    }, /linked, shared/)
  }
  await refused('excessive depth inside backup', async (h) => {
    let directory = await nativeBackup(h.model, '.tensorfold-original')
    for (let i = 0; i < 9; i++) { directory = path.join(directory, 'nested'); await fs.mkdir(directory, { mode: 0o700 }) }
  }, /too many nested folders/)
  console.log('Living Weights: completed/partial native backups support reuse and erasure; public, writable, redirected, shared and deeply nested folders fail closed; original checkpoints are preserved')
}

main().catch((error) => { console.error(error); process.exit(1) })
