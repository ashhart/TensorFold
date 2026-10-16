import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { allocatedBytes, freeDiskBytes, hasCachedSnapshot, hubCacheDir, isRepoId, isRevision, repoCacheDir, resolveEngineBinaryPath } from './paths'

async function main() {
  // Binary: packaged apps always use their own Resources copy; the override is a dev-only seam.
  assert.equal(
    resolveEngineBinaryPath({ packaged: true, resourcesPath: '/App.app/Contents/Resources', appRoot: '/repo', env: { TENSORFOLD_STUDIO_ENGINE_BIN: '/tmp/evil' } }),
    '/App.app/Contents/Resources/engine/bin/tensorfold-native',
  )
  assert.equal(resolveEngineBinaryPath({ packaged: false, resourcesPath: '', appRoot: '/repo', env: {} }), '/repo/build/engine/bin/tensorfold-native')
  assert.equal(resolveEngineBinaryPath({ packaged: false, resourcesPath: '', appRoot: '/repo', env: { TENSORFOLD_STUDIO_ENGINE_BIN: '/opt/tf/bin/tensorfold-native' } }), '/opt/tf/bin/tensorfold-native')

  // Cache: the engine's own order.
  assert.equal(hubCacheDir({ HF_HUB_CACHE: '/c/hub', HF_HOME: '/h', HOME: '/u' }), '/c/hub')
  assert.equal(hubCacheDir({ HF_HOME: '/h', HOME: '/u' }), '/h/hub')
  assert.equal(hubCacheDir({ HOME: '/u' }), '/u/.cache/huggingface/hub')

  for (const good of ['TensorFold/Qwen3.8-27B-MLX-4bit', 'mlx-community/Qwen3.5-2B-MLX-4bit', 'z-lab/Qwen3.8-27B-DFlash2', 'a/b']) assert.equal(isRepoId(good), true, good)
  for (const bad of ['plain', 'a/b/c', '../x', 'a/..', 'a/.hidden', 'a--b/c', 'a/b--c', 'a/b..c', '/abs', 'a/', 'a b/c', 'a/b\0', 42, null]) assert.equal(isRepoId(bad), false, String(bad))
  assert.equal(repoCacheDir('/hub', 'mlx-community/Qwen3.5-2B-MLX-4bit'), '/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit')
  assert.throws(() => repoCacheDir('/hub', '../escape'), /owner\/name/)

  for (const good of ['main', 'v1.0', '93760be4f1f69842a46bc13dbdc0f19e291392a3']) assert.equal(isRevision(good), true, good)
  for (const bad of ['', 'a/b', '..', 'x..y', 'a b', 'refs%2Fpr']) assert.equal(isRevision(bad), false, bad)

  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-paths-test-'))
  try {
    // pull.zig sizes a big file to its full length first: only written blocks count as progress.
    const blobs = path.join(root, 'models--Org--Name', 'blobs')
    await fs.mkdir(blobs, { recursive: true })
    const sparse = await fs.open(path.join(blobs, 'big.ranges'), 'w')
    await sparse.truncate(256 * 1024 * 1024)
    await sparse.write(Buffer.alloc(1024 * 1024, 1), 0, 1024 * 1024, 0)
    await sparse.close()
    await fs.writeFile(path.join(blobs, 'small'), Buffer.alloc(10_000, 1))
    const counted = await allocatedBytes(blobs)
    assert.ok(counted >= 1024 * 1024 + 10_000, `written bytes count (${counted})`)
    assert.ok(counted < 8 * 1024 * 1024, `unwritten sparse length does not count (${counted})`)
    assert.equal(await allocatedBytes(path.join(root, 'missing')), 0)

    assert.equal(typeof await freeDiskBytes(path.join(root, 'not', 'yet', 'created')), 'number', 'free space comes from the nearest existing folder')

    assert.equal(await hasCachedSnapshot(root, 'Org/Name'), false)
    await fs.mkdir(path.join(root, 'models--Org--Name', 'refs'), { recursive: true })
    await fs.writeFile(path.join(root, 'models--Org--Name', 'refs', 'main'), '../../escape')
    assert.equal(await hasCachedSnapshot(root, 'Org/Name'), false, 'a ref that is not a sha is never joined into a path')
    await fs.writeFile(path.join(root, 'models--Org--Name', 'refs', 'main'), 'b'.repeat(40))
    await fs.mkdir(path.join(root, 'models--Org--Name', 'snapshots', 'b'.repeat(40)), { recursive: true })
    assert.equal(await hasCachedSnapshot(root, 'Org/Name'), true)
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
  console.log('engine paths tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
