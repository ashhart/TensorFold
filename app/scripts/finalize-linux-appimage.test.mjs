import assert from 'node:assert/strict'
import { mkdtemp, writeFile, readFile, readdir, rm, mkdir, symlink } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { finalizeLinuxAppImage } from './finalize-linux-appimage.mjs'

const folder = await mkdtemp(path.join(os.tmpdir(), 'tf-appimage-build-'))
const image = path.join(folder, 'fixture.AppImage'), sentinel = path.join(folder, 'sentinel')
const bytes = Buffer.alloc(1024)
bytes.write('7f454c4602010100414902', 0, 'hex'); bytes.writeUInt16LE(62, 18)
bytes.write('hsqs', 128); bytes.writeUInt32LE(131072, 140); bytes.writeUInt16LE(4, 156); bytes.writeBigUInt64LE(192n, 168)
try {
  await writeFile(image, bytes); await writeFile(sentinel, 'unchanged')
  await assert.rejects(finalizeLinuxAppImage(image, { exec: async () => { throw new Error('Missing extractor') } }), /squashfs-tools/)
  assert.deepEqual(await readFile(image), bytes)
  assert.deepEqual((await readdir(folder)).sort(), ['fixture.AppImage', 'sentinel'])
  if (process.platform !== 'win32') {
    const extract = async (command, args) => {
      assert.equal(command, 'unsquashfs', 'An unsafe payload must be refused before any repacking tool runs')
      const tree = args[args.indexOf('-d') + 1]
      await mkdir(tree)
      await symlink(sentinel, path.join(tree, 'AppRun'))
      await writeFile(path.join(tree, 'tensorfold-studio'), 'fixture')
    }
    await assert.rejects(finalizeLinuxAppImage(image, { exec: extract }), /ordinary files/)
    assert.equal(await readFile(sentinel, 'utf8'), 'unchanged')
    assert.deepEqual(await readFile(image), bytes)
  }
  bytes[4] = 1; await writeFile(image, bytes)
  await assert.rejects(finalizeLinuxAppImage(image, { exec: async () => { throw new Error('Must not run') } }), /type-2/)
} finally { await rm(folder, { recursive: true, force: true }) }
console.log('AppImage finalization refuses malformed/symlink payloads, leaves originals intact on failure and removes only its own temporary files.')
