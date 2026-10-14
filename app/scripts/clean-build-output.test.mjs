import assert from 'node:assert/strict'
import { access, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { cleanBuildOutputs } from './clean-build-output.mjs'

const root = await mkdtemp(path.join(os.tmpdir(), 'vontra-clean-build-'))
try {
  await mkdir(path.join(root, 'dist', 'assets'), { recursive: true })
  await mkdir(path.join(root, 'dist-electron'), { recursive: true })
  await mkdir(path.join(root, 'release'), { recursive: true })
  await writeFile(path.join(root, 'dist', 'assets', 'stale.js'), 'obsolete renderer')
  await writeFile(path.join(root, 'dist-electron', 'main-stale.js'), 'obsolete main process')
  await writeFile(path.join(root, 'release', 'keep.dmg'), 'historical artifact')

  await cleanBuildOutputs(root)
  await assert.rejects(access(path.join(root, 'dist')), /ENOENT/)
  await assert.rejects(access(path.join(root, 'dist-electron')), /ENOENT/)
  await access(path.join(root, 'release', 'keep.dmg'))
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log('build output cleanup tests passed')
