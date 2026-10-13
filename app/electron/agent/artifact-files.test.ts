// Run: jiti electron/agent/artifact-files.test.ts
import assert from 'node:assert/strict'
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { readArtifactTextBounded, readArtifactTextPrefix } from './artifact-files.ts'

const root = await mkdtemp(path.join(os.tmpdir(), 'vontra-artifact-files-'))
try {
  const file = path.join(root, 'large.txt')
  await writeFile(file, 'prefix-' + 'x'.repeat(4_096))
  assert.equal(await readArtifactTextPrefix(file, 7), 'prefix-', 'preview reads only its byte prefix')
  assert.equal((await readArtifactTextBounded(file, 8_000)).length, 4_103, 'bounded read returns a file below the cap')
  await assert.rejects(() => readArtifactTextBounded(file, 1_000), /exceeds the 1000-byte IPC text limit/, 'oversized artifacts are refused before IPC serialization')
  await assert.rejects(() => readArtifactTextPrefix(root, 10), /not a regular file/)
  const outside = path.join(root, 'outside.txt')
  const swapped = path.join(root, 'swapped.txt')
  await writeFile(outside, 'sensitive replacement')
  await symlink(outside, swapped)
  await assert.rejects(() => readArtifactTextPrefix(swapped, 100), /ELOOP|symbolic link/i, 'a registered artifact path swapped to a symlink is never followed')
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log('artifact file boundary tests passed')
