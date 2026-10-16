import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { parseApplyPatchText, stagePatchOperations, writeAppliedChanges } from './tool-patch'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tool-patch-race-'))
try {
  await fs.writeFile(path.join(root, 'a.txt'), 'before\n')
  const changes = await stagePatchOperations(root, parseApplyPatchText(`*** Begin Patch
*** Update File: a.txt
@@
-before
+after
*** Add File: raced.txt
+agent
*** End Patch`))

  // Another actor creates the add target after staging but before commit. The add
  // must fail without clobbering that file, and the earlier update must roll back.
  await fs.writeFile(path.join(root, 'raced.txt'), 'concurrent\n')
  await assert.rejects(() => writeAppliedChanges(root, changes), /EEXIST|exist/i)
  assert.equal(await fs.readFile(path.join(root, 'raced.txt'), 'utf8'), 'concurrent\n')
  assert.equal(await fs.readFile(path.join(root, 'a.txt'), 'utf8'), 'before\n')

  const edit = await stagePatchOperations(root, parseApplyPatchText(`*** Begin Patch
*** Update File: a.txt
@@
-before
+agent edit
*** End Patch`))
  await fs.writeFile(path.join(root, 'a.txt'), 'concurrent edit\n')
  await assert.rejects(() => writeAppliedChanges(root, edit), /changed after (?:the patch was staged|it was read)/)
  assert.equal(await fs.readFile(path.join(root, 'a.txt'), 'utf8'), 'concurrent edit\n', 'staged edits use a digest preflight')

  await fs.writeFile(path.join(root, 'source.txt'), 'source\n')
  const move = await stagePatchOperations(root, parseApplyPatchText(`*** Begin Patch
*** Update File: source.txt
*** Move to: destination.txt
*** End Patch`))
  await fs.writeFile(path.join(root, 'destination.txt'), 'concurrent destination\n')
  await assert.rejects(() => writeAppliedChanges(root, move), /EEXIST|exist/i)
  assert.equal(await fs.readFile(path.join(root, 'source.txt'), 'utf8'), 'source\n', 'a rejected move keeps its source')
  assert.equal(await fs.readFile(path.join(root, 'destination.txt'), 'utf8'), 'concurrent destination\n', 'a rejected move keeps the raced destination')

  await fs.rm(path.join(root, 'destination.txt'))
  const driftedMove = await stagePatchOperations(root, parseApplyPatchText(`*** Begin Patch
*** Update File: source.txt
*** Move to: destination.txt
*** End Patch`))
  await fs.writeFile(path.join(root, 'source.txt'), 'concurrent source edit\n')
  await assert.rejects(() => writeAppliedChanges(root, driftedMove), /changed after the patch was staged/)
  assert.equal(await fs.readFile(path.join(root, 'source.txt'), 'utf8'), 'concurrent source edit\n', 'a move refuses a drifted source')
  await assert.rejects(() => fs.access(path.join(root, 'destination.txt')), 'a rejected drifted move creates no destination')
} finally {
  await fs.rm(root, { recursive: true, force: true })
}

console.log('tool patch race tests passed')
