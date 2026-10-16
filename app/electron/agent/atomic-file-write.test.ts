import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { atomicWriteWorkspaceFile, workspaceContentSha256 } from './atomic-file-write'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-atomic-write-'))
const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-atomic-outside-'))
try {
  const target = path.join(root, 'target.txt')
  const victim = path.join(outside, 'victim.txt')
  await fs.writeFile(target, 'before')
  await fs.writeFile(victim, 'outside-safe')

  await atomicWriteWorkspaceFile(root, target, 'after', {
    beforeRename: async () => {
      await fs.unlink(target)
      await fs.symlink(victim, target)
    },
  })
  assert.equal(await fs.readFile(target, 'utf8'), 'after', 'the target entry is atomically replaced')
  assert.equal(await fs.readFile(victim, 'utf8'), 'outside-safe', 'a last-moment symlink target is never followed')
  assert.equal((await fs.lstat(target)).isSymbolicLink(), false)

  const casTarget = path.join(root, 'cas.txt')
  await fs.writeFile(casTarget, 'staged-version')
  await assert.rejects(
    () => atomicWriteWorkspaceFile(root, casTarget, 'agent-version', {
      expectedSha256: workspaceContentSha256('staged-version'),
      beforeRename: async () => { await fs.writeFile(casTarget, 'concurrent-version') },
    }),
    /changed after it was read/,
  )
  assert.equal(await fs.readFile(casTarget, 'utf8'), 'concurrent-version', 'digest preflight preserves a concurrent regular-file edit observed before commit')

  const createTarget = path.join(root, 'create-only.txt')
  await fs.writeFile(createTarget, 'concurrent-user-content')
  await assert.rejects(
    () => atomicWriteWorkspaceFile(root, createTarget, 'agent-content', { ifExists: 'error' }),
    /EEXIST|exist/i,
  )
  assert.equal(await fs.readFile(createTarget, 'utf8'), 'concurrent-user-content', 'create-only writes never clobber an entry raced in by another actor')

  const racedTarget = path.join(root, 'raced-create.txt')
  await atomicWriteWorkspaceFile(root, racedTarget, 'agent-content', {
    ifExists: 'error',
    beforeRename: async () => { await fs.symlink(victim, racedTarget) },
  }).then(
    () => assert.fail('a final symlink raced into a create-only write must be rejected'),
    () => undefined,
  )
  assert.equal((await fs.lstat(racedTarget)).isSymbolicLink(), true, 'the concurrent final entry is preserved')
  assert.equal(await fs.readFile(victim, 'utf8'), 'outside-safe', 'create-only final symlink races never reach the link target')

  const swapParent = path.join(root, 'swap-parent')
  const movedParent = path.join(root, 'swap-parent-original')
  await fs.mkdir(swapParent)
  await assert.rejects(
    () => atomicWriteWorkspaceFile(root, path.join(swapParent, 'file.txt'), 'secret-content', {
      beforeRename: async () => {
        await fs.rename(swapParent, movedParent)
        await fs.symlink(outside, swapParent)
      },
    }),
    /changed through a symlink|changed parent directory/,
  )
  assert.equal(await fs.readFile(victim, 'utf8'), 'outside-safe', 'a swapped parent cannot redirect the final write outside')
  assert.equal(await fs.stat(path.join(movedParent, (await fs.readdir(movedParent))[0])).then((s) => s.size), 0, 'an unreachable failed temp inode is truncated before close')

  const nested = path.join(root, 'new', 'deep', 'file.txt')
  await atomicWriteWorkspaceFile(root, nested, 'nested')
  assert.equal(await fs.readFile(nested, 'utf8'), 'nested', 'missing parent directories are created inside the canonical workspace')

  await assert.rejects(
    () => atomicWriteWorkspaceFile(root, path.join(outside, 'escape.txt'), 'nope'),
    /escapes the workspace/,
  )
} finally {
  await fs.rm(root, { recursive: true, force: true })
  await fs.rm(outside, { recursive: true, force: true })
}

console.log('atomic workspace write tests passed')
