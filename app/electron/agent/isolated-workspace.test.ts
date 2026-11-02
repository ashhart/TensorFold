import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { prepareIsolatedWorkspace, publishIsolatedWorkspace, removeIsolatedWorkspace } from './isolated-workspace'

const parent = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-isolation-test-'))
const root = path.join(parent, 'project')
await fs.mkdir(root)
await fs.writeFile(path.join(root, 'source.txt'), 'original')
await fs.chmod(path.join(root, 'source.txt'), 0o664)
await fs.writeFile(path.join(root, 'deleted.txt'), 'delete me')
await fs.mkdir(path.join(root, '.git'))
await fs.writeFile(path.join(root, '.git/config'), 'protected config')
await fs.writeFile(path.join(parent, 'private.txt'), 'outside fixture')
try {
  const first = await prepareIsolatedWorkspace(root)
  try {
    assert.notEqual((await fs.stat(path.join(first.workspace, 'source.txt'))).ino, (await fs.stat(path.join(root, 'source.txt'))).ino)
    await fs.writeFile(path.join(first.workspace, 'source.txt'), 'updated')
    await fs.unlink(path.join(first.workspace, 'deleted.txt'))
    await fs.mkdir(path.join(first.workspace, 'output'))
    await fs.writeFile(path.join(first.workspace, 'output/result.txt'), 'built')
    const result = await publishIsolatedWorkspace(first)
    assert.deepEqual(result.sort(), ['deleted.txt', 'output/result.txt', 'source.txt'])
    assert.equal(await fs.readFile(path.join(root, 'source.txt'), 'utf8'), 'updated')
    if (process.platform !== 'win32') assert.equal((await fs.stat(path.join(root, 'source.txt'))).mode & 0o777, 0o664)
    assert.equal(await fs.readFile(path.join(root, '.git/config'), 'utf8'), 'protected config')
  } finally { await removeIsolatedWorkspace(first) }

  for (const attack of ['new-git', 'git-pointer', 'outside-link', 'hardlink', 'concurrent-edit']) {
    const stage = await prepareIsolatedWorkspace(root)
    try {
      await fs.writeFile(path.join(stage.workspace, 'source.txt'), 'must not be published')
      if (attack === 'new-git') {
        await fs.mkdir(path.join(stage.workspace, 'nested/.git/hooks'), { recursive: true })
        await fs.writeFile(path.join(stage.workspace, 'nested/.git/hooks/pre-commit'), 'unsafe hook')
      } else if (attack === 'git-pointer') {
        await fs.rm(path.join(stage.workspace, '.git'), { recursive: true })
        await fs.writeFile(path.join(stage.workspace, '.git'), 'gitdir: ../private')
      } else if (attack === 'outside-link') {
        if (process.platform === 'win32') continue
        await fs.symlink(path.join(parent, 'private.txt'), path.join(stage.workspace, 'escape'))
      } else if (attack === 'hardlink') {
        await fs.link(path.join(parent, 'private.txt'), path.join(stage.workspace, 'escape'))
      } else await fs.writeFile(path.join(root, 'source.txt'), 'newer user edit')
      await assert.rejects(publishIsolatedWorkspace(stage), /Git|link|changed|concurrent/i, attack)
      assert.equal(await fs.readFile(path.join(root, 'source.txt'), 'utf8'), attack === 'concurrent-edit' ? 'newer user edit' : 'updated')
      assert.equal(await fs.readFile(path.join(parent, 'private.txt'), 'utf8'), 'outside fixture')
      await fs.writeFile(path.join(root, 'source.txt'), 'updated')
    } finally { await removeIsolatedWorkspace(stage) }
  }
  await fs.writeFile(path.join(root, 'a.txt'), 'original a')
  await fs.writeFile(path.join(root, 'b.txt'), 'original b')
  const rollback = await prepareIsolatedWorkspace(root)
  try {
    await fs.writeFile(path.join(rollback.workspace, 'a.txt'), 'isolated a')
    await fs.writeFile(path.join(rollback.workspace, 'b.txt'), 'isolated b')
    await assert.rejects(publishIsolatedWorkspace(rollback, undefined, async relative => {
      if (relative === 'b.txt') await fs.writeFile(path.join(root, 'b.txt'), 'newer user edit')
    }), /changed/)
    assert.equal(await fs.readFile(path.join(root, 'a.txt'), 'utf8'), 'original a')
    assert.equal(await fs.readFile(path.join(root, 'b.txt'), 'utf8'), 'newer user edit')
    assert.ok(!(await fs.readdir(root)).some(name => name.startsWith('.tensorfold-isolation-')))
  } finally { await removeIsolatedWorkspace(rollback) }
  if (process.platform !== 'win32') {
    const permissions = await prepareIsolatedWorkspace(root)
    try {
      await fs.chmod(path.join(permissions.workspace, 'output'), 0o500)
      await assert.rejects(publishIsolatedWorkspace(permissions), /directory permissions/)
    } finally { await removeIsolatedWorkspace(permissions) }
    const links = await prepareIsolatedWorkspace(root)
    try {
      await fs.symlink('source.txt', path.join(links.workspace, 'source-link.txt'))
      await publishIsolatedWorkspace(links)
      assert.equal(await fs.readlink(path.join(root, 'source-link.txt')), 'source.txt')
      assert.equal(await fs.readFile(path.join(root, 'source-link.txt'), 'utf8'), 'updated')
    } finally { await removeIsolatedWorkspace(links) }
  }
  console.log('Isolated workspace publishes ordinary changes and refuses Git controls, links and concurrent edits without partial publication.')
} finally { await fs.rm(parent, { recursive: true, force: true }) }
