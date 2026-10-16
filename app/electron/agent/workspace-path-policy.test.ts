import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { LocalBackend } from './workspace-backend'
import { atomicWriteWorkspaceFile } from './atomic-file-write'
import { stagePatchOperations, writeAppliedChanges } from './tool-patch'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-path-policy-'))
const backend = new LocalBackend(root, async () => ({ code: 0, stdout: '', stderr: '' }))
try {
  await fs.mkdir(path.join(root, '.git'))
  await fs.writeFile(path.join(root, '.git', 'config'), 'synthetic metadata\n')
  await fs.writeFile(path.join(root, '.env'), 'SYNTHETIC_VALUE=fixture\n')
  await fs.symlink('.git', path.join(root, 'metadata-alias'))
  await fs.symlink('.git/config', path.join(root, 'leaf-alias'))
  await fs.symlink('.env', path.join(root, 'ordinary.txt'))
  for (const name of ['.git/config', 'metadata-alias/config', 'leaf-alias']) {
    const target = path.join(root, name)
    await assert.rejects(() => atomicWriteWorkspaceFile(root, target, 'changed'), /safety policy/)
    await assert.rejects(() => backend.writeFile(target, 'changed'), /safety policy/)
    await assert.rejects(() => backend.remove(target), /safety policy/)
    await assert.rejects(() => backend.rename(target, path.join(root, 'moved')), /safety policy/)
    await assert.rejects(() => stagePatchOperations(root, [{ kind: 'delete', filePath: name }]), /safety policy/)
  }
  await assert.rejects(() => backend.readFile(path.join(root, 'ordinary.txt')), /safety policy/)
  await assert.rejects(() => backend.mkdirp(path.join(root, 'metadata-alias', 'nested')), /safety policy/)
  await assert.rejects(() => atomicWriteWorkspaceFile(root, path.join(root, 'metadata-alias', 'new', 'file'), 'changed'), /safety policy/)
  assert.equal(await fs.stat(path.join(root, '.git', 'new')).then(() => true, () => false), false, 'denied writes create no parents')
  await fs.writeFile(path.join(root, 'source.txt'), 'source')
  await assert.rejects(() => backend.rename(path.join(root, 'source.txt'), path.join(root, 'metadata-alias', 'new')), /safety policy/)
  await assert.rejects(() => writeAppliedChanges(root, [{ filePath: 'metadata-alias/config', finalPath: 'metadata-alias/config', before: 'synthetic metadata\n', after: '', kind: 'delete' }]), /safety policy/)
  assert.equal(await fs.readFile(path.join(root, '.git', 'config'), 'utf8'), 'synthetic metadata\n')
  await fs.mkdir(path.join(root, 'normal'))
  await fs.symlink('normal', path.join(root, 'normal-alias'))
  await backend.writeFile(path.join(root, 'normal-alias', 'file.txt'), 'allowed')
  assert.equal(await backend.readFile(path.join(root, 'normal-alias', 'file.txt')), 'allowed')
  // Worktree indirection can name a control directory without a .git component.
  await fs.rename(path.join(root, '.git'), path.join(root, 'control'))
  await fs.writeFile(path.join(root, '.git'), 'gitdir: control\n')
  await assert.rejects(() => backend.writeFile(path.join(root, 'control', 'config'), 'changed'), /safety policy/)
  await assert.rejects(() => backend.readFile(path.join(root, 'control', 'config')), /safety policy/)
  await fs.rename(path.join(root, '.git'), path.join(root, 'git-pointer'))
  await fs.symlink('git-pointer', path.join(root, '.git'))
  await assert.rejects(() => backend.writeFile(path.join(root, 'control', 'config'), 'changed'), /safety policy/)
} finally {
  await fs.rm(root, { recursive: true, force: true })
}
console.log('workspace canonical path policy tests passed')
