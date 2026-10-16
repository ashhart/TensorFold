import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { canonicalHomeDirectory, resolvePairedWorkspace } from './local-workspace-policy'

const fixture = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-home-policy-'))
try {
  const home = path.join(fixture, 'home')
  const inside = path.join(home, 'project')
  const other = path.join(home, 'other')
  const outside = path.join(fixture, 'outside')
  await fs.mkdir(home)
  await Promise.all([fs.mkdir(inside), fs.mkdir(other), fs.mkdir(outside)])
  await fs.symlink(outside, path.join(home, 'outside-alias'))
  await fs.symlink(inside, path.join(home, 'inside-alias'))
  const canonical = await fs.realpath(inside)
  assert.equal(await resolvePairedWorkspace(inside, { kind: 'local', path: path.join(home, 'inside-alias') }, home), canonical)
  assert.equal(await canonicalHomeDirectory(home, home), await fs.realpath(home))
  for (const target of ['/', home, outside, path.join(home, 'outside-alias')]) {
    await assert.rejects(() => resolvePairedWorkspace(target, null, home), /home|directory/)
  }
  await assert.rejects(() => resolvePairedWorkspace(inside, { kind: 'local', path: other }, home), /same directory/)
  await assert.rejects(() => resolvePairedWorkspace(inside, { kind: 'local', path: outside }, home), /home/)
  await assert.rejects(() => resolvePairedWorkspace(inside, { kind: 'ssh', path: inside, hostId: 'fixture' }, home), /local/)
  await assert.rejects(() => canonicalHomeDirectory(path.join(home, 'outside-alias'), home), /home/)
} finally { await fs.rm(fixture, { recursive: true, force: true }) }
console.log('paired canonical workspace tests passed')
