import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { projectMemorySourceSnapshots } from './project-memory-renderer'

const root = mkdtempSync(path.join(os.tmpdir(), 'vontra-remote-memory-'))
try {
  writeFileSync(path.join(root, 'AGENTS.md'), 'unrelated local lookalike')
  const entry = { id: 'm', projectId: 'p', sourceType: 'file', sourceRef: 'AGENTS.md' } as any
  const local = projectMemorySourceSnapshots({ id: 'p', cwd: root, workspace: null } as any, [entry])
  assert.equal(local['AGENTS.md'].exists, true, 'local source remains verifiable')
  const remote = projectMemorySourceSnapshots({ id: 'p', cwd: root, workspace: { kind: 'ssh', hostId: 'pgx', path: root } } as any, [entry])
  assert.deepEqual(remote['AGENTS.md'], { sourceRef: 'AGENTS.md', exists: false, updatedAt: null }, 'SSH source ignores the local lookalike')
} finally {
  rmSync(root, { recursive: true, force: true })
}

console.log('project memory marks SSH file sources unverifiable without local stat trust')
