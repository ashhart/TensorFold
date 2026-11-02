import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { buildWorkspaceRetrievalIndex, searchRetrievalIndex } from './retrieval-index'

const fixture = await mkdtemp(path.join(tmpdir(), 'tensorfold-retrieval-code-'))
const cwd = path.join(fixture, 'workspace')
try {
  await mkdir(path.join(cwd, 'src'), { recursive: true })
  await mkdir(path.join(cwd, 'node_modules'), { recursive: true })
  await mkdir(path.join(fixture, 'outside'), { recursive: true })
  const sourceFiles: Record<string, string> = {
    'build.zig': 'const tensorfold_allocator = @import("std").heap.page_allocator;',
    'build.zig.zon': '.{ .name = .tensorfold_allocator, .version = "0.0.0" }',
    'src/kernel.metal': 'kernel void tensorfold_allocator(device float* values) {}',
    'src/kernel.cu': '__global__ void tensorfold_allocator(float* values) {}',
    'src/kernel.cuh': '__device__ float tensorfold_allocator(float value);',
    'src/control.ts': 'export const tensorfold_allocator = true',
  }
  for (const [name, content] of Object.entries(sourceFiles)) await writeFile(path.join(cwd, name), content)
  await writeFile(path.join(cwd, '.env.zig'), 'private_marker')
  await writeFile(path.join(cwd, 'secret.key'), 'private_marker')
  await writeFile(path.join(cwd, 'node_modules', 'dependency.zig'), 'private_marker')
  await writeFile(path.join(cwd, 'oversized.zig'), 'x'.repeat(240_001))
  await writeFile(path.join(fixture, 'outside', 'external.zig'), 'private_marker')
  await symlink(path.join(fixture, 'outside'), path.join(cwd, 'external'))
  await symlink(path.join(fixture, 'outside', 'external.zig'), path.join(cwd, 'linked.zig'))

  const index = await buildWorkspaceRetrievalIndex({ cwd, kinds: ['code'] })
  assert.deepEqual(index.records.map(record => record.path).sort(), Object.keys(sourceFiles).sort())
  const result = searchRetrievalIndex(index, { query: 'tensorfold_allocator', kinds: ['code'], limit: 20 })
  assert.equal(result.hits.length, 6, 'Zig, its manifest and GPU kernels participate in real workspace search')
  assert.ok(result.hits.every(hit => hit.record.sourceIds.includes(`code:${hit.record.path}`)))
  assert.equal(searchRetrievalIndex(index, { query: 'private_marker' }).hits.length, 0)

  const scoped = await buildWorkspaceRetrievalIndex({ cwd, path: 'src', kinds: ['code'] })
  assert.ok(scoped.records.every(record => record.path?.startsWith('src/')))
  assert.equal(scoped.records.length, 4)
  const limited = await buildWorkspaceRetrievalIndex({ cwd, kinds: ['code'], maxCodeFiles: 2 })
  assert.equal(limited.records.length, 2, 'collection honors the caller file budget')
  const invalidLimit = await buildWorkspaceRetrievalIndex({ cwd, kinds: ['code'], maxCodeFiles: Number.NaN })
  assert.equal(invalidLimit.records.length, 6, 'non-finite budgets use the bounded default')
  await assert.rejects(buildWorkspaceRetrievalIndex({ cwd, path: '../outside', kinds: ['code'] }), /escapes the workspace/)
  await assert.rejects(buildWorkspaceRetrievalIndex({ cwd, path: 'external', kinds: ['code'] }), /symlink/)
  console.log('Workspace retrieval covers Zig, ZON, Metal and CUDA with source IDs, limits and path exclusions intact')
} finally {
  await rm(fixture, { recursive: true, force: true })
}
