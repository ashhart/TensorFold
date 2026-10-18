import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { saveBenchmarkExport } from './model-benchmark-export'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'benchmark-export-'))
const file = path.join(root, 'receipt.json')
try {
  await saveBenchmarkExport(file, '{"synthetic":1}\n', () => {})
  assert.equal(await fs.readFile(file, 'utf8'), '{"synthetic":1}\n')
  assert.equal((await fs.stat(file)).mode & 0o777, 0o600)
  await assert.rejects(saveBenchmarkExport(file, '{"synthetic":2}', () => { throw new Error('Owner access changed') }), /access changed/)
  assert.equal(await fs.readFile(file, 'utf8'), '{"synthetic":1}\n', 'failed admission preserves the existing export')
  assert.deepEqual(await fs.readdir(root), ['receipt.json'], 'failed writes leave no temporary files')
  await fs.symlink(file, path.join(root, 'symlink.json'))
  await assert.rejects(saveBenchmarkExport(path.join(root, 'symlink.json'), '{}', () => {}), /linked destinations/)
  await fs.link(file, path.join(root, 'hardlink.json'))
  await assert.rejects(saveBenchmarkExport(file, '{}', () => {}), /linked destinations/)
  await fs.unlink(path.join(root, 'hardlink.json'))
  await assert.rejects(saveBenchmarkExport(file, 'x'.repeat(1_048_577), () => {}), /too large/)
  await saveBenchmarkExport(file, '{"synthetic":3}', () => {})
  assert.equal(await fs.readFile(file, 'utf8'), '{"synthetic":3}')
} finally { await fs.rm(root, { recursive: true, force: true }) }
console.log('Benchmark native export atomicity, owner recheck, size, link and cleanup regressions passed')
