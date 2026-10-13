import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { boundedRegexSearch } from './bounded-regex-search'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-regex-search-'))
try {
  await fs.writeFile(path.join(root, 'code.txt'), `safe marker\n${'a'.repeat(100_000)}!\n`)
  const normal = await boundedRegexSearch(root, 'SAFE (marker|value)', [])
  assert.equal(normal.length, 1)
  assert.match(normal[0], /code\.txt:1:safe marker/)

  let eventLoopTicked = false
  const tick = new Promise<void>((resolve) => setTimeout(() => { eventLoopTicked = true; resolve() }, 20))
  const started = Date.now()
  await Promise.allSettled([boundedRegexSearch(root, '(a+)+$', []), tick])
  assert.equal(eventLoopTicked, true, 'model regex evaluation does not block the Electron event loop')
  assert.ok(Date.now() - started < 4_500, 'pathological regex is bounded by the child-process timeout')
  await assert.rejects(() => boundedRegexSearch(root, 'x'.repeat(2_001), []), /exceeds 2000/)
} finally {
  await fs.rm(root, { recursive: true, force: true })
}

console.log('bounded regex search tests passed')
