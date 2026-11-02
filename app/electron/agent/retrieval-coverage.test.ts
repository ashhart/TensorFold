import assert from 'node:assert/strict'
import { promises as fs, type Dirent } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { collectCodeFiles } from './retrieval-code'
import {
  buildRetrievalIndex, buildWorkspaceRetrievalIndex, filterRetrievalProjectMemorySources,
  filterRetrievalThreadSources, formatRetrievalSearchResult, searchRetrievalIndex,
} from './retrieval-index'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'retrieval-fair-coverage-'))
try {
  const cwd = path.join(tmp, 'monorepo')
  await fs.mkdir(path.join(cwd, 'app'), { recursive: true })
  await fs.mkdir(path.join(cwd, 'native'), { recursive: true })
  for (let i = 0; i < 350; i++) {
    await fs.writeFile(path.join(cwd, 'app', `file-${String(i).padStart(3, '0')}.ts`), `export const item${i} = ${i}`)
  }
  await fs.writeFile(path.join(cwd, 'native', 'compiler.zig'), 'const native_probe = @import("std");')
  const index = await buildWorkspaceRetrievalIndex({ cwd, kinds: ['code'] })
  const scan = index.stats.codeScan!
  assert.equal(index.records.length, 300)
  assert.ok(index.records.some(record => record.path === 'native/compiler.zig'), 'app cannot starve a sibling native tree')
  assert.equal(scan.readFiles, 300)
  assert.equal(scan.discoveredFiles, 351)
  assert.equal(scan.truncated, true)
  assert.ok(scan.limits.includes('files'))
  const result = searchRetrievalIndex(index, { query: 'native_probe', kinds: ['code'] })
  assert.equal(result.hits[0]?.record.path, 'native/compiler.zig')
  assert.match(formatRetrievalSearchResult(result), /Partial code coverage: 300-file read budget.*Narrow the path/)

  const repeated = await buildWorkspaceRetrievalIndex({ cwd, kinds: ['code'] })
  assert.deepEqual(repeated.records.map(record => record.path), index.records.map(record => record.path))
  const scoped = await buildWorkspaceRetrievalIndex({ cwd, path: 'native', kinds: ['code'] })
  assert.equal(scoped.stats.codeScan?.scope, 'native')
  assert.equal(scoped.stats.codeScan?.truncated, false)
  assert.equal(scoped.records.length, 1)
  assert.deepEqual(filterRetrievalThreadSources(index, []).stats.codeScan, scan)
  assert.deepEqual(filterRetrievalProjectMemorySources(index, []).stats.codeScan, scan)
  assert.equal('codeScan' in buildRetrievalIndex({ codeFiles: [] }).stats, false, 'synchronous fixture shape stays compatible')

  const wide = path.join(tmp, 'wide')
  await fs.mkdir(wide)
  for (let i = 0; i < scan.maxDirectories + 3; i++) await fs.mkdir(path.join(wide, `dir-${i}`))
  const directoryLimited = await collectCodeFiles(wide, undefined, 1)
  assert.ok(directoryLimited.scan.visitedDirectories <= scan.maxDirectories)
  assert.ok(directoryLimited.scan.limits.includes('directories'))

  const deep = path.join(tmp, 'deep')
  const leaf = path.join(deep, ...Array.from({ length: scan.maxDepth + 1 }, () => 'next'))
  await fs.mkdir(leaf, { recursive: true })
  await fs.writeFile(path.join(leaf, 'deep.zig'), 'const hidden_by_depth = true;')
  const depthLimited = await collectCodeFiles(deep, undefined, 1)
  assert.equal(depthLimited.files.length, 0)
  assert.ok(depthLimited.scan.limits.includes('depth'))

  // A virtual directory stream exercises global and per-directory bounds without creating thousands of files.
  const realOpendir = fs.opendir
  let openStreams = 0
  const entry = (name: string, directory: boolean) => ({
    name, isDirectory: () => directory, isSymbolicLink: () => false,
  }) as Dirent
  fs.opendir = (async (dir: string) => (async function* () {
    openStreams++
    try {
      if (dir === cwd) {
        for (let i = 0; i < 20; i++) yield entry(`branch-${i}`, true)
      } else {
        for (let i = 0; i < scan.maxEntriesPerDirectory + 5; i++) yield entry(`ignored-${i}.txt`, false)
      }
    } finally { openStreams-- }
  })()) as unknown as typeof fs.opendir
  try {
    const entryLimited = await collectCodeFiles(cwd, undefined, 1)
    assert.equal(entryLimited.scan.visitedEntries, scan.maxEntries)
    assert.ok(entryLimited.scan.limits.includes('entries'))
    assert.ok(entryLimited.scan.limits.includes('directory_entries'))
    assert.equal(openStreams, 0, 'capped directory streams are closed')
  } finally {
    fs.opendir = realOpendir
  }
  console.log('Fair code scans reach sibling native sources, expose partial coverage and honor file/directory/entry/depth caps.')
} finally {
  await fs.rm(tmp, { recursive: true, force: true })
}
