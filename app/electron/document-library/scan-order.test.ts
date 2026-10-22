// Run: node electron/document-library/scan-order.test.ts
import assert from 'node:assert/strict'
import { promises as fs, type Dirent } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { collectDocumentFiles } from './scanner'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-scan-order-'))
try {
  for (const name of ['charlie.txt', 'alpha.txt', 'bravo.txt']) await fs.writeFile(path.join(root, name), `${name} evidence`)
  await fs.mkdir(path.join(root, 'delta'))
  await fs.writeFile(path.join(root, 'delta', 'echo.txt'), 'echo evidence')
  // A filesystem may list a folder in any order; this one answers in reverse name order.
  const opendir = fs.opendir
  fs.opendir = (async (dir: string) => {
    const entries: Dirent[] = []
    for await (const entry of await opendir(dir)) entries.push(entry)
    entries.sort((a, b) => (a.name < b.name ? 1 : a.name > b.name ? -1 : 0))
    return { async *[Symbol.asyncIterator]() { yield* entries } }
  }) as unknown as typeof fs.opendir
  try {
    const files = (await collectDocumentFiles(root, 'dir')).map(file => path.relative(root, file))
    assert.deepEqual(files, ['alpha.txt', 'bravo.txt', 'charlie.txt', path.join('delta', 'echo.txt')], 'files come out in name order whatever order the folder is listed in')
  } finally { fs.opendir = opendir }
  console.log('scan order: a folder is walked in name order on every filesystem')
} finally { await fs.rm(root, { recursive: true, force: true }) }
