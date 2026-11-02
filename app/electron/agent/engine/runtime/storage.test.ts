import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { c } from 'tar'
import { fixture } from './test-fixtures'
import { extractRuntime, verifyRuntimeTree, privateDirectory } from './storage'
const f = await fixture()
try {
  const dir = path.join(f.root, 'good')
  await extractRuntime(f.archive, dir, f.pkg)
  await verifyRuntimeTree(dir, f.pkg)
  await fs.writeFile(path.join(dir, 'lib/libtfgrammar.dylib'), 'tampered')
  await assert.rejects(verifyRuntimeTree(dir, f.pkg), /integrity/)
  await fs.rm(path.join(f.source, 'lib/libtfgrammar.dylib'))
  await fs.symlink('/etc/passwd', path.join(f.source, 'lib/libtfgrammar.dylib'))
  const linked = path.join(f.root, 'linked.tar.gz')
  await c({ gzip: true, file: linked, cwd: f.source }, f.pkg.files.map(p => p.path))
  await assert.rejects(extractRuntime(linked, path.join(f.root, 'bad-link'), f.pkg), /links|unexpected/)
  assert.deepEqual(await fs.readdir(path.join(f.root, 'bad-link')), [], 'preflight rejects links before writing any files')
  const duplicate = path.join(f.root, 'duplicate.tar.gz')
  await c({ gzip: true, file: duplicate, cwd: f.source }, ['NOTICE', 'NOTICE'])
  await assert.rejects(extractRuntime(duplicate, path.join(f.root, 'duplicate'), f.pkg), /duplicates|incomplete/)
  await fs.symlink(dir, path.join(f.root, 'linked-storage'))
  await assert.rejects(privateDirectory(path.join(f.root, 'linked-storage')), /without links/)
  const missing = { ...f.pkg, files: [...f.pkg.files, { path: 'missing.so', size: 1, sha256: 'a'.repeat(64), executable: false }] }
  await assert.rejects(extractRuntime(f.archive, path.join(f.root, 'missing'), missing), /incomplete/)
  const controller = new AbortController(); controller.abort()
  await assert.rejects(extractRuntime(f.archive, path.join(f.root, 'cancelled'), f.pkg, controller.signal), /abort/i)
} finally { await f.cleanup() }
console.log('Runtime archive inventory, symlink, duplicate, missing-file and cancellation checks passed.')
