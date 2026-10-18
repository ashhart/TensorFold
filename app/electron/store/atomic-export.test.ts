import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { atomicExport } from './atomic-export'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-synthetic-export-review-'))
const destination = path.join(root, 'readable.txt'), payload = Buffer.from('Reviewed synthetic export.'), originalOpen = fs.open
const leftovers = async () => (await fs.readdir(root)).filter((name) => name.startsWith('.tensorfold-export-'))
async function duringTempSync(action: () => Promise<void>, run: () => Promise<void>) {
  ;(fs as any).open = async (...args: any[]) => {
    const handle = await (originalOpen as any)(...args)
    if (String(args[0]).includes('.tensorfold-export-')) { const sync = handle.datasync.bind(handle); handle.datasync = async () => { await action(); await sync() } }
    return handle
  }
  try { await run() } finally { fs.open = originalOpen }
}
try {
  let validated = 0
  await atomicExport(destination, payload, 1000, () => { validated++ })
  assert.equal(await fs.readFile(destination, 'utf8'), payload.toString()); assert.equal(validated, 1)
  assert.equal((await fs.stat(destination)).mode & 0o777, 0o600)
  await assert.rejects(atomicExport(destination, payload, 2, () => {}), /size limit/)
  assert.equal(await fs.readFile(destination, 'utf8'), payload.toString())
  await assert.rejects(atomicExport(destination, Buffer.from('Stale attempted replacement.'), 1000, () => { throw new Error('Synthetic owner/profile/revision drift.') }), /drift/)
  assert.equal(await fs.readFile(destination, 'utf8'), payload.toString()); assert.deepEqual(await leftovers(), [])
  const replacement = path.join(root, 'replacement.txt')
  await duringTempSync(async () => { await fs.writeFile(replacement, 'Intervening synthetic edit.'); await fs.rename(replacement, destination) }, async () => {
    await assert.rejects(atomicExport(destination, payload, 1000, () => {}), /destination changed/)
  })
  assert.equal(await fs.readFile(destination, 'utf8'), 'Intervening synthetic edit.'); assert.deepEqual(await leftovers(), [])
  const absent = path.join(root, 'initially-absent.txt')
  await duringTempSync(async () => { await fs.writeFile(absent, 'New synthetic user file.') }, async () => {
    await assert.rejects(atomicExport(absent, payload, 1000, () => {}), /destination changed/)
  })
  assert.equal(await fs.readFile(absent, 'utf8'), 'New synthetic user file.'); assert.deepEqual(await leftovers(), [])
  const linked = path.join(root, 'linked.txt'), symlink = path.join(root, 'symlink.txt')
  await fs.link(destination, linked); await fs.symlink(destination, symlink)
  await assert.rejects(atomicExport(linked, payload, 1000, () => {}), /linked files/)
  await assert.rejects(atomicExport(symlink, payload, 1000, () => {}), /linked files/)
  assert.equal(await fs.readFile(destination, 'utf8'), 'Intervening synthetic edit.')
  console.log('Actual atomic exporter preserves old/user-raced bytes on owner/profile/revision refusal and in-write destination changes, rejects hard/symbolic links, writes mode0600, bounds size and removes failed temporaries.')
} finally { fs.open = originalOpen; await fs.rm(root, { recursive: true, force: true }) }
