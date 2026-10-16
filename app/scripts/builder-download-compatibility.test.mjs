import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createRequire } from 'node:module'
import { createHash } from 'node:crypto'

const require = createRequire(import.meta.url)
const { downloadElectronArtifactZip } = require('app-builder-lib/out/util/electronGet.js')
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-builder-download-'))
try {
  // Exercise the real builder's download/cache/checksum consumer after replacing
  // its old downloader dependency; the synthetic downloader substitutes only I/O.
  const archive = Buffer.from('synthetic artifact bytes')
  const digest = createHash('sha256').update(archive).digest('hex')
  let downloads = 0
  const downloader = { download: async (_url, destination) => { downloads++; await fs.writeFile(destination, archive, { mode: 0o600 }) } }
  const options = {
    version: '42.11.12', arch: 'arm64', platformName: 'darwin', artifactName: 'electron', cacheDir: tmp,
    electronDownload: { mirrorOptions: { mirror: 'https://artifact.example.test/' }, checksums: { 'electron-v42.11.12-darwin-arm64.zip': digest }, downloader, tempDirectory: tmp },
  }
  const first = await downloadElectronArtifactZip(options)
  assert.deepEqual(await fs.readFile(first), archive)
  assert.equal(downloads, 1)
  const cached = await downloadElectronArtifactZip(options)
  assert.deepEqual(await fs.readFile(cached), archive)
  assert.equal(downloads, 1, 'validated cached artifacts do not download again')
  await assert.rejects(() => downloadElectronArtifactZip({ ...options, cacheDir: path.join(tmp, 'bad'), electronDownload: { ...options.electronDownload, checksums: { 'electron-v42.11.12-darwin-arm64.zip': '0'.repeat(64) } } }), /checksum|digest|hash/i, 'the replacement downloader must still reject wrong checksums')
  console.log('real builder download, validated cache and rejected checksum compatibility passed')
} finally {
  await fs.rm(tmp, { recursive: true, force: true })
}
