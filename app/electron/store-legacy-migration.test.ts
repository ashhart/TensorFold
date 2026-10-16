import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'

// Exercise the actual scheduler/provider/trust consumers and real path/migration
// helpers, substituting only Electron's path lookup with an owned private profile.
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-legacy-migration-'))
const oldHome = process.env.VONTRA_HOME
delete process.env.VONTRA_HOME
try {
  const home = path.join(tmp, 'home')
  const legacy = path.join(tmp, 'legacy')
  await fs.mkdir(home, { recursive: true, mode: 0o700 })
  await fs.mkdir(legacy, { recursive: true, mode: 0o700 })
  const electron = path.join(tmp, 'electron.mjs')
  await fs.writeFile(electron, `export const app={getPath:(kind)=>kind==='home'?${JSON.stringify(home)}:${JSON.stringify(legacy)}};`, { mode: 0o600 })
  const jiti = createJiti(__filename, { alias: { electron }, moduleCache: false, fsCache: false })
  const { loadSchedulerLease, loadProviderStatus, loadSkillTrust } = await jiti.import<any>(path.join(__dirname, 'store/persistence.ts'))
  const primary = (name: string) => path.join(home, '.vontra', 'state', name)

  const leaseFile = path.join(legacy, 'scheduler-lease.json')
  await fs.writeFile(leaseFile, '{', { mode: 0o600 })
  assert.equal(await loadSchedulerLease(), null)
  await assert.rejects(fs.stat(primary('scheduler-lease.json')), { code: 'ENOENT' }, 'a failed legacy read cannot create authoritative fallback state')
  const repaired = { owner: 'synthetic-owner', acquiredAt: 1, heartbeatAt: 2, expiresAt: 3 }
  await fs.writeFile(leaseFile, JSON.stringify(repaired))
  assert.deepEqual(await loadSchedulerLease(), repaired, 'repairing the legacy file remains sufficient for recovery')
  assert.deepEqual(JSON.parse(await fs.readFile(primary('scheduler-lease.json'), 'utf8')), repaired)

  const status = { version: 1, statuses: {} }
  await fs.writeFile(path.join(legacy, 'provider-status.json'), JSON.stringify(status))
  assert.deepEqual(await loadProviderStatus(), status, 'valid legacy JSON still migrates normally')
  // JSON null is a real persisted value, unlike an undefined failed-read sentinel.
  await fs.writeFile(path.join(legacy, 'skill-trust.json'), 'null')
  assert.equal(await loadSkillTrust(), null)
  assert.equal(await fs.readFile(primary('skill-trust.json'), 'utf8'), 'null')
  console.log('legacy state migration corruption/recovery and valid/null controls passed')
} finally {
  if (oldHome === undefined) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = oldHome
  await fs.rm(tmp, { recursive: true, force: true })
}
