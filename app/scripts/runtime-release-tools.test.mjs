import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { createJiti } from 'jiti'
const jiti = createJiti(import.meta.url)
const { fixture, keys } = await jiti.import('../electron/agent/engine/runtime/test-fixtures.ts')
const f = await fixture()
try {
  const key = path.join(f.root, 'fixture-key.pem')
  await fs.writeFile(key, keys.privateKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o600 })
  await fs.copyFile(f.archive, path.join(f.root, `${f.pkg.id}.tar.gz`))
  const metadata = path.join(f.root, `${f.pkg.id}.json`)
  await fs.writeFile(metadata, JSON.stringify(f.pkg), { mode: 0o600 })
  const output = path.join(f.root, 'runtimes.json')
  const result = spawnSync(process.execPath, ['scripts/sign-runtime-catalog.mjs', '--package', metadata, '--signing-key', key, '--output', output], { encoding: 'utf8' })
  assert.notEqual(result.status, 0)
  assert.match(result.stderr, /does not match the embedded production root/)
  assert.doesNotMatch(result.stderr, /BEGIN PRIVATE KEY|END PRIVATE KEY/)
  await assert.rejects(fs.access(output))
  const badOutput = path.join(f.root, 'bad-output')
  const invalid = spawnSync(process.execPath, ['scripts/make-runtime-package.mjs', '--source', f.source, '--output', badOutput, '--id', '../escape', '--version', '1.1.7', '--platform', process.platform, '--arch', process.arch, '--backend', 'metal', '--min-app', '1.0.0-beta.10', '--url', f.pkg.url], { encoding: 'utf8' })
  assert.notEqual(invalid.status, 0); assert.match(invalid.stderr, /verified catalog/)
  await assert.rejects(fs.access(badOutput), 'invalid IDs cannot write outside the output directory')
} finally { await f.cleanup() }
console.log('Runtime publisher keeps keys off build hosts and rejects untrusted signing keys and escaping output IDs.')
