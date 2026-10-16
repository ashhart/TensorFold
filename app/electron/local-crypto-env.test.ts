import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import path from 'node:path'

// Module-load confinement needs a fresh process: once local-crypto is imported,
// the captured value is intentionally immutable for the lifetime of that process.
const key = Buffer.alloc(32, 19).toString('base64')
const script = `(async () => {
  const imported = await import(${JSON.stringify(path.resolve('electron/local-crypto.ts'))});
  const mod = imported.default ?? imported;
  if (process.env.VONTRA_SECRET_KEY_BASE64) throw new Error('credential remained in process.env');
  if (mod.capturedExternalSecretKeyValue() !== ${JSON.stringify(key)}) throw new Error('captured value mismatch');
  if (mod.configuredLocalSecretKeyMode() !== 'external-env-key') throw new Error('wrong key mode');
})().catch((error) => { console.error(error); process.exit(1) })`
const result = spawnSync(process.execPath, ['--import', 'jiti/register', '--eval', script], {
  cwd: process.cwd(),
  env: { ...process.env, VONTRA_SECRET_KEY_BASE64: key },
  encoding: 'utf8',
})
assert.equal(result.status, 0, `${result.stderr || result.stdout}`)

console.log('local-crypto-env.test.ts passed')
