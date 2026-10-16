import assert from 'node:assert/strict'
import { existsSync, mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const home = mkdtempSync(path.join(os.tmpdir(), 'vontra-acp-main-required-key-'))
process.env.VONTRA_HOME = home
delete process.env.VONTRA_SECRET_KEY_BASE64
process.env.VONTRA_REQUIRE_EXTERNAL_SECRET_KEY = '1'

const { resolveDefaultProviderModel } = await import('./acp-server-main.ts')

assert.throws(
  () => resolveDefaultProviderModel(),
  /VONTRA_REQUIRE_EXTERNAL_SECRET_KEY.*VONTRA_SECRET_KEY_BASE64 is missing/,
  'standalone ACP aborts instead of silently falling back when managed external-key policy is unsatisfied',
)
assert.equal(existsSync(path.join(home, '.secretkey')), false, 'fail-closed ACP startup does not create a fallback profile key')

console.log('ACP standalone required-external-key fail-closed tests passed')
