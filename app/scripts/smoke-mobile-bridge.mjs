#!/usr/bin/env node
import { createRequire } from 'node:module'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url)), require = createRequire(import.meta.url)
const jiti = path.join(path.dirname(require.resolve('jiti/package.json')), 'lib', 'jiti-cli.mjs')
const result = spawnSync(process.execPath, [jiti, path.join(root, 'electron', 'agent', 'mobile-bridge.integration.ts')], { cwd: root, stdio: 'inherit', timeout: 30_000 })
if (result.error) {
  console.error('Private LAN TLS integration did not complete; check the runtime\'s Local Network access and rerun the explicit smoke check.')
  process.exit(1)
}
process.exit(result.status ?? 1)
