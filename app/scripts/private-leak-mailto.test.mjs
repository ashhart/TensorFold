import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
const root = await mkdtemp('/tmp/tensorfold-email-guard-')
try {
  await mkdir(path.join(root, 'scripts'))
  const scanner = path.join(root, 'scripts/check-private-leaks.mjs')
  await writeFile(scanner, await readFile(fileURLToPath(new URL('./check-private-leaks.mjs', import.meta.url))))
  await writeFile(path.join(root, 'scripts/public-vendor-attributions.json'), JSON.stringify({ matches: [] }))
  const fixture = path.join(root, 'fixture.txt')
  const email = ['person', 'blocked.invalid'].join('@')
  const run = async text => {
    await writeFile(fixture, text)
    return spawnSync(process.execPath, [scanner], { encoding: 'utf8', timeout: 10000 })
  }
  const mailto = await run('mailto:' + email)
  assert.equal(mailto.status, 1, 'mailto contacts must be rejected even though a colon precedes the address')
  assert.ok(!mailto.stderr.includes(email), 'the guard must not print the address it rejected')
  assert.equal((await run('Contact:' + email)).status, 1, 'an arbitrary label colon cannot exempt a private address')
  assert.equal((await run('mailto:person@example.test')).status, 0, 'reserved fictional mailto contacts remain valid fixtures')
  const credentialUri = 'postgres://user:' + ['password', 'database.invalid'].join('@') + '/app'
  assert.equal((await run(credentialUri)).status, 0, 'URI user-info belongs to the separate credential scanner, not email classification')
  console.log('email guard CLI regression passed: mailto, arbitrary label, reserved fixture, user-info, redacted diagnostics')
} finally { await rm(root, { recursive: true, force: true }) }
