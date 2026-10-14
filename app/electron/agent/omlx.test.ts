import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { discoverOmlx } from './omlx'

async function withRoot(settings: Record<string, any>, run: (root: string) => Promise<void>) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'omlx-test-'))
  try {
    await fs.writeFile(path.join(root, 'settings.json'), JSON.stringify(settings), 'utf8')
    await run(root)
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

async function main() {
  // IPv6 loopback literal must yield a URL parseable by new URL().
  await withRoot({ server: { host: '::1', port: 9000 } }, async (root) => {
    const d = await discoverOmlx(root)
    assert.equal(d.baseUrl, 'http://[::1]:9000/v1')
    const u = new URL(d.baseUrl) // would throw on the malformed 'http://::1:9000/v1'
    assert.equal(u.hostname.replace(/^\[|\]$/g, ''), '::1')
    console.log('ok: IPv6 literal ::1 ->', d.baseUrl)
  })

  // Arbitrary global IPv6 literal also bracketed.
  await withRoot({ server: { host: '2001:db8::1', port: 1234 } }, async (root) => {
    const d = await discoverOmlx(root)
    assert.equal(d.baseUrl, 'http://[2001:db8::1]:1234/v1')
    new URL(d.baseUrl)
    console.log('ok: IPv6 literal 2001:db8::1 ->', d.baseUrl)
  })

  // Already-bracketed input is not double-bracketed.
  await withRoot({ server: { host: '[fe80::1]', port: 9000 } }, async (root) => {
    const d = await discoverOmlx(root)
    assert.equal(d.baseUrl, 'http://[fe80::1]:9000/v1')
    new URL(d.baseUrl)
    console.log('ok: pre-bracketed [fe80::1] ->', d.baseUrl)
  })

  // IPv4 host is left untouched (no spurious brackets).
  await withRoot({ server: { host: '127.0.0.1', port: 9000 } }, async (root) => {
    const d = await discoverOmlx(root)
    assert.equal(d.baseUrl, 'http://127.0.0.1:9000/v1')
    console.log('ok: IPv4 untouched ->', d.baseUrl)
  })

  // Wildcard hosts still normalize to loopback.
  await withRoot({ server: { host: '::', port: 9000 } }, async (root) => {
    const d = await discoverOmlx(root)
    assert.equal(d.baseUrl, 'http://127.0.0.1:9000/v1')
    console.log('ok: :: wildcard -> 127.0.0.1')
  })

  console.log('all omlx tests passed')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
