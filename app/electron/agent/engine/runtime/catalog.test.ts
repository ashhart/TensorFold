import assert from 'node:assert/strict'
import { verifyRuntimeCatalog, compatibleRuntime, safeRuntimePath, runtimeId } from './catalog'
import { fixture, signed, document, publicKey } from './test-fixtures'
const f = await fixture()
try {
  const doc = document(f.pkg), raw = signed(doc)
  assert.deepEqual(verifyRuntimeCatalog(raw, { publicKey }), doc)
  assert.throws(() => verifyRuntimeCatalog(raw), /signature/, 'a fixture key is never a production trust root')
  for (const platform of ['darwin', 'linux', 'win32'] as const) for (const arch of ['x64', 'arm64'] as const) {
    const p = { ...f.pkg, platform, arch }
    assert.equal(compatibleRuntime(p, '1.0.0-beta.10', platform, arch), true)
    assert.equal(compatibleRuntime(p, '1.0.0-beta.9', platform, arch), false)
    assert.equal(compatibleRuntime(p, '1.0.0-beta.10', 'other', arch), false)
  }
  const newer = document({ ...f.pkg, version: '1.2.1' })
  assert.doesNotThrow(() => verifyRuntimeCatalog(signed(newer), { publicKey }), 'compatible future 1.x releases need no GUI version allowlist')
  assert.equal(compatibleRuntime(newer.packages[0], '1.0.0-beta.10', f.pkg.platform, f.pkg.arch), true)
  const changed = JSON.parse(raw); changed.payload = Buffer.from(JSON.stringify({ ...doc, packages: [] })).toString('base64')
  assert.throws(() => verifyRuntimeCatalog(JSON.stringify(changed), { publicKey }), /signature/)
  for (const mutate of [
    (d: any) => d.expiresAt = Date.now() - 1,
    (d: any) => d.publishedAt = Date.now() + 900_000,
    (d: any) => d.packages[0].protocol = 2,
    (d: any) => d.packages[0].files[0].path = '../escape',
    (d: any) => d.packages[0].files.push({ ...d.packages[0].files[0], path: 'BIN/TENSORFOLD-NATIVE' }),
    (d: any) => d.packages[0].url = 'https://example.test/engine.tar.gz',
    (d: any) => d.packages[0].url = 'https://github.com/ashhart/TensorFold/releases/download/latest/engine.tar.gz',
    (d: any) => d.packages[0].files = d.packages[0].files.filter((x: any) => x.path !== 'NOTICE'),
  ]) { const bad = structuredClone(doc); mutate(bad); assert.throws(() => verifyRuntimeCatalog(signed(bad), { publicKey })) }
  for (const p of ['../escape', '/absolute', 'C:/bad', 'a\\b', 'a//b', 'CON.txt', 'lib/aux', 'a.', 'a:b']) assert.equal(safeRuntimePath(p), false, p)
  for (const id of ['CON', 'foo.', '../bad', '/absolute']) assert.throws(() => runtimeId(id))
  const expired = { ...doc, expiresAt: doc.publishedAt + 1 }
  assert.doesNotThrow(() => verifyRuntimeCatalog(signed(expired), { publicKey, installed: true }), 'installed immutable packages remain usable offline after catalog expiry')
} finally { await f.cleanup() }
console.log('Runtime catalog trust, compatibility and portable paths passed.')
