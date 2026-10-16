import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { readFileSync, readdirSync, lstatSync } from 'node:fs'
import { createHash } from 'node:crypto'
const root = fileURLToPath(new URL('..', import.meta.url))
const requireMobile = createRequire(path.join(root, 'mobile/package.json'))
const braces = requireMobile('braces')
const sprintf = requireMobile('sprintf-js').sprintf
const forge = requireMobile('node-forge')
for (const name of ['braces', 'sprintf-js', 'node-forge']) {
  const resolved = requireMobile.resolve(name)
  assert.ok(resolved.includes(`${path.sep}mobile${path.sep}vendor${path.sep}${name}${path.sep}`), `${name} must resolve to the reviewed backport`)
}
assert.deepEqual(braces.expand('item-{a,b}'), ['item-a', 'item-b'])
for (const method of ['compile', 'expand', 'stringify']) {
  let ast = { type: 'text', value: 'ok' }
  for (let i = 0; i < 130; i++) ast = { type: 'root', nodes: [ast] }
  assert.throws(() => braces[method](ast), /safety limits/)
}
assert.throws(() => braces.parse('{'.repeat(130) + 'x' + '}'.repeat(130)), /safety limits/)
assert.equal(sprintf('%.2f %s', 1.25, 'ok'), '1.25 ok')
assert.throws(() => sprintf('%.101f', 1), /safety limits/)
assert.throws(() => sprintf('%1000001s', 'x'), /safety limits/)
// Exercise RSA structure validation using a normal local signature and a
// parser test seam, without generating or accepting forged signatures.
const keys = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 65537 })
const md = forge.md.sha256.create().update('local regression fixture')
const signature = keys.privateKey.sign(md)
const digest = forge.md.sha256.create().update('local regression fixture').digest().getBytes()
assert.equal(keys.publicKey.verify(digest, signature), true)
const originalFromDer = forge.asn1.fromDer
forge.asn1.fromDer = (...args) => {
  const parsed = originalFromDer(...args)
  parsed.value[0].value.push(forge.asn1.create(forge.asn1.Class.UNIVERSAL, forge.asn1.Type.NULL, false, ''))
  return parsed
}
try {
  assert.throws(() => keys.publicKey.verify(digest, signature), /DigestInfo/)
} finally {
  forge.asn1.fromDer = originalFromDer
}
assert.equal(keys.publicKey.verify(digest, signature), true)
const records = JSON.parse(readFileSync(path.join(root, 'mobile/vendor/security-backports.json'), 'utf8'))
assert.equal(records.length, 3)
for (const record of records) {
  const directory = path.join(root, 'mobile/vendor', record.name)
  const seen = new Set()
  function inspect(relative = '') {
    for (const name of readdirSync(path.join(directory, relative))) {
      const rel = relative ? `${relative}/${name}` : name
      const full = path.join(directory, rel)
      const info = lstatSync(full)
      assert.equal(info.isSymbolicLink(), false, `${record.name}/${rel} must not be a link`)
      if (info.isDirectory()) inspect(rel)
      else {
        assert.equal(info.isFile(), true)
        assert.equal(createHash('sha256').update(readFileSync(full)).digest('hex'), record.files[rel], `${record.name}/${rel} differs from the reviewed backport`)
        seen.add(rel)
      }
    }
  }
  inspect()
  assert.deepEqual([...seen].sort(), Object.keys(record.files).sort())
}
console.log('mobile security dependency backports passed: upstream integrity, installed resolution, bounded format/AST, strict RSA structure')
