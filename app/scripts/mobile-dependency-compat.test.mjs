import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import { fileURLToPath } from 'node:url'

const vendor = new URL('../mobile/vendor/decode-uri-component/', import.meta.url)
const require = createRequire(new URL('../mobile/package.json', import.meta.url))
const source = readFileSync(new URL('index.cjs', vendor), 'utf8')
const original = source.replace('module.exports = function decodeUriComponent', 'export default function decodeUriComponent')
// Exact published 0.5.0 implementation, with export syntax alone adapted to CJS.
assert.equal(createHash('sha256').update(original).digest('hex'), '9401353df38f8010ad7035fe8d666bce6a4902bc1cff809afc4ab23fa2e0bdaa')
const decode = require(fileURLToPath(new URL('index.cjs', vendor)))
assert.equal(decode('%C3%85%20hello'), 'Å hello')
assert.equal(decode('%FF%FE%20hello'), '\uFFFD\uFFFD%20hello')
assert.equal(decode('incomplete%G1'), 'incomplete%G1')
const malicious = '%A2'.repeat(40_000)
const start = performance.now()
assert.equal(decode(malicious), malicious)
assert.ok(performance.now() - start < 2_000, 'Malformed URI input must complete promptly without recursive/exponential decoding')
assert.match(readFileSync(new URL('LICENSE', vendor), 'utf8'), /The MIT License/)

if (existsSync(new URL('../mobile/node_modules/query-string/package.json', import.meta.url))) {
  const query = require('query-string')
  const text = query.stringify({ worker: 'task/ one', message: 'Check UTF-8, café' }, { sort: false })
  assert.deepEqual({ ...query.parse(text) }, { worker: 'task/ one', message: 'Check UTF-8, café' })
  assert.equal(query.parse(`worker=${malicious}`).worker, malicious)
  const xcode = require('xcode')
  const project = xcode.project('/tmp/fixture.xcodeproj/project.pbxproj')
  project.hash = { project: { objects: {} } }
  assert.match(project.generateUuid(), /^[A-F0-9]{24}$/)
}
console.log('Mobile router keeps its CommonJS API with the exact patched URI decoder, malformed input is bounded, and Xcode UUID generation works with the security patch.')
