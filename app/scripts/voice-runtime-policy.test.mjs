import assert from 'node:assert/strict'
import { existsSync, readFileSync, mkdtempSync, mkdirSync, copyFileSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import os from 'node:os'
import { VOICE_REQUIRED_FILES, VOICE_RUNTIME_EXCLUSIONS } from './voice-runtime-policy.mjs'
const app = path.resolve(new URL('..', import.meta.url).pathname)
const req = createRequire(new URL('../package.json', import.meta.url))
const pkg = JSON.parse(readFileSync(path.join(app, 'package.json'), 'utf8'))
const lock = JSON.parse(readFileSync(path.join(app, 'package-lock.json'), 'utf8'))
const filters = new Set(pkg.build.files)
for (const entry of VOICE_RUNTIME_EXCLUSIONS) assert.ok(filters.has('!' + entry), `Missing narrow exclusion ${entry}`)
for (const kind of ['dependencies', 'devDependencies']) assert.deepEqual(pkg[kind], lock.packages[''][kind], `Root lock ${kind} must exactly match the manifest`)
for (const name of ['@types/diff', '@types/qrcode']) {
  assert.equal(pkg.dependencies[name], undefined)
  assert.equal(lock.packages['node_modules/' + name].dev, true)
}
const minimatch = req('minimatch')
const matches = typeof minimatch === 'function' ? minimatch : minimatch.minimatch
for (const [name, files] of Object.entries(VOICE_REQUIRED_FILES)) {
  for (const file of files) {
    const rel = `node_modules/${name}/${file}`
    assert.ok(existsSync(path.join(app, rel)), `Required dependency file missing: ${rel}`)
    assert.ok(!VOICE_RUNTIME_EXCLUSIONS.some(pattern => matches(rel, pattern)), `Required dependency file excluded: ${rel}`)
  }
}
assert.ok([...filters].every(pattern => !/^!node_modules\/(?:onnxruntime-web|onnxruntime-node)(?:\/\*\*)?$/.test(pattern)), 'ONNX packages cannot be removed wholesale')
// Resolve the eager Node and web backend imports without loading ONNX or its native binding.
for (const [from, name, suffix] of [
  ['kokoro-js', '@huggingface/transformers', 'transformers.node.cjs'],
  ['@huggingface/transformers', 'onnxruntime-web', 'ort.node.min.js'],
  ['@huggingface/transformers', 'onnxruntime-node', 'dist/index.js'],
]) {
  const resolveFrom = createRequire(req.resolve(from))
  assert.ok(resolveFrom.resolve(name).endsWith(suffix), `Unexpected retained Node entrypoint for ${name}`)
}
// A real filtered fixture keeps English module-relative imports, dictionary and legal notices.
const fixture = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-voice-manifest-'))
try {
  for (const [name, files] of Object.entries(VOICE_REQUIRED_FILES)) {
    for (const file of files) {
      const target = path.join(fixture, 'node_modules', name, file)
      mkdirSync(path.dirname(target), { recursive: true })
      // Native libraries are verified in place, not copied or executed in this code-only test.
      if (!/\.(?:node|dylib)$/.test(file)) copyFileSync(path.join(app, 'node_modules', name, file), target)
    }
  }
  for (const file of ['language-en-us.mjs', 'language.mjs', 'utils.mjs']) {
    const source = readFileSync(path.join(fixture, 'node_modules/@met4citizen/headtts/modules', file), 'utf8')
    for (const match of source.matchAll(/(?:from|import)\s*[('"\s]+(\.\/[^'"\s)]+)/g)) {
      assert.ok(existsSync(path.resolve(fixture, 'node_modules/@met4citizen/headtts/modules', match[1])), `Filtered English module import missing: ${match[1]}`)
    }
  }
  const fixtureRequire = createRequire(path.join(fixture, 'package.json'))
  assert.ok(fixtureRequire.resolve('@met4citizen/headtts/package.json'))
  assert.ok(existsSync(path.join(fixture, 'node_modules/@met4citizen/headtts/dictionaries/en-us.txt')))
  for (const notice of ['CMU-DICTIONARY-BSD.txt', 'HEADTTS-MIT.txt']) assert.ok(existsSync(path.join(app, 'build/voice', notice)), `Voice notice missing: ${notice}`)
} finally { rmSync(fixture, { recursive: true, force: true }) }
console.log('Voice dependency manifest, retained resolution and filtered English fixture passed; no app or native binding loaded')
