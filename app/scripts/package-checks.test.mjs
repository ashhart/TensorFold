import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { FUSE_ORDER, assertFuses, readFuseWire, verifyPackagedEngine } from './package-checks.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
const fuses = pkg.build?.electronFuses

// Release fuse requirements: no RunAsNode, NODE_OPTIONS or --inspect; encrypted cookies; asar integrity
// and asar-only loading; file:// keeps its privileges because the renderer loads module chunks from it.
assert.deepEqual(fuses, {
  runAsNode: false,
  enableCookieEncryption: true,
  enableNodeOptionsEnvironmentVariable: false,
  enableNodeCliInspectArguments: false,
  enableEmbeddedAsarIntegrityValidation: true,
  onlyLoadAppFromAsar: true,
  grantFileProtocolExtraPrivileges: true,
})
assert.deepEqual(pkg.build?.extraResources, [
  { from: 'build/engine', to: 'engine', filter: ['**/*'] },
  { from: 'build/computer-use', to: 'computer-use', filter: ['tensorfold-computer'] },
], 'bin/ and lib/ stay side by side under Contents/Resources/engine; only the helper binary ships from build/computer-use')
for (const script of ['pack:mac:dev', 'pack:mac', 'dist:mac']) {
  assert.match(pkg.scripts[script], /node scripts\/fetch-engine\.mjs && node scripts\/build-computer-helper\.mjs && node scripts\/package-mac\.mjs/, `${script} stages the pinned engine and builds the computer-use helper before packaging`)
}
const engineManifest = JSON.parse(await readFile(path.join(root, 'build', 'engine.json'), 'utf8'))
assert.equal(engineManifest.version, '1.0.9')
assert.equal(engineManifest.sha256, '4892b4690df9f1d5c367c5cb4a8e831b1cfbac23d04c9bb8d42fde9b05234fc8')
assert.equal(engineManifest.url, 'https://github.com/ashhart/TensorFold/releases/download/v1.0.9/tensorfold-1.0.9-macos-arm64.tar.gz')
const ignore = await readFile(path.join(root, '.gitignore'), 'utf8')
assert.match(ignore, /^build\/engine\/$/m, 'the staged engine is never committed')

const tmp = await mkdtemp(path.join(os.tmpdir(), 'package-checks-test-'))
try {
  const app = path.join(tmp, 'Fixture.app')
  const framework = path.join(app, 'Contents', 'Frameworks', 'Electron Framework.framework')
  await mkdir(framework, { recursive: true })
  const wire = Buffer.from([1, 9, ...FUSE_ORDER.map((key) => (fuses[key] ?? false) ? 0x31 : 0x30), 0x72])
  await writeFile(path.join(framework, 'Electron Framework'), Buffer.concat([Buffer.alloc(64, 7), Buffer.from('dL7pKGdnNz796PbbjQWNKmHXBZaB9tsX'), wire, Buffer.alloc(16)]))
  const read = await readFuseWire(app)
  assert.equal(read.runAsNode, false)
  assert.equal(read.grantFileProtocolExtraPrivileges, true)
  await assertFuses(app, fuses)
  await assert.rejects(() => assertFuses(app, { ...fuses, runAsNode: true }), /fuse runAsNode reads false/)
  await assert.rejects(() => assertFuses(app, undefined), /electronFuses is required/)

  const manifest = { version: '1.0.7', executables: ['bin/tensorfold-native', 'lib/libtfgrammar.dylib'] }
  await assert.rejects(() => verifyPackagedEngine(app, { manifest, run: async () => ({ stdout: '' }) }), /no bundled engine/)
  const engineRoot = path.join(app, 'Contents', 'Resources', 'engine')
  await mkdir(path.join(engineRoot, 'bin'), { recursive: true })
  await mkdir(path.join(engineRoot, 'lib'), { recursive: true })
  await writeFile(path.join(engineRoot, 'bin', 'tensorfold-native'), 'bin')
  const calls = []
  const run = async (command, args) => { calls.push([command, ...args]); return { stdout: 'tensorfold-native 1.0.7\n', stderr: '' } }
  await assert.rejects(() => verifyPackagedEngine(app, { manifest, run }), /missing the bundled engine file lib\/libtfgrammar.dylib/)
  await symlink('/usr/lib/libSystem.B.dylib', path.join(engineRoot, 'lib', 'libtfgrammar.dylib'))
  await assert.rejects(() => verifyPackagedEngine(app, { manifest, run }), /must be a regular file/)
  await rm(path.join(engineRoot, 'lib', 'libtfgrammar.dylib'))
  await writeFile(path.join(engineRoot, 'lib', 'libtfgrammar.dylib'), 'lib')
  const ok = await verifyPackagedEngine(app, { manifest, run })
  assert.equal(ok.version, 'tensorfold-native 1.0.7')
  assert.deepEqual(calls.map((call) => call.slice(1)), [['--version']], 'dev packaging only runs --version')
  await assert.rejects(() => verifyPackagedEngine(app, { manifest, run: async () => ({ stdout: 'tensorfold-native 1.0.6' }) }), /expected tensorfold-native 1.0.7/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}

console.log('package checks tests passed')
