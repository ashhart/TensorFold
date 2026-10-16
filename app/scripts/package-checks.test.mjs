import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { FUSE_ORDER, assertFuses, readFuseWire, verifyPackagedEngine } from './package-checks.mjs'
import { assertPackagedMacOSMinimum, macOSBuildMinimums } from './macos-runtime-minimum.mjs'

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
  { from: 'build/voice', to: 'voice-notices', filter: ['*.txt'] },
], 'the engine, computer helper and offline voice notices have explicit packaging destinations')
assert.match(await readFile(path.join(root, 'build', 'voice', 'NOTICE.txt'), 'utf8'), /HeadTTS 1\.2\.0/)
assert.match(await readFile(path.join(root, 'build', 'voice', 'CMU-DICTIONARY-BSD.txt'), 'utf8'), /Redistribution and use in source and binary forms/)
assert.match(await readFile(path.join(root, 'build', 'voice', 'HEADTTS-MIT.txt'), 'utf8'), /MIT License/)
assert.match(await readFile(path.join(root, 'build', 'voice', 'APACHE-2.0.txt'), 'utf8'), /Apache License/)
assert.equal(pkg.dependencies['kokoro-js'], '1.2.1')
assert.equal(pkg.dependencies['@met4citizen/headtts'], '1.2.0')
assert.equal(pkg.overrides.phonemizer, '$phonemizer', 'the eSpeak-based upstream phonemizer must stay replaced in every dependency branch')
assert.deepEqual(pkg.build.asarUnpack, ['node_modules/onnxruntime-node/bin/napi-v3/darwin/arm64/*'], 'ONNX native bindings and their adjacent dylib must remain available outside asar')
for (const script of ['pack:mac:dev', 'pack:mac', 'dist:mac']) {
  assert.match(pkg.scripts[script], /node scripts\/fetch-engine\.mjs && node scripts\/build-computer-helper\.mjs && node scripts\/package-mac\.mjs/, `${script} stages the pinned engine and builds the computer-use helper before packaging`)
}
const engineManifest = JSON.parse(await readFile(path.join(root, 'build', 'engine.json'), 'utf8'))
assert.equal(engineManifest.version, '1.0.10')
assert.equal(engineManifest.sha256, '85987885bd3ba11a41ea0e9aea94f6207b09f0df9b1c38d992753e494c06c806')
assert.equal(engineManifest.url, 'https://github.com/ashhart/TensorFold/releases/download/v1.0.10/tensorfold-1.0.10-macos-arm64.tar.gz')
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

const nativeBuild = (minimum, platform = 'MACOS') => `Load command 1\n cmd LC_BUILD_VERSION\n platform ${platform}\n minos ${minimum}\n sdk 26.5\n ntools 1\n tool CLANG\n version 17.0\n`
assert.deepEqual(macOSBuildMinimums(nativeBuild('13.3')), ['13.3'])
assert.deepEqual(macOSBuildMinimums(`${nativeBuild('12.0')}\n${nativeBuild('13.3')}`), ['12.0', '13.3'])
assert.deepEqual(macOSBuildMinimums('Load command 1\n cmd LC_VERSION_MIN_MACOSX\n version 10.15\n sdk 11.0\n'), ['10.15'])
assert.throws(() => macOSBuildMinimums(nativeBuild('13.3', 'IOS')), /non-macOS/)
assert.throws(() => macOSBuildMinimums(nativeBuild('not-a-version')), /Invalid minimum/)
assert.throws(() => macOSBuildMinimums('cmd LC_BUILD_VERSION\n platform MACOS\n sdk 26.5'), /Invalid minimum/)
assert.throws(() => macOSBuildMinimums('no load commands'), /no recognized/)

const minimumTmp = await mkdtemp(path.join(os.tmpdir(), 'macos-minimum-test-'))
try {
  const app = path.join(minimumTmp, 'Fixture.app')
  const nested = path.join(app, 'Contents', 'Resources', 'onnxruntime')
  await mkdir(nested, { recursive: true })
  await writeFile(path.join(app, 'Contents', 'Info.plist'), 'fixture plist')
  await writeFile(path.join(nested, 'runtime.dylib'), Buffer.from('cffaedfe00000000', 'hex'))
  await writeFile(path.join(nested, 'notice.txt'), 'not native')
  await symlink(path.join(nested, 'runtime.dylib'), path.join(nested, 'runtime-alias'))
  let declared = '13.0'
  let build = nativeBuild('13.3')
  const calls = []
  const run = async (command, args) => {
    calls.push([command, ...args])
    return { stdout: command.endsWith('plutil') ? declared : build, stderr: '' }
  }
  await assert.rejects(() => assertPackagedMacOSMinimum(app, { run }), /requires macOS 13.3, but the app declares 13.0/)
  declared = pkg.build.mac.minimumSystemVersion
  const ok = await assertPackagedMacOSMinimum(app, { run })
  assert.equal(ok.declaredMinimum, '13.3')
  assert.equal(ok.binaries.length, 1, 'regular native files are checked once, including non-executable dylibs')
  assert.ok(calls.every(call => call[0] === '/usr/bin/plutil' || call[0] === '/usr/bin/xcrun'), 'native code is never executed')
  build = `${nativeBuild('12.0')}\n${nativeBuild('14.0')}`
  await assert.rejects(() => assertPackagedMacOSMinimum(app, { run }), /requires macOS 14.0/)
  build = nativeBuild('13.3', 'IOS')
  await assert.rejects(() => assertPackagedMacOSMinimum(app, { run }), /non-macOS/)
  declared = ''
  await assert.rejects(() => assertPackagedMacOSMinimum(app, { run }), /Invalid minimum/)
  declared = '13.3'
  await rm(path.join(nested, 'runtime.dylib'))
  await assert.rejects(() => assertPackagedMacOSMinimum(app, { run }), /no native binaries/)
} finally {
  await rm(minimumTmp, { recursive: true, force: true })
}
console.log('macOS native minimum-version checks passed')
