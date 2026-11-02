import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { windowsPackageArgs } from './package-windows.mjs'
import { resolvePackagedWindowsExecutable, windowsSmokeEnvironment } from './smoke-packaged-windows.mjs'

const require = createRequire(import.meta.url)
const config = require('./windows-builder-config.cjs')
const shared = require('../package.json').build
assert.ok(windowsPackageArgs(['--dir'], 'win32', 'x64').includes('--dir'))
assert.ok(windowsPackageArgs(['--nsis'], 'win32', 'x64').includes('nsis'))
assert.equal(windowsPackageArgs(['--nsis'], 'win32', 'x64').at(-1), 'never')
assert.throws(() => windowsPackageArgs(['--nsis'], 'darwin', 'arm64'), /Windows x64 build host/)
assert.throws(() => windowsPackageArgs(['--publish', 'always'], 'win32', 'x64'), /exactly --dir or --nsis/)
assert.throws(() => resolvePackagedWindowsExecutable(undefined, 'linux'), /requires Windows/)
assert.deepEqual(config.files.slice(0, shared.files.length), shared.files)
assert.ok(config.files.includes('!node_modules/**/*.pdb'))
assert.ok(config.files.includes('!node_modules/onnxruntime-node/bin/napi-v3/{darwin,linux}/**/*'))
assert.deepEqual(config.electronFuses, shared.electronFuses)
assert.deepEqual(config.extraResources.map(item => item.to), ['engine', 'voice-notices', 'third-party-notices', 'sandbox/TensorFoldSandbox.exe'])
assert.ok(config.files.some(item => typeof item === 'object' && item.from.endsWith('/asar') && item.to === '.'
  && item.filter.includes('sandbox-native.json')), 'The helper pin uses a directory mapping inside the protected ASAR')
assert.equal(typeof config.afterPack, 'function')
assert.equal(config.nsis.deleteAppDataOnUninstall, false)
assert.equal(config.toolsets.nsis, '1.2.1')
assert.equal(shared.forceCodeSigning, true)
const env = windowsSmokeEnvironment({
  SystemRoot: 'C:\\Windows', PATH: 'C:\\Windows\\System32',
  OPENAI_API_KEY: 'synthetic-secret', GITHUB_TOKEN: 'synthetic-token', NODE_OPTIONS: '--inspect', ELECTRON_RUN_AS_NODE: '1',
}, process.cwd(), 43210)
assert.equal(env.SystemRoot, 'C:\\Windows')
assert.equal(env.VONTRA_HOME, process.cwd())
for (const key of ['OPENAI_API_KEY', 'GITHUB_TOKEN', 'NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE']) assert.equal(env[key], undefined)
const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-win-package-pin-'))
try {
  const resources = path.join(temporary, 'resources'), source = path.join(temporary, 'source')
  await fs.mkdir(path.join(resources, 'sandbox'), { recursive: true }); await fs.mkdir(source)
  const bytes = Buffer.from('MZsynthetic fixture'), binary = path.join(resources, 'sandbox/TensorFoldSandbox.exe')
  const manifest = { abi: 1, platform: 'win32', arch: 'x64', sha256: createHash('sha256').update(bytes).digest('hex') }
  await fs.writeFile(binary, bytes)
  await fs.writeFile(path.join(source, 'sandbox-native.json'), JSON.stringify(manifest))
  await require('@electron/asar').createPackage(source, path.join(resources, 'app.asar'))
  await config.afterPack({ appOutDir: temporary })
  await fs.writeFile(binary, 'MZtampered fixture')
  await assert.rejects(config.afterPack({ appOutDir: temporary }), /mismatching helper bytes/)
  await fs.rm(path.join(source, 'sandbox-native.json'))
  await require('@electron/asar').createPackage(source, path.join(resources, 'empty.asar'))
  await fs.rename(path.join(resources, 'empty.asar'), path.join(resources, 'app.asar'))
  require('@electron/asar').uncacheAll()
  await assert.rejects(config.afterPack({ appOutDir: temporary }), /not found/)
} finally { await fs.rm(temporary, { recursive: true, force: true }) }
console.log('Windows package policy, shared assets/fuses, platform refusal and isolated smoke environment passed')
