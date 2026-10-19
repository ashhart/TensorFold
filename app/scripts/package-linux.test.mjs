import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { linuxPackageArgs } from './package-linux.mjs'

const require = createRequire(import.meta.url)
const root = fileURLToPath(new URL('..', import.meta.url))
const shared = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
const linux = require(path.join(root, 'scripts/linux-builder-config.cjs'))

assert.equal(linux.extends, null, 'the Linux config never inherits the Mac release configuration')
const resources = linux.extraResources.map((entry) => entry.to)
assert.ok(!resources.includes('engine'), 'no bundled inference engine in the Linux artifact')
assert.ok(!resources.includes('computer-use'), 'no Apple computer-use helper in the Linux artifact')
assert.deepEqual(resources, ['voice-notices', 'third-party-notices'], 'only platform-neutral notices are packaged')
assert.equal(linux.forceCodeSigning, false, 'unsigned Linux development artifacts are allowed')
assert.equal(shared.build.forceCodeSigning, true, 'the Mac release signing requirement is untouched')
assert.equal(shared.build.mac.notarize, true, 'Mac notarization is untouched')
assert.deepEqual(linux.asarUnpack, ['node_modules/onnxruntime-node/bin/napi-v3/linux/x64/**/*'])
assert.equal(linux.linux.target[0].target, 'AppImage')
assert.deepEqual(linux.linux.target[0].arch, ['x64'])
assert.deepEqual(linux.files, shared.build.files, 'the renderer/main file set matches the shared configuration')
assert.equal(linux.electronFuses.onlyLoadAppFromAsar, shared.build.electronFuses.onlyLoadAppFromAsar, 'fuses are not relaxed for Linux')
assert.equal(linux.electronFuses.enableNodeOptionsEnvironmentVariable, false)

const args = linuxPackageArgs(['--dir'], 'linux', 'x64')
assert.deepEqual(args.slice(0, 2), ['--config', 'scripts/linux-builder-config.cjs'])
assert.ok(args.includes('--dir') && args.includes('--x64'))
assert.ok(args.includes('never'), 'packaging never publishes')
assert.ok(linuxPackageArgs(['--appimage'], 'linux', 'x64').includes('AppImage'))
assert.throws(() => linuxPackageArgs(['--dir'], 'darwin', 'arm64'), /Linux x64 build host/)
assert.throws(() => linuxPackageArgs(['--dir'], 'linux', 'arm64'), /Linux x64 build host/)
assert.throws(() => linuxPackageArgs([], 'linux', 'x64'), /exactly --dir or --appimage/)
assert.throws(() => linuxPackageArgs(['--dir', '--appimage'], 'linux', 'x64'), /exactly --dir or --appimage/)
assert.throws(() => linuxPackageArgs(['--publish', 'always'], 'linux', 'x64'), /exactly --dir or --appimage/)

console.log('Linux packaging configuration tests passed')
