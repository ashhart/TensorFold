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
assert.ok(resources.includes('engine'), 'Linux packages the pinned native engine')
assert.ok(!resources.includes('computer-use'), 'no Apple computer-use helper in the Linux artifact')
assert.deepEqual(resources, ['engine', 'voice-notices', 'third-party-notices'], 'Linux includes its own engine and shared notices')
assert.equal(linux.forceCodeSigning, false, 'unsigned Linux development artifacts are allowed')
assert.equal(shared.build.forceCodeSigning, true, 'the Mac release signing requirement is untouched')
assert.equal(shared.build.mac.notarize, true, 'Mac notarization is untouched')
assert.deepEqual(linux.asarUnpack, ['node_modules/onnxruntime-node/bin/napi-v3/linux/${arch}/**/*'])
assert.equal(linux.linux.target[0].target, 'AppImage')
assert.deepEqual(linux.linux.target[0].arch, ['x64'])
assert.deepEqual(linux.files.slice(0, shared.build.files.length), shared.build.files, 'the renderer/main file set matches the shared configuration')
assert.ok(linux.files.includes('!node_modules/**/*.pdb'))
assert.ok(linux.files.includes('!node_modules/onnxruntime-node/bin/napi-v3/{darwin,win32}/**/*'))
assert.equal(linux.electronFuses.onlyLoadAppFromAsar, shared.build.electronFuses.onlyLoadAppFromAsar, 'fuses are not relaxed for Linux')
assert.equal(linux.electronFuses.enableNodeOptionsEnvironmentVariable, false)

const args = linuxPackageArgs(['--dir'], 'linux', 'x64')
assert.deepEqual(args.slice(0, 2), ['--config', 'scripts/linux-builder-config.cjs'])
assert.ok(args.includes('--dir') && args.includes('--x64'))
assert.ok(args.includes('never'), 'packaging never publishes')
assert.ok(linuxPackageArgs(['--appimage'], 'linux', 'x64').includes('AppImage'))
assert.ok(linuxPackageArgs(['--deb'], 'linux', 'arm64').includes('deb'))
assert.equal(linux.deb.maintainer, 'TensorFold')
assert.ok(linux.deb.depends.includes('bubblewrap'), 'Native command and MCP isolation require their kernel sandbox launcher')
assert.ok(linux.deb.depends.includes('libasound2t64 | libasound2'), 'A clean Debian install must resolve Electron audio linking on current and older Ubuntu/Debian hosts')
for (const library of ['libgtk-3-0', 'libnss3', 'libsecret-1-0']) assert.ok(linux.deb.depends.includes(library), 'Explicit dependencies retain the desktop and protected-storage runtimes')
assert.throws(() => linuxPackageArgs(['--dir'], 'darwin', 'arm64'), /Linux x64 or arm64 build host/)
assert.ok(linuxPackageArgs(['--dir'], 'linux', 'arm64').includes('--arm64'))
assert.throws(() => linuxPackageArgs([], 'linux', 'x64'), /exactly --dir or --appimage/)
assert.throws(() => linuxPackageArgs(['--dir', '--appimage'], 'linux', 'x64'), /exactly --dir or --appimage/)
assert.throws(() => linuxPackageArgs(['--publish', 'always'], 'linux', 'x64'), /exactly --dir or --appimage/)

console.log('Linux packaging configuration tests passed')
