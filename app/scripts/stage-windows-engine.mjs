import { assertWindowsPe, embeddedCudaImageCount, assertWindowsEngineCapabilities } from './windows-engine-checks.mjs'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'

const source = path.resolve(process.argv[2] || '')
const manifest = JSON.parse(await readFile('build/engine.json', 'utf8'))
const expectedSource = manifest.sourceCommit
if (typeof expectedSource !== 'string' || !/^[a-f0-9]{40}$/.test(expectedSource)) throw new Error('Pin the engine source commit before staging Windows.')
if (process.platform !== 'win32') throw new Error('Stage the native Windows engine on its Windows build host.')
if (execFileSync('git', ['-C', source, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() !== expectedSource) {
  throw new Error(`Windows engine source does not match the pinned ${manifest.version} commit.`)
}
const binary = path.join(source, 'zig-out/native/bin/tensorfold-native.exe')
if (execFileSync(binary, ['--version'], { encoding: 'utf8', timeout: 10_000 }).trim() !== `tensorfold-native ${manifest.version}`) {
  throw new Error('Built Windows engine version does not match the app pin.')
}
const grammar = path.join(source, 'zig-out/native/lib/tfgrammar.dll')
const executableBytes = await readFile(binary), grammarBytes = await readFile(grammar)
assertWindowsPe(executableBytes); assertWindowsPe(grammarBytes, 'Windows grammar library')
const cudaImages = embeddedCudaImageCount(executableBytes)
if (cudaImages < 103) throw new Error('Build the complete pinned CUDA image set before staging Windows.')
const xgrammar = path.resolve(process.argv[3] || path.join(source, 'xgrammar'))
if (execFileSync('git', ['-C', xgrammar, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() !== '97787376faee5ed8466cfad57c99855e4ce2f6aa') throw new Error('Use the pinned xgrammar v0.2.8 source and notices.')
const destination = path.resolve('build/engine')
await mkdir(path.join(destination, 'bin'), { recursive: true })
await copyFile(binary, path.join(destination, 'bin/tensorfold-native.exe'))
await mkdir(path.join(destination, 'lib'), { recursive: true })
// Protocol 1 uses this library name on all non-macOS platforms, including Windows.
await copyFile(grammar, path.join(destination, 'lib/libtfgrammar.so'))
assertWindowsEngineCapabilities(JSON.parse(execFileSync(path.join(destination, 'bin/tensorfold-native.exe'), ['capabilities', '--json'], { encoding: 'utf8', timeout: 10_000 })), manifest.version)
await mkdir(path.join(destination, 'LICENSES'), { recursive: true })
for (const [from, to] of [['LICENSE', 'xgrammar-Apache-2.0.txt'], ['NOTICE', 'xgrammar-NOTICE.txt'], ['3rdparty/dlpack/LICENSE', 'dlpack-Apache-2.0.txt']]) await copyFile(path.join(xgrammar, from), path.join(destination, 'LICENSES', to))
for (const name of ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md']) await copyFile(path.join(source, name), path.join(destination, name))
await writeFile(path.join(destination, 'VERSION'), `${manifest.version}\n`)
await writeFile(path.join(destination, 'BUILD-PROVENANCE.json'), JSON.stringify({
  sourceCommit: expectedSource, version: manifest.version, platform: 'windows-x64',
  sha256: createHash('sha256').update(await readFile(binary)).digest('hex'),
  grammarSha256: createHash('sha256').update(grammarBytes).digest('hex'), cudaImages,
  qualification: 'Native build and CLI checks only; GPU inference requires a Windows NVIDIA device',
}, null, 2) + '\n')
