import { assertWindowsPe, embeddedCudaImageCount, assertWindowsEngineCapabilities } from './windows-engine-checks.mjs'
import { buildWindowsSandbox } from './build-windows-sandbox.mjs'
import { spawn, execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const root = fileURLToPath(new URL('..', import.meta.url))

export function windowsPackageArgs(args, platform = process.platform, arch = process.arch) {
  if (platform !== 'win32' || arch !== 'x64') throw new Error('Windows packaging requires a Windows x64 build host.')
  if (args.length !== 1 || !['--dir', '--nsis'].includes(args[0])) throw new Error('Choose exactly --dir or --nsis.')
  return ['--config', 'scripts/windows-builder-config.cjs', '--win',
    ...(args[0] === '--dir' ? ['--dir'] : ['nsis']), '--x64', '--publish', 'never']
}

export async function packageWindows(args) {
  const builderArgs = windowsPackageArgs(args)
  await buildWindowsSandbox()
  const manifest = JSON.parse(await readFile(path.join(root, 'build/engine.json'), 'utf8'))
  const binary = path.join(root, 'build/engine/bin/tensorfold-native.exe')
  const bytes = await readFile(binary)
  assertWindowsPe(bytes)
  assertWindowsPe(await readFile(path.join(root, 'build/engine/lib/libtfgrammar.so')), 'Windows grammar library')
  if (embeddedCudaImageCount(bytes) < 103) throw new Error('The pinned Windows engine is missing its complete CUDA image set.')
  for (const name of ['xgrammar-Apache-2.0.txt', 'xgrammar-NOTICE.txt', 'dlpack-Apache-2.0.txt']) await readFile(path.join(root, 'build/engine/LICENSES', name))
  const version = execFileSync(binary, ['--version'], { encoding: 'utf8', timeout: 10_000 }).trim()
  if (version !== `tensorfold-native ${manifest.version}`) throw new Error('The Windows engine does not match the app pin.')
  assertWindowsEngineCapabilities(JSON.parse(execFileSync(binary, ['capabilities', '--json'], { encoding: 'utf8', timeout: 10_000 })), manifest.version)
  const cli = require.resolve('electron-builder/cli.js')
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, ...builderArgs], { cwd: root, env: process.env, stdio: 'inherit' })
    child.once('error', reject)
    child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`Windows packaging failed (${signal ?? code}).`)))
  })
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  packageWindows(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1 })
}
