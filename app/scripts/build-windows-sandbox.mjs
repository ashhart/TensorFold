import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { createHash } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { assertWindowsPe } from './windows-engine-checks.mjs'

const run = promisify(execFile)
const root = fileURLToPath(new URL('..', import.meta.url))
export async function buildWindowsSandbox(compiler = process.env.TENSORFOLD_SANDBOX_CC || 'zig') {
  const destination = path.join(root, 'build/sandbox/windows-x64')
  await mkdir(destination, { recursive: true })
  const temporary = path.join(destination, 'TensorFoldSandbox.next.exe'), output = path.join(destination, 'TensorFoldSandbox.exe')
  const sources = ['main.c', 'policy.c', 'process.c', 'export.c'].map(name => path.join(root, 'native/windows-sandbox', name))
  try {
    await run(compiler, ['cc', '-target', 'x86_64-windows-gnu', '-O2', '-Wall', '-Wextra', '-Werror', '-municode',
      ...sources, '-luserenv', '-ladvapi32', '-lole32', '-Wl,--stack,33554432', '-o', temporary], { timeout: 120000, maxBuffer: 1000000 })
    const bytes = await readFile(temporary)
    assertWindowsPe(bytes, 'Windows sandbox helper')
    await rename(temporary, output)
    const manifest = { abi: 1, platform: 'win32', arch: 'x64', sha256: createHash('sha256').update(bytes).digest('hex') }
    await writeFile(path.join(destination, 'manifest.json'), JSON.stringify(manifest) + '\n')
    await mkdir(path.join(destination, 'asar'), { recursive: true })
    await writeFile(path.join(destination, 'asar/sandbox-native.json'), JSON.stringify(manifest) + '\n')
    return { output, ...manifest }
  } finally { await rm(temporary, { force: true }) }
}
if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  buildWindowsSandbox().then(result => console.log(JSON.stringify(result)), error => { console.error(error.stderr || error.message); process.exitCode = 1 })
}
