#!/usr/bin/env node
// Compiles native/computer-use/*.swift into build/computer-use/tensorfold-computer, then runs its --self-test.
// A stamp beside the binary records the source digest and the binary's own sha256; swiftc is skipped only
// when both still match. Sources go to swiftc by relative name so no local build path lands in the binary.
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chmod, lstat, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const HELPER_NAME = 'tensorfold-computer'
export const SWIFT_TARGET = 'arm64-apple-macos13.0'
const MODULE_NAME = 'TensorFoldComputer'
const FRAMEWORKS = ['AppKit', 'ApplicationServices', 'CoreGraphics', 'ScreenCaptureKit', 'Vision']
const STAMP = `.${HELPER_NAME}.stamp.json`
const COMPILE_TIMEOUT_MS = 10 * 60_000

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

export function helperPaths(root) {
  const outDir = path.join(root, 'build', 'computer-use')
  return { sourceDir: path.join(root, 'native', 'computer-use'), outDir, binary: path.join(outDir, HELPER_NAME), stamp: path.join(outDir, STAMP) }
}

export function swiftcArgs(sourceNames, output) {
  return ['swiftc', '-O', '-swift-version', '5', '-module-name', MODULE_NAME, '-target', SWIFT_TARGET, ...FRAMEWORKS.flatMap((name) => ['-framework', name]), '-o', output, ...sourceNames]
}

export async function readSources(sourceDir) {
  const names = (await readdir(sourceDir)).filter((name) => name.endsWith('.swift')).sort()
  if (!names.length) throw new Error(`No Swift sources in ${sourceDir}.`)
  return Promise.all(names.map(async (name) => ({ name, bytes: await readFile(path.join(sourceDir, name)) })))
}

// Names, contents and compiler flags all feed the digest, so renaming a file or changing a flag rebuilds.
export function sourceDigest(sources) {
  const hash = createHash('sha256')
  hash.update(JSON.stringify(swiftcArgs([], 'out')))
  for (const { name, bytes } of sources) {
    hash.update(`\0${name}\0${bytes.length}\0`)
    hash.update(bytes)
  }
  return hash.digest('hex')
}

export async function helperIsCurrent(paths, digest) {
  try {
    const stamp = JSON.parse(await readFile(paths.stamp, 'utf8'))
    if (stamp.digest !== digest) return false
    const info = await lstat(paths.binary)
    if (!info.isFile() || (info.mode & 0o111) === 0) return false
    return sha256(await readFile(paths.binary)) === stamp.binarySha256
  } catch {
    return false
  }
}

export function privateBuildPath(bytes) {
  return /\/(?:Users|home)\/[^/\x00\s]+\//.exec(bytes.toString('latin1'))?.[0] ?? null
}

function defaultRun(command, args, options = {}) {
  return spawnSync(command, args, { cwd: options.cwd, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, timeout: options.timeout ?? COMPILE_TIMEOUT_MS })
}

function output(result) {
  return [result?.stdout, result?.stderr].filter(Boolean).join('\n').trim().split('\n').slice(-40).join('\n')
}

export function parseSelfTest(stdout) {
  const line = String(stdout ?? '').trim().split('\n').at(-1) ?? ''
  try {
    const report = JSON.parse(line)
    return report?.ok === true && Number.isSafeInteger(report.checks) && report.checks > 0 ? report.checks : null
  } catch {
    return null
  }
}

async function removeBuild(paths) {
  await rm(paths.stamp, { force: true })
  await rm(paths.binary, { force: true })
}

export async function buildComputerHelper({ root, force = false, run = defaultRun, log = console.log, platform = process.platform } = {}) {
  if (platform !== 'darwin') throw new Error('The computer-use helper builds only on macOS.')
  const paths = helperPaths(root)
  const sources = await readSources(paths.sourceDir)
  const digest = sourceDigest(sources)
  let compiled = false
  if (force || !(await helperIsCurrent(paths, digest))) {
    const found = run('xcrun', ['--find', 'swiftc'], { timeout: 60_000 })
    if (found.error || found.status !== 0) {
      throw new Error('xcrun could not find swiftc. Install Xcode or the Command Line Tools (xcode-select --install), then run this again.')
    }
    await mkdir(paths.outDir, { recursive: true })
    await removeBuild(paths)
    const temp = `${paths.binary}.tmp-${process.pid}`
    const result = run('xcrun', swiftcArgs(sources.map((source) => source.name), temp), { cwd: paths.sourceDir, timeout: COMPILE_TIMEOUT_MS })
    if (result.error || result.status !== 0) {
      await rm(temp, { force: true })
      throw new Error(`swiftc failed to build ${HELPER_NAME}${result.error ? `: ${result.error.message}` : ''}\n${output(result)}`)
    }
    const leaked = privateBuildPath(await readFile(temp))
    if (leaked) {
      await rm(temp, { force: true })
      throw new Error(`${HELPER_NAME} contains the local build path ${leaked}; refusing to stage it.`)
    }
    await chmod(temp, 0o755)
    await rename(temp, paths.binary)
    compiled = true
  } else {
    log(`[computer-helper] ${HELPER_NAME} is current in build/computer-use.`)
  }
  const test = run(paths.binary, ['--self-test'], { timeout: 60_000 })
  const checks = test.error || test.status !== 0 ? null : parseSelfTest(test.stdout)
  if (!checks) {
    await removeBuild(paths)
    throw new Error(`${HELPER_NAME} --self-test failed${test.error ? `: ${test.error.message}` : ` (exit ${test.status})`}\n${output(test)}`)
  }
  if (compiled) {
    const stamp = { digest, binarySha256: sha256(await readFile(paths.binary)), target: SWIFT_TARGET }
    await writeFile(paths.stamp, `${JSON.stringify(stamp, null, 2)}\n`)
    log(`[computer-helper] built ${HELPER_NAME} for ${SWIFT_TARGET}; self-test passed ${checks} checks.`)
  } else {
    log(`[computer-helper] self-test passed ${checks} checks.`)
  }
  return { binary: paths.binary, compiled, checks }
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  const root = fileURLToPath(new URL('..', import.meta.url))
  buildComputerHelper({ root, force: process.argv.includes('--force') }).catch((error) => {
    console.error(`[computer-helper] ${error?.message || error}`)
    process.exitCode = 1
  })
}
