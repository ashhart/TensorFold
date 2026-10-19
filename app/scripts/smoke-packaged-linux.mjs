#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { lstat, mkdtemp, readFile, realpath, rm, stat } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

// Smoke the packaged Linux desktop app: isolated profile, loopback-only headless server, health probe,
// then a bounded shutdown and exact-profile cleanup. No inference engine is expected in the artifact.

export const LINUX_SMOKE_PROFILE_PREFIX = 'tensorfold-linux-smoke-'
const READY_TIMEOUT_MS = 45_000
const SHUTDOWN_TIMEOUT_MS = 8_000

export function assertSafeLinuxSmokeTemp(tempParent, protectedHomes) {
  if (!path.isAbsolute(tempParent)) throw new Error('Linux smoke requires an absolute temporary directory.')
  const parent = path.resolve(tempParent)
  if (parent === path.parse(parent).root) throw new Error('Linux smoke refuses to create a profile under the filesystem root.')
  for (const rawHome of protectedHomes) {
    if (!rawHome || !path.isAbsolute(rawHome)) continue
    const home = path.resolve(rawHome)
    if (parent === home) throw new Error('Linux smoke refuses to use a user home directory as its temporary parent.')
    const profile = path.join(home, '.vontra')
    const relative = path.relative(profile, parent)
    if (parent === profile || (relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative))) {
      throw new Error('Linux smoke refuses to create anything in the real TensorFold profile.')
    }
  }
  return parent
}

export function linuxSmokeEnvironment(baseEnv, profileRoot, port) {
  if (!path.isAbsolute(profileRoot)) throw new Error('Linux smoke VONTRA_HOME must be absolute.')
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error('Linux smoke port is invalid.')
  const env = {}
  for (const key of ['HOME', 'PATH', 'SHELL', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TZ', 'XDG_RUNTIME_DIR']) {
    if (typeof baseEnv[key] === 'string' && baseEnv[key]) env[key] = baseEnv[key]
  }
  return { ...env, VONTRA_HOME: profileRoot, VONTRA_HEADLESS: '1', VONTRA_HEADLESS_PORT: String(port) }
}

export async function resolvePackagedLinuxExecutable(root = fileURLToPath(new URL('..', import.meta.url)), platform = process.platform, arch = process.arch) {
  if (platform !== 'linux') throw new Error('The packaged Linux smoke command can run only on Linux.')
  if (arch !== 'x64') throw new Error('Initial packaged Linux smoke covers x64 only.')
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  // electron-builder names the Linux executable from linux.executableName, else package.json name.
  const executable = String(pkg?.build?.linux?.executableName || pkg?.name || '').trim()
  if (!executable || executable.includes('/') || executable.includes('\\')) throw new Error('package.json does not define a packaged Linux executable name.')
  return path.join(root, 'release', 'linux', 'linux-unpacked', executable)
}

async function probe(port) {
  const health = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1_500) })
  const body = await health.json().catch(() => null)
  if (!health.ok || body?.ok !== true || body?.listening !== true) throw new Error('health did not confirm ok=true and listening=true')
  const ui = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(1_500) })
  const uiText = await ui.text()
  if (ui.status !== 200 || !uiText.trim()) throw new Error(`UI returned HTTP ${ui.status}${uiText.trim() ? '' : ' with an empty body'}`)
  // Status only: an accidentally authorized snapshot body may contain private state.
  const snapshot = await fetch(`http://127.0.0.1:${port}/v1/snapshot`, { signal: AbortSignal.timeout(1_500) })
  await snapshot.arrayBuffer().catch(() => {})
  if (snapshot.status !== 401) throw new Error(`unauthenticated snapshot returned HTTP ${snapshot.status}, expected 401`)
  return { healthStatus: health.status, uiStatus: ui.status, snapshotStatus: snapshot.status }
}

async function waitForHealthy(port, timeoutMs, child) {
  const deadline = Date.now() + timeoutMs
  let lastFailure = 'listener did not become ready'
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Packaged Linux app exited early (code ${child.exitCode}).`)
    try {
      return await probe(port)
    } catch (error) {
      lastFailure = error?.message || String(error)
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Packaged Linux app did not pass readiness checks on loopback port ${port}: ${lastFailure}.`)
}

export async function runPackagedLinuxSmoke(options = {}) {
  const executable = options.executablePath ?? await resolvePackagedLinuxExecutable()
  const info = await stat(executable).catch(() => null)
  if (!info?.isFile()) throw new Error(`No packaged Linux executable at ${executable}. Run npm run pack:linux:dev first.`)
  const tempParent = assertSafeLinuxSmokeTemp(options.tempDir ?? '/tmp', [os.homedir(), process.env.HOME].filter(Boolean))
  const profile = await realpath(await mkdtemp(path.join(tempParent, LINUX_SMOKE_PROFILE_PREFIX)))
  const port = options.port ?? 20_000 + Math.floor(Math.random() * 20_000)
  let child = null
  try {
    child = spawn(executable, ['--headless-server', '--ozone-platform=headless'], {
      env: linuxSmokeEnvironment(options.baseEnv ?? process.env, profile, port),
      stdio: ['ignore', 'ignore', 'ignore'],
    })
    await waitForHealthy(port, options.readyTimeoutMs ?? READY_TIMEOUT_MS, child)
    const bundled = await lstat(path.join(path.dirname(executable), 'resources'))
    if (!bundled.isDirectory()) throw new Error('Packaged resources directory is missing.')
    return { executable, port, profile }
  } finally {
    if (child && child.exitCode === null) {
      child.kill('SIGTERM')
      const deadline = Date.now() + SHUTDOWN_TIMEOUT_MS
      while (child.exitCode === null && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 100))
      if (child.exitCode === null) child.kill('SIGKILL')
    }
    await rm(profile, { recursive: true, force: true })
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  if (process.argv.length > 2) {
    console.error('The packaged Linux smoke accepts no arguments.')
    process.exitCode = 1
  } else {
    runPackagedLinuxSmoke().then(({ executable, port }) => {
      console.log(`Packaged Linux smoke passed: ${executable} answered /health on 127.0.0.1:${port}.`)
    }, (error) => {
      console.error(error?.message || error)
      process.exitCode = 1
    })
  }
}
