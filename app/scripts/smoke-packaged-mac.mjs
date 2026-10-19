#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { constants } from 'node:fs'
import { lstat, mkdtemp, open, readFile, realpath, rm, stat } from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const PACKAGED_SMOKE_PROFILE_PREFIX = 'vontra-packaged-smoke-'
export const PACKAGED_SMOKE_LOOPBACK_HOST = '127.0.0.1'

const SAFE_OS_ENV_KEYS = new Set([
  'HOME', 'PATH', 'SHELL', 'USER', 'LOGNAME', 'LANG', 'TZ',
  'XPC_SERVICE_NAME', 'XPC_FLAGS', '__CF_USER_TEXT_ENCODING', '__CFBundleIdentifier',
])
const SAFE_OS_ENV_PREFIXES = ['LC_']

const DEFAULT_READY_TIMEOUT_MS = 30_000
const DEFAULT_REQUEST_TIMEOUT_MS = 1_500
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000
const DEFAULT_LISTENER_CLOSE_TIMEOUT_MS = 5_000
const profileHandles = new WeakMap()
// The packaged command is macOS-only. Use the system /tmp root directly instead of
// inheriting TMPDIR, which a caller could point at a non-temporary or profile path.
const DEFAULT_SMOKE_TEMP_DIR = process.platform === 'win32' ? os.tmpdir() : '/tmp'

export function parsePackagedSmokeArgs(argv) {
  if (argv.length > 0) {
    throw new Error('The packaged smoke command accepts no arguments or profile overrides; it always creates its own isolated temporary VONTRA_HOME.')
  }
  return {}
}

export function isPathInside(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate))
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
}

export function assertSafeTemporaryParent(tempParent, protectedHomes) {
  if (!path.isAbsolute(tempParent)) throw new Error('Packaged smoke requires an absolute OS temporary directory.')
  const parent = path.resolve(tempParent)
  if (parent === path.parse(parent).root) throw new Error('Packaged smoke refuses to create its profile directly under the filesystem root.')

  for (const rawHome of protectedHomes) {
    if (!rawHome || !path.isAbsolute(rawHome)) continue
    const home = path.resolve(rawHome)
    const realProfile = path.join(home, '.vontra')
    if (parent === home) throw new Error('Packaged smoke refuses to use a user home directory as its temporary parent.')
    if (parent === realProfile || isPathInside(realProfile, parent)) {
      throw new Error('Packaged smoke refuses to create anything in the real TensorFold profile.')
    }
  }
  return parent
}

export async function createIsolatedSmokeProfile(options = {}) {
  const requestedTemp = options.tempDir ?? DEFAULT_SMOKE_TEMP_DIR
  const homes = options.protectedHomes ?? defaultProtectedHomes()
  const canonicalTemp = await realpath(requestedTemp)
  const canonicalHomes = await Promise.all(homes.map(async (home) => {
    if (!home || !path.isAbsolute(home)) return home
    try { return await realpath(home) } catch { return path.resolve(home) }
  }))
  assertSafeTemporaryParent(canonicalTemp, [...homes, ...canonicalHomes])

  const root = await mkdtemp(path.join(canonicalTemp, PACKAGED_SMOKE_PROFILE_PREFIX))
  const profile = { root: path.resolve(root), tempParent: path.resolve(canonicalTemp) }
  let handle
  try {
    handle = await open(profile.root, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
    await handle.chmod(0o700)
    const [info, entry, canonicalRoot] = await Promise.all([
      handle.stat({ bigint: true }), lstat(profile.root, { bigint: true }), realpath(profile.root),
    ])
    if (!info.isDirectory() || entry.isSymbolicLink() || info.dev !== entry.dev || info.ino !== entry.ino) {
      throw new Error('Packaged smoke profile is not a private directory.')
    }
    if ((info.mode & 0o777n) !== 0o700n) throw new Error('Packaged smoke profile permissions are not mode 0700.')
    if (canonicalRoot !== profile.root || path.dirname(profile.root) !== profile.tempParent || !path.basename(profile.root).startsWith(PACKAGED_SMOKE_PROFILE_PREFIX)) {
      throw new Error('Packaged smoke profile escaped its exact OS temporary directory.')
    }
    const owned = { ...profile, identity: { dev: info.dev, ino: info.ino, uid: info.uid } }
    profileHandles.set(owned, handle)
    return owned
  } catch (error) {
    await handle?.close()
    await rm(profile.root, { recursive: true, force: true })
    throw error
  }
}

export async function removeExactSmokeProfile(profile, remove = rm) {
  assertExactSmokeProfile(profile)
  try {
    let current
    try { current = await lstat(profile.root, { bigint: true }) } catch (error) {
      if (error?.code === 'ENOENT') return
      throw error
    }
    const pinned = await profileHandles.get(profile)?.stat({ bigint: true })
    const expected = profile.identity
    const processUid = typeof process.getuid === 'function' ? BigInt(process.getuid()) : expected.uid
    if (!pinned || !current.isDirectory() || current.isSymbolicLink() ||
        current.dev !== pinned.dev || current.ino !== pinned.ino ||
        current.dev !== expected.dev || current.ino !== expected.ino ||
        current.uid !== expected.uid || current.uid !== processUid ||
        (current.mode & 0o777n) !== 0o700n) {
      throw new Error('Packaged smoke profile identity or ownership changed; refusing cleanup of the replaced path.')
    }
    await remove(profile.root, { recursive: true, force: true })
  } finally {
    await releaseSmokeProfile(profile)
  }
}

export async function releaseSmokeProfile(profile) {
  const handle = profileHandles.get(profile)
  profileHandles.delete(profile)
  await handle?.close()
}

export function buildPackagedSmokeEnvironment(baseEnv, profileRoot, port) {
  if (!path.isAbsolute(profileRoot)) throw new Error('Packaged smoke VONTRA_HOME must be absolute.')
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error('Packaged smoke port is invalid.')
  const env = {}
  // Pass only ordinary OS/session values. Provider credentials, release secrets (APPLE_*/CSC_*),
  // common *_API_KEY/*_TOKEN/*_PASSWORD/*_SECRET variables, proxy credentials, SSH agents,
  // NODE_OPTIONS, Electron debug controls, and every inherited VONTRA_* value are absent by
  // construction. HOME stays the normal OS home and is never repurposed as isolation.
  for (const [key, value] of Object.entries(baseEnv ?? {})) {
    if (typeof value !== 'string') continue
    if (SAFE_OS_ENV_KEYS.has(key) || SAFE_OS_ENV_PREFIXES.some((prefix) => key.startsWith(prefix))) env[key] = value
  }
  return {
    ...env,
    // Keep child-created temporary files inside the exact disposable root as well.
    TMPDIR: profileRoot,
    VONTRA_HOME: profileRoot,
    VONTRA_HEADLESS: '1',
    VONTRA_HEADLESS_PORT: String(port),
    VONTRA_HEADLESS_BIND_HOST: PACKAGED_SMOKE_LOOPBACK_HOST,
  }
}

export async function selectUnusedLoopbackPort(createServer = net.createServer) {
  const server = createServer()
  server.unref?.()
  await new Promise((resolve, reject) => {
    const onError = (error) => {
      server.removeListener('listening', onListening)
      reject(error)
    }
    const onListening = () => {
      server.removeListener('error', onError)
      resolve()
    }
    server.once('error', onError)
    server.once('listening', onListening)
    server.listen({ host: PACKAGED_SMOKE_LOOPBACK_HOST, port: 0, exclusive: true })
  })
  const address = server.address()
  if (!address || typeof address === 'string' || address.address !== PACKAGED_SMOKE_LOOPBACK_HOST || !Number.isInteger(address.port)) {
    await closeServer(server)
    throw new Error('Packaged smoke could not reserve a loopback port.')
  }
  const port = address.port
  await closeServer(server)
  return port
}

export async function probePackagedApp(options) {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch
  const requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
  const baseUrl = `http://${PACKAGED_SMOKE_LOOPBACK_HOST}:${options.port}`
  const request = async (pathname, accept) => {
    const response = await fetchImpl(`${baseUrl}${pathname}`, {
      method: 'GET',
      redirect: 'manual',
      cache: 'no-store',
      headers: { Accept: accept },
      signal: AbortSignal.timeout(requestTimeoutMs),
    })
    return response
  }

  const health = await request('/health', 'application/json')
  if (health.status !== 200) {
    await discardResponse(health)
    throw new Error(`health returned HTTP ${health.status}`)
  }
  let healthPayload
  try { healthPayload = JSON.parse(await readResponseTextBounded(health, 4_096)) } catch {
    throw new Error('health returned invalid JSON')
  }
  if (healthPayload?.ok !== true || healthPayload?.listening !== true) {
    throw new Error('health did not confirm ok=true and listening=true')
  }

  const ui = await request('/', 'text/html')
  if (ui.status !== 200) {
    await discardResponse(ui)
    throw new Error(`UI returned HTTP ${ui.status}`)
  }
  const uiNonEmpty = await responseHasContent(ui)
  if (!uiNonEmpty) throw new Error('UI returned an empty response')

  // Never consume a mistakenly authorized snapshot body. Status is the only evidence
  // needed here, and a 200 body could contain private application state.
  const snapshot = await request('/v1/snapshot', 'application/json')
  const snapshotStatus = snapshot.status
  await discardResponse(snapshot)
  if (snapshotStatus !== 401) throw new Error(`unauthenticated snapshot returned HTTP ${snapshotStatus}, expected 401`)

  return { healthStatus: health.status, uiStatus: ui.status, uiNonEmpty, snapshotStatus }
}

export async function waitForPackagedAppReady(options) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_READY_TIMEOUT_MS
  const pollIntervalMs = options.pollIntervalMs ?? 100
  const now = options.now ?? Date.now
  const sleep = options.sleep ?? delay
  const probe = options.probe ?? (() => probePackagedApp({
    port: options.port,
    requestTimeoutMs: options.requestTimeoutMs,
  }))
  const deadline = now() + timeoutMs
  let lastFailure = 'listener did not become ready'

  while (now() <= deadline) {
    const outcome = options.monitor.outcome
    if (outcome) throw childOutcomeError(outcome)
    try {
      return await probe()
    } catch (error) {
      lastFailure = safeProbeFailure(error)
    }
    if (options.monitor.outcome) throw childOutcomeError(options.monitor.outcome)
    const remaining = deadline - now()
    if (remaining <= 0) break
    await Promise.race([
      sleep(Math.min(pollIntervalMs, remaining)),
      options.monitor.done,
    ])
  }
  throw new Error(`Packaged app did not pass its bounded readiness checks: ${lastFailure}.`)
}

export async function waitForLoopbackListenerClosed(port, options = {}) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_LISTENER_CLOSE_TIMEOUT_MS
  const pollIntervalMs = options.pollIntervalMs ?? 100
  const now = options.now ?? Date.now
  const sleep = options.sleep ?? delay
  const isClosed = options.isClosed ?? (() => isLoopbackPortClosed(port))
  const deadline = now() + timeoutMs
  while (now() <= deadline) {
    if (await isClosed()) return
    const remaining = deadline - now()
    if (remaining <= 0) break
    await sleep(Math.min(pollIntervalMs, remaining))
  }
  throw new Error(`Packaged smoke listener on loopback port ${port} did not close after process termination.`)
}

export async function runPackagedMacSmoke(options) {
  const dependencies = options.dependencies ?? {}
  const createProfile = dependencies.createProfile ?? createIsolatedSmokeProfile
  const choosePort = dependencies.choosePort ?? selectUnusedLoopbackPort
  const spawnApp = dependencies.spawnApp ?? spawn
  const validateExecutable = dependencies.validateExecutable ?? requireExecutable
  const waitUntilReady = dependencies.waitUntilReady ?? waitForPackagedAppReady
  const waitUntilClosed = dependencies.waitUntilClosed ?? waitForLoopbackListenerClosed
  const removeProfile = dependencies.removeProfile ?? removeExactSmokeProfile
  let profile = null
  let port = null
  let child = null
  let monitor = null
  let result = null
  const failures = []

  try {
    await validateExecutable(options.executablePath)
    profile = await createProfile()
    assertExactSmokeProfile(profile)
    port = await choosePort()
    const env = buildPackagedSmokeEnvironment(options.baseEnv ?? process.env, profile.root, port)
    // With the cookie-encryption fuse on, Chromium keys cookies from the login keychain; the mock
    // keychain keeps the smoke from creating or reading a real keychain item.
    child = spawnApp(options.executablePath, ['--headless-server', '--use-mock-keychain'], {
      env,
      stdio: ['ignore', 'ignore', 'ignore'],
      detached: false,
      windowsHide: true,
    })
    monitor = monitorChild(child)
    result = await waitUntilReady({
      port,
      monitor,
      timeoutMs: options.readinessTimeoutMs,
      requestTimeoutMs: options.requestTimeoutMs,
    })
  } catch (error) {
    failures.push(asError(error))
  } finally {
    if (child && monitor) {
      try { await terminateAndAwaitChild(child, monitor, options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS) } catch (error) { failures.push(asError(error)) }
      if (port !== null) {
        try { await waitUntilClosed(port, { timeoutMs: options.listenerCloseTimeoutMs }) } catch (error) { failures.push(asError(error)) }
      }
    }
    if (profile) {
      try { await removeProfile(profile) } catch (error) { failures.push(asError(error)) }
    }
  }

  if (failures.length === 1) throw failures[0]
  if (failures.length > 1) throw new AggregateError(failures, 'Packaged macOS smoke failed and one or more safety cleanup checks also failed.')
  return { ...result, port }
}

export async function resolvePackagedMacExecutable(root = fileURLToPath(new URL('..', import.meta.url)), platform = process.platform, arch = process.arch) {
  if (platform !== 'darwin') throw new Error('The packaged macOS smoke command can run only on macOS.')
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const productName = String(pkg?.build?.productName || pkg?.productName || pkg?.name || '').trim()
  const outputDir = String(pkg?.build?.directories?.output || 'release').trim()
  if (!productName || !outputDir) throw new Error('package.json does not define a packaged macOS product/output path.')
  const appFolder = arch === 'x64' ? 'mac' : `mac-${arch}`
  return path.join(root, outputDir, appFolder, `${productName}.app`, 'Contents', 'MacOS', productName)
}

function defaultProtectedHomes() {
  const values = []
  try { values.push(os.userInfo().homedir) } catch { /* best-effort additional guard */ }
  if (process.env.HOME) values.push(process.env.HOME)
  return [...new Set(values.filter(Boolean))]
}

function assertExactSmokeProfile(profile) {
  if (!profile || !path.isAbsolute(profile.root) || !path.isAbsolute(profile.tempParent)) {
    throw new Error('Packaged smoke received an invalid temporary profile record.')
  }
  const root = path.resolve(profile.root)
  const parent = path.resolve(profile.tempParent)
  assertSafeTemporaryParent(parent, defaultProtectedHomes())
  if (path.dirname(root) !== parent || !path.basename(root).startsWith(PACKAGED_SMOKE_PROFILE_PREFIX)) {
    throw new Error('Packaged smoke refuses to clean a path outside its exact generated temporary profile.')
  }
  if (!profile.identity || typeof profile.identity.dev !== 'bigint' || typeof profile.identity.ino !== 'bigint' || typeof profile.identity.uid !== 'bigint') {
    throw new Error('Packaged smoke refuses a temporary profile without its creation identity.')
  }
}

async function requireExecutable(executablePath) {
  if (!executablePath || !path.isAbsolute(executablePath)) throw new Error('Packaged macOS executable path is invalid.')
  let info
  try { info = await stat(executablePath) } catch { throw new Error(`No packaged macOS app executable exists at ${executablePath}. Run npm run pack:mac:dev first.`) }
  if (!info.isFile()) throw new Error(`Packaged macOS executable is not a file at ${executablePath}.`)
}

function monitorChild(child) {
  let outcome = child.exitCode === null || child.exitCode === undefined
    ? null
    : { type: 'exit', code: child.exitCode, signal: child.signalCode ?? null }
  let resolveDone
  const done = new Promise((resolve) => { resolveDone = resolve })
  const settle = (next) => {
    if (outcome) return
    outcome = next
    resolveDone(next)
  }
  child.once('error', (error) => settle({ type: 'error', code: String(error?.code || 'spawn-error') }))
  child.once('exit', (code, signal) => settle({ type: 'exit', code, signal }))
  if (outcome) resolveDone(outcome)
  return { get outcome() { return outcome }, done }
}

async function terminateAndAwaitChild(child, monitor, timeoutMs) {
  if (monitor.outcome) return monitor.outcome
  try { child.kill('SIGTERM') } catch { /* bounded hard-stop below */ }
  let outcome = await waitForChildOutcome(monitor, timeoutMs)
  if (outcome) return outcome
  try { child.kill('SIGKILL') } catch { /* final bounded wait reports failure */ }
  outcome = await waitForChildOutcome(monitor, timeoutMs)
  if (!outcome) throw new Error('Packaged app did not exit after SIGTERM and SIGKILL.')
  return outcome
}

async function waitForChildOutcome(monitor, timeoutMs) {
  if (monitor.outcome) return monitor.outcome
  const timedOut = Symbol('timed-out')
  let timer
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve(timedOut), timeoutMs)
    timer.unref?.()
  })
  const result = await Promise.race([monitor.done, timeout])
  clearTimeout(timer)
  return result === timedOut ? null : result
}

function childOutcomeError(outcome) {
  if (outcome.type === 'error') return new Error(`Packaged app could not be spawned (${outcome.code}).`)
  return new Error(`Packaged app exited before readiness checks completed (${outcome.signal || (outcome.code ?? 'unknown')}).`)
}

async function responseHasContent(response) {
  if (!response.body?.getReader) return Boolean((await response.text()).length)
  const reader = response.body.getReader()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) return false
      if (value?.byteLength > 0) return true
    }
  } finally {
    try { await reader.cancel() } catch { /* response already closed */ }
  }
}

async function readResponseTextBounded(response, limit) {
  if (!response.body?.getReader) {
    const text = await response.text()
    if (Buffer.byteLength(text) > limit) throw new Error('response exceeded its safety bound')
    return text
  }
  const reader = response.body.getReader()
  const chunks = []
  let length = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (!value?.byteLength) continue
      length += value.byteLength
      if (length > limit) throw new Error('response exceeded its safety bound')
      chunks.push(value)
    }
    const bytes = new Uint8Array(length)
    let offset = 0
    for (const chunk of chunks) {
      bytes.set(chunk, offset)
      offset += chunk.byteLength
    }
    return new TextDecoder().decode(bytes)
  } finally {
    try { await reader.cancel() } catch { /* response already closed */ }
  }
}

async function discardResponse(response) {
  try { await response.body?.cancel?.() } catch { /* response already closed */ }
}

function safeProbeFailure(error) {
  const message = error instanceof Error ? error.message : ''
  if (/^(?:health|UI|unauthenticated snapshot) returned HTTP \d+(?:, expected 401)?$/.test(message) ||
      message === 'UI returned an empty response' || message === 'health returned invalid JSON' ||
      message === 'health did not confirm ok=true and listening=true') return message
  return 'loopback endpoint was not ready'
}

async function isLoopbackPortClosed(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: PACKAGED_SMOKE_LOOPBACK_HOST, port })
    let settled = false
    const finish = (closed) => {
      if (settled) return
      settled = true
      socket.destroy()
      resolve(closed)
    }
    socket.setTimeout(250)
    socket.once('connect', () => finish(false))
    socket.once('error', () => finish(true))
    socket.once('timeout', () => finish(false))
  })
}

function closeServer(server) {
  return new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function asError(error) {
  return error instanceof Error ? error : new Error(String(error))
}

async function main() {
  parsePackagedSmokeArgs(process.argv.slice(2))
  const executablePath = await resolvePackagedMacExecutable()
  const result = await runPackagedMacSmoke({ executablePath })
  console.log(`Packaged macOS headless smoke passed on loopback port ${result.port}: health 200, UI 200/nonempty, unauthenticated snapshot 401; isolated profile removed and listener closed.`)
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  main().catch((error) => {
    console.error(`[smoke-packaged-mac] ${error?.message || error}`)
    process.exitCode = 1
  })
}
