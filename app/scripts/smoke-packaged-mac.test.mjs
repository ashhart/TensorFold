import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { chmod, lstat, mkdir, mkdtemp, readFile, realpath, rename, rm, stat, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {
  PACKAGED_SMOKE_LOOPBACK_HOST,
  PACKAGED_SMOKE_PROFILE_PREFIX,
  assertSafeTemporaryParent,
  buildPackagedSmokeEnvironment,
  createIsolatedSmokeProfile,
  parsePackagedSmokeArgs,
  probePackagedApp,
  releaseSmokeProfile,
  removeExactSmokeProfile,
  runPackagedMacSmoke,
  selectUnusedLoopbackPort,
  waitForLoopbackListenerClosed,
  waitForPackagedAppReady,
} from './smoke-packaged-mac.mjs'

assert.deepEqual(parsePackagedSmokeArgs([]), {})
assert.throws(
  () => parsePackagedSmokeArgs(['--vontra-home', '/tmp/override']),
  /accepts no arguments or profile overrides/,
  'the CLI has no route for a caller-selected profile',
)
assert.throws(() => assertSafeTemporaryParent('/Users/example', ['/Users/example']), /user home directory/)
assert.throws(() => assertSafeTemporaryParent('/Users/example/.vontra', ['/Users/example']), /real TensorFold profile/)
assert.throws(() => assertSafeTemporaryParent('/Users/example/.vontra/smoke', ['/Users/example']), /real TensorFold profile/)

const inheritedTmp = process.env.TMPDIR
process.env.TMPDIR = '/Users/example/.vontra'
try {
  const ignoresInheritedTmp = await createIsolatedSmokeProfile({ protectedHomes: ['/Users/example'] })
  assert.notEqual(ignoresInheritedTmp.tempParent, '/Users/example/.vontra', 'an inherited TMPDIR cannot select the profile parent')
  await removeExactSmokeProfile(ignoresInheritedTmp)
} finally {
  if (inheritedTmp === undefined) delete process.env.TMPDIR
  else process.env.TMPDIR = inheritedTmp
}

const testTemp = await mkdtemp(path.join(os.tmpdir(), 'vontra-smoke-harness-test-'))
try {
  await chmod(testTemp, 0o700)
  const profile = await createIsolatedSmokeProfile({ tempDir: testTemp, protectedHomes: ['/Users/example'] })
  assert.equal(path.dirname(profile.root), profile.tempParent)
  assert.ok(path.basename(profile.root).startsWith(PACKAGED_SMOKE_PROFILE_PREFIX))
  assert.equal((await lstat(profile.root)).mode & 0o777, 0o700)
  assert.equal(typeof profile.identity.dev, 'bigint')
  assert.equal(typeof profile.identity.ino, 'bigint')
  assert.equal(typeof profile.identity.uid, 'bigint')
  await writeFile(path.join(profile.root, 'child-created-file'), 'ordinary smoke profile changes')
  await removeExactSmokeProfile(profile)
  await assert.rejects(() => stat(profile.root), /ENOENT/)
  assert.equal((await stat(testTemp)).isDirectory(), true, 'cleanup removes only the exact generated profile')

  await assert.rejects(
    () => removeExactSmokeProfile({ root: testTemp, tempParent: path.dirname(testTemp) }),
    /refuses to clean/,
    'cleanup cannot be widened to an arbitrary temporary directory',
  )

  // Retain the original inode so replacement is deterministic even on inode-reusing filesystems.
  // Profiles carry realpathed roots, so compare against the canonical temp (/private/var on macOS).
  const canonicalTemp = await realpath(testTemp)
  const replaced = await createIsolatedSmokeProfile({ tempDir: testTemp, protectedHomes: ['/Users/example'] })
  const retained = path.join(canonicalTemp, 'retained-original')
  assert.equal(path.dirname(replaced.root), canonicalTemp)
  assert.equal(path.dirname(retained), canonicalTemp)
  await rename(replaced.root, retained)
  await mkdir(replaced.root, { mode: 0o700 })
  await assert.rejects(() => removeExactSmokeProfile(replaced), /identity or ownership changed/)
  assert.equal((await stat(replaced.root)).isDirectory(), true, 'an inode-replaced path is left untouched')
  await rm(replaced.root, { recursive: true })

  const retained = await createIsolatedSmokeProfile({ tempDir: testTemp, protectedHomes: ['/Users/example'] })
  const retainedFile = path.join(retained.root, 'retained.txt')
  await writeFile(retainedFile, 'retained native transaction fixture')
  await releaseSmokeProfile(retained)
  await assert.rejects(() => removeExactSmokeProfile(retained), /identity or ownership changed/)
  assert.equal(await readFile(retainedFile, 'utf8'), 'retained native transaction fixture')
  await rm(retained.root, { recursive: true })

  const symlinkReplaced = await createIsolatedSmokeProfile({ tempDir: testTemp, protectedHomes: ['/Users/example'] })
  const victim = path.join(testTemp, 'must-not-delete')
  await mkdir(victim)
  await rm(symlinkReplaced.root, { recursive: true })
  await symlink(victim, symlinkReplaced.root, 'dir')
  await assert.rejects(() => removeExactSmokeProfile(symlinkReplaced), /identity or ownership changed/)
  assert.equal((await stat(victim)).isDirectory(), true, 'a symlink target is never traversed or removed')
  assert.equal((await lstat(symlinkReplaced.root)).isSymbolicLink(), true, 'the swapped symlink is left untouched')
  await rm(symlinkReplaced.root, { force: true })
} finally {
  await rm(testTemp, { recursive: true, force: true })
}

const env = buildPackagedSmokeEnvironment({
  HOME: '/Users/real',
  VONTRA_HOME: '/Users/real/.vontra',
  VONTRA_HEADLESS_PORT: '8765',
  VONTRA_HEADLESS_BIND_HOST: '0.0.0.0',
  VONTRA_HEADLESS_BOOTSTRAP: '1',
  ELECTRON_RUN_AS_NODE: '1',
  NODE_OPTIONS: '--inspect',
  VONTRA_SECRET_KEY_BASE64: 'secret-key',
  VONTRA_REQUIRE_EXTERNAL_SECRET_KEY: '1',
  APPLE_KEYCHAIN_PROFILE: 'notary-profile',
  CSC_NAME: 'Developer ID Application: Example',
  OPENAI_API_KEY: 'provider-secret',
  GITHUB_TOKEN: 'source-token',
  DATABASE_PASSWORD: 'database-secret',
  CLIENT_SECRET: 'oauth-secret',
  AWS_ACCESS_KEY_ID: 'access-key',
  HTTPS_PROXY: 'https://user:password@example.test',
  SSH_AUTH_SOCK: '/private/tmp/agent.sock',
  XPC_SERVICE_NAME: '0',
  XPC_SECRET: 'must-not-cross',
  __CF_USER_TEXT_ENCODING: '0x1F5:0:0',
  __CF_SECRET: 'must-not-cross',
}, '/private/tmp/vontra-packaged-smoke-fixture', 43_210)
assert.equal(env.HOME, '/Users/real', 'HOME remains a normal OS home and is never repurposed as isolation')
assert.equal(env.VONTRA_HOME, '/private/tmp/vontra-packaged-smoke-fixture')
assert.equal(env.VONTRA_HEADLESS_PORT, '43210')
assert.equal(env.VONTRA_HEADLESS_BIND_HOST, PACKAGED_SMOKE_LOOPBACK_HOST)
assert.equal(env.VONTRA_HEADLESS, '1')
assert.equal(env.TMPDIR, '/private/tmp/vontra-packaged-smoke-fixture')
assert.equal(env.XPC_SERVICE_NAME, '0')
assert.equal(env.__CF_USER_TEXT_ENCODING, '0x1F5:0:0')
for (const sensitive of [
  'VONTRA_HEADLESS_BOOTSTRAP', 'ELECTRON_RUN_AS_NODE', 'NODE_OPTIONS',
  'VONTRA_SECRET_KEY_BASE64', 'VONTRA_REQUIRE_EXTERNAL_SECRET_KEY',
  'APPLE_KEYCHAIN_PROFILE', 'CSC_NAME', 'OPENAI_API_KEY', 'GITHUB_TOKEN',
  'DATABASE_PASSWORD', 'CLIENT_SECRET', 'AWS_ACCESS_KEY_ID', 'HTTPS_PROXY', 'SSH_AUTH_SOCK',
  'XPC_SECRET', '__CF_SECRET',
]) assert.equal(sensitive in env, false, `${sensitive} must not cross into the smoke child`)

const selectedPort = await selectUnusedLoopbackPort()
assert.ok(Number.isInteger(selectedPort) && selectedPort > 0)
await waitForLoopbackListenerClosed(selectedPort, { timeoutMs: 100 })

const requested = []
const fetchOk = async (url, init) => {
  requested.push({ url, init })
  const pathname = new URL(url).pathname
  if (pathname === '/health') return new Response('{"ok":true,"listening":true}', { status: 200 })
  if (pathname === '/') return new Response('<!doctype html>', { status: 200 })
  if (pathname === '/v1/snapshot') return new Response('private-state-must-not-be-read', { status: 401 })
  return new Response('', { status: 404 })
}
assert.deepEqual(await probePackagedApp({ port: 43_210, fetchImpl: fetchOk }), {
  healthStatus: 200,
  uiStatus: 200,
  uiNonEmpty: true,
  snapshotStatus: 401,
})
assert.deepEqual(requested.map(({ url }) => new URL(url).pathname), ['/health', '/', '/v1/snapshot'])
assert.ok(requested.every(({ url }) => new URL(url).hostname === PACKAGED_SMOKE_LOOPBACK_HOST))
assert.ok(requested.every(({ init }) => !('Authorization' in init.headers)), 'the snapshot request is explicitly unauthenticated')

await assert.rejects(
  () => probePackagedApp({
    port: 43_210,
    fetchImpl: async (url) => {
      const pathname = new URL(url).pathname
      const body = pathname === '/health' ? '{"ok":true,"listening":true}' : pathname === '/' ? '' : '{}'
      return new Response(body, { status: pathname === '/v1/snapshot' ? 401 : 200 })
    },
  }),
  /UI returned an empty response/,
)
await assert.rejects(
  () => probePackagedApp({
    port: 43_210,
    fetchImpl: async (url) => new Response(
      new URL(url).pathname === '/health' ? '{"ok":true,"listening":true}' : 'x',
      { status: 200 },
    ),
  }),
  /unauthenticated snapshot returned HTTP 200, expected 401/,
)
await assert.rejects(
  () => probePackagedApp({
    port: 43_210,
    fetchImpl: async () => new Response('{"ok":true,"listening":false}', { status: 200 }),
  }),
  /health did not confirm ok=true and listening=true/,
)

let clock = 0
await assert.rejects(
  () => waitForPackagedAppReady({
    port: 43_210,
    monitor: { outcome: null, done: new Promise(() => {}) },
    timeoutMs: 10,
    pollIntervalMs: 5,
    now: () => clock,
    sleep: async (ms) => { clock += ms },
    probe: async () => { throw new Error('Host bearer token: MUST_NOT_APPEAR') },
  }),
  (error) => error instanceof Error && /loopback endpoint was not ready/.test(error.message) && !error.message.includes('MUST_NOT_APPEAR'),
  'arbitrary child/network failure strings are not reflected into harness output',
)

class FakeChild extends EventEmitter {
  constructor() {
    super()
    this.exitCode = null
    this.signalCode = null
    this.kills = []
  }
  kill(signal) {
    this.kills.push(signal)
    queueMicrotask(() => {
      this.exitCode = 0
      this.signalCode = signal
      this.emit('exit', 0, signal)
    })
    return true
  }
}

const harnessTemp = await mkdtemp(path.join(os.tmpdir(), 'vontra-smoke-run-test-'))
let generatedProfile
let spawned
let closedPort
let removedRoot
const fakeChild = new FakeChild()
try {
  const result = await runPackagedMacSmoke({
    executablePath: '/Applications/TensorFold Studio.app/Contents/MacOS/TensorFold Studio',
    baseEnv: { HOME: '/Users/real', VONTRA_HOME: '/Users/real/.vontra' },
    dependencies: {
      validateExecutable: async () => {},
      createProfile: async () => {
        generatedProfile = await createIsolatedSmokeProfile({ tempDir: harnessTemp, protectedHomes: ['/Users/real'] })
        return generatedProfile
      },
      choosePort: async () => 43_211,
      spawnApp: (executable, args, options) => {
        spawned = { executable, args, options }
        return fakeChild
      },
      waitUntilReady: async ({ port, monitor }) => {
        assert.equal(port, 43_211)
        assert.equal(monitor.outcome, null)
        return { healthStatus: 200, uiStatus: 200, uiNonEmpty: true, snapshotStatus: 401 }
      },
      waitUntilClosed: async (port) => { closedPort = port },
      removeProfile: async (profile) => {
        removedRoot = profile.root
        await removeExactSmokeProfile(profile)
      },
    },
  })
  assert.equal(result.port, 43_211)
  assert.deepEqual(spawned.args, ['--headless-server', '--use-mock-keychain'])
  assert.deepEqual(spawned.options.stdio, ['ignore', 'ignore', 'ignore'], 'bootstrap credentials cannot reach parent output')
  assert.equal(spawned.options.env.HOME, '/Users/real')
  assert.equal(spawned.options.env.VONTRA_HOME, generatedProfile.root)
  assert.notEqual(spawned.options.env.VONTRA_HOME, '/Users/real/.vontra')
  assert.equal(spawned.options.env.VONTRA_HEADLESS_BIND_HOST, PACKAGED_SMOKE_LOOPBACK_HOST)
  assert.equal(spawned.options.env.VONTRA_HEADLESS_PORT, '43211')
  assert.deepEqual(fakeChild.kills, ['SIGTERM'])
  assert.equal(closedPort, 43_211)
  assert.equal(removedRoot, generatedProfile.root)
  await assert.rejects(() => stat(generatedProfile.root), /ENOENT/)
} finally {
  await rm(harnessTemp, { recursive: true, force: true })
}

const failureTemp = await mkdtemp(path.join(os.tmpdir(), 'vontra-smoke-failure-test-'))
try {
  const failureCases = [
    {
      name: 'health',
      expected: /health returned HTTP 503/,
      fetchImpl: async (url) => new Response('x', {
        status: new URL(url).pathname === '/health' ? 503 : 200,
      }),
    },
    {
      name: 'authentication',
      expected: /unauthenticated snapshot returned HTTP 200, expected 401/,
      fetchImpl: async (url) => new Response(
        new URL(url).pathname === '/health' ? '{"ok":true,"listening":true}' : 'x',
        { status: 200 },
      ),
    },
  ]
  for (const [index, failureCase] of failureCases.entries()) {
    let failureProfile
    let failureRemoved = false
    let failureListenerChecked = false
    const failureChild = new FakeChild()
    await assert.rejects(
      () => runPackagedMacSmoke({
        executablePath: '/Applications/TensorFold Studio.app/Contents/MacOS/TensorFold Studio',
        dependencies: {
          validateExecutable: async () => {},
          createProfile: async () => {
            failureProfile = await createIsolatedSmokeProfile({ tempDir: failureTemp, protectedHomes: ['/Users/real'] })
            return failureProfile
          },
          choosePort: async () => 43_212 + index,
          spawnApp: () => failureChild,
          waitUntilReady: async ({ port }) => probePackagedApp({ port, fetchImpl: failureCase.fetchImpl }),
          waitUntilClosed: async (port) => {
            assert.equal(port, 43_212 + index)
            failureListenerChecked = true
          },
          removeProfile: async (profile) => {
            assert.equal(profile.root, failureProfile.root)
            failureRemoved = true
            await removeExactSmokeProfile(profile)
          },
        },
      }),
      failureCase.expected,
    )
    assert.equal(failureRemoved, true, `${failureCase.name} failure still removes only the generated profile`)
    assert.equal(failureListenerChecked, true, `${failureCase.name} failure still verifies listener closure`)
    assert.deepEqual(failureChild.kills, ['SIGTERM'], `${failureCase.name} failure still terminates and awaits the child`)
    await assert.rejects(() => stat(failureProfile.root), /ENOENT/)
  }
} finally {
  await rm(failureTemp, { recursive: true, force: true })
}

console.log('packaged macOS smoke harness tests passed')
