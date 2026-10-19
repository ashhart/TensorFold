import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { assertSafeLinuxSmokeTemp, linuxSmokeEnvironment, resolvePackagedLinuxExecutable } from './smoke-packaged-linux.mjs'

assert.equal(assertSafeLinuxSmokeTemp('/tmp', ['/home/example']), '/tmp')
assert.throws(() => assertSafeLinuxSmokeTemp('tmp', ['/home/example']), /absolute temporary directory/)
assert.throws(() => assertSafeLinuxSmokeTemp('/', ['/home/example']), /filesystem root/)
assert.throws(() => assertSafeLinuxSmokeTemp('/home/example', ['/home/example']), /user home directory/)
assert.throws(() => assertSafeLinuxSmokeTemp('/home/example/.vontra', ['/home/example']), /real TensorFold profile/)
assert.throws(() => assertSafeLinuxSmokeTemp('/home/example/.vontra/smoke', ['/home/example']), /real TensorFold profile/)
assert.equal(assertSafeLinuxSmokeTemp('/home/example/projects/tmp', ['/home/example']), '/home/example/projects/tmp', 'sibling paths outside the profile are allowed')

const env = linuxSmokeEnvironment({
  HOME: '/home/example', PATH: '/usr/bin', LANG: 'C.UTF-8',
  VONTRA_HOME: '/home/example/.vontra', VONTRA_HEADLESS_PORT: '8765',
  OPENAI_API_KEY: 'provider-secret', AWS_SECRET_ACCESS_KEY: 'aws-secret', GITHUB_TOKEN: 'token',
  NODE_OPTIONS: '--inspect', ELECTRON_RUN_AS_NODE: '1',
}, '/tmp/vontra-linux-smoke-fixture', 43_210)
assert.equal(env.HOME, '/home/example')
assert.equal(env.VONTRA_HOME, '/tmp/vontra-linux-smoke-fixture')
assert.equal(env.VONTRA_HEADLESS, '1')
assert.equal(env.VONTRA_HEADLESS_PORT, '43210')
for (const leaked of ['OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN', 'NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE']) {
  assert.equal(env[leaked], undefined, `${leaked} never crosses into the smoke profile`)
}
assert.throws(() => linuxSmokeEnvironment({}, 'relative/profile', 1234), /VONTRA_HOME must be absolute/)
assert.throws(() => linuxSmokeEnvironment({}, '/tmp/profile', 0), /port is invalid/)

await assert.rejects(() => resolvePackagedLinuxExecutable(undefined, 'darwin', 'arm64'), /only on Linux/)
await assert.rejects(() => resolvePackagedLinuxExecutable(undefined, 'linux', 'arm64'), /x64 only/)

const root = await mkdtemp(path.join(os.tmpdir(), 'tf-linux-smoke-resolve-'))
try {
  await mkdir(path.join(root, 'release'))
  await writeFile(path.join(root, 'package.json'), JSON.stringify({ name: 'tensorfold-studio', build: { productName: 'TensorFold' } }))
  const resolved = await resolvePackagedLinuxExecutable(root, 'linux', 'x64')
  assert.equal(resolved, path.join(root, 'release', 'linux', 'linux-unpacked', 'tensorfold-studio'))
  await writeFile(path.join(root, 'package.json'), JSON.stringify({ name: 'tensorfold-studio', build: { linux: { executableName: 'tensorfold' } } }))
  assert.equal(await resolvePackagedLinuxExecutable(root, 'linux', 'x64'), path.join(root, 'release', 'linux', 'linux-unpacked', 'tensorfold'))
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log('packaged Linux smoke helper tests passed')
