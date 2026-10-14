import assert from 'node:assert/strict'
import { MEMORY_QUALIFICATION_PROFILES_GIB, parseMemoryQualificationArgs, qualificationEnvironment } from './qualify-packaged-memory.mjs'

assert.deepEqual(MEMORY_QUALIFICATION_PROFILES_GIB, [8, 16, 24, 32, 64])
assert.throws(() => parseMemoryQualificationArgs([]), /required/)
assert.throws(() => parseMemoryQualificationArgs(['--app', '/tmp/App', '--out', '/tmp/evidence']), /\.app/)
const env = qualificationEnvironment({ HOME: '/Users/example', PATH: '/usr/bin', OPENAI_API_KEY: 'private', APPLE_API_KEY: 'private', NODE_OPTIONS: '--inspect' }, '/tmp/vontra-packaged-smoke-fixture', 50000, 8)
assert.equal(env.HOME, '/Users/example', 'qualification never repurposes the OS home')
assert.equal(env.VONTRA_HEADLESS, '0')
assert.equal(env.TENSORFOLD_STUDIO_MEMORY_BUDGET_BYTES, String(8 * 1024 ** 3))
assert.equal(env.OPENAI_API_KEY, undefined)
assert.equal(env.APPLE_API_KEY, undefined)
assert.equal(env.NODE_OPTIONS, undefined)
assert.throws(() => qualificationEnvironment({}, '/tmp/fixture', 50000, 256), /fixed/)
console.log('Packaged memory qualification uses fixed lowered budgets and isolated sanitized profiles')
