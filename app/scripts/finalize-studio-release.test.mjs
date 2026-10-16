import assert from 'node:assert/strict'
import { parseFinalizationArgs } from './finalize-studio-release.mjs'

assert.throws(() => parseFinalizationArgs([]), /notarized app/)
assert.throws(() => parseFinalizationArgs(['--unsigned-dev', '/tmp/example']), /notarized app/)
assert.throws(() => parseFinalizationArgs(['--artifacts', '/tmp/example', '--private-key', '/tmp/another']), /notarized app/)
assert.equal(parseFinalizationArgs(['--artifacts', '/tmp/release-artifacts']), '/tmp/release-artifacts')
console.log('GUI release finalization cannot override signing key, version or trust gates')
