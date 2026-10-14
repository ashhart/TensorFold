import assert from 'node:assert/strict'
import { parseUpdateSmokeArgs } from './smoke-update-install.mjs'
assert.throws(() => parseUpdateSmokeArgs([]), /required/)
assert.throws(() => parseUpdateSmokeArgs(['--feed', 'https://example.com/update.json']), /cannot be overridden/)
assert.throws(() => parseUpdateSmokeArgs(['--from-app', '/tmp/source', '--target-app', '/tmp/target.app', '--expected-version', '1.0.1', '--out', '/tmp/evidence']), /\.app/)
assert.equal(parseUpdateSmokeArgs(['--from-app', '/tmp/source.app', '--target-app', '/tmp/target.app', '--expected-version', '1.0.1', '--out', '/tmp/evidence'])['expected-version'], '1.0.1')
console.log('Native update smoke requires two real signed bundles and cannot override release trust/feed')
