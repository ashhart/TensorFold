import assert from 'node:assert/strict'
import { createPersistenceFailureReporter } from './persistence'

const published: unknown[] = []
const warnings: string[] = []
const secret = 'sk_live_this_value_must_never_escape_1234567890'
const reporter = createPersistenceFailureReporter({
  now: () => 42,
  publish: (failure) => { published.push(failure) },
  warn: (message) => { warnings.push(message) },
})

const failure = reporter.record('threads', new Error(`disk full; apiKey=${secret}`))
assert.equal(failure.label, 'threads')
assert.equal(failure.at, 42)
assert.match(failure.message, /disk full/)
assert.ok(!JSON.stringify(failure).includes(secret), 'returned failure is redacted')
assert.ok(!JSON.stringify(published).includes(secret), 'renderer notification is redacted')
assert.ok(!warnings.join('\n').includes(secret), 'diagnostic warning is redacted')
assert.deepEqual(reporter.lastFailure(), failure, 'last failure is available to readiness snapshots')

const isolated = createPersistenceFailureReporter({
  publish: () => { throw new Error('renderer has disappeared') },
  warn: () => { throw new Error('console unavailable') },
})
assert.doesNotThrow(() => isolated.record('jobs', new Error('write failed')), 'reporting failures cannot recurse into the persistence path')
assert.equal(isolated.lastFailure()?.label, 'jobs')

console.log('persistence failure visibility + redaction tests passed')
