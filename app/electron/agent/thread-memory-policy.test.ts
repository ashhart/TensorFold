import assert from 'node:assert/strict'
import { captureThreadMemoryPolicy, effectiveThreadMemoryPolicy, parseThreadMemoryPolicy, assertThreadMemoryTransport, inspectThreadMemoryPolicy } from '../../shared/thread-memory-policy'

assert.deepEqual(parseThreadMemoryPolicy(undefined), { useExisting: true, contribute: true })
for (const useExisting of [false, true]) for (const contribute of [false, true]) {
  const policy = { useExisting, contribute }
  assert.deepEqual(parseThreadMemoryPolicy(policy), policy)
  assert.deepEqual(effectiveThreadMemoryPolicy({ memoryPolicy: policy, incognito: true }), { useExisting, contribute: false })
  const snapshot = captureThreadMemoryPolicy({ memoryPolicy: policy })
  policy.useExisting = !useExisting
  assert.equal(snapshot.useExisting, useExisting)
  assert.ok(Object.isFrozen(snapshot))
}
for (const malformed of [null, true, [], {}, { useExisting: false }, { useExisting: 'false', contribute: false }, { useExisting: false, contribute: false, extra: true }]) {
  assert.throws(() => parseThreadMemoryPolicy(malformed), /memory/i)
}
assert.deepEqual(effectiveThreadMemoryPolicy({ memoryPolicy: { useExisting: true, contribute: true } }, { useExisting: false, contribute: false }), { useExisting: false, contribute: false })
assert.deepEqual(effectiveThreadMemoryPolicy({ memoryPolicy: { useExisting: false, contribute: false } }, { useExisting: true, contribute: true }), { useExisting: false, contribute: false })
const restricted = { memoryPolicy: { useExisting: false, contribute: true } }
assert.throws(() => assertThreadMemoryTransport(restricted, { transport: 'codex-app-server' } as any), /API/)
assert.throws(() => assertThreadMemoryTransport({ ...restricted, orchestratorHostId: 'synthetic-host' }, { transport: 'chat-completions' } as any), /remote/)
assert.doesNotThrow(() => assertThreadMemoryTransport({}, { transport: 'codex-app-server' } as any))
assert.doesNotThrow(() => assertThreadMemoryTransport(restricted, { transport: 'chat-completions' } as any))
console.log('thread memory policy tests passed')

assert.deepEqual(inspectThreadMemoryPolicy({ memoryPolicy: { useExisting: 'false', contribute: true } as any }), { policy: null, revision: 0, problem: 'Saved memory choices need review. Save both choices explicitly before continuing this chat.' })
assert.equal(inspectThreadMemoryPolicy({ memoryPolicyRevision: -1 }).revision, null)
assert.equal(inspectThreadMemoryPolicy({ memoryPolicyRevision: -1 }).policy, null)
assert.deepEqual(inspectThreadMemoryPolicy({}).policy, { useExisting: true, contribute: true })
