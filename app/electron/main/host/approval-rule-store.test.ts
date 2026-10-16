import assert from 'node:assert/strict'
import type { ApprovalRule } from '../../agent/types'
import { createDurableApprovalRuleStore } from './approval-rule-store'

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

const now = 10_000
const activeRule = (overrides: Partial<ApprovalRule> = {}): ApprovalRule => ({
  id: 'rule-1',
  scope: 'tool',
  key: 'run:bounded',
  decision: 'allow',
  duration: 'always',
  projectId: 'project-1',
  workspace: null,
  expiresAt: now + 60_000,
  maxUses: 2,
  useCount: 0,
  createdAt: now - 1_000,
  updatedAt: now - 1_000,
  lastUsedAt: null,
  ...overrides,
})

// A bounded rule's in-memory use-count mutation is not authority. The caller cannot receive
// (and therefore cannot apply) the allow until the critical persistence promise settles.
{
  let rules = [activeRule()]
  const disk = deferred<void>()
  const store = createDurableApprovalRuleStore({
    getRules: () => rules,
    setRules: (next) => { rules = next },
    persist: () => disk.promise,
    now: () => now,
    newId: () => 'unused',
  })
  let settled = false
  const lookup = store.find('tool', 'run:bounded', { projectId: 'project-1' }).then((rule) => { settled = true; return rule })
  await Promise.resolve()
  assert.equal(rules[0]!.useCount, 1, 'the use is reserved in memory while persistence is pending')
  assert.equal(settled, false, 'the allow is not returned before its incremented use count is durable')
  disk.resolve()
  assert.equal((await lookup)?.useCount, 1)
}

// A failed consume is fail-closed and restores the approval slice. The persistence layer is
// expected to report the failure; the lookup itself returns null so the normal prompt appears.
{
  let rules = [activeRule()]
  const store = createDurableApprovalRuleStore({
    getRules: () => rules,
    setRules: (next) => { rules = next },
    persist: async () => { throw new Error('disk full') },
    now: () => now,
    newId: () => 'unused',
  })
  assert.equal(await store.find('tool', 'run:bounded', { projectId: 'project-1' }), null)
  assert.equal(rules[0]!.useCount, 0, 'failed durable consumption is rolled back')
}

// Concurrent lookups serialize the complete consume+save cycle. A one-use rule can never be
// handed to two callers merely because both arrived before the first write completed.
{
  let rules = [activeRule({ maxUses: 1 })]
  const firstWrite = deferred<void>()
  let writes = 0
  const store = createDurableApprovalRuleStore({
    getRules: () => rules,
    setRules: (next) => { rules = next },
    persist: () => (++writes === 1 ? firstWrite.promise : Promise.resolve()),
    now: () => now,
    newId: () => 'unused',
  })
  const first = store.find('tool', 'run:bounded', { projectId: 'project-1' })
  const second = store.find('tool', 'run:bounded', { projectId: 'project-1' })
  await Promise.resolve()
  assert.equal(writes, 1, 'the second lookup waits behind the first critical write')
  firstWrite.resolve()
  assert.ok(await first)
  assert.equal(await second, null, 'the exhausted rule is not returned to the queued caller')
  assert.equal(writes, 1, 'an exhausted lookup does not perform a redundant write')
}

// Remembering is also durable-before-acknowledgement. Unlike a lookup (which can fall back to a
// manual prompt), save failure rejects so an allow_always caller cannot execute under a phantom rule.
{
  let rules: ApprovalRule[] = []
  const store = createDurableApprovalRuleStore({
    getRules: () => rules,
    setRules: (next) => { rules = next },
    persist: async () => { throw new Error('read only') },
    now: () => now,
    newId: () => 'rule-new',
  })
  await assert.rejects(
    () => store.remember('tool', 'run:new', 'allow', 'always', { projectId: 'project-1' }),
    /read only/,
  )
  assert.deepEqual(rules, [], 'failed remember rolls the in-memory rule back')
}

console.log('approval-rule-store.test.ts passed')
