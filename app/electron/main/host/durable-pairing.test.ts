import assert from 'node:assert/strict'
import { createPairingRegistry } from '../../agent/pairing-sessions'
import { durablePairingMutation } from './durable-pairing'

let token = 0
const registry = createPairingRegistry({ newToken: () => `token-${++token}` })

await assert.rejects(
  () => durablePairingMutation(registry, () => registry.issuePairingLink('Phone'), async () => { throw new Error('disk full') }),
  /disk full/,
)
assert.deepEqual(registry.snapshot(), { links: [], sessions: [] }, 'a pairing link is not acknowledged/in-memory after its protected save fails')

const link = await durablePairingMutation(registry, () => registry.issuePairingLink('Phone'), async () => {})
const beforeRedeem = registry.snapshot()
await assert.rejects(
  () => durablePairingMutation(registry, () => registry.redeemPairingLink(link.credential), async () => { throw new Error('read only') }),
  /read only/,
)
assert.deepEqual(registry.snapshot(), beforeRedeem, 'failed session-token persistence restores the unconsumed link and removes the new session')

// The mutation+save+rollback cycle is serialized per registry. Without this, the first failed
// issue would restore its old snapshot over the second link even after the second save succeeded.
{
  let generated = 0
  const concurrent = createPairingRegistry({ newToken: () => `concurrent-${++generated}` })
  let rejectFirst!: (reason?: unknown) => void
  const firstPersist = new Promise<void>((_resolve, reject) => { rejectFirst = reject })
  const first = durablePairingMutation(
    concurrent,
    () => concurrent.issuePairingLink('First'),
    () => firstPersist,
  )
  const second = durablePairingMutation(
    concurrent,
    () => concurrent.issuePairingLink('Second'),
    async () => {},
  )
  await Promise.resolve()
  assert.equal(concurrent.snapshot().links.length, 1, 'second issue waits behind the first durable mutation')
  rejectFirst(new Error('first write failed'))
  await assert.rejects(() => first, /first write failed/)
  const secondLink = await second
  assert.equal(secondLink.credential, 'concurrent-2')
  assert.deepEqual(concurrent.snapshot().links.map((item) => item.label), ['Second'], 'failed rollback cannot erase a later successful issue')
}

// Invalid credentials do not mutate state and therefore do not depend on an unrelated disk write.
{
  const invalid = createPairingRegistry()
  let writes = 0
  const result = await durablePairingMutation(
    invalid,
    () => invalid.redeemPairingLink('not-issued'),
    async () => { writes++ },
    { changed: (redeem) => redeem.ok },
  )
  assert.equal(result.ok, false)
  assert.equal(writes, 0)
}

console.log('durable-pairing.test.ts passed')
