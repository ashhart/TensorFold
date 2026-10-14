// Run: node electron/store/seeded-cli-accounts.test.ts
import assert from 'node:assert'
import { dropUnusedSeededCliAccounts } from './seeded-cli-accounts.ts'
import { defaultSettings } from './defaults.ts'
import type { Provider } from '../agent/types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

type Account = NonNullable<Provider['accounts']>[number]
function account(id: string, cliHome?: string): Account {
  return { id, name: id, kind: 'subscription', enabled: true, cliHome, models: [], createdAt: 1, updatedAt: 1 }
}
function codex(accounts: Account[], defaultAccountId: string | null): Provider {
  return { id: 'codex-cli', name: 'Codex CLI OAuth', transport: 'codex-cli', baseUrl: '', apiKey: '', models: [], accounts, defaultAccountId }
}

{
  const defaults = defaultSettings().providers
  for (const id of ['codex-cli', 'claude-cli']) {
    const p = defaults.find((x) => x.id === id)!
    assert.deepEqual(p.accounts, [])
    assert.equal(p.defaultAccountId, undefined)
  }
  ok('fresh defaults use the CLI default home: no seeded accounts')
}

{
  const before = [codex([account('codex-personal', '~/.codex-personal'), account('codex-work', '~/.codex-work')], 'codex-personal')]
  const { providers, removedIds } = dropUnusedSeededCliAccounts(before, () => false, '/home/u')
  assert.deepEqual(removedIds, ['codex-personal', 'codex-work'])
  assert.deepEqual(providers[0].accounts, [])
  assert.equal(providers[0].defaultAccountId, null)
  ok('seeded accounts whose home never existed are dropped')
}

{
  const before = [codex([account('codex-personal', '~/.codex-personal'), account('codex-work', '~/.codex-work')], 'codex-personal')]
  const { providers, removedIds } = dropUnusedSeededCliAccounts(before, (p) => p === '/home/u/.codex-personal', '/home/u')
  assert.deepEqual(removedIds, ['codex-work'])
  assert.deepEqual(providers[0].accounts?.map((a) => a.id), ['codex-personal'])
  assert.equal(providers[0].defaultAccountId, 'codex-personal')
  ok('an authorized seeded account (home exists) is kept')
}

{
  const before = [codex([account('codex-personal', '~/.my-codex'), account('team', '~/.codex-work')], 'team')]
  const { providers, removedIds } = dropUnusedSeededCliAccounts(before, () => false, '/home/u')
  assert.deepEqual(removedIds, [])
  assert.equal(providers[0], before[0])
  assert.deepEqual(providers[0].accounts?.map((a) => a.id), ['codex-personal', 'team'])
  ok('user-edited or user-created accounts are never touched')
}

console.log(`\n${passed} passed`)
