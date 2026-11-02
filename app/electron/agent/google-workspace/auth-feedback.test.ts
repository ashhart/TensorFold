import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import { requestedScopes } from './scopes'
import type { GoogleWorkspacePrivateState } from './private-state'

const now = 1_800_000_000_000
for (const code of ['invalid_client', 'invalid_grant', 'invalid_scope']) {
  let state: GoogleWorkspacePrivateState = { version: 1,
    client: { clientId: '123456789012-synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' },
    accounts: [{ id: 'account', subject: 'subject', email: 'person@example.com', revision: 1,
      status: 'connected', connectedAt: now, capabilities: ['gmail.read'],
      tokens: { accessToken: 'synthetic-expired', refreshToken: 'synthetic-refresh', expiresAt: now,
        scopes: requestedScopes(['gmail.read']) } }], drafts: [], actions: [] }
  const service = new GoogleWorkspaceService({ now: () => now,
    store: { read: async () => structuredClone(state), write: async next => { state = structuredClone(next) } },
    openExternal: async () => { assert.fail('No real sign-in in this test') },
    lookup: async () => [{ address: '8.8.8.8', family: 4 }],
    fetchImpl: (async () => Response.json({ error: code, error_description: 'synthetic-private-token-and-path' },
      { status: 400 })) as typeof fetch,
  })
  try {
    const expected = code === 'invalid_client' ? /Desktop.*client/i : code === 'invalid_scope' ? /permissions/i : /Reconnect/i
    await assert.rejects(service.listMessages('account'), expected)
    const account = (await service.snapshot()).accounts[0]
    assert.equal(account.status, 'needs-reauthorization')
    assert.match(account.lastError!, expected)
    assert.doesNotMatch(account.lastError!, /synthetic-private/)
  } finally { service.dispose() }
}
console.log('Actual expired Google refresh distinguishes client, revoked grant and permission recovery without server text')
