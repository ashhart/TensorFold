// Run: jiti electron/agent/pairing-sessions.test.ts
//
// Pairing + device sessions (VST-035): one-time pairing links → durable per-device session
// tokens with revoke. Deterministic via injected clock + id/token generators.
import assert from 'node:assert/strict'
import { createPairingRegistry, type PairingRegistryState } from './pairing-sessions'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('pairing-sessions tests:')

// Production defaults must never derive credentials from time/counters. IDs are UUIDs and
// bearer credentials carry 256 bits of base64url CSPRNG output.
{
  const reg = createPairingRegistry()
  const first = reg.issuePairingLink('Phone')
  const second = reg.issuePairingLink('Laptop')
  assert.match(first.credential, /^tok_[A-Za-z0-9_-]{43}$/)
  assert.match(second.credential, /^tok_[A-Za-z0-9_-]{43}$/)
  assert.notEqual(first.credential, second.credential, 'independent links get independent random credentials')
  const session = reg.redeemPairingLink(first.credential).session!
  assert.match(session.id, /^dev_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
  assert.match(session.token, /^tok_[A-Za-z0-9_-]{43}$/)
  assert.notEqual(session.token, first.credential, 'session token is independently random')
  ok('production defaults use UUID/CSPRNG entropy for ids and bearer credentials')
}

// Deterministic clock + sequential id/token generators.
function harness(startAt = 1_000_000) {
  let clock = startAt
  let ids = 0
  let toks = 0
  const reg = createPairingRegistry({
    now: () => clock,
    newId: () => `dev-${++ids}`,
    newToken: () => `tok-${++toks}`,
    linkTtlMs: 600_000, // 10 min
  })
  return { reg, advance: (ms: number) => { clock += ms }, at: () => clock }
}

// --- happy path: issue → redeem → authorize ---
{
  const { reg } = harness()
  const link = reg.issuePairingLink('Alex iPhone')
  assert.equal(link.credential, 'tok-1', 'pairing credential minted')
  const redeem = reg.redeemPairingLink('tok-1', 'iOS Safari')
  assert.equal(redeem.ok, true)
  assert.equal(redeem.session?.id, 'dev-1')
  assert.equal(redeem.session?.token, 'tok-2', 'a fresh session token, distinct from the credential')
  assert.equal(redeem.session?.label, 'Alex iPhone', 'the link label carries to the session')
  const authed = reg.authorizeSessionToken('tok-2')
  assert.deepEqual(authed, { ok: true, sessionId: 'dev-1' }, 'the session token authorizes')
  assert.deepEqual(reg.authorizeSessionToken('tok-1'), { ok: false }, 'the (consumed) pairing credential is NOT a session token')
  assert.deepEqual(reg.authorizeSessionToken('nope'), { ok: false })
  ok('issue → redeem → authorize with a distinct per-device token')
}

// --- one-time: a credential cannot be redeemed twice ---
{
  const { reg } = harness()
  reg.issuePairingLink('Device')
  assert.equal(reg.redeemPairingLink('tok-1').ok, true)
  const second = reg.redeemPairingLink('tok-1')
  assert.equal(second.ok, false)
  assert.equal(second.error, 'already_consumed', 'a consumed pairing link is refused — no second token minted')
  assert.equal(reg.listSessions().length, 1, 'only one session exists')
  ok('pairing link is strictly one-time')
}

// --- expiry: an unredeemed credential dies after its TTL ---
{
  const { reg, advance } = harness()
  reg.issuePairingLink('Slow device')
  advance(600_001)
  const late = reg.redeemPairingLink('tok-1')
  assert.equal(late.ok, false)
  assert.equal(late.error, 'expired')
  assert.equal(reg.authorizeSessionToken('tok-2').ok, false, 'no session token was minted')
  assert.equal(reg.listSessions().length, 0, 'an expired redeem creates no session')
  ok('unredeemed pairing link expires')
}

// --- unknown credential ---
{
  const { reg } = harness()
  assert.equal(reg.redeemPairingLink('never-issued').error, 'unknown_credential')
  ok('unknown credential is refused')
}

// --- revoke a single device (lost phone) ---
{
  const { reg } = harness()
  reg.issuePairingLink('Phone'); const s1 = reg.redeemPairingLink('tok-1').session!
  reg.issuePairingLink('Laptop'); const s2 = reg.redeemPairingLink('tok-3').session!
  assert.equal(reg.authorizeSessionToken(s1.token).ok, true)
  assert.equal(reg.revokeSession(s1.id), true, 'revoke flips a live session')
  assert.equal(reg.authorizeSessionToken(s1.token).ok, false, 'a revoked token no longer authorizes')
  assert.equal(reg.authorizeSessionToken(s2.token).ok, true, 'the OTHER device is unaffected')
  assert.equal(reg.revokeSession(s1.id), false, 'revoking again is a no-op (idempotent)')
  assert.equal(reg.revokeSession('nonexistent'), false)
  const views = reg.listSessions()
  assert.equal(views.find((v) => v.id === s1.id)?.revoked, true)
  assert.ok(!('token' in (views[0] as any)), 'list views never expose the raw token')
  ok('revokeSession: targeted, idempotent, token-safe views')
}

// --- sign out everywhere else ---
{
  const { reg } = harness()
  reg.issuePairingLink('A'); const keep = reg.redeemPairingLink('tok-1').session!
  reg.issuePairingLink('B'); reg.redeemPairingLink('tok-3')
  reg.issuePairingLink('C'); reg.redeemPairingLink('tok-5')
  const revoked = reg.revokeAllexcept(keep.id)
  assert.equal(revoked, 2, 'the other two sessions are revoked')
  assert.equal(reg.authorizeSessionToken(keep.token).ok, true, 'the caller keeps its session')
  assert.equal(reg.listSessions().filter((v) => !v.revoked).length, 1)
  assert.equal(reg.revokeAllexcept(keep.id), 0, 'nothing left to revoke → 0')
  ok('revokeAllexcept: keeps the caller, revokes the rest')
}

// --- lastSeen touch on authorize ---
{
  const { reg, advance } = harness()
  reg.issuePairingLink('Phone'); const s = reg.redeemPairingLink('tok-1').session!
  const created = reg.listSessions()[0].lastSeenAt
  advance(30_000)
  reg.authorizeSessionToken(s.token)
  assert.equal(reg.listSessions()[0].lastSeenAt, created + 30_000, 'authorize refreshes lastSeenAt for the sessions list')
  ok('authorize touches lastSeenAt')
}

// --- prune ---
{
  const { reg, advance } = harness()
  reg.issuePairingLink('unredeemed')       // tok-1, will expire unconsumed
  reg.issuePairingLink('redeemed'); reg.redeemPairingLink('tok-2')  // consumed → session tok-3
  advance(600_001)
  reg.pruneExpiredLinks()
  // Redeeming the now-expired unconsumed link must fail (it was pruned / expired anyway).
  assert.equal(reg.redeemPairingLink('tok-1').ok, false)
  // The consumed link's session is untouched by link pruning.
  assert.equal(reg.authorizeSessionToken('tok-3').ok, true, 'pruning links never touches live sessions')
  ok('pruneExpiredLinks: drops dead links, keeps sessions')
}

// --- clear all live + pending auth material ---
{
  const { reg } = harness()
  reg.issuePairingLink('Phone'); const session = reg.redeemPairingLink('tok-1').session!
  reg.issuePairingLink('Pending')
  assert.equal(reg.clearAll(), 3, 'one consumed link, one pending link, and one session removed')
  assert.equal(reg.authorizeSessionToken(session.token).ok, false)
  assert.deepEqual(reg.snapshot(), { links: [], sessions: [] })
  assert.equal(reg.clearAll(), 0, 'clearing an empty registry is idempotent')
  ok('clearAll removes every pending credential and device session')
}

// --- snapshot / restore ---
{
  const { reg } = harness()
  reg.issuePairingLink('Phone'); const s = reg.redeemPairingLink('tok-1').session!
  const state: PairingRegistryState = reg.snapshot()
  assert.equal(state.sessions.length, 1)
  assert.equal(state.links.length, 1)

  const restored = createPairingRegistry({ now: () => 2_000_000, newId: () => 'dev-x', newToken: () => 'tok-x' }, state)
  assert.equal(restored.authorizeSessionToken(s.token).ok, true, 'a restored registry still authorizes existing device tokens')
  assert.equal(restored.revokeSession(s.id), true, 'and can revoke them')
  // Snapshot is a copy — mutating the restored registry does not change the original snapshot array.
  assert.equal(state.sessions[0].revokedAt, null, 'the taken snapshot is an immutable copy')
  ok('snapshot/restore preserves sessions across restarts')
}

// --- restore() at boot (VST-035 persistence): merge, live-wins, defensive ---
{
  const { reg } = harness()
  reg.issuePairingLink('Phone')
  const live = reg.redeemPairingLink('tok-1').session!
  reg.revokeSession(live.id)

  // A stale disk copy still holds the session UNREVOKED — restore must not resurrect it.
  const stale: PairingRegistryState = { links: [], sessions: [{ ...live, revokedAt: null }] }
  const added = reg.restore(stale)
  assert.equal(added, 0, 'live entries win over the disk copy')
  assert.equal(reg.authorizeSessionToken(live.token).ok, false, 'the revocation sticks')

  // A session only on disk comes back and authorizes.
  const diskOnly: PairingRegistryState = {
    links: [],
    sessions: [{ id: 'disk-1', label: 'Old laptop', token: 'tok-disk', createdAt: 1, lastSeenAt: 1, revokedAt: null }],
  }
  assert.equal(reg.restore(diskOnly), 1)
  assert.equal(reg.authorizeSessionToken('tok-disk').ok, true, 'a restored pairing survives the restart')
  // Restore deep-copies: mutating the source state can't reach the registry.
  diskOnly.sessions[0].revokedAt = 999
  assert.equal(reg.authorizeSessionToken('tok-disk').ok, true)

  // Garbage in -> nothing restored, never a throw.
  assert.equal(reg.restore(null), 0)
  assert.equal(reg.restore({ sessions: [{ id: '', token: '' } as never, 42 as never] }), 0)
  ok('restore: merges disk sessions, live-wins on conflicts, defensive on garbage')
}

console.log(`\n${passed} passed`)
