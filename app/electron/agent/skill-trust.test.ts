// Run: jiti electron/agent/skill-trust.test.ts
//
// Project-skill trust registry (VST-010): hash-pinned trust — untrusted by default,
// trusted content stays trusted, CHANGED content drops out until re-reviewed.
import assert from 'node:assert/strict'
import {
  emptySkillTrustRegistry,
  evaluateSkillTrust,
  normalizeSkillTrustRegistry,
  revokeProjectSkillTrust,
  trustProjectSkill,
} from './skill-trust'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('skill-trust tests:')

{
  const reg = emptySkillTrustRegistry()
  assert.equal(evaluateSkillTrust(reg, 'p1', 'deploy', 'hash-a'), 'untrusted', 'repo content is never trusted by default')
  const trusted = trustProjectSkill(reg, { projectId: 'p1', skillId: 'deploy', contentHash: 'hash-a', now: 1000 })
  assert.equal(evaluateSkillTrust(trusted, 'p1', 'deploy', 'hash-a'), 'trusted')
  assert.equal(evaluateSkillTrust(trusted, 'p2', 'deploy', 'hash-a'), 'untrusted', 'trust is per-project')
  ok('default untrusted → explicit trust, scoped to the project')
}

{
  const reg = trustProjectSkill(emptySkillTrustRegistry(), { projectId: 'p1', skillId: 'deploy', contentHash: 'hash-a', now: 1000 })
  assert.equal(evaluateSkillTrust(reg, 'p1', 'deploy', 'hash-b'), 'changed', 'a repo edit after review is the DANGEROUS case — flagged, not silently run')
  // Re-trusting pins the new content.
  const retrusted = trustProjectSkill(reg, { projectId: 'p1', skillId: 'deploy', contentHash: 'hash-b', now: 2000 })
  assert.equal(evaluateSkillTrust(retrusted, 'p1', 'deploy', 'hash-b'), 'trusted')
  assert.equal(retrusted.records.length, 1, 're-trust replaces, never accumulates')
  ok('content change → changed until re-reviewed; re-trust pins the new hash')
}

{
  const reg = trustProjectSkill(emptySkillTrustRegistry(), { projectId: 'p1', skillId: 'deploy', contentHash: 'hash-a', now: 1000 })
  const revoked = revokeProjectSkillTrust(reg, 'p1', 'deploy')
  assert.equal(evaluateSkillTrust(revoked, 'p1', 'deploy', 'hash-a'), 'untrusted')
  ok('revoke withdraws trust regardless of content')
}

{
  assert.deepEqual(normalizeSkillTrustRegistry(null), { version: 1, records: [] })
  assert.deepEqual(normalizeSkillTrustRegistry('garbage'), { version: 1, records: [] })
  const mixed = normalizeSkillTrustRegistry({
    records: [
      { projectId: 'p1', skillId: 's1', contentHash: 'h', trustedAt: 1 },
      { projectId: '', skillId: 's2', contentHash: 'h', trustedAt: 1 },
      { projectId: 'p1', skillId: 's3', contentHash: '', trustedAt: 1 },
      42,
    ],
  })
  assert.equal(mixed.records.length, 1, 'only well-formed records survive a corrupt file')
  ok('normalize: corrupt registries degrade to empty, never throw')
}

console.log(`  ${passed} passed`)
