// Run: jiti electron/agent/suggestions.test.ts
import assert from 'node:assert/strict'
import { addSuggestion, createSkillSuggestion, skillSuggestionDiff, isDuplicateSuggestion, acceptSuggestion, dismissSuggestion, pendingSuggestions } from './suggestions.ts'
import type { Skill, SkillSuggestion } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('suggestions tests:')

{
  // Files a fresh suggestion (pending), newest-first.
  const r1 = addSuggestion([], { kind: 'command', name: 'deploy cloudflare', body: 'deploy to cloudflare', rationale: 'you did this' }, {}, 1000, 'a')
  assert.equal(r1.added?.status, 'pending')
  assert.equal(r1.added?.kind, 'command')
  assert.equal(r1.list.length, 1)
  ok('addSuggestion: files a pending suggestion')
}

{
  // Dedup: never re-suggest something that already exists as a skill/command or is pending.
  const corpus = { skills: [{ name: 'Deploy Notes' }], commands: [{ name: 'release' }] }
  assert.equal(isDuplicateSuggestion({ kind: 'command', name: '/release', body: 'x' }, corpus), true, 'matches existing command (slash + case ignored)')
  assert.equal(isDuplicateSuggestion({ kind: 'skill', name: 'deploy-notes', body: 'x' }, corpus), true, 'matches existing skill')
  assert.equal(isDuplicateSuggestion({ kind: 'command', name: 'brand new', body: 'x' }, corpus), false)
  assert.equal(isDuplicateSuggestion({ kind: 'command', name: '', body: 'x' }, {}), true, 'nameless not filed')

  // A second identical pending suggestion is skipped.
  const r1 = addSuggestion([], { kind: 'skill', name: 'How we deploy', body: 'steps' }, {}, 1, 'a')
  const r2 = addSuggestion(r1.list, { kind: 'skill', name: 'how we deploy', body: 'steps' }, {}, 2, 'b')
  assert.equal(r2.added, null, 'duplicate pending skipped')
  assert.equal(r2.list.length, 1)
  ok('dedup: existing skill/command + duplicate pending are skipped')
}

{
  // Accept a command suggestion -> a real Command, suggestion marked accepted (idempotent).
  const seed = addSuggestion([], { kind: 'command', name: 'Ship It', description: 'run the release', body: 'run release.sh', argHint: '<env>', rationale: 'reuse' }, {}, 5, 'sg1').list
  const acc = acceptSuggestion(seed, 'sg1', 10, 'cmd1')
  assert.ok(acc.command, 'creates a Command')
  assert.equal(acc.command?.name, 'ship-it', 'command name normalized for /invocation')
  assert.equal(acc.command?.prompt, 'run release.sh')
  assert.equal(acc.command?.argHint, '<env>')
  assert.equal(acc.list.find((s) => s.id === 'sg1')?.status, 'accepted')
  assert.equal(acc.list.find((s) => s.id === 'sg1')?.acceptedId, 'cmd1')
  // Idempotent: accepting again does nothing.
  const again = acceptSuggestion(acc.list, 'sg1', 11, 'cmd2')
  assert.equal(again.command, undefined, 'second accept is a no-op')
  ok('acceptSuggestion (command): mints a normalized Command, idempotent')
}

{
  // Accept a skill suggestion -> a real enabled Skill.
  const seed = addSuggestion([], { kind: 'skill', name: 'Cloudflare setup', description: 'how to set up CF', body: '1. login\n2. add zone' }, {}, 5, 'sg2').list
  const acc = acceptSuggestion(seed, 'sg2', 10, 'sk1')
  assert.ok(acc.skill, 'creates a Skill')
  assert.equal(acc.skill?.name, 'Cloudflare setup')
  assert.equal(acc.skill?.body, '1. login\n2. add zone')
  assert.equal(acc.skill?.enabled, true)
  ok('acceptSuggestion (skill): mints an enabled Skill')
}

{
  // Dismiss is terminal; a dismissed one no longer blocks the same name from being re-filed.
  const seed = addSuggestion([], { kind: 'skill', name: 'maybe', body: 'x' }, {}, 1, 'd1').list
  const after = dismissSuggestion(seed, 'd1', 2)
  assert.equal(after.find((s) => s.id === 'd1')?.status, 'dismissed')
  assert.equal(pendingSuggestions(after).length, 0)
  const refile = addSuggestion(after, { kind: 'skill', name: 'maybe', body: 'x2' }, {}, 3, 'd2')
  assert.ok(refile.added, 'a dismissed suggestion does not block re-filing later')
  ok('dismissSuggestion: terminal, does not block future re-file')
}

{
  // A skill IMPROVEMENT (targetSkillId set) is NOT blocked by an existing skill of the
  // same name — matching the existing skill is the whole point.
  const corpus = { skills: [{ name: 'Cloudflare setup' }] }
  const blocked = addSuggestion([], { kind: 'skill', name: 'Cloudflare setup', body: 'v2' }, corpus, 1, 'x1')
  assert.equal(blocked.added, null, 'plain skill suggestion still deduped against an existing skill')
  const improve = addSuggestion([], { kind: 'skill', name: 'Cloudflare setup', body: 'v2', targetSkillId: 'sk-cf' }, corpus, 1, 'x2')
  assert.ok(improve.added, 'an improvement of that skill is allowed through')
  // But only one pending improvement per target.
  const dup = addSuggestion(improve.list, { kind: 'skill', name: 'Cloudflare setup', body: 'v3', targetSkillId: 'sk-cf' }, corpus, 2, 'x3')
  assert.equal(dup.added, null, 'a second pending improvement for the same target is deduped')
  ok('improvement suggestions bypass the same-name skill block but dedup per target')
}

{
  // Accepting an improvement rewrites the existing skill in place (same id), not a new one.
  const seed = addSuggestion([], { kind: 'skill', name: 'Cloudflare setup', description: 'sharper', body: 'better body', targetSkillId: 'sk-cf' }, {}, 5, 'imp1').list
  const acc = acceptSuggestion(seed, 'imp1', 10, 'ignored-new-id', new Set(['sk-cf']))
  assert.ok(acc.skill, 'returns a skill to persist')
  assert.equal(acc.skill?.id, 'sk-cf', 'keeps the existing skill id (in-place update)')
  assert.equal(acc.skill?.body, 'better body')
  assert.equal(acc.skill?.description, 'sharper')
  assert.equal(acc.list.find((s) => s.id === 'imp1')?.status, 'accepted')
  ok('acceptSuggestion (improvement): rewrites the targeted skill, preserving its id')
}

{
  const seed = addSuggestion([], { kind: 'skill', name: 'Deleted skill', body: 'stale update', targetSkillId: 'sk-deleted' }, {}, 5, 'imp-deleted').list
  const acc = acceptSuggestion(seed, 'imp-deleted', 10, 'ignored-new-id', new Set(['some-other-skill']))
  assert.equal(acc.skill, undefined, 'a deleted target is not recreated')
  assert.equal(acc.accepted, undefined, 'the stale suggestion is not falsely marked accepted')
  assert.equal(acc.list.find((s) => s.id === 'imp-deleted')?.status, 'pending')
  ok('acceptSuggestion refuses an improvement whose target skill was deleted')
}

// ---- VST-010: what accepting a suggestion would DO ----
{
  const skill = { id: 's1', name: 'deploy', description: 'old desc', body: 'old body', enabled: true, createdAt: 1, updatedAt: 1 } as Skill
  const create = createSkillSuggestion({ kind: 'skill', name: 'new-skill', description: 'fresh', body: 'steps', rationale: 'r' }, 10, 'sg_new')
  const createDiff = skillSuggestionDiff([skill], create)
  assert.equal(createDiff.action, 'create')
  assert.ok(createDiff.changes.some((c) => c.field === 'body' && c.after === 'steps' && c.before === null))

  const update = createSkillSuggestion({ kind: 'skill', name: 'deploy', description: 'old desc', body: 'better body', rationale: 'r', targetSkillId: 's1' }, 20, 'sg_upd')
  const updateDiff = skillSuggestionDiff([skill], update)
  assert.equal(updateDiff.action, 'update')
  assert.equal(updateDiff.targetSkillId, 's1')
  assert.deepEqual(updateDiff.changes.map((c) => c.field), ['body'], 'unchanged fields never appear')
  assert.equal(updateDiff.changes[0].before, 'old body')
  ok('skillSuggestionDiff: create lists fields; update lists only real changes')
}

console.log(`\n${passed} passed`)
