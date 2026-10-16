import assert from 'node:assert/strict'
import {
  approveProjectMemoryCandidate,
  createProjectMemoryCandidate,
  detectProjectMemoryWarnings,
  formatProjectMemoryEntryForPrompt,
  forgetProjectMemoryEntry,
  formatProjectMemoryWarnings,
  projectMemoryCandidateDiff,
  rejectProjectMemoryCandidate,
  rollbackProjectMemoryEntry,
  scopeProjectMemoryCandidates,
  scopeProjectMemoryEntries,
  type ProjectMemorySourceSnapshot,
} from './project-memory.ts'
import type { ProjectMemoryEntry } from './types'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

function memory(patch: Partial<ProjectMemoryEntry> = {}): ProjectMemoryEntry {
  return {
    id: patch.id ?? 'mem_1',
    projectId: patch.projectId ?? 'project_1',
    kind: patch.kind ?? 'decision',
    text: patch.text ?? 'Use AGENTS.md as the shared instruction source.',
    sourceType: patch.sourceType ?? 'file',
    sourceRef: Object.prototype.hasOwnProperty.call(patch, 'sourceRef') ? patch.sourceRef! : 'AGENTS.md',
    sourceIds: patch.sourceIds,
    confidence: patch.confidence ?? 'high',
    privacyScope: patch.privacyScope ?? 'project',
    owner: patch.owner ?? null,
    lastVerifiedAt: patch.lastVerifiedAt ?? 100,
    sourceThreadId: patch.sourceThreadId ?? null,
    sourceTurn: patch.sourceTurn ?? null,
    createdAt: patch.createdAt ?? 1,
    updatedAt: patch.updatedAt ?? 90,
  }
}

function source(patch: Partial<ProjectMemorySourceSnapshot> = {}): ProjectMemorySourceSnapshot {
  return {
    sourceRef: patch.sourceRef ?? 'AGENTS.md',
    exists: patch.exists ?? true,
    updatedAt: patch.updatedAt ?? 90,
  }
}

console.log('project-memory tests:')

{
  const warnings = detectProjectMemoryWarnings([memory()], { 'AGENTS.md': source() })
  assert.deepEqual(warnings, [])
  ok('keeps verified source-backed memory quiet when the live source is unchanged')
}

{
  const warnings = detectProjectMemoryWarnings([memory()], { 'AGENTS.md': source({ updatedAt: 150 }) })
  assert.equal(warnings.length, 1)
  assert.equal(warnings[0].code, 'source_newer_than_verification')
  assert.match(warnings[0].message, /changed after this memory was verified/)
  ok('warns when a live source changed after memory verification')
}

{
  const warnings = detectProjectMemoryWarnings([memory()], { 'AGENTS.md': source({ exists: false, updatedAt: null }) })
  assert.equal(warnings.length, 1)
  assert.equal(warnings[0].code, 'source_missing')
  ok('warns when source-backed memory points at a missing live source')
}

{
  const warnings = detectProjectMemoryWarnings([memory({ sourceRef: null })])
  assert.equal(warnings.length, 1)
  assert.equal(warnings[0].code, 'missing_source_ref')
  ok('warns when source-backed memory has no source reference')
}

{
  const warnings = detectProjectMemoryWarnings([memory({ updatedAt: 140 })], { 'AGENTS.md': source() })
  assert.equal(warnings.length, 1)
  assert.equal(warnings[0].code, 'memory_changed_after_verification')
  ok('warns when memory text changed after its last verification')
}

{
  const warnings = detectProjectMemoryWarnings([memory({ id: 'manual_1', sourceType: 'manual', sourceRef: null })])
  assert.deepEqual(warnings, [])
  ok('does not require source references for manual memory')
}

{
  const formatted = formatProjectMemoryWarnings([
    {
      entryId: 'mem_1',
      code: 'source_newer_than_verification',
      sourceRef: 'AGENTS.md',
      message: 'Memory source AGENTS.md changed after this memory was verified.',
    },
  ])
  assert.match(formatted, /\[source_newer_than_verification; src: AGENTS\.md; memory: mem_1\]/)
  ok('formats warnings for prompt-visible memory context')
}

{
  const entries = [
    memory({ id: 'project_a_1', projectId: 'project_a', text: 'A memory' }),
    memory({ id: 'project_b_1', projectId: 'project_b', text: 'B memory' }),
    memory({ id: 'project_a_2', projectId: 'project_a', text: 'Second A memory' }),
  ]
  assert.deepEqual(scopeProjectMemoryEntries(entries, 'project_a').map((entry) => entry.id), ['project_a_1', 'project_a_2'])
  assert.deepEqual(scopeProjectMemoryEntries(entries, 'project_b').map((entry) => entry.id), ['project_b_1'])
  assert.deepEqual(scopeProjectMemoryEntries(entries, 'missing'), [])
  ok('scopes project memory so one project cannot receive another project memory')
}

{
  const legacy = memory({ id: 'legacy', privacyScope: undefined as any, kind: 'bogus' as any })
  const scoped = scopeProjectMemoryEntries([legacy], 'project_1')
  assert.equal(scoped[0].kind, 'summary')
  assert.equal(scoped[0].privacyScope, 'project')
  ok('normalizes legacy memory kind and privacy scope safely')
}

{
  const entries = [
    memory({ id: 'keep_1', projectId: 'project_a' }),
    memory({ id: 'forget_me', projectId: 'project_a' }),
    memory({ id: 'keep_2', projectId: 'project_b' }),
  ]
  const forgotten = forgetProjectMemoryEntry(entries, 'forget_me')
  assert.deepEqual(forgotten.map((entry) => entry.id), ['keep_1', 'keep_2'])
  assert.deepEqual(scopeProjectMemoryEntries(forgotten, 'project_a').map((entry) => entry.id), ['keep_1'])
  ok('forgets individual memory entries before they can be reused in project context')
}

{
  const candidate = createProjectMemoryCandidate({
    projectId: 'project_a',
    kind: 'decision',
    text: '  Prefer source-backed memory candidates.  ',
    sourceType: 'file',
    sourceRef: 'AGENTS.md',
    reason: 'Observed in project guidance',
  }, 10, 'cand_1')
  assert.equal(candidate.status, 'pending')
  assert.equal(candidate.text, 'Prefer source-backed memory candidates.')
  assert.equal(candidate.privacyScope, 'project')
  assert.deepEqual(scopeProjectMemoryEntries([], 'project_a'), [])
  assert.deepEqual(scopeProjectMemoryCandidates([candidate], 'project_a').map((item) => item.id), ['cand_1'])
  ok('creates pending memory candidates without durable memory writes')
}

{
  const candidate = createProjectMemoryCandidate({
    projectId: 'project_a',
    kind: 'summary',
    text: 'Launch owner needs the updated rollout note.',
    sourceType: 'connector',
    sourceRef: 'email:inbox:message-1',
    sourceIds: ['email:inbox:message-1', 'calendar:work:event-1'],
    reason: 'Connector memory has cited sources.',
  } as any, 10, 'cand_sources')
  assert.deepEqual((candidate as any).sourceIds, ['email:inbox:message-1', 'calendar:work:event-1'])
  const result = approveProjectMemoryCandidate([], [candidate], 'cand_sources', 20, 'reviewer', 'mem_sources')
  assert.deepEqual((result.entry as any).sourceIds, ['email:inbox:message-1', 'calendar:work:event-1'])
  assert.match(formatProjectMemoryEntryForPrompt(result.entry), /src: email:inbox:message-1, calendar:work:event-1/)
  ok('preserves connector source ids as prompt-visible memory citations')
}

{
  const candidate = createProjectMemoryCandidate({
    projectId: 'project_a',
    kind: 'preference',
    text: 'Maya likes compact status updates with exact percentages.',
    sourceType: 'manual',
    privacyScope: 'personal',
    owner: 'Maya',
  }, 10, 'cand_pref')
  const result = approveProjectMemoryCandidate([], [candidate], 'cand_pref', 20, 'reviewer', 'mem_pref')
  assert.equal(result.entry.kind, 'preference')
  assert.equal(result.entry.privacyScope, 'personal')
  assert.equal(result.entry.owner, 'Maya')
  ok('stores people and preference memory with explicit privacy scope')
}

{
  const candidate = createProjectMemoryCandidate({ projectId: 'project_a', text: 'Use candidates before durable writes.', sourceType: 'manual' }, 10, 'cand_approve')
  const result = approveProjectMemoryCandidate([], [candidate], 'cand_approve', 20, 'reviewer', 'mem_approved')
  assert.equal(result.candidate.status, 'approved')
  assert.equal(result.candidate.durableEntryId, 'mem_approved')
  assert.equal(result.entry.id, 'mem_approved')
  assert.equal(result.entries.length, 1)
  assert.equal(scopeProjectMemoryCandidates(result.candidates, 'project_a').length, 0)
  ok('approval turns a pending memory candidate into a durable entry')
}

{
  const existing = memory({ id: 'mem_edit', projectId: 'project_a', text: 'Old memory', createdAt: 1, updatedAt: 2 })
  const candidate = createProjectMemoryCandidate({ projectId: 'project_a', targetEntryId: 'mem_edit', text: 'Updated memory', confidence: 'high' }, 10, 'cand_edit')
  const result = approveProjectMemoryCandidate([existing], [candidate], 'cand_edit', 20, 'reviewer')
  assert.equal(result.entries.length, 1)
  assert.equal(result.entries[0].id, 'mem_edit')
  assert.equal(result.entries[0].text, 'Updated memory')
  assert.equal(result.entries[0].confidence, 'high')
  assert.equal(result.entries[0].createdAt, 1)
  ok('approval can review edits before updating an existing durable entry')
}

{
  const existing = memory({ id: 'mem_src_edit', projectId: 'project_a', text: 'Old source-backed memory', sourceType: 'file', sourceRef: 'AGENTS.md', lastVerifiedAt: 100, createdAt: 1, updatedAt: 90 })
  const candidate = createProjectMemoryCandidate({ projectId: 'project_a', targetEntryId: 'mem_src_edit', text: 'Reviewed against the live source again', sourceType: 'file', sourceRef: 'AGENTS.md' }, 200, 'cand_src_edit')
  const result = approveProjectMemoryCandidate([existing], [candidate], 'cand_src_edit', 300, 'reviewer')
  assert.equal(result.entries[0].updatedAt, 300)
  assert.equal(result.entries[0].lastVerifiedAt, 300)
  const warnings = detectProjectMemoryWarnings(result.entries, { 'AGENTS.md': source({ updatedAt: 90 }) })
  assert.deepEqual(warnings, [])
  ok('re-approving a source-backed memory refreshes lastVerifiedAt so it is not flagged unverified')
}

{
  const existing = memory({ id: 'mem_manual_edit', projectId: 'project_a', text: 'Old manual memory', sourceType: 'manual', sourceRef: null, lastVerifiedAt: 100, createdAt: 1, updatedAt: 90 })
  const candidate = createProjectMemoryCandidate({ projectId: 'project_a', targetEntryId: 'mem_manual_edit', text: 'Edited manual memory', sourceType: 'manual' }, 200, 'cand_manual_edit')
  const result = approveProjectMemoryCandidate([existing], [candidate], 'cand_manual_edit', 300, 'reviewer')
  assert.equal(result.entries[0].lastVerifiedAt, 100)
  ok('editing a non-source-backed memory leaves lastVerifiedAt untouched')
}

{
  const candidate = createProjectMemoryCandidate({ projectId: 'project_a', text: 'Do not save this.' }, 10, 'cand_reject')
  const result = rejectProjectMemoryCandidate([candidate], 'cand_reject', 20, 'reviewer')
  assert.equal(result.candidate.status, 'rejected')
  assert.deepEqual(scopeProjectMemoryCandidates(result.candidates, 'project_a'), [])
  ok('rejected memory candidates never become prompt-visible project memory')
}

// ---- VST-009 reviewed memory loop: diff, provenance, revisions, rollback, scope ----

{
  // Provenance flows candidate → entry: the review UI can always answer "which turn
  // noticed this?".
  const candidate = createProjectMemoryCandidate(
    { projectId: 'project_a', text: 'We always use pnpm here.', sourceType: 'agent', sourceThreadId: 'thread-9', sourceTurn: 4 },
    1_000,
    'mc_prov',
  )
  assert.equal(candidate.sourceThreadId, 'thread-9')
  assert.equal(candidate.sourceTurn, 4)
  const approved = approveProjectMemoryCandidate([], [candidate], 'mc_prov', 2_000, 'user', 'm_prov')
  assert.equal(approved.entry.sourceThreadId, 'thread-9')
  assert.equal(approved.entry.sourceTurn, 4)
  ok('provenance: source thread/turn survive from proposal to durable entry')
}

{
  // Diff: a CREATE lists the set fields; an UPDATE lists only what actually changes.
  const entry: ProjectMemoryEntry = {
    id: 'm_base', projectId: 'project_a', kind: 'convention', text: 'Use tabs.',
    sourceType: 'manual', sourceRef: null, confidence: 'medium', privacyScope: 'project',
    owner: null, lastVerifiedAt: null, createdAt: 1, updatedAt: 1,
  }
  const create = createProjectMemoryCandidate({ projectId: 'project_a', text: 'New fact.' }, 10, 'mc_new')
  const createDiff = projectMemoryCandidateDiff([entry], create)
  assert.equal(createDiff.action, 'create')
  assert.equal(createDiff.targetEntryId, null)
  assert.ok(createDiff.changes.some((c) => c.field === 'text' && c.after === 'New fact.' && c.before === null))

  const update = createProjectMemoryCandidate(
    { projectId: 'project_a', targetEntryId: 'm_base', text: 'Use spaces.', kind: 'convention', confidence: 'medium' },
    20,
    'mc_upd',
  )
  const updateDiff = projectMemoryCandidateDiff([entry], update)
  assert.equal(updateDiff.action, 'update')
  assert.equal(updateDiff.targetEntryId, 'm_base')
  assert.deepEqual(updateDiff.changes.map((c) => c.field), ['text'], 'unchanged fields never appear in an update diff')
  assert.equal(updateDiff.changes[0].before, 'Use tabs.')
  assert.equal(updateDiff.changes[0].after, 'Use spaces.')
  ok('candidate diff: informed review — create lists fields, update lists only real changes')
}

{
  // Approving an UPDATE pushes the superseded state as a revision; rollback restores it,
  // flips the approval to rolled_back, and drops the stale verification claim.
  const base = createProjectMemoryCandidate({ projectId: 'project_a', text: 'v1 rule.' }, 100, 'mc_v1')
  const first = approveProjectMemoryCandidate([], [base], 'mc_v1', 200, 'user', 'm_roll')
  const upd = createProjectMemoryCandidate({ projectId: 'project_a', targetEntryId: 'm_roll', text: 'v2 rule.' }, 300, 'mc_v2')
  const second = approveProjectMemoryCandidate(first.entries, [...first.candidates, upd], 'mc_v2', 400, 'user', 'm_ignored')
  const updated = second.entries.find((e) => e.id === 'm_roll')!
  assert.equal(updated.text, 'v2 rule.')
  assert.equal(updated.revisions?.length, 1)
  assert.equal(updated.revisions?.[0].text, 'v1 rule.')

  const rollback = rollbackProjectMemoryEntry(second.entries, second.candidates, 'm_roll', 500)
  assert.equal(rollback.action, 'restored_previous')
  assert.equal(rollback.entry?.text, 'v1 rule.')
  assert.equal(rollback.entry?.lastVerifiedAt, null, 'a restored state is unverified until rechecked')
  assert.equal(rollback.entry?.revisions?.length, 0)
  const flipped = rollback.candidates.find((c) => c.id === 'mc_v2')
  assert.equal(flipped?.status, 'rolled_back', 'the reverted approval is visible in the audit trail')
  assert.equal(rollback.candidates.find((c) => c.id === 'mc_v1')?.status, 'approved', 'earlier approvals stay intact')

  // Rolling back AGAIN removes the entry — its creation was the remaining change.
  const again = rollbackProjectMemoryEntry(rollback.entries, rollback.candidates, 'm_roll', 600)
  assert.equal(again.action, 'removed')
  assert.equal(again.entries.some((e) => e.id === 'm_roll'), false)
  assert.equal(again.candidates.find((c) => c.id === 'mc_v1')?.status, 'rolled_back')

  assert.throws(() => rollbackProjectMemoryEntry([], [], 'missing', 700), /not found/)
  ok('rollback: revision restore → creation removal, approvals flip to rolled_back')
}

{
  const existing = memory({
    id: 'm_owned', projectId: 'project_a', text: 'Original', sourceType: 'connector',
    sourceRef: 'mail:1', sourceIds: ['mail:1', 'thread:old'], owner: 'Old owner',
    lastVerifiedAt: 150, sourceThreadId: 'thread-old', sourceTurn: 2,
  })
  const candidate = createProjectMemoryCandidate({
    projectId: 'project_a', targetEntryId: existing.id, text: 'Replacement',
    kind: 'decision', confidence: 'high', sourceType: 'file', sourceRef: 'AGENTS.md', sourceIds: ['AGENTS.md'], owner: 'New owner',
    sourceThreadId: 'thread-new', sourceTurn: 7,
  }, 200, 'mc_owned')
  const diff = projectMemoryCandidateDiff([existing], candidate)
  assert.deepEqual(
    diff.changes.map((change) => change.field),
    ['text', 'sourceType', 'sourceRef', 'sourceIds', 'owner', 'lastVerifiedAt', 'sourceThreadId', 'sourceTurn'],
    'review diff includes every changed source, owner, and provenance field',
  )
  const approved = approveProjectMemoryCandidate([existing], [candidate], candidate.id, 300, 'reviewer')
  const revision = approved.entry.revisions?.[0]
  assert.equal(revision?.sourceType, 'connector')
  assert.deepEqual(revision?.sourceIds, ['mail:1', 'thread:old'])
  assert.equal(revision?.owner, 'Old owner')
  assert.equal(revision?.sourceThreadId, 'thread-old')
  assert.equal(revision?.sourceTurn, 2)
  const rolledBack = rollbackProjectMemoryEntry(approved.entries, approved.candidates, existing.id, 400)
  assert.equal(rolledBack.entry?.sourceType, 'connector')
  assert.equal(rolledBack.entry?.sourceRef, 'mail:1')
  assert.deepEqual(rolledBack.entry?.sourceIds, ['mail:1', 'thread:old'])
  assert.equal(rolledBack.entry?.owner, 'Old owner')
  assert.equal(rolledBack.entry?.lastVerifiedAt, 150)
  assert.equal(rolledBack.entry?.sourceThreadId, 'thread-old')
  assert.equal(rolledBack.entry?.sourceTurn, 2)
  ok('rollback restores the full superseded source, owner, verification, and provenance state')
}

{
  const target = memory({ id: 'm_project_b', projectId: 'project_b', text: 'Project B only' })
  const hostile = createProjectMemoryCandidate({
    projectId: 'project_a', targetEntryId: target.id, text: 'Cross-project overwrite',
  }, 10, 'mc_cross_project')
  assert.throws(() => projectMemoryCandidateDiff([target], hostile), /different project/)
  assert.throws(() => approveProjectMemoryCandidate([target], [hostile], hostile.id), /different project/)
  assert.equal(target.text, 'Project B only')
  ok('candidate approval cannot target a durable entry owned by another project')
}

{
  // Revisions stay bounded: overwrite an entry more times than the cap keeps.
  let entries: ProjectMemoryEntry[] = []
  let candidates = [createProjectMemoryCandidate({ projectId: 'project_a', text: 'rev 0' }, 0, 'mc_r0')]
  let result = approveProjectMemoryCandidate(entries, candidates, 'mc_r0', 1, 'user', 'm_cap')
  entries = result.entries
  candidates = result.candidates
  for (let i = 1; i <= 8; i++) {
    const c = createProjectMemoryCandidate({ projectId: 'project_a', targetEntryId: 'm_cap', text: `rev ${i}` }, i * 10, `mc_r${i}`)
    result = approveProjectMemoryCandidate(entries, [...candidates, c], `mc_r${i}`, i * 10 + 1, 'user', 'm_x')
    entries = result.entries
    candidates = result.candidates
  }
  const capped = entries.find((e) => e.id === 'm_cap')!
  assert.equal(capped.revisions?.length, 5, 'revision ledger is bounded')
  assert.equal(capped.revisions?.[0].text, 'rev 7', 'newest superseded state first')
  ok('revisions: bounded ledger, newest-first')
}

{
  // The cross-project guarantee: project A memory can never render into project B\'s
  // prompt bucket — scoping + formatting compose to an empty injection.
  const entryA = approveProjectMemoryCandidate(
    [],
    [createProjectMemoryCandidate({ projectId: 'project_a', text: 'Secret-A convention.' }, 1, 'mc_a')],
    'mc_a', 2, 'user', 'm_a',
  ).entries
  const scopedForB = scopeProjectMemoryEntries(entryA, 'project_b')
  assert.deepEqual(scopedForB, [])
  const renderedForB = scopedForB.map(formatProjectMemoryEntryForPrompt).join('\n')
  assert.equal(renderedForB, '', 'nothing from project A reaches project B\'s prompt')
  assert.ok(scopeProjectMemoryEntries(entryA, 'project_a').length === 1)
  ok('scope: memory is never injected across unrelated projects')
}

console.log(`\n${passed} checks passed.`)
