// Consent-first "suggestion spine". When the agent finishes a workflow the user is
// likely to repeat (e.g. it just wired up a deploy), it FILES a suggestion — it never
// creates anything itself. The user accepts it into a real Skill or Command. Adapted
// from the project-memory candidate pattern; pure + unit-tested (no IO here).

import type { Command, Skill, SkillSuggestion, SkillSuggestionKind, SkillSuggestionDiff } from './types'

export interface SuggestionInput {
  kind: SkillSuggestionKind
  name: string
  description?: string
  body: string
  argHint?: string
  rationale?: string
  sourceThreadId?: string | null
  // Set to an existing skill's id to file an in-place improvement of it (kind:'skill').
  targetSkillId?: string | null
}

// Existing artifacts a new suggestion is deduped against, so the agent never nags
// with something the user already has (or already declined/queued).
export interface SuggestionCorpus {
  skills?: Pick<Skill, 'name'>[]
  commands?: Pick<Command, 'name'>[]
  pending?: Pick<SkillSuggestion, 'name' | 'kind' | 'status' | 'targetSkillId'>[]
}

const MAX_PENDING = 25

function norm(name: string): string {
  return (name || '').trim().toLowerCase().replace(/^\//, '').replace(/\s+/g, '-')
}

/** True when an equivalent skill/command/pending-suggestion already exists. */
export function isDuplicateSuggestion(input: SuggestionInput, corpus: SuggestionCorpus): boolean {
  const key = norm(input.name)
  if (!key) return true // nameless → not worth filing
  // An IMPROVEMENT intentionally matches an existing skill by name — it must not be
  // blocked by it. It's a duplicate only if another improvement of the SAME skill is
  // already pending (one revision in flight per skill).
  if (input.targetSkillId) {
    return (corpus.pending ?? []).some((p) => p.status === 'pending' && p.targetSkillId === input.targetSkillId)
  }
  if ((corpus.skills ?? []).some((s) => norm(s.name) === key)) return true
  if ((corpus.commands ?? []).some((c) => norm(c.name) === key)) return true
  // Same kind already pending with the same name = duplicate; dismissed ones don't block.
  if ((corpus.pending ?? []).some((p) => p.status === 'pending' && p.kind === input.kind && norm(p.name) === key)) return true
  return false
}

export function createSkillSuggestion(input: SuggestionInput, now = Date.now(), id = `sg_${Math.random().toString(36).slice(2, 9)}`): SkillSuggestion {
  return {
    id,
    kind: input.kind,
    name: input.name.trim(),
    description: (input.description ?? '').trim(),
    body: input.body,
    argHint: input.kind === 'command' ? input.argHint : undefined,
    rationale: (input.rationale ?? '').trim(),
    sourceThreadId: input.sourceThreadId ?? null,
    targetSkillId: input.kind === 'skill' ? (input.targetSkillId ?? null) : null,
    status: 'pending',
    createdAt: now,
    updatedAt: now,
  }
}

/**
 * File a suggestion: skipped (added:false) when it duplicates an existing artifact or
 * when the pending queue is full (oldest-pending cap), so suggestions never pile up.
 */
export function addSuggestion(
  list: SkillSuggestion[],
  input: SuggestionInput,
  corpus: SuggestionCorpus,
  now = Date.now(),
  id?: string,
): { list: SkillSuggestion[]; added: SkillSuggestion | null; reason?: string } {
  if (isDuplicateSuggestion(input, { ...corpus, pending: list })) return { list, added: null, reason: 'already exists or pending' }
  const pendingCount = list.filter((s) => s.status === 'pending').length
  if (pendingCount >= MAX_PENDING) return { list, added: null, reason: 'pending queue full' }
  const suggestion = createSkillSuggestion(input, now, id)
  return { list: [suggestion, ...list], added: suggestion }
}

/** Pending suggestions, newest-first. */
export function pendingSuggestions(list: SkillSuggestion[]): SkillSuggestion[] {
  return list.filter((s) => s.status === 'pending').sort((a, b) => b.createdAt - a.createdAt)
}

/**
 * Accept a suggestion into a real Skill or Command. Marks it accepted (idempotent: a
 * non-pending suggestion yields created:null) and returns the artifact to persist.
 */
export function acceptSuggestion(
  list: SkillSuggestion[],
  id: string,
  now = Date.now(),
  newId = `${id}_accepted`,
  existingSkillIds?: ReadonlySet<string>,
): { list: SkillSuggestion[]; skill?: Skill; command?: Command; accepted?: SkillSuggestion } {
  const target = list.find((s) => s.id === id)
  if (!target || target.status !== 'pending') return { list }
  // An improvement is consent to replace one EXISTING skill. If that skill was
  // deleted after the suggestion was filed, accepting the stale proposal must not
  // resurrect it as a new artifact under the deleted id.
  if (target.kind === 'skill' && target.targetSkillId && existingSkillIds && !existingSkillIds.has(target.targetSkillId)) {
    return { list }
  }
  const accepted: SkillSuggestion = { ...target, status: 'accepted', acceptedId: newId, updatedAt: now }
  const next = list.map((s) => (s.id === id ? accepted : s))
  if (target.kind === 'command') {
    const command: Command = {
      id: newId,
      name: norm(target.name),
      description: target.description,
      prompt: target.body,
      argHint: target.argHint,
      createdAt: now,
      updatedAt: now,
    }
    return { list: next, command, accepted }
  }
  // An improvement rewrites the targeted skill in place: keep its existing id so the
  // persistence layer overwrites rather than creating a duplicate. createdAt is only
  // meaningful for new skills; the runtime preserves the original when updating.
  const skill: Skill = {
    id: target.targetSkillId || newId,
    name: target.name,
    description: target.description,
    body: target.body,
    enabled: true,
    createdAt: now,
    updatedAt: now,
  }
  return { list: next, skill, accepted }
}

/** Dismiss a pending suggestion (terminal; idempotent). */
/**
 * What accepting `suggestion` would DO to the skill/command corpus (VST-010): a create, or
 * an in-place update with per-field before/after — the review UI shows this diff so
 * accepting a procedural change is an informed action, mirroring the memory loop (VST-009).
 */
export function skillSuggestionDiff(skills: Skill[], suggestion: SkillSuggestion): SkillSuggestionDiff {
  const target = suggestion.targetSkillId ? skills.find((skill) => skill.id === suggestion.targetSkillId) ?? null : null
  const fields: SkillSuggestionDiff['changes'] = [
    { field: 'name', before: target?.name ?? null, after: suggestion.name },
    { field: 'description', before: target?.description ?? null, after: suggestion.description },
    { field: 'body', before: target?.body ?? null, after: suggestion.body },
    { field: 'argHint', before: null, after: suggestion.argHint ?? null },
  ]
  return {
    suggestionId: suggestion.id,
    action: target ? 'update' : 'create',
    targetSkillId: target?.id ?? null,
    changes: fields.filter(({ before, after }) => (target ? before !== after : after != null && after !== '')),
  }
}

export function dismissSuggestion(list: SkillSuggestion[], id: string, now = Date.now()): SkillSuggestion[] {
  return list.map((s) => (s.id === id && s.status === 'pending' ? { ...s, status: 'dismissed', updatedAt: now } : s))
}
