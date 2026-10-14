// Room outcomes (VST-019): a room MODE says what the room is FOR — and therefore when it
// may stop, what its final message must contain, how seats should behave, how the
// moderator should push, and how much budget it gets by default. Orthogonal to
// collaborationMode (who plans vs builds) and deliberationEffort (how long): the mode is
// the outcome contract. A room with no mode keeps legacy behavior (no outcome gate) —
// the gate never silently applies to rooms created before modes existed.

import type { RoomDeliberationEffort } from './types'

export type RoomMode = 'plan' | 'implement' | 'review' | 'debug' | 'incident' | 'research'

export type RoomArtifactKind = 'plan' | 'implementation_summary' | 'review_verdict' | 'root_cause' | 'incident_report' | 'findings'

export interface RoomOutcomeProfile {
  id: RoomMode
  label: string
  // When the room may FINAL — shown to every seat and enforced by the outcome gate.
  stopCondition: string
  expectedArtifact: { kind: RoomArtifactKind; description: string }
  seatBehavior: string
  moderatorBehavior: string
  // Default deliberation budget when the room does not set its own effort.
  budget: RoomDeliberationEffort
}

export const ROOM_OUTCOME_PROFILES: Record<RoomMode, RoomOutcomeProfile> = {
  plan: {
    id: 'plan',
    label: 'Plan',
    stopCondition: 'Stop when a concrete, ordered plan exists and major objections are resolved or named as dissent.',
    expectedArtifact: { kind: 'plan', description: 'A numbered/step plan: what to do, in what order, and who or what does each step.' },
    seatBehavior: 'Propose and challenge approaches; converge on ONE plan rather than listing alternatives forever.',
    moderatorBehavior: 'Force ranking and a decision; cut circular debate; demand the final plan be enumerated step by step.',
    budget: 'normal',
  },
  implement: {
    id: 'implement',
    label: 'Implement',
    stopCondition: 'Stop when the change is implemented AND verified — tests or checks ran and their outcome is stated.',
    expectedArtifact: { kind: 'implementation_summary', description: 'What changed (files/behavior) and the verification evidence (tests/checks run and their result).' },
    seatBehavior: 'Build in small verifiable steps; claim tickets; report file-level changes and check results, not intentions.',
    moderatorBehavior: 'Block FINAL without verification evidence; route unproven claims back to the builder seats.',
    budget: 'deep',
  },
  review: {
    id: 'review',
    label: 'Review',
    stopCondition: 'Stop when an explicit verdict is reached: approved, or changes requested with each required change named.',
    expectedArtifact: { kind: 'review_verdict', description: 'An explicit verdict (approve / changes requested / blocked) plus the concrete findings that justify it.' },
    seatBehavior: 'Read the actual work; report concrete findings with locations; no verdicts without evidence.',
    moderatorBehavior: 'Demand a committed verdict — a review that ends with "looks fine overall" and no verdict is not done.',
    budget: 'normal',
  },
  debug: {
    id: 'debug',
    label: 'Debug',
    stopCondition: 'Stop when the root cause is identified with evidence and a fix (or precise next step) is stated.',
    expectedArtifact: { kind: 'root_cause', description: 'The root cause (what breaks and why) and the fix or the exact next diagnostic step.' },
    seatBehavior: 'Form hypotheses, test them against the code/logs, and kill disproven ones out loud.',
    moderatorBehavior: 'Keep seats hypothesis-driven; block FINAL that names symptoms without a cause.',
    budget: 'deep',
  },
  incident: {
    id: 'incident',
    label: 'Incident',
    stopCondition: 'Stop when the incident is mitigated, the cause is understood, and follow-ups are assigned.',
    expectedArtifact: { kind: 'incident_report', description: 'Mitigation status, root cause, and the follow-up actions that prevent recurrence.' },
    seatBehavior: 'Mitigate FIRST, investigate second; timestamp findings; never block stabilization on root-causing.',
    moderatorBehavior: 'Triage hard: stabilization outranks elegance; demand follow-ups before the room closes.',
    budget: 'max',
  },
  research: {
    id: 'research',
    label: 'Research',
    stopCondition: 'Stop when the question is answered with sourced findings and a conclusion, or the unknowns are precisely named.',
    expectedArtifact: { kind: 'findings', description: 'Findings with their sources/evidence and a conclusion (or the named open unknowns).' },
    seatBehavior: 'Divide the search space; report findings WITH where they came from; distinguish fact from inference.',
    moderatorBehavior: 'Push for coverage and sourcing; block conclusions that cite no evidence.',
    budget: 'normal',
  },
}

export const ROOM_MODES: RoomMode[] = ['plan', 'implement', 'review', 'debug', 'incident', 'research']
export const DEFAULT_ROOM_MODE: RoomMode = 'implement'

/** Null for unset/unknown — legacy rooms (no mode) get NO outcome gate. */
export function normalizeRoomMode(mode?: string | null): RoomMode | null {
  return mode && (ROOM_MODES as string[]).includes(mode) ? (mode as RoomMode) : null
}

export function roomOutcomeProfile(mode?: string | null): RoomOutcomeProfile | null {
  const normalized = normalizeRoomMode(mode)
  return normalized ? ROOM_OUTCOME_PROFILES[normalized] : null
}

/** Canonical defaults for a newly-created room or a mode change with no manual override. */
export function roomModeDefaults(mode: RoomMode = DEFAULT_ROOM_MODE): { mode: RoomMode; deliberationEffort: RoomDeliberationEffort } {
  return { mode, deliberationEffort: ROOM_OUTCOME_PROFILES[mode].budget }
}

/** The prompt block seats + moderator receive when the room has a mode. */
export function roomModePromptBlock(profile: RoomOutcomeProfile): string {
  return [
    `Room mode: ${profile.label}.`,
    `Stop condition: ${profile.stopCondition}`,
    `Expected final artifact: ${profile.expectedArtifact.description}`,
    `Seat behavior: ${profile.seatBehavior}`,
    `Moderator behavior: ${profile.moderatorBehavior}`,
  ].join('\n')
}

export interface RoomOutcomeGateResult {
  satisfied: boolean
  // What the FINAL text is missing, in the words a seat can act on. Empty when satisfied.
  missing: string[]
}

// Textual outcome gates: honest, cheap signals over the FINAL message. They cannot prove
// semantic quality — they prove the artifact SHAPE is present (a verdict word, enumerated
// steps, a named cause) so a room cannot close out with an empty gesture.
const ENUMERATED_STEP = /^\s*(?:\d+[.)]|[-*•])\s+\S/m
const VERDICT = /\b(approved?|lgtm|ship it|changes requested|request(?:ing)? changes|blocked|rejected?)\b/i
const VERIFICATION = /\b(test(s|ed)?|check(s|ed)?|verif(y|ied|ication)|lint(ed)?|typecheck(ed)?|build (passed|succeeded|green)|npm run|pytest|cargo test)\b/i
const CHANGE_EVIDENCE = /\b(implement(ed)?|changed?|created?|updated?|wrote|added|edited|patch(ed)?|diff)\b/i
const ROOT_CAUSE = /\b(root cause|caused by|the (bug|failure|crash|leak) (is|was)|because)\b/i
const FIX_OR_NEXT = /\b(fix(ed)?|patch(ed)?|resolv(e|ed)|workaround|next (step|diagnostic)|will (change|update))\b/i
const MITIGATION = /\b(mitigat(e|ed|ion)|stabiliz(e|ed)|contained|rolled back|restored|recovered)\b/i
const FOLLOW_UP = /\b(follow[- ]?ups?|action items?|prevent(ion)?|postmortem|to avoid recurrence)\b/i
const SOURCED = /\b(source[sd]?|according to|from [`'"]?[\w./-]+|file[s]?:|see |evidence|reference[sd]?|documented in)\b/i
const CONCLUSION = /\b(conclusion|finding(s)?|we learned|answer(ed)?|unknowns?|open question)\b/i

export function roomOutcomeGate(mode: RoomMode, signals: { finalText: string }): RoomOutcomeGateResult {
  const text = signals.finalText ?? ''
  const missing: string[] = []
  const enumeratedSteps = (text.match(new RegExp(ENUMERATED_STEP.source, 'gm')) ?? []).length
  switch (mode) {
    case 'plan':
      if (enumeratedSteps < 2) missing.push('an ordered plan (at least two enumerated steps)')
      break
    case 'implement':
      if (!CHANGE_EVIDENCE.test(text)) missing.push('what actually changed (files/behavior)')
      if (!VERIFICATION.test(text)) missing.push('verification evidence (which tests/checks ran and their result)')
      break
    case 'review':
      if (!VERDICT.test(text)) missing.push('an explicit verdict (approved / changes requested / blocked)')
      break
    case 'debug':
      if (!ROOT_CAUSE.test(text)) missing.push('the root cause (what breaks and why)')
      if (!FIX_OR_NEXT.test(text)) missing.push('the fix or the exact next diagnostic step')
      break
    case 'incident':
      if (!MITIGATION.test(text)) missing.push('mitigation status')
      if (!ROOT_CAUSE.test(text)) missing.push('the root cause')
      if (!FOLLOW_UP.test(text)) missing.push('follow-up actions to prevent recurrence')
      break
    case 'research':
      if (!CONCLUSION.test(text)) missing.push('a conclusion (or the named open unknowns)')
      if (!SOURCED.test(text)) missing.push('sources/evidence for the findings')
      break
  }
  return { satisfied: missing.length === 0, missing }
}
