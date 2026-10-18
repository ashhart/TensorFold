// Event-triggered ambient reactions. The connector spine (connector-spine.ts) already
// classifies an inbound ConnectorEvent (priority / trustLevel / classification) with a
// flat, connector-agnostic policy. This module sits ONE layer above: given that already
// classified event plus a set of user SUBSCRIPTIONS, it decides — purely — what ambient
// reaction the assistant should have: Notify, Question, Review, or ignore. Every decision
// carries machine-readable `reasons` so the UI/log can explain "why did it react".
//
// Pure + dependency-injected: no fetch, no clock reads except an injectable `now`, no I/O.
// The host feeds live events + subscriptions; the caller turns a Question/Review reaction
// into an actual assistant turn or approval draft. This never mutates its inputs.

// Structurally typed so this module does NOT import from types/* (a SHARED file). It only
// reads the fields it needs off the spine's ConnectorEvent — assignable from the real type.
export type ReactionEventDirection = 'inbound' | 'outbound' | 'internal'
export type ReactionEventPriority = 'critical' | 'high' | 'normal' | 'low'
export type ReactionEventTrust = 'notify' | 'suggest' | 'approve' | 'auto_logged' | 'auto_silent'
export type ReactionEventClass = 'communication' | 'document' | 'schedule' | 'approval' | 'system' | 'unknown'

export interface ReactionEvent {
  id: string
  connectorId: string
  direction: ReactionEventDirection
  classification: ReactionEventClass
  priority: ReactionEventPriority
  trustLevel: ReactionEventTrust
  action: string
  title: string
  summary: string
  actor: string
  threadKey: string
  createdAt: number
  requiresApproval?: boolean
  /** Optional lowercased keyword corpus the caller may pre-extract; if absent we derive
   *  it from title+summary+actor. Never trust raw webhook payload text here. */
  keywords?: string[]
}

/** What kind of ambient reaction to have. `ignore` = the event is not worth surfacing. */
export type ReactionKind = 'notify' | 'question' | 'review' | 'ignore'

/**
 * A user-configured rule for reacting to events. Matching is AND across the provided
 * facets (a facet with an empty/absent list matches everything). `minPriority` gates on
 * the spine's priority ladder. A `mute` (or `react: 'ignore'`) match forces ignore even
 * if another subscription would have reacted.
 */
export interface ReactionSubscription {
  id: string
  /** Reaction to emit when this subscription matches. */
  react: ReactionKind
  /** Match only these connectors (empty = any). */
  connectorIds?: string[]
  /** Match only these classifications (empty = any). */
  classifications?: ReactionEventClass[]
  /** Match only when the event actor matches one of these (case-insensitive substring). */
  actors?: string[]
  /** Match only when title/summary/keywords contain one of these (case-insensitive). */
  keywords?: string[]
  /** Match only when the thread key is one of these (exact). */
  threadKeys?: string[]
  /** Minimum spine priority to react to (default 'low' = react to anything). */
  minPriority?: ReactionEventPriority
  /** If true, a match forces `ignore` (a mute rule) regardless of `react`. */
  mute?: boolean
  /** Higher wins when several subscriptions match one event (default 0). */
  weight?: number
  /** Disabled subscriptions never match. */
  enabled?: boolean
}

export interface ReactionReason {
  code:
    | 'outbound_skipped'
    | 'own_action_skipped'
    | 'stale_skipped'
    | 'no_subscription_match'
    | 'muted'
    | 'subscription_match'
    | 'approval_forces_question'
    | 'critical_default_notify'
  detail: string
  subscriptionId?: string
}

export interface Reaction {
  eventId: string
  kind: ReactionKind
  /** The subscription that produced a non-ignore reaction, if any. */
  subscriptionId: string | null
  /** Higher = more urgent; derived from priority + reaction kind. Stable sort key. */
  urgency: number
  reasons: ReactionReason[]
}

const PRIORITY_RANK: Record<ReactionEventPriority, number> = { low: 0, normal: 1, high: 2, critical: 3 }
const KIND_RANK: Record<ReactionKind, number> = { ignore: 0, notify: 1, question: 2, review: 3 }

function keywordCorpus(event: ReactionEvent): string {
  if (event.keywords && event.keywords.length) return event.keywords.join(' ').toLowerCase()
  return `${event.title} ${event.summary} ${event.actor}`.toLowerCase()
}

function matchesActor(event: ReactionEvent, actors: string[]): boolean {
  const hay = event.actor.toLowerCase()
  return actors.some((a) => hay.includes(a.toLowerCase()))
}

function matchesKeyword(event: ReactionEvent, keywords: string[]): boolean {
  const corpus = keywordCorpus(event)
  return keywords.some((k) => corpus.includes(k.toLowerCase()))
}

/** Does a subscription's facets all match the event? (facet absent/empty = wildcard) */
function subscriptionMatches(event: ReactionEvent, sub: ReactionSubscription): boolean {
  if (sub.enabled === false) return false
  if (sub.connectorIds?.length && !sub.connectorIds.includes(event.connectorId)) return false
  if (sub.classifications?.length && !sub.classifications.includes(event.classification)) return false
  if (sub.threadKeys?.length && !sub.threadKeys.includes(event.threadKey)) return false
  if (sub.actors?.length && !matchesActor(event, sub.actors)) return false
  if (sub.keywords?.length && !matchesKeyword(event, sub.keywords)) return false
  const min = sub.minPriority ?? 'low'
  if (PRIORITY_RANK[event.priority] < PRIORITY_RANK[min]) return false
  return true
}

function ignored(event: ReactionEvent, reasons: ReactionReason[]): Reaction {
  return { eventId: event.id, kind: 'ignore', subscriptionId: null, urgency: 0, reasons }
}

function urgencyFor(priority: ReactionEventPriority, kind: ReactionKind): number {
  // Priority dominates; reaction kind breaks ties (review > question > notify).
  return PRIORITY_RANK[priority] * 10 + KIND_RANK[kind]
}

export interface DecideOptions {
  /** Ignore events older than this many ms (relative to `now`). 0/undefined = no cutoff. */
  maxAgeMs?: number
  now?: number
  /**
   * When true (default), an approval-gated inbound event with no matching subscription
   * still yields a `question` (the assistant should ask before it can act). When false,
   * approval events follow the same subscription rules as everything else.
   */
  approvalDefaultsToQuestion?: boolean
  /**
   * When true (default), a `critical` inbound event with no matching subscription still
   * yields a `notify` so genuinely urgent things are never silently dropped.
   */
  criticalDefaultsToNotify?: boolean
}

/**
 * Decide the ambient reaction for ONE event against the user's subscriptions. Pure.
 *
 * Order of precedence:
 *  1. Non-inbound (outbound/internal) events never react — the assistant doesn't react to
 *     its own or system-internal traffic.
 *  2. Stale events (older than maxAgeMs) are ignored.
 *  3. A matching `mute` subscription forces ignore (mutes beat reactions).
 *  4. The highest-weight matching non-mute subscription wins and sets the reaction kind.
 *  5. With no match: approval-gated events default to `question`, critical events default
 *     to `notify`, everything else is ignored. (Both defaults are toggleable.)
 */
export function decideReaction(
  event: ReactionEvent,
  subscriptions: ReactionSubscription[],
  options: DecideOptions = {},
): Reaction {
  const approvalDefault = options.approvalDefaultsToQuestion ?? true
  const criticalDefault = options.criticalDefaultsToNotify ?? true

  if (event.direction !== 'inbound') {
    return ignored(event, [{
      code: event.direction === 'outbound' ? 'outbound_skipped' : 'own_action_skipped',
      detail: `${event.direction} events do not trigger ambient reactions`,
    }])
  }

  if (options.maxAgeMs && options.maxAgeMs > 0) {
    const now = options.now ?? Date.now()
    if (now - event.createdAt > options.maxAgeMs) {
      return ignored(event, [{ code: 'stale_skipped', detail: `event older than ${options.maxAgeMs}ms cutoff` }])
    }
  }

  const matches = subscriptions.filter((sub) => subscriptionMatches(event, sub))

  // Mutes win outright.
  const mute = matches.find((sub) => sub.mute || sub.react === 'ignore')
  if (mute) {
    return ignored(event, [{ code: 'muted', detail: `muted by subscription ${mute.id}`, subscriptionId: mute.id }])
  }

  // Highest weight (then most-urgent reaction kind) among matching non-mute subs wins.
  const ranked = matches
    .filter((sub) => sub.react !== 'ignore')
    .sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0) || KIND_RANK[b.react] - KIND_RANK[a.react] || a.id.localeCompare(b.id))
  const winner = ranked[0]

  if (winner) {
    const reasons: ReactionReason[] = [{
      code: 'subscription_match',
      detail: `matched subscription ${winner.id} → ${winner.react}`,
      subscriptionId: winner.id,
    }]
    return {
      eventId: event.id,
      kind: winner.react,
      subscriptionId: winner.id,
      urgency: urgencyFor(event.priority, winner.react),
      reasons,
    }
  }

  // No matching subscription — safety-net defaults.
  if (approvalDefault && (event.requiresApproval || event.trustLevel === 'approve' || event.classification === 'approval')) {
    return {
      eventId: event.id,
      kind: 'question',
      subscriptionId: null,
      urgency: urgencyFor(event.priority, 'question'),
      reasons: [{ code: 'approval_forces_question', detail: 'approval-gated event with no rule defaults to a question' }],
    }
  }

  if (criticalDefault && event.priority === 'critical') {
    return {
      eventId: event.id,
      kind: 'notify',
      subscriptionId: null,
      urgency: urgencyFor(event.priority, 'notify'),
      reasons: [{ code: 'critical_default_notify', detail: 'critical event with no rule defaults to a notify' }],
    }
  }

  return ignored(event, [{ code: 'no_subscription_match', detail: 'no subscription matched and no safety-net default applied' }])
}

/**
 * Decide reactions for a batch of events, drop the ignores, and return the actionable
 * reactions sorted most-urgent-first (stable by eventId on ties). Convenience wrapper the
 * host uses to turn an event feed into an ambient-reaction queue.
 */
export function decideReactions(
  events: ReactionEvent[],
  subscriptions: ReactionSubscription[],
  options: DecideOptions = {},
): Reaction[] {
  return events
    .map((event) => decideReaction(event, subscriptions, options))
    .filter((r) => r.kind !== 'ignore')
    .sort((a, b) => b.urgency - a.urgency || a.eventId.localeCompare(b.eventId))
}
