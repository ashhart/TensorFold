// Run: jiti electron/agent/ambient-reactions.test.ts
import assert from 'node:assert/strict'
import {
  decideReaction,
  decideReactions,
  type ReactionEvent,
  type ReactionSubscription,
} from './ambient-reactions.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('ambient-reactions tests:')

function mkEvent(over: Partial<ReactionEvent> = {}): ReactionEvent {
  return {
    id: 'e1',
    connectorId: 'email',
    direction: 'inbound',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'suggest',
    action: 'message_received',
    title: 'Renewal thread needs a reply',
    summary: 'A client asked for pricing clarification.',
    actor: 'Maya Chen',
    threadKey: 'email:renewal',
    createdAt: 1_000,
    ...over,
  }
}

{
  // Outbound/internal events never react.
  const r1 = decideReaction(mkEvent({ direction: 'outbound' }), [])
  const r2 = decideReaction(mkEvent({ direction: 'internal' }), [])
  assert.equal(r1.kind, 'ignore')
  assert.equal(r1.reasons[0].code, 'outbound_skipped')
  assert.equal(r2.kind, 'ignore')
  assert.equal(r2.reasons[0].code, 'own_action_skipped')
  ok('non-inbound events ignored')
}

{
  // A matching subscription sets the reaction kind + carries its id in reasons.
  const sub: ReactionSubscription = { id: 's-review', react: 'review', connectorIds: ['email'], keywords: ['renewal'] }
  const r = decideReaction(mkEvent(), [sub])
  assert.equal(r.kind, 'review')
  assert.equal(r.subscriptionId, 's-review')
  assert.equal(r.reasons[0].code, 'subscription_match')
  assert.equal(r.reasons[0].subscriptionId, 's-review')
  ok('matching subscription drives the reaction kind')
}

{
  // Facet AND: a keyword that is not present means no match → falls through to ignore.
  const sub: ReactionSubscription = { id: 's', react: 'notify', connectorIds: ['email'], keywords: ['invoice'] }
  const r = decideReaction(mkEvent(), [sub])
  assert.equal(r.kind, 'ignore')
  assert.equal(r.reasons[0].code, 'no_subscription_match')
  ok('facets are ANDed; a missing keyword blocks the match')
}

{
  // Wrong connector → no match.
  const sub: ReactionSubscription = { id: 's', react: 'notify', connectorIds: ['slack'] }
  assert.equal(decideReaction(mkEvent({ connectorId: 'email' }), [sub]).kind, 'ignore')
  ok('connector facet gates the match')
}

{
  // minPriority gates: a normal event does not clear a high threshold.
  const sub: ReactionSubscription = { id: 's', react: 'notify', minPriority: 'high' }
  assert.equal(decideReaction(mkEvent({ priority: 'normal' }), [sub]).kind, 'ignore')
  assert.equal(decideReaction(mkEvent({ priority: 'critical' }), [sub]).kind, 'notify')
  ok('minPriority threshold gates matching')
}

{
  // Actor substring match is case-insensitive.
  const sub: ReactionSubscription = { id: 's', react: 'question', actors: ['maya'] }
  assert.equal(decideReaction(mkEvent({ actor: 'Maya Chen' }), [sub]).kind, 'question')
  assert.equal(decideReaction(mkEvent({ actor: 'Bob' }), [sub]).kind, 'ignore')
  ok('actor facet matches case-insensitive substring')
}

{
  // A mute subscription forces ignore even when another subscription would react.
  const react: ReactionSubscription = { id: 's-notify', react: 'notify', connectorIds: ['email'] }
  const mute: ReactionSubscription = { id: 's-mute', react: 'notify', mute: true, actors: ['maya'] }
  const r = decideReaction(mkEvent(), [react, mute])
  assert.equal(r.kind, 'ignore')
  assert.equal(r.reasons[0].code, 'muted')
  assert.equal(r.reasons[0].subscriptionId, 's-mute')
  // react:'ignore' behaves as a mute too.
  const muteViaKind: ReactionSubscription = { id: 's-ig', react: 'ignore', actors: ['maya'] }
  assert.equal(decideReaction(mkEvent(), [react, muteViaKind]).kind, 'ignore')
  ok('mute (and react:ignore) beats reacting subscriptions')
}

{
  // Highest weight wins among several matching subscriptions.
  const low: ReactionSubscription = { id: 'a-low', react: 'notify', weight: 1 }
  const high: ReactionSubscription = { id: 'b-high', react: 'review', weight: 5 }
  const r = decideReaction(mkEvent(), [low, high])
  assert.equal(r.kind, 'review')
  assert.equal(r.subscriptionId, 'b-high')
  ok('highest-weight subscription wins')
}

{
  // Tie on weight → more-urgent reaction kind wins (review > question > notify).
  const a: ReactionSubscription = { id: 'a', react: 'notify', weight: 2 }
  const b: ReactionSubscription = { id: 'b', react: 'review', weight: 2 }
  assert.equal(decideReaction(mkEvent(), [a, b]).subscriptionId, 'b')
  ok('weight tie broken by reaction-kind urgency')
}

{
  // Disabled subscriptions never match.
  const sub: ReactionSubscription = { id: 's', react: 'review', enabled: false, connectorIds: ['email'] }
  assert.equal(decideReaction(mkEvent(), [sub]).kind, 'ignore')
  ok('disabled subscriptions are inert')
}

{
  // Safety-net default: approval-gated event with no rule becomes a question.
  const r = decideReaction(mkEvent({ requiresApproval: true, trustLevel: 'approve' }), [])
  assert.equal(r.kind, 'question')
  assert.equal(r.reasons[0].code, 'approval_forces_question')
  // Toggle off → back to ignore.
  const off = decideReaction(mkEvent({ requiresApproval: true, trustLevel: 'approve' }), [], { approvalDefaultsToQuestion: false })
  assert.equal(off.kind, 'ignore')
  ok('approval events default to question (toggleable)')
}

{
  // Safety-net default: critical event with no rule becomes a notify.
  const r = decideReaction(mkEvent({ priority: 'critical' }), [])
  assert.equal(r.kind, 'notify')
  assert.equal(r.reasons[0].code, 'critical_default_notify')
  const off = decideReaction(mkEvent({ priority: 'critical' }), [], { criticalDefaultsToNotify: false })
  assert.equal(off.kind, 'ignore')
  ok('critical events default to notify (toggleable)')
}

{
  // Explicit subscription beats the safety-net default (a rule can downgrade a critical
  // to a notify or, via mute, silence an approval).
  const mute: ReactionSubscription = { id: 'm', react: 'notify', mute: true, connectorIds: ['email'] }
  assert.equal(decideReaction(mkEvent({ requiresApproval: true }), [mute]).kind, 'ignore')
  ok('subscription overrides safety-net defaults')
}

{
  // Stale events are dropped via maxAgeMs relative to injected now.
  const fresh = decideReaction(mkEvent({ createdAt: 900 }), [{ id: 's', react: 'notify' }], { now: 1_000, maxAgeMs: 500 })
  const stale = decideReaction(mkEvent({ createdAt: 100 }), [{ id: 's', react: 'notify' }], { now: 1_000, maxAgeMs: 500 })
  assert.equal(fresh.kind, 'notify')
  assert.equal(stale.kind, 'ignore')
  assert.equal(stale.reasons[0].code, 'stale_skipped')
  ok('maxAgeMs drops stale events (injectable now)')
}

{
  // Batch: ignores dropped, sorted most-urgent-first, stable by eventId on ties.
  const events: ReactionEvent[] = [
    mkEvent({ id: 'low', priority: 'low', threadKey: 't1' }),
    mkEvent({ id: 'crit', priority: 'critical', threadKey: 't2' }),
    mkEvent({ id: 'out', direction: 'outbound' }), // dropped
    mkEvent({ id: 'high', priority: 'high', threadKey: 't3' }),
  ]
  const subs: ReactionSubscription[] = [{ id: 's', react: 'notify' }]
  const out = decideReactions(events, subs)
  assert.deepEqual(out.map((r) => r.eventId), ['crit', 'high', 'low'])
  assert.ok(out.every((r) => r.kind !== 'ignore'))
  ok('decideReactions drops ignores and sorts by urgency')
}

{
  // Purity: inputs are not mutated.
  const ev = mkEvent()
  const subs: ReactionSubscription[] = [{ id: 's', react: 'review' }]
  const snapshotEv = JSON.stringify(ev)
  const snapshotSubs = JSON.stringify(subs)
  decideReactions([ev], subs)
  assert.equal(JSON.stringify(ev), snapshotEv)
  assert.equal(JSON.stringify(subs), snapshotSubs)
  ok('inputs are not mutated')
}

console.log(`\n${passed} assertions passed.`)
