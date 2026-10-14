import { makeRemoteControlToken } from '../../store'
import { isSecretMask } from '../../agent/security'
import { stampSecretRotations } from '../../agent/secret-rotation'
import { spineEligible } from '../../agent/incognito'
import { spineInputForAgentEvent, type EventSpineHost } from '../../agent/event-spine-host'
import { ensureAgentEventTrace } from '../../agent/trace'
import { observerEventForAgentEvent, type ObserverHub } from '../../agent/observer-hooks'
import { emitOutboundWebhooks } from './webhooks'
import { emitJobLifecycle } from './orchestration-events'
import { fanOutHostEvent, sendToRenderer } from './renderer-events'
import { store } from '../state'
import type { AgentEvent, Automation, Incident, Job, Mission, RoomStreamEvent } from '../../agent/types'

export interface EmitHelpersDeps {
  // remoteServer is constructed at module scope before the helpers are first invoked, but the
  // late binding (vs a captured value) keeps the dep stable across the eventual assignment.
  remoteServer(): { publish(event: { kind: string; data: unknown }): void }
  // spineHost is forward-declared and only assigned during app-ready; accessor keeps the
  // pre-ready window safe (the helper degrades to a no-op record when it is null).
  spineHost(): EventSpineHost | null
  // observerHub is created later in the bootstrap than the emit helpers; accessor lets the
  // factory be constructed before it exists while lookups stay lazy (it is only read inside
  // emitToRenderer, which is never invoked until after boot completes).
  observerHub(): ObserverHub
  persistMissions(): void
  // Sees every agent event after delivery (native run notifications and the dock badge).
  onAgentEvent?(e: AgentEvent): void
}

export interface EmitHelpers {
  emitRoom(e: RoomStreamEvent): void
  emitTicket(t: import('../../agent/types').Ticket): void
  emitAutomation(a: Automation): void
  emitJob(j: Job): void
  emitMission(m: Mission): void
  emitIncident(inc: Incident): void
  replaceMission(next: Mission): Mission
  getMission(id: string): Mission
  ensureRemoteControlToken(): void
  emitToRenderer(e: AgentEvent): void
}

/**
 * The fan-out emit layer: every domain mutation (room/ticket/automation/job/mission/incident)
 * is mirrored to (a) the renderer over IPC, (b) the remote-control SSE projection, and (c) the
 * outbound webhooks — via the shared `fanOutHostEvent` helper. `emitToRenderer` is the agent-event
 * equivalent PLUS the ordered-spine + observer bridges.
 *
 * Extracted verbatim from main-app.ts. The factory closes over `store` (singleton), the webhooks
 * /orchestration/renderer helpers (module bindings), and the two late-bound collaborators
 * (`remoteServer`, `spineHost`) via accessors so identity and lifecycle stay exactly as before.
 */
export function createEmitHelpers(deps: EmitHelpersDeps): EmitHelpers {
  const { remoteServer, spineHost, observerHub: observerHubAccessor, persistMissions, onAgentEvent } = deps

  function emitRoom(e: RoomStreamEvent) {
    fanOutHostEvent('room:event', e, [
      { label: 'remote room projection', deliver: () => remoteServer().publish({ kind: 'room', data: e }) },
      { label: 'room webhook', deliver: () => emitOutboundWebhooks('room', e) },
    ])
  }
  function emitTicket(t: import('../../agent/types').Ticket) {
    fanOutHostEvent('ticket:update', t, [
      { label: 'remote ticket projection', deliver: () => remoteServer().publish({ kind: 'ticket', data: t }) },
      { label: 'ticket webhook', deliver: () => emitOutboundWebhooks('ticket', t) },
    ])
  }
  function emitAutomation(a: Automation) {
    fanOutHostEvent('automation:update', a, [
      { label: 'remote automation projection', deliver: () => remoteServer().publish({ kind: 'automation', data: a }) },
      { label: 'automation webhook', deliver: () => emitOutboundWebhooks('automation', a) },
    ])
  }
  function emitJob(j: Job) {
    if (j.incognito || store.threads.some((thread) => thread.incognito && [j.threadId, j.currentThreadId, j.kind === 'thread' ? j.owner : null].includes(thread.id))) {
      sendToRenderer('job:update', j)
      return
    }
    fanOutHostEvent('job:update', j, [
      { label: 'remote job projection', deliver: () => remoteServer().publish({ kind: 'job', data: j }) },
      // Live push (additive, alongside the existing job:update mirror): a typed
      // job.status_changed broker event so the Command Centre can update its job list the
      // moment status flips, without re-polling jobs:get.
      { label: 'job lifecycle', deliver: () => emitJobLifecycle(j.id, j.status, j.kind, { missionId: j.missionId ?? null, roomId: j.roomId ?? null }) },
    ])
  }

  function ensureRemoteControlToken(): void {
    if (!store.settings.remoteControl.authToken || isSecretMask(store.settings.remoteControl.authToken)) {
      const current = store.settings.remoteControl
      const next = { ...current, authToken: makeRemoteControlToken() }
      store.settings = { ...store.settings, remoteControl: stampSecretRotations(next, current, { authToken: [next.authToken, current.authToken] }) }
    }
  }

  function emitMission(m: Mission) {
    fanOutHostEvent('mission:update', m, [
      { label: 'remote mission projection', deliver: () => remoteServer().publish({ kind: 'mission', data: m }) },
      { label: 'mission webhook', deliver: () => emitOutboundWebhooks('mission', m) },
    ])
  }
  function emitIncident(inc: Incident) { sendToRenderer('incident:update', inc) }
  function replaceMission(next: Mission): Mission {
    store.missions = store.missions.map((mission) => mission.id === next.id ? next : mission)
    persistMissions()
    emitMission(next)
    return next
  }

  function getMission(id: string): Mission {
    const mission = store.missions.find((m) => m.id === id)
    if (!mission) throw new Error('Mission not found')
    return mission
  }

  function emitToRenderer(e: AgentEvent) {
    const traced = ensureAgentEventTrace(e, 'main.emit')
    const thread = store.threads.find((thread) => thread.id === traced.threadId)
    // Incognito is local live state, including its titles, question answers and
    // derived tool metadata; no remote replay, webhook or observer may retain it.
    if (thread?.incognito) {
      sendToRenderer('agent:event', traced)
      return
    }
    // Phone-derived approval labels can contain captured screen text. Only the
    // owner's UI projection may receive them; automatic sinks need a known,
    // non-phone conversation, including after a phone grant during the run.
    const automaticSinksAllowed = !!thread && !thread.phoneLocalOnly
    fanOutHostEvent('agent:event', traced, [
      { label: 'remote agent projection', deliver: () => remoteServer().publish({ kind: 'agent', data: { ...traced, trace: { ...traced.trace, surface: traced.trace?.surface ?? 'remote' } } }) },
      // Per-token deltas never reach webhooks: each would be a DNS-checked POST, a persist and an audit entry.
      ...(!automaticSinksAllowed || traced.type === 'item_delta' ? [] : [{ label: 'agent webhook', deliver: () => emitOutboundWebhooks('agent', { ...traced, trace: { ...traced.trace, surface: traced.trace?.surface ?? 'webhook' } }) }]),
    ])
    if (!automaticSinksAllowed) return
    // Observer bridge into the ordered spine: structural events only (deltas skipped), lean
    // payloads (ids/states — never message text or tool args). Failures never block the run.
    try {
      // Incognito threads leave no journal trace — drop their events pre-spine.
      if (spineEligible(store.threads.find((t) => t.id === traced.threadId))) {
        const input = spineInputForAgentEvent(traced)
        const host = spineHost()
        if (input && host) host.record(input)
      }
    } catch (error) {
      console.error('[event-spine] record failed:', error)
    }
    // Observer hooks (VST-012): the same structural whitelist, sanitized + fail-open. A
    // run 'done'/'error' also closes the session for observers.
    const observerHub = observerHubAccessor()
    const observed = observerEventForAgentEvent(traced)
    if (observed) observerHub.emit(observed.kind, { threadId: traced.threadId, runId: traced.runId, payload: observed.payload })
    if (traced.type === 'done' || traced.type === 'error') {
      observerHub.emit('session.end', { threadId: traced.threadId, runId: traced.runId, payload: { ok: traced.type === 'done' } })
    }
    if (traced.type !== 'item_delta') {
      try { onAgentEvent?.(traced) } catch (error) { console.error('[run-attention] failed:', error) }
    }
  }

  return {
    emitRoom,
    emitTicket,
    emitAutomation,
    emitJob,
    emitMission,
    emitIncident,
    replaceMission,
    getMission,
    ensureRemoteControlToken,
    emitToRenderer,
  }
}
