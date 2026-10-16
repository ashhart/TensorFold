import { reconcileJobsAfterRestart } from '../../agent/jobs'
import { pauseMission } from '../../agent/missions'
import { newIncidentEventId } from './incidents-ipc'
import type { Mission, MissionStatus, Incident } from '../../agent/types'
import { store } from '../state'

export interface ReconcileJobsAndMissionsDeps {
  persistJobs(): void
  persistMissions(): void
}

export interface ReconcileIncidentsDeps {
  persistIncidents(): void
}

/**
 * Restart orphan reconciliation (jobs + missions). After the persisted jobs/missions load,
 * walk each collection and fix up state left mid-flight by a prior crash/quit:
 *
 *  - jobs: dropped lease owners are reaped; reconciled set is persisted if it changed.
 *  - missions: anything still 'running'/'waiting_approval'/'stalled' (with no live executor
 *    or approval resolver) is demoted to 'paused' so the user can resume it deliberately.
 *    A mission still awaiting an approval is left alone — its durable approval can still land.
 *
 * Extracted verbatim from the main-app.ts whenReady boot sequence. Operates on the shared
 * `store` in place and re-assigns the collection arrays (store.jobs = …) so every reader
 * observes the reconciled state. The caller still runs its own `reconcileStuckJobs()` +
 * `setInterval(reconcileStuckJobs, 60_000)` cadence around this — that is unaffected.
 */
export function reconcileJobsAndMissionsAfterRestart(deps: ReconcileJobsAndMissionsDeps): void {
  const { persistJobs, persistMissions } = deps

  // Jobs: reap lease owners from the prior process.
  {
    const reconciled = reconcileJobsAfterRestart(store.jobs)
    if (reconciled.changed) {
      store.jobs = reconciled.jobs
      persistJobs()
    }
  }

  // Missions: any mission left mid-flight (running / waiting_approval / stalled) by a prior
  // crash/quit has no live executor or approval resolver — demote to paused so the user can
  // resume it deliberately.
  const orphanStatuses = new Set<MissionStatus>(['running', 'waiting_approval', 'stalled'])
  let demotedOrphan = false
  store.missions = store.missions.map((m: Mission) => {
    if (m.status === 'waiting_approval' && m.pendingApprovals.length) return m
    if (orphanStatuses.has(m.status)) { demotedOrphan = true; return pauseMission(m, 'Paused after restart — no live executor. Resume to continue.') }
    return m
  })
  if (demotedOrphan) persistMissions()
}

/**
 * Restart orphan reconciliation (incidents). Runs AFTER `reconcileJobsAndMissionsAfterRestart`
 * (and after the remaining persisted collections load), because it reads reconciled mission
 * status. Reconciles incident state orphaned by a crash/restart: the in-process scan timer
 * AND the mission completion hook both die with the previous process, so a persisted 'running'
 * scan or a mitigation still pointing at a now-dead mission would otherwise be stuck in the
 * UI forever. (missions were already orphan-reconciled above, so a previously-running mission
 * now reads 'paused'.)
 *
 * Extracted verbatim from the main-app.ts whenReady boot sequence; operates on the shared
 * `store` in place.
 */
export function reconcileIncidentsAfterRestart(deps: ReconcileIncidentsDeps): void {
  const { persistIncidents } = deps
  let dirty = false
  for (const inc of store.incidents as Incident[]) {
    if (inc.scan.status === 'running') { inc.scan = { status: 'error', error: 'Scan interrupted by a restart — run it again.', ranAt: Date.now() }; inc.updatedAt = Date.now(); dirty = true }
    for (const m of inc.mitigations) {
      if (!m.missionId || m.status === 'applied') continue
      const mis = store.missions.find((x) => x.id === m.missionId)
      if (mis && mis.status === 'completed') { m.status = 'applied' }
      else { m.missionId = undefined; inc.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: 'Mitigation execution was interrupted by a restart — re-run it if still needed.', at: Date.now(), source: 'agent' }) }
      inc.updatedAt = Date.now(); dirty = true
    }
  }
  if (dirty) persistIncidents()
}
