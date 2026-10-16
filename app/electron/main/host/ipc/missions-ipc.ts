import { buildMissionInspectionSnapshot, buildMissionRuntimeSnapshot, createMission, evaluateMissionHeartbeat } from '../../../agent/missions'
import { recordMissionCreated } from '../../../agent/mission-supervisor'
import type { CreateMissionInput, CreatePrLoopInput, Mission } from '../../../agent/types'
import { store } from '../../state'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { persistMissions } from '../persistence'
import { selectedProviderAccountId } from '../provider-runtime'
import { assertMissionWorkspaceSupported } from '../mission-runtime'

function refreshMissionHeartbeats() {
  store.missions = store.missions.map((mission) => evaluateMissionHeartbeat(mission))
  persistMissions()
}

export interface MissionsIpcDeps {
  getMission(id: string): Mission
  createLocalPrLoop(input: CreatePrLoopInput): Promise<unknown>
}

export function registerMissionsIpc(deps: MissionsIpcDeps) {
  const { getMission, createLocalPrLoop } = deps

handle('missions:snapshot', async () => {
  refreshMissionHeartbeats()
  return buildMissionRuntimeSnapshot(store.missions)
})

handle('missions:inspect', async (_e, id: string) => {
  refreshMissionHeartbeats()
  return buildMissionInspectionSnapshot(getMission(id))
})

handle('missions:create', async (_e, input: CreateMissionInput) => {
  requireSecurityRole('operator', 'create missions')
  // A durable mission must be owned by live host records before it is persisted. In
  // particular, do not accept a missing/archived/mismatched source or an SSH workspace that
  // this local mission runtime cannot execute safely. Derive an omitted project from the
  // source thread so later archive/routing policy sees the same ownership we validated here.
  assertMissionWorkspaceSupported({ projectId: input.projectId ?? null, threadId: input.threadId ?? null })
  const sourceThread = input.threadId ? store.threads.find((thread) => thread.id === input.threadId) ?? null : null
  const projectId = input.projectId ?? sourceThread?.projectId ?? null
  const provider = store.settings.providers.find((p) => p.id === (input.providerId ?? store.settings.activeProviderId))
  const mission = createMission({
    ...input,
    projectId,
    providerId: input.providerId ?? store.settings.activeProviderId,
    providerAccountId: provider ? selectedProviderAccountId(provider, input.providerAccountId) : (input.providerAccountId ?? null),
    model: input.model ?? store.settings.activeModel,
  })
  store.missions.unshift(mission)
  persistMissions()
  // Durable 'created' journal entry + live 'mission.created' broker event (additive —
  // the Command Centre still reconciles via missions:snapshot as the fallback).
  recordMissionCreated(mission)
  return mission
})

handle('prLoop:createLocal', async (_e, input: CreatePrLoopInput) => createLocalPrLoop(input))

}
