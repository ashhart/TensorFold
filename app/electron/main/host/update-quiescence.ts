export interface UpdateQuiescence {
  ready: boolean
  threadControllers: number
  roomControllers: number
  missionControllers: number
  runningMissions: number
  pendingApprovals: number
  runningThreads: number
  activeJobs: number
  runningAutomations: number
  liveTerminals: number
  mobileOperations: number
  engineRequests: number
  modelLoading: boolean
  modelDownloads: number
}

export function assertUpdateQuiescent(state: UpdateQuiescence): void {
  if (!state.ready) throw new Error('Wait for TensorFold to finish starting before installing an update.')
  const counts = [state.threadControllers, state.roomControllers, state.missionControllers, state.runningMissions, state.pendingApprovals, state.runningThreads, state.activeJobs, state.runningAutomations, state.liveTerminals, state.mobileOperations, state.engineRequests, state.modelDownloads]
  if (state.modelLoading || counts.some((count) => !Number.isSafeInteger(count) || count < 0 || count > 0)) {
    throw new Error('Finish or stop active workers, approvals, model downloads and terminals before installing this update.')
  }
}
