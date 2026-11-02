import { assertClusterWorkspaceIdle } from '../host/cluster-runtime'
import { registerModelHelperRuntime } from '../host/model-helper-runtime'
import { registerModelQualificationRuntime } from '../host/model-qualification-runtime'
import type { BackupHostState } from '../host/workspace-backup/repository'
import { registerWorkspaceBackupRuntime } from '../host/workspace-backup/runtime'
import { readProtectedJson, writeProtectedJson } from '../../store'
import { beginComposerDraftPurge, endComposerDraftPurge } from '../../store/composer-drafts'
import { normalizeAgentPresets } from '../../../shared/bot-policy'
import { app, dialog } from 'electron'
import { saveSettings, saveThreads, saveProjects } from '../../store'
import { vontraHomeDir } from '../../store'
import { store } from '../state'
import { handle } from '../host/ipc'
import { preserveStateForUpdate } from '../host/persistence'
import { requireSecurityRole } from '../host/audit'
import { configureUpdateInstallSafety, registerUpdateInstallIpc } from '../host/updates'
import { assertUpdateQuiescent } from '../host/update-quiescence'
import { WorkspaceMutationGuard } from '../host/workspace-mutation-guard'
import { liveTerminalCount } from '../host/terminal-runtime'
import { createMobileBridgeRuntime } from '../host/mobile-bridge-runtime'
import { sendToRenderer } from '../host/renderer-events'
import { engineService } from '../host/engine-runtime'
import { requestRendererDraftFlush } from '../host/ipc/composer-draft-ipc'

interface WorkspaceServicesDeps {
  aborters: ReadonlyMap<string, AbortController>
  roomAborters: ReadonlyMap<string, AbortController>
  missionAborters: ReadonlyMap<string, AbortController>
  runningMissions: ReadonlySet<string>
  pendingApprovals: { readonly size: number }
  pendingApprovalMeta: { readonly size: number }
  missionApprovalResolvers: { readonly size: number }
  mobileBridgeRuntime: Pick<ReturnType<typeof createMobileBridgeRuntime>, 'activeOperations'>
  workspaceMutationGuard: WorkspaceMutationGuard
}

export async function persistWorkspaceSnapshot(next: BackupHostState): Promise<void> {
  next.settings.agentPresets = normalizeAgentPresets(next.settings.agentPresets ?? [], true)
  const results = await Promise.allSettled([saveSettings(next.settings), saveThreads(next.threads), saveProjects(next.projects)])
  if (results.some(result => result.status === 'rejected')) throw new Error('Workspace storage could not complete; restore will roll back the reviewed items.')
}

export function registerApplicationWorkspaceServices(deps: WorkspaceServicesDeps) {
  const { aborters, roomAborters, missionAborters, runningMissions, pendingApprovals,
    pendingApprovalMeta, missionApprovalResolvers, mobileBridgeRuntime, workspaceMutationGuard } = deps
  function assertWorkspaceIdle(): void {
    assertClusterWorkspaceIdle()
    assertUpdateQuiescent({
      ready: app.isReady() && !!store.settings,
      threadControllers: aborters.size,
      roomControllers: roomAborters.size,
      missionControllers: missionAborters.size,
      runningMissions: runningMissions.size,
      pendingApprovals: pendingApprovals.size + pendingApprovalMeta.size + missionApprovalResolvers.size,
      runningThreads: store.threads.filter((thread) => thread.status === 'running').length,
      activeJobs: store.jobs.filter((job) => ['queued', 'running', 'waiting_approval'].includes(job.status)).length,
      runningAutomations: store.automations.filter((automation) => automation.runs?.some((run) => run.status === 'running')).length,
      liveTerminals: liveTerminalCount(),
      mobileOperations: mobileBridgeRuntime.activeOperations(),
      engineRequests: engineService()?.supervisor.activeRequestCount ?? 0,
      modelLoading: engineService()?.supervisor.status().state === 'starting',
      modelDownloads: engineService()?.pullList().filter((pull) => ['queued', 'running', 'verifying'].includes(pull.state)).length ?? 0,
    })
  }
  registerModelHelperRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['models:helper'] }), assertIdle: assertWorkspaceIdle })
  registerModelQualificationRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['models:qualification'] }), assertIdle: assertWorkspaceIdle })
  registerWorkspaceBackupRuntime({
    handle,
    authorize: () => requireSecurityRole('owner', 'back up or restore this workspace'),
    profileRoot: vontraHomeDir,
    getState: () => ({ settings: store.settings, threads: store.threads, projects: store.projects }),
    setState: next => { store.settings = next.settings; store.threads = next.threads; store.projects = next.projects },
    persist: persistWorkspaceSnapshot,
    readProtectedJson, writeProtectedJson,
    flushDrafts: event => requestRendererDraftFlush(event.sender),
    pauseDraftWrites: beginComposerDraftPurge,
    resumeDraftWrites: endComposerDraftPurge,
    acquireLease: () => workspaceMutationGuard.acquireGlobal('Workspace backup or restore'),
    assertIdle: assertWorkspaceIdle,
    chooseOpen: async () => { const result = await dialog.showOpenDialog({ title: 'Choose an encrypted workspace backup', properties: ['openFile'], filters: [{ name: 'TensorFold backup', extensions: ['tfbackup'] }] }); return result.canceled ? null : result.filePaths[0] ?? null },
    chooseSave: async () => { const result = await dialog.showSaveDialog({ title: 'Save encrypted workspace backup', defaultPath: 'tensorfold-workspace.tfbackup', filters: [{ name: 'TensorFold backup', extensions: ['tfbackup'] }] }); return result.canceled ? null : result.filePath ?? null },
    publishRestored: () => { sendToRenderer('composer-drafts:reset', null); setImmediate(() => store.win?.webContents.reload()) },
  })
  configureUpdateInstallSafety({
    acquireLease: () => workspaceMutationGuard.acquireGlobal('App update installation'),
    assertIdle: assertWorkspaceIdle,
    preserve: preserveStateForUpdate,
  })
  registerUpdateInstallIpc()

}
