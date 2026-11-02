import { assertClusterWorkspaceIdle, quiesceClusterWorkspace } from '../host/cluster-runtime'
import { quiesceModelHelpers, resumeModelHelpers } from '../host/model-helper-runtime'
import { quiesceModelQualifications, resumeModelQualifications, purgeModelQualifications } from '../host/model-qualification-runtime'
import { quiesceDocumentLibrary, resumeDocumentLibrary, purgeDocumentLibrary } from '../host/document-library-runtime'
import type { SecurityPurgeResult } from '../../agent/types'
import { requireSecurityRole } from '../host/audit'
import { createSecurityPurgeRuntime, buildSecurityPurgeConfig } from '../host/security-purge-runtime'
import { engineService } from '../host/engine-runtime'
import { phoneHost } from '../host/phone-host'
import { getBotVmHost } from '../host/bot-vm-host'
import { getUserQuestionHost } from '../host/user-question-host'
import type { createUserQuestionPersistence } from '../host/user-question-persistence'

type PurgeConfig = Parameters<typeof buildSecurityPurgeConfig>[0]
interface ApplicationSecurityPurgeDeps extends Pick<PurgeConfig, 'aborters' | 'roomAborters' | 'missionAborters' | 'pendingApprovals' | 'pendingApprovalMeta' | 'missionApprovalResolvers' | 'workspaceMutationGuard' | 'automationRuntime' | 'clearThreadCues' | 'checkpoints' | 'pairingRegistry' | 'persistPairingSessions' | 'spineHost' | 'providerStatusRuntime' | 'loopHost'> {
  questionPersistence(): ReturnType<typeof createUserQuestionPersistence>
}

export function createApplicationSecurityPurge(deps: ApplicationSecurityPurgeDeps) {
  const { aborters, roomAborters, missionAborters, pendingApprovals, pendingApprovalMeta, missionApprovalResolvers, workspaceMutationGuard, automationRuntime, clearThreadCues, checkpoints, pairingRegistry, persistPairingSessions, spineHost, providerStatusRuntime, loopHost, questionPersistence } = deps
  const securityPurgeRuntime = createSecurityPurgeRuntime(buildSecurityPurgeConfig({
    quiesceDocumentLibrary: async () => { await Promise.all([quiesceDocumentLibrary(), quiesceModelQualifications(), quiesceModelHelpers(), quiesceClusterWorkspace(), quiesceQuickAsk()]) },
    resumeDocumentLibrary: () => { resumeDocumentLibrary(); resumeModelQualifications(); resumeModelHelpers(); resumeQuickAsk() },
    purgeDocumentLibrary: async () => { await purgeDocumentLibrary(); await purgeModelQualifications(); await purgeQuickAsk() },
    quiescePhoneControl: async () => { await phoneHost()?.service.disconnect() },
    quiesceBotVms: async () => { await getBotVmHost()?.quiesce() },
    purgeBotVmData: async () => { await getBotVmHost()?.purgeOwnedData() },
    resumeBotVms: () => { getBotVmHost()?.resume() },
    quiesceUserQuestions: async () => { await getUserQuestionHost()?.quiesce(); await questionPersistence().drain() },
    resumeUserQuestions: () => { getUserQuestionHost()?.resume() },
    aborters,
    roomAborters,
    missionAborters,
    pendingApprovals,
    pendingApprovalMeta,
    missionApprovalResolvers,
    workspaceMutationGuard,
    automationRuntime,
    clearThreadCues,
    checkpoints,
    pairingRegistry,
    persistPairingSessions,
    spineHost,
    providerStatusRuntime,
    loopHost,
    engineService,
  }))
  async function purgeLocalMemory(): Promise<SecurityPurgeResult> {
    requireSecurityRole('owner', 'purge local memory')
    assertClusterWorkspaceIdle()
    await quiesceClusterWorkspace()
    return securityPurgeRuntime.purgeLocalMemory()
  }

  return purgeLocalMemory
}
import { quiesceQuickAsk, resumeQuickAsk, purgeQuickAsk } from '../host/quick-ask/runtime'
