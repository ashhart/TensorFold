import { quiesceClusterWorkspace } from '../host/cluster-runtime'
import { quiesceQuickAsk, resumeQuickAsk } from '../host/quick-ask/runtime'
import { quiesceModelHelpers, resumeModelHelpers } from '../host/model-helper-runtime'
import { quiesceModelQualifications, resumeModelQualifications } from '../host/model-qualification-runtime'
import { quiesceDocumentLibrary, resumeDocumentLibrary } from '../host/document-library-runtime'
import { stopAllWhatsAppBridges } from '../../agent/whatsapp-bridge'
import { app, dialog } from 'electron'
import { McpHost } from '../../agent/mcp-client'
import { createWebhookServer } from '../../agent/webhook-server'
import { createRemoteControlServer } from '../../agent/remote-control'
import { store } from '../state'
import { stopCaffeinate } from '../host/power'
import { closeBrowserAutomationSessions } from '../host/browser-automation'
import { WorkspaceMutationGuard } from '../host/workspace-mutation-guard'
import { killAllTerminals } from '../host/terminal-runtime'
import { createMobileBridgeRuntime } from '../host/mobile-bridge-runtime'
import { closeAllCodexAppServerSessions } from '../../agent/codex-app-server'
import { type AutomationRuntime } from '../host/automations'
import { createDesktopModeRuntime } from '../host/desktop-mode'
import { type BotPollingRuntime } from '../host/bot-polling-runtime'
import { createTelegramRemoteRuntime } from '../host/telegram-remote-runtime'
import { engineService } from '../host/engine-runtime'
import { createLearningQuitGuard } from '../host/learning-quit'
import { shutdownComputerUse } from '../host/computer-use-host'
import { requestRendererDraftFlush } from '../host/ipc/composer-draft-ipc'
import { createAttachmentTextService } from '../host/attachment-text'
import { initializePhoneHost } from '../host/phone-host'
import { initializeBotVmHost } from '../host/bot-vm-host'
import { initializeUserQuestionHost } from '../host/user-question-host'
import { createUserQuestionPersistence } from '../host/user-question-persistence'
import { HEADLESS_MODE } from '../early-electron-paths'

interface ApplicationShutdownDeps {
  aborters: Map<string, AbortController>
  roomAborters: Map<string, AbortController>
  missionAborters: Map<string, AbortController>
  workspaceMutationGuard: WorkspaceMutationGuard
  automationRuntime: AutomationRuntime
  telegramRemoteRuntime: ReturnType<typeof createTelegramRemoteRuntime>
  botPollingApi: Pick<BotPollingRuntime, 'stopBotControlPolling'>
  desktopModeRuntime: ReturnType<typeof createDesktopModeRuntime>
  attachmentTextService: ReturnType<typeof createAttachmentTextService>
  phones: ReturnType<typeof initializePhoneHost>
  botVms: ReturnType<typeof initializeBotVmHost>
  userQuestions: ReturnType<typeof initializeUserQuestionHost> | null
  questionPersistence: ReturnType<typeof createUserQuestionPersistence>
  mcpHost: McpHost
  webhookServer: ReturnType<typeof createWebhookServer>
  mobileBridgeRuntime: ReturnType<typeof createMobileBridgeRuntime>
  remoteServer: ReturnType<typeof createRemoteControlServer>
}

export function registerApplicationShutdown(deps: ApplicationShutdownDeps) {
  const { aborters, roomAborters, missionAborters, workspaceMutationGuard, automationRuntime,
    telegramRemoteRuntime, botPollingApi, desktopModeRuntime, attachmentTextService, phones,
    botVms, userQuestions, questionPersistence, mcpHost, webhookServer, mobileBridgeRuntime, remoteServer } = deps
  let quitAdmissionPause: { release(): void } | null = null
  function resumeAfterFailedQuit() {
    resumeQuickAsk()
    draftQuitSettled = false
    if (store.win && !store.win.isDestroyed()) store.win.setEnabled(true)

    try {
      resumeDocumentLibrary(); resumeModelQualifications(); resumeModelHelpers(); documentQuitSettled = false
      userQuestions?.resume(); questionQuitSettled = false
      phones.service.resume(); phoneQuitSettled = false
      botVms.resume(); botVmQuitSettled = false
      quitAdmissionPause?.release(); quitAdmissionPause = null
    } catch {
      // Retain the gate until the still-stopping service settles and Quit is retried.
    }
  }
  const deferLearningQuit = createLearningQuitGuard({
    service: engineService,
    quit: () => app.quit(),
    failed: (error) => {
      resumeAfterFailedQuit()
      console.error('[engine] Could not finish learning before quit', error)
      dialog.showErrorBox('Living Weights is still saving', 'TensorFold could not confirm that your private model is safely saved, so it remains open; inspect the learning graph before quitting again.')
    },
  })
  let phoneQuitSettled = false
  const deferPhoneQuit = createLearningQuitGuard({
    service: () => ({ isLearningBusy: () => !phoneQuitSettled, prepareToQuit: async () => { await phones.shutdown(); phoneQuitSettled = true } }),
    quit: () => { setImmediate(() => app.quit()) },
    failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Phone control is stopping', 'TensorFold is waiting for its phone connection to close safely; try Quit again after it stops.') },
  })
  let botVmQuitSettled = false
  const deferBotVmQuit = createLearningQuitGuard({
    service: () => ({ isLearningBusy: () => !botVmQuitSettled, prepareToQuit: async () => { await botVms.shutdown(); botVmQuitSettled = true } }),
    quit: () => { setImmediate(() => app.quit()) },
    failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Bot VMs are stopping', 'TensorFold could not confirm that its isolated bot work has stopped; your VM workspaces are preserved, so inspect the bot status before quitting again.') },
  })
  let questionQuitSettled = false
  const deferQuestionQuit = createLearningQuitGuard({
    service: () => userQuestions ? { isLearningBusy: () => !questionQuitSettled, prepareToQuit: async () => { await userQuestions.shutdown(); await questionPersistence.drain(); questionQuitSettled = true } } : null,
    quit: () => { setImmediate(() => app.quit()) },
    failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Questions are stopping', 'TensorFold could not confirm that its pending questions were safely interrupted; try Quit again after the current save finishes.') },
  })
  let documentQuitSettled = false
  const deferDocumentQuit = createLearningQuitGuard({
    service: () => ({ isLearningBusy: () => !documentQuitSettled, prepareToQuit: async () => { await Promise.all([quiesceDocumentLibrary(), quiesceModelQualifications(), quiesceModelHelpers(), quiesceClusterWorkspace(), quiesceQuickAsk()]); documentQuitSettled = true } }),
    quit: () => { setImmediate(() => app.quit()) },
    failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Document indexing is stopping', 'TensorFold is waiting for document indexing to stop safely; try Quit again after the current scan finishes.') },
  })
  let draftQuitSettled = false
  const deferDraftQuit = createLearningQuitGuard({
    service: () => ({ isLearningBusy: () => !draftQuitSettled, prepareToQuit: async () => { await requestRendererDraftFlush(store.win?.webContents ?? null); draftQuitSettled = true } }),
    quit: () => { setImmediate(() => app.quit()) },
    failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Your draft needs attention', 'TensorFold could not save an unfinished message, so it remains open; retry saving the draft before quitting.') },
  })
  app.on('before-quit', (event) => {
    if (store.win && !store.win.isDestroyed()) store.win.setEnabled(false)
    quitAdmissionPause ??= workspaceMutationGuard.pauseAdmission('TensorFold is quitting')
    telegramRemoteRuntime.cancelPending()

    for (const controller of aborters.values()) controller.abort()
    for (const controller of roomAborters.values()) controller.abort()
    for (const controller of missionAborters.values()) controller.abort()
    automationRuntime?.abortAll()
    if (deferLearningQuit(event)) return
    if (deferQuestionQuit(event)) return
    if (deferPhoneQuit(event)) return
    if (deferBotVmQuit(event)) return

    if (deferDocumentQuit(event)) return
    if (deferDraftQuit(event)) return
    desktopModeRuntime.dispose()
    botPollingApi.stopBotControlPolling()
    stopAllWhatsAppBridges()
    closeBrowserAutomationSessions()
    stopCaffeinate()

    try { for (const controller of aborters.values()) controller.abort() } catch {}
    try { for (const controller of roomAborters.values()) controller.abort() } catch {}
    try { for (const controller of missionAborters.values()) controller.abort() } catch {}
    try { automationRuntime?.abortAll() } catch {}

    try { shutdownComputerUse() } catch {}
    attachmentTextService.shutdown()

    try { killAllTerminals() } catch {}

    try { closeAllCodexAppServerSessions() } catch {}

    try { engineService()?.shutdown(false) } catch {}
    void mcpHost.shutdown()
    void webhookServer.stop()
    void mobileBridgeRuntime.stop()
    void remoteServer.stop()
  })
  app.on('window-all-closed', () => { if (!HEADLESS_MODE && process.platform !== 'darwin' && !desktopModeRuntime.canBackground()) app.quit() })

  return { isQuitting: () => !!quitAdmissionPause, isDraftQuitSettled: () => draftQuitSettled }
}
