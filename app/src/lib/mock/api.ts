import { createMockGmailInboxApi } from './gmail-inbox'
import { createMockMailConnectionsApi } from './mail-connections'
import { createMockEngineApi } from './engine'
import { createMockClusterApi } from './clusters'
import { createMockModelQualificationApi } from './model-qualification'
import { createMockWorkspaceBackupApi } from './workspace-backup'
import { createMockReadableExportApi } from './readable-export'
import { createMockConversationSearchApi } from './conversation-search'
import { createMockDocumentLibraryApi } from './document-library'
import { createMockGoogleWorkspaceApi } from './google-workspace'
import { createMockTelegramRemoteApi } from './telegram-remote'
import './fixtures-core'
import './fixtures-data'
import { mockState, uid } from './state'
import { updateEmit } from './fixtures-data'
import { createMockKnowledgeApi } from './api-knowledge'
import { createMockModelBenchmarkApi } from './model-benchmark'
import { createMockPhoneApi } from './phone'
import { createMockBotVmApi } from './bot-vm'
import { createMockUserQuestionApi } from './user-question'
import { createMockBotRecipeApi } from './bot-recipes'
import { createMockWhatsAppApi } from './whatsapp'
import type { TensorFoldStudioApi } from '../ipc'
import { createMockSettingsApi } from './api-settings'
import { createMockRemoteApi } from './api-remote'
import { createMockProvidersApi } from './api-providers'
import { createMockWorkspaceApi } from './api-workspace'
import { createMockConnectionsApi } from './api-connections'
import { createMockProjectsApi } from './api-projects'
import { createMockToolsApi } from './api-tools'

export function createMockApi(): TensorFoldStudioApi {

  const { mockEngineReadiness, ...engineApi } = createMockEngineApi()
  return {
    modelHelperRun: async () => { throw new Error('Local helper jobs require an eligible desktop model; browser preview does not run them.') },
    modelHelperCancel: async () => { },
    reviewRunOutcome: async () => { throw new Error('Outcome review requires a completed desktop run.') },
    getComposerDraft: async () => { throw new Error('Persistent drafts require the desktop app.') },
    saveComposerDraft: async () => { throw new Error('Persistent drafts require the desktop app.') },
    clearComposerDraft: async () => { throw new Error('Persistent drafts require the desktop app.') },
    onComposerDraftFlush: () => () => { },
    onComposerDraftReset: () => () => { },
    acknowledgeComposerDraftFlush: async () => { },
    setDeviceWatchSpeech: async () => { throw new Error('Watch speech permissions require the desktop app.') },
    getAttachmentTextCapability: async () => ({ available: false, method: 'apple-vision', reason: 'Local text extraction requires the macOS desktop app.', maxBytes: 20 * 1024 * 1024, maxPages: 8 }),
    extractAttachmentText: async () => ({ ok: false, error: 'Local text extraction requires the macOS desktop app.' }),
    prepareMessageEdit: async () => { throw new Error('Editing saved messages requires the desktop app.') },
    editMessage: async () => { throw new Error('Editing saved messages requires the desktop app.') },
    ...createMockBotRecipeApi(),
    ...createMockWhatsAppApi(() => mockState.settings, uid),
    ...createMockGoogleWorkspaceApi(uid),
    ...createMockMailConnectionsApi(),
    ...createMockGmailInboxApi(),
    ...createMockTelegramRemoteApi(() => mockState.settings, () => mockState.threads, () => mockState.mockRooms),
    ...createMockBotVmApi(),
    ...createMockUserQuestionApi(),
    ...createMockPhoneApi(),
    platform: 'browser',
    writeClipboard: async (text: string) => {
      try {
        await navigator.clipboard?.writeText(text)
        return { ok: true }
      } catch {
        return { ok: false }
      }
    },
    getUpdateStatus: async () => mockState.mockUpdateStatus,
    checkForUpdates: async () => {
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, state: 'checking', checkedAt: Date.now(), error: null }
      updateEmit(mockState.mockUpdateStatus)
      await new Promise((r) => setTimeout(r, 600))
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, state: 'available', checkedAt: Date.now() }
      updateEmit(mockState.mockUpdateStatus)
      return mockState.mockUpdateStatus
    },
    openUpdateDownload: async () => ({ ok: false }),
    downloadAppUpdate: async () => {
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, automaticInstall: { supported: false, reason: 'Updates can be downloaded and installed in the desktop app.' }, error: 'Browser demo cannot download or install app updates.' }
      updateEmit(mockState.mockUpdateStatus)
      return { ok: false }
    },
    cancelAppUpdateDownload: async () => ({ cancelled: false }),
    installAppUpdate: async () => ({ ok: false }),
    onUpdateStatus: (cb) => { mockState.updateListener = cb; return () => { mockState.updateListener = null } },
    ...createMockKnowledgeApi(),
    ...engineApi,
    ...createMockModelBenchmarkApi(),
    ...createMockReadableExportApi(),
    ...createMockWorkspaceBackupApi(),
    ...createMockClusterApi(),
    ...createMockModelQualificationApi(),
    ...createMockConversationSearchApi(() => mockState.threads),
    ...createMockDocumentLibraryApi(() => mockState.settings),
    getInputHistory: async () => ({ revision: 0, entries: [], partial: false }),
    clearInputHistory: async () => ({ revision: 0, entries: [], partial: false }),
    recallInputHistory: async () => { throw new Error('Input history needs a local desktop conversation.') },
    prepareMessageQuote: async () => { throw new Error('Quoting needs a local desktop conversation.') },
    ...createMockSettingsApi(),
    ...createMockRemoteApi(),
    ...createMockProvidersApi(mockEngineReadiness),
    ...createMockWorkspaceApi(),
    ...createMockConnectionsApi(),
    ...createMockProjectsApi(),
    ...createMockToolsApi(),
  }
}
