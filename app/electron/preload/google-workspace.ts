import type { GoogleWorkspaceApi } from '../../shared/google-workspace-api'

export function googleWorkspacePreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): GoogleWorkspaceApi {
  return {
    googleWorkspaceStatus: () => invoke('googleWorkspace:status'),
    googleWorkspaceImportClient: () => invoke('googleWorkspace:importClient'),
    googleWorkspaceAuthorize: (request) => invoke('googleWorkspace:authorize', request),
    googleWorkspaceCancelAuth: (id) => invoke('googleWorkspace:cancelAuth', id),
    googleWorkspaceDisconnect: (accountId) => invoke('googleWorkspace:disconnect', accountId),
    googleWorkspaceMessages: (accountId, query) => invoke('googleWorkspace:messages', { accountId, query }),
    googleWorkspaceMessage: (accountId, id) => invoke('googleWorkspace:message', { accountId, id }),
    googleWorkspaceThread: (accountId, id) => invoke('googleWorkspace:thread', { accountId, id }),
    googleWorkspaceCalendars: (accountId, query) => invoke('googleWorkspace:calendars', { accountId, query }),
    googleWorkspaceEvents: (accountId, query) => invoke('googleWorkspace:events', { accountId, query }),
    googleWorkspaceAvailability: (accountId, query) => invoke('googleWorkspace:availability', { accountId, query }),
    googleWorkspaceSaveDraft: (accountId, email, id) => invoke('googleWorkspace:saveDraft', { accountId, email, id }),
    googleWorkspaceDrafts: (accountId) => invoke('googleWorkspace:drafts', accountId),
    googleWorkspaceDeleteDraft: (accountId, id) => invoke('googleWorkspace:deleteDraft', { accountId, id }),
    googleWorkspacePrepareAction: (accountId, action) => invoke('googleWorkspace:prepareAction', { accountId, action }),
    googleWorkspaceCancelAction: (receipt) => invoke('googleWorkspace:cancelAction', receipt),
    googleWorkspaceExecuteAction: (receipt) => invoke('googleWorkspace:executeAction', receipt),
  }
}
