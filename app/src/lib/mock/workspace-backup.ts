import type { WorkspaceBackupApi } from '../../../shared/workspace-backup'

/** Browser demos never pretend to create a recoverable desktop backup. */
export function createMockWorkspaceBackupApi(): WorkspaceBackupApi {
  const unavailable = async (): Promise<never> => { throw new Error('Encrypted workspace backup requires the local desktop app.') }
  return {
    async workspaceBackupList() { return { available: false, reason: 'Open the local desktop app to create or restore an encrypted workspace backup.', conversations: [], projects: [], bots: [] } },
    workspaceBackupPreviewExport: unavailable,
    workspaceBackupExport: unavailable,
    workspaceBackupPreviewRestore: unavailable,
    workspaceBackupApplyRestore: unavailable,
    async workspaceBackupCancel() {},
  }
}
