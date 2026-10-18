import type { WorkspaceBackupApi } from '../../shared/workspace-backup'

export function workspaceBackupPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): WorkspaceBackupApi {
  return {
    workspaceBackupList: () => invoke('workspace-backup:list'),
    workspaceBackupPreviewExport: selection => invoke('workspace-backup:preview-export', selection),
    workspaceBackupExport: input => invoke('workspace-backup:export', input),
    workspaceBackupPreviewRestore: input => invoke('workspace-backup:preview-restore', input),
    workspaceBackupApplyRestore: input => invoke('workspace-backup:apply-restore', input),
    workspaceBackupCancel: () => invoke('workspace-backup:cancel'),
  }
}
