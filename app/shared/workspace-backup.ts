export type BackupJson = null | boolean | number | string | BackupJson[] | { [key: string]: BackupJson }
export type BackupRecord = { [key: string]: BackupJson }
export type WorkspaceRestoreMode = 'merge' | 'replace'

export interface WorkspaceBackupAttachment {
  id: string
  name: string
  mediaType: string
  originalBlobId: string | null
  parsedBlobId: string | null
}

/** Logical data only; no profile keys, filesystem trees, active jobs or connection credentials. */
export interface WorkspaceBackupData {
  settings: BackupRecord
  conversations: BackupRecord[]
  projects: BackupRecord[]
  bots: BackupRecord[]
  drafts: BackupRecord[]
  attachments: WorkspaceBackupAttachment[]
}

export interface WorkspaceBackupSelection {
  settings?: boolean
  conversationIds?: string[]
  projectIds?: string[]
  botIds?: string[]
  drafts?: boolean
  attachments?: boolean
}

export interface WorkspaceBackupCounts {
  conversations: number
  projects: number
  bots: number
  drafts: number
  attachments: number
  blobs: number
  blobBytes: number
}

export interface WorkspaceBackupPreview {
  approvalId: string
  operation: 'export' | 'restore'
  mode?: WorkspaceRestoreMode
  counts: WorkspaceBackupCounts
  warnings: string[]
  conflicts: Array<{ collection: 'conversations' | 'projects' | 'bots' | 'drafts' | 'attachments'; id: string; action: 'keep-existing' | 'replace' }>
  expiresAt: number
}

export interface WorkspaceRestoreResult {
  restored: boolean
  cancelled: boolean
  counts: WorkspaceBackupCounts
  warnings: string[]
}

export interface WorkspaceBackupInventory {
  conversations: Array<{ id: string; title: string; archived: boolean }>
  projects: Array<{ id: string; name: string }>
  bots: Array<{ id: string; name: string }>
  available: boolean
  reason?: string
}

export interface WorkspaceBackupApi {
  workspaceBackupList(): Promise<WorkspaceBackupInventory>
  workspaceBackupPreviewExport(selection: WorkspaceBackupSelection): Promise<WorkspaceBackupPreview>
  workspaceBackupExport(input: { approvalId: string; password: string }): Promise<{ saved: boolean; cancelled: boolean; bytes?: number }>
  workspaceBackupPreviewRestore(input: { password: string; mode: WorkspaceRestoreMode }): Promise<WorkspaceBackupPreview | null>
  workspaceBackupApplyRestore(input: { approvalId: string }): Promise<WorkspaceRestoreResult>
  workspaceBackupCancel(): Promise<void>
}
