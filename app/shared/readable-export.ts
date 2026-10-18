export type ReadableExportFormat = 'markdown' | 'text' | 'html'
export interface ReadableExportRequest {
  threadId: string
  format: ReadableExportFormat
  scope: 'conversation' | 'selected'
  itemIds?: string[]
  /** Only the active preserved timeline is exported; other branches need their own preview. */
  branch: 'active'
}
export interface ReadableExportOmission { kind: string; label: string; count: number }
export interface ReadableExportMessage { role: 'user' | 'assistant'; text: string; ordinal: number }
export interface ReadableExportPreview {
  approvalId: string
  expiresAt: number
  revision: string
  threadId: string
  format: ReadableExportFormat
  scope: 'conversation' | 'selected'
  branchLabel: string
  title: string
  messages: ReadableExportMessage[]
  body: string
  bytes: number
  messageCount: number
  omissions: ReadableExportOmission[]
  redactions: { kind: string; count: number }[]
  suggestedName: string
  mimeType: string
  redacted: true
  anonymized: true
}
export interface ReadableExportSaveRequest { approvalId: string }
export type ReadableExportSaveResult = { saved: true; bytes: number } | { saved: false; canceled?: boolean; error?: string }
export interface ReadableExportApi {
  readableExportPreview(request: ReadableExportRequest): Promise<ReadableExportPreview>
  readableExportSave(request: ReadableExportSaveRequest): Promise<ReadableExportSaveResult>
}
