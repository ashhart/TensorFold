import type { BackupRecord, WorkspaceBackupData } from '../../shared/workspace-backup'
import { normalizeBotToolScope } from '../../shared/bot-tool-policy'
import { TIMELINE_FIELDS, backupList, backupObject, validateApiMessages, validateBotRecord, validateConversation, validateTimeline } from './record-validation'
import { portableRunRecovery } from './recovery-validation'

const pick = (value: BackupRecord, fields: string[]): BackupRecord => Object.fromEntries(Object.entries(value).filter(([key]) => fields.includes(key)))

function portableTimeline(input: unknown): BackupRecord[] {
  backupList(input)
  const items = input.map(value => pick(backupObject(value) as BackupRecord, TIMELINE_FIELDS))
  validateTimeline(items)
  return items.map(item => ({ ...item, ...(['pending', 'awaiting', 'running'].includes(String(item.state)) ? { state: 'denied', meta: [item.meta, 'Historical operation was not resumed from backup.'].filter(Boolean).join(' ') } : {}),
    ...(Array.isArray(item.attachments) ? { attachments: item.attachments.map(input => { const attachment = backupObject(input); return attachment.status === 'processing' ? { ...attachment, status: 'needs-reselection', error: 'Saved file processing was not resumed; select the original again.' } : attachment }) } : {}) }))
}

export function portableBot(input: unknown, captured = false): BackupRecord {
  validateBotRecord(input, captured)
  const value = input as BackupRecord
  const scope = normalizeBotToolScope(value.toolScope)
  return { ...(captured ? { presetId: value.presetId } : { id: value.id }), name: value.name, role: value.role ?? '',
    ...(!captured ? { lane: value.lane ?? 'operations' } : {}), providerId: value.providerId ?? '', providerAccountId: null, model: value.model ?? '', instructions: value.instructions ?? '',
    approvalScope: captured ? 'readonly' : value.approvalScope ?? 'readonly', executionTarget: value.executionTarget ?? 'workspace',
    ...(captured ? { toolScope: { mode: 'selected', names: [] } } : scope ? { toolScope: scope as unknown as BackupRecord } : {}),
    ...(captured ? { capturedAt: value.capturedAt ?? 0 } : { paused: true, ...(value.archivedAt !== undefined ? { archivedAt: value.archivedAt } : {}), createdAt: value.createdAt ?? 0, updatedAt: value.updatedAt ?? 0 }) }
}

export function portableConversation(input: BackupRecord): BackupRecord {
  const hadPending = Array.isArray(input.timeline) && input.timeline.some(value => value && typeof value === 'object' && !Array.isArray(value) && ['pending', 'awaiting', 'running'].includes(String(value.state)))
  const output: BackupRecord = { ...input, documentSourceIds: [], kind: input.kind ?? 'chat', title: input.title ?? 'Untitled conversation', timeline: portableTimeline(input.timeline ?? []), apiMessages: input.apiMessages ?? [], cwd: '', status: 'idle' }
  validateApiMessages(output.apiMessages)
  output.apiMessages = (output.apiMessages as BackupRecord[]).map(message => message.content === undefined && message.role === 'assistant' ? { ...message, content: null } : message)
  const unanswered = new Set<string>()
  for (const message of output.apiMessages as BackupRecord[]) {
    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) for (const call of message.tool_calls) unanswered.add(String(backupObject(call).id))
    if (message.role === 'tool' && typeof message.tool_call_id === 'string') unanswered.delete(message.tool_call_id)
  }
  const unsettledHistory = hadPending || unanswered.size > 0
  if (input.bot !== undefined) output.bot = portableBot(input.bot, true)
  if (input.runRecovery !== undefined) {
    const snapshot = portableRunRecovery(input.runRecovery)
    output.runRecovery = { ...snapshot, ...(unsettledHistory ? { state: 'needs_review', reason: 'restart_interrupted', message: 'The saved history contained an unsettled operation; verify its outcome before continuing.', delivery: 'unknown', retry: 'requires_review', retryReason: 'Historical actions were not resumed.' } : {}) } as unknown as BackupRecord
  }
  else if (unsettledHistory) output.runRecovery = { runId: `backup-${input.id}`, state: 'needs_review', reason: 'restart_interrupted', message: 'The saved history contained an unsettled operation; verify its outcome before continuing.',
    completedTools: 0, completedMutations: 0, pendingOperations: 0, pendingMutations: 0, delivery: 'unknown', retry: 'requires_review', retryReason: 'Historical actions were not resumed.', canStartNewRun: true, updatedAt: 0 }
  if (input.messageEditRevisions !== undefined) {
    backupList(input.messageEditRevisions, 1_000)
    output.messageEditRevisions = input.messageEditRevisions.map(value => { const revision = backupObject(value); return { ...revision, timeline: portableTimeline(revision.timeline) } })
  }
  validateConversation(output)
  return output
}

/** Normalize validated legacy defaults and lifecycle metadata without broadening permissions. */
export function portableImportedRecovery(data: WorkspaceBackupData): WorkspaceBackupData {
  return { ...data, conversations: data.conversations.map(portableConversation), bots: data.bots.map(value => portableBot(value)) }
}
