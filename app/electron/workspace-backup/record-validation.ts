import { validateMessageQuotes } from '../../shared/message-quotes'
import { validateDocumentSourceIds } from '../../shared/document-collections'
import { LANGUAGE_IDS } from '../../shared/languages'
import type { WorkspaceBackupData } from '../../shared/workspace-backup'
import { normalizeBotToolScope } from '../../shared/bot-tool-policy'
import { parseThreadMemoryPolicy } from '../../shared/thread-memory-policy'
import { validateComposerDraftScope, validateComposerDraftSnapshot } from '../../shared/composer-drafts'
import { validateEngineModelProfileSets } from '../../shared/model-profiles'
import { validateBackupRunRecovery } from './recovery-validation'

export const invalidRecord = (): never => { throw new Error('Invalid or unsupported nested workspace backup record.') }
export function backupObject(value: unknown): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalidRecord()
  return value as Record<string, any>
}
export function backupFields(value: Record<string, any>, allowed: readonly string[]): void { if (Object.keys(value).some(key => !allowed.includes(key))) invalidRecord() }
export function backupText(value: unknown, max = 1_000_000, empty = true): asserts value is string {
  if (typeof value !== 'string' || value.length > max || !empty && !value.trim() || /\0/u.test(value)) invalidRecord()
}
export function backupId(value: unknown): asserts value is string {
  backupText(value, 512, false)
  if (!/^[A-Za-z0-9][A-Za-z0-9_.:/-]*$/u.test(value)) invalidRecord()
}
export function backupNumber(value: unknown, integer = false, min = 0): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || integer && !Number.isSafeInteger(value)) invalidRecord()
}
export function backupList(value: unknown, max = 10_000): asserts value is any[] { if (!Array.isArray(value) || value.length > max) invalidRecord() }
function optionalText(value: Record<string, any>, keys: string[], max = 1_000_000) { for (const key of keys) if (value[key] !== undefined) backupText(value[key], max) }
function optionalNumbers(value: Record<string, any>, keys: string[]) { for (const key of keys) if (value[key] !== undefined) backupNumber(value[key], true) }
function enumValue(value: unknown, values: readonly unknown[]) { if (!values.includes(value)) invalidRecord() }
function bool(value: unknown) { if (typeof value !== 'boolean') invalidRecord() }
function strings(value: unknown, max = 200) { backupList(value, max); for (const item of value) backupText(item) }
export const TIMELINE_FIELDS = ['id', 'kind', 'text', 'tool', 'callId', 'title', 'args', 'command', 'path', 'output', 'approval', 'visualization', 'diff', 'changedPaths', 'state', 'meta', 'createdAt', 'attachments', 'documentEvidence', 'memorySource', 'apiMessageRef', 'quotedMessages']
export const CONVERSATION_FIELDS = ['id', 'title', 'kind', 'projectId', 'providerId', 'model', 'createdAt', 'updatedAt', 'apiMessages', 'timeline', 'messageEditRevisions', 'conversationRevision', 'documentSourceIds', 'memoryPolicy', 'phoneLocalOnly', 'googleLocalOnly', 'archivedAt', 'notepad', 'annotations', 'parentId', 'forkedAt', 'bot', 'runRecovery', 'cwd', 'status']

function imageUrl(value: unknown): void {
  backupText(value, 1_000_000, false)
  if (/^backup-image:(?:[a-f0-9]{64}|unavailable)$/u.test(value)) return
  if (/^data:image\/(?:png|jpeg|jpg|gif|webp|bmp|x-ms-bmp);base64,[A-Za-z0-9+/]*={0,2}$/iu.test(value)) return
  try { const url = new URL(value); if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) invalidRecord() } catch { invalidRecord() }
}
export function validateApiMessages(value: unknown): void {
  backupList(value)
  for (const input of value) {
    const message = backupObject(input)
    backupFields(message, ['role', 'content', 'tool_calls', 'tool_call_id', 'name'])
    enumValue(message.role, ['system', 'user', 'assistant', 'tool'])
    const omittedToolContent = message.content === undefined && message.role === 'assistant' && Array.isArray(message.tool_calls) && message.tool_calls.length > 0
    if (!omittedToolContent && message.content !== null && typeof message.content !== 'string') {
      backupList(message.content, 200)
      for (const inputBlock of message.content) {
        const block = backupObject(inputBlock)
        if (block.type === 'text') { backupFields(block, ['type', 'text']); backupText(block.text) }
        else if (block.type === 'image_url') {
          backupFields(block, ['type', 'image_url'])
          const image = backupObject(block.image_url); backupFields(image, ['url', 'detail']); imageUrl(image.url)
          if (image.detail !== undefined) enumValue(image.detail, ['low', 'high', 'auto'])
        } else invalidRecord()
      }
    } else if (typeof message.content === 'string') backupText(message.content)
    if (message.tool_calls !== undefined) {
      if (message.role !== 'assistant') invalidRecord()
      backupList(message.tool_calls, 200)
      const ids = new Set<string>()
      for (const inputCall of message.tool_calls) {
        const call = backupObject(inputCall), fn = backupObject(call.function)
        backupFields(call, ['id', 'type', 'function']); backupFields(fn, ['name', 'arguments'])
        backupId(call.id); if (ids.has(call.id) || call.type !== 'function') invalidRecord(); ids.add(call.id)
        backupId(fn.name); backupText(fn.arguments)
      }
    }
    if (message.tool_call_id !== undefined) backupId(message.tool_call_id)
    if (message.name !== undefined) backupId(message.name)
  }
}

function validateAttachment(input: unknown): void {
  const value = backupObject(input)
  backupFields(value, ['id', 'kind', 'name', 'mime', 'dataUrl', 'text', 'bytes', 'status', 'error', 'warnings', 'previewDataUrl', 'source', 'extraction', 'managedBackupRef'])
  backupId(value.id); enumValue(value.kind, ['image', 'document']); backupText(value.name, 512, false); backupText(value.mime, 160, false)
  if (!/^[a-z0-9!#$&^_.+*-]+\/[a-z0-9!#$&^_.+*-]+$/iu.test(value.mime)) invalidRecord()
  optionalText(value, ['text', 'error']); optionalNumbers(value, ['bytes'])
  if (value.status !== undefined) enumValue(value.status, ['processing', 'ready', 'error', 'needs-reselection', 'review'])
  if (value.warnings !== undefined) strings(value.warnings)
  for (const key of ['dataUrl', 'previewDataUrl']) if (value[key] !== undefined) {
    backupText(value[key]); if (!/^data:image\//iu.test(value[key])) invalidRecord(); imageUrl(value[key])
  }
  if (value.managedBackupRef !== undefined && (typeof value.managedBackupRef !== 'string' || !/^[a-f0-9]{64}$/u.test(value.managedBackupRef))) invalidRecord()
  if (value.source !== undefined) { const source = backupObject(value.source); backupFields(source, ['lastModified', 'relativePath']); backupNumber(source.lastModified); optionalText(source, ['relativePath']) }
  if (value.extraction !== undefined) {
    const extraction = backupObject(value.extraction)
    backupFields(extraction, ['method', 'state', 'text', 'confidence', 'warnings', 'pageCount', 'processedPages', 'truncated'])
    if (extraction.method !== 'apple-vision' || value.kind !== 'document') invalidRecord()
    enumValue(extraction.state, ['review', 'accepted']); backupText(extraction.text)
    if (extraction.confidence !== null) { backupNumber(extraction.confidence); if (extraction.confidence > 1) invalidRecord() }
    strings(extraction.warnings); optionalNumbers(extraction, ['pageCount', 'processedPages']); bool(extraction.truncated)
    if (!Number.isSafeInteger(extraction.pageCount) || !Number.isSafeInteger(extraction.processedPages) || extraction.processedPages > extraction.pageCount) invalidRecord()
  }
}

function validateVisualization(input: unknown): void {
  const value = backupObject(input)
  backupFields(value, ['kind', 'title', 'subtitle', 'caption', 'nodes', 'edges', 'bars', 'metrics', 'events', 'legend'])
  enumValue(value.kind, ['flow', 'bar', 'stats', 'timeline']); backupText(value.title); optionalText(value, ['subtitle', 'caption'])
  for (const group of ['nodes', 'edges', 'bars', 'metrics', 'events', 'legend']) if (value[group] !== undefined) {
    backupList(value[group], 2_000)
    for (const inputItem of value[group]) {
      const item = backupObject(inputItem)
      const allowed: Record<string, string[]> = { nodes: ['id', 'title', 'subtitle', 'tone', 'column', 'row'], edges: ['from', 'to', 'label'], bars: ['label', 'value', 'max', 'detail', 'tone'], metrics: ['label', 'value', 'detail', 'tone'], events: ['title', 'time', 'detail', 'tone'], legend: ['label', 'tone'] }
      backupFields(item, allowed[group])
      optionalText(item, ['title', 'subtitle', 'label', 'detail', 'time'])
      if (item.tone !== undefined) enumValue(item.tone, ['accent', 'muted', 'positive', 'warning', 'danger'])
      if (group === 'nodes') { backupId(item.id); backupText(item.title); optionalNumbers(item, ['column', 'row']) }
      if (group === 'edges') { backupId(item.from); backupId(item.to) }
      if (group === 'bars') { backupText(item.label); backupNumber(item.value, false, -Number.MAX_VALUE); if (item.max !== undefined) backupNumber(item.max, false, -Number.MAX_VALUE) }
      if (group === 'metrics') { backupText(item.label); backupText(item.value) }
      if (group === 'events') backupText(item.title)
      if (group === 'legend') backupText(item.label)
    }
  }
}

export function validateTimeline(value: unknown): void {
  backupList(value)
  const ids = new Set<string>()
  for (const input of value) {
    const item = backupObject(input); backupFields(item, TIMELINE_FIELDS); backupId(item.id)
    if (ids.has(item.id)) invalidRecord(); ids.add(item.id)
    enumValue(item.kind, ['user', 'cue', 'assistant', 'reasoning', 'tool', 'visualization', 'error', 'notice'])
    optionalText(item, ['text', 'tool', 'callId', 'title', 'command', 'path', 'output', 'approval', 'meta']); optionalNumbers(item, ['createdAt'])
    if (item.args !== undefined) backupObject(item.args)
    if (item.state !== undefined) enumValue(item.state, ['pending', 'awaiting', 'running', 'ok', 'error', 'denied'])
    if (item.quotedMessages !== undefined) { if (item.kind !== 'user') invalidRecord(); validateMessageQuotes(item.quotedMessages) }
    if (item.memorySource !== undefined) enumValue(item.memorySource, ['user', 'external'])
    if (item.changedPaths !== undefined) strings(item.changedPaths, 10_000)
    if (item.visualization !== undefined) validateVisualization(item.visualization)
    if (item.diff !== undefined) { backupList(item.diff); for (const rawLine of item.diff) { const line = backupObject(rawLine); backupFields(line, ['type', 'text', 'oldNo', 'newNo']); enumValue(line.type, ['add', 'del', 'ctx', 'hunk']); backupText(line.text); optionalNumbers(line, ['oldNo', 'newNo']) } }
    if (item.attachments !== undefined) { backupList(item.attachments, 200); const seen = new Set<string>(); for (const attachment of item.attachments) { validateAttachment(attachment); if (seen.has(attachment.id)) invalidRecord(); seen.add(attachment.id) } }
    if (item.documentEvidence !== undefined) { backupList(item.documentEvidence, 200); for (const inputEvidence of item.documentEvidence) { const evidence = backupObject(inputEvidence); backupFields(evidence, ['documentId', 'revision', 'passageId']); for (const key of ['documentId', 'revision', 'passageId']) backupText(evidence[key], 512, false) } }
    if (item.apiMessageRef !== undefined) { const reference = backupObject(item.apiMessageRef); backupFields(reference, ['ordinal', 'fingerprint', 'compactionCount']); backupNumber(reference.ordinal, true); backupNumber(reference.compactionCount, true); backupText(reference.fingerprint, 256, false) }
  }
}

export function validateBotRecord(input: unknown, captured = false): void {
  const value = backupObject(input)
  const common = ['name', 'role', 'instructions', 'providerId', 'providerAccountId', 'model', 'approvalScope', 'executionTarget', 'toolScope']
  backupFields(value, captured ? ['presetId', ...common, 'capturedAt'] : ['id', ...common, 'lane', 'paused', 'archivedAt', 'createdAt', 'updatedAt'])
  backupId(captured ? value.presetId : value.id)
  if (!/^[A-Za-z0-9_-]{1,100}$/u.test(String(captured ? value.presetId : value.id))) invalidRecord()
  backupText(value.name, 80, false)
  optionalText(value, ['role'], 240); optionalText(value, ['instructions'], 16_000); optionalText(value, ['providerId', 'model'], 4096)
  if (value.providerAccountId !== undefined && value.providerAccountId !== null) backupText(value.providerAccountId, 512, false)
  if (value.approvalScope !== undefined) enumValue(value.approvalScope, ['readonly', 'review'])
  if (value.executionTarget !== undefined) enumValue(value.executionTarget, ['workspace', 'vm'])
  normalizeBotToolScope(value.toolScope)
  if (captured) optionalNumbers(value, ['capturedAt'])
  else { if (value.lane !== undefined) enumValue(value.lane, ['planning', 'development', 'review', 'research', 'operations']); if (value.paused !== undefined) bool(value.paused); optionalNumbers(value, ['createdAt', 'updatedAt']); if (value.archivedAt !== undefined && value.archivedAt !== null) backupNumber(value.archivedAt, true) }
}

export function validateConversation(value: unknown): void {
  const thread = backupObject(value); backupFields(thread, CONVERSATION_FIELDS); backupId(thread.id); backupText(thread.title)
  enumValue(thread.kind, ['chat', 'task']); validateTimeline(thread.timeline); validateApiMessages(thread.apiMessages)
  for (const key of ['projectId', 'providerId', 'model', 'parentId']) if (thread[key] !== undefined && thread[key] !== null) backupText(thread[key], 4096, false)
  optionalText(thread, ['cwd', 'notepad', 'annotations']); optionalNumbers(thread, ['createdAt', 'updatedAt', 'conversationRevision', 'forkedAt'])
  for (const key of ['phoneLocalOnly', 'googleLocalOnly']) if (thread[key] !== undefined) bool(thread[key])
  if (thread.archivedAt !== undefined && thread.archivedAt !== null) backupNumber(thread.archivedAt, true)
  if (thread.status !== undefined) enumValue(thread.status, ['idle'])
  if (thread.cwd !== undefined && thread.cwd !== '') invalidRecord()
  if (thread.documentSourceIds !== undefined) validateDocumentSourceIds(thread.documentSourceIds)
  if (thread.memoryPolicy !== undefined) parseThreadMemoryPolicy(thread.memoryPolicy)
  if (thread.bot !== undefined) validateBotRecord(thread.bot, true)
  if (thread.runRecovery !== undefined) validateBackupRunRecovery(thread.runRecovery)
  if (thread.messageEditRevisions !== undefined) {
    backupList(thread.messageEditRevisions, 1_000); const ids = new Set<string>()
    for (const input of thread.messageEditRevisions) { const revision = backupObject(input); backupFields(revision, ['id', 'itemId', 'editedAt', 'conversationRevision', 'timeline', 'apiMessages']); backupId(revision.id); backupId(revision.itemId); if (ids.has(revision.id)) invalidRecord(); ids.add(revision.id); backupNumber(revision.editedAt, true); backupNumber(revision.conversationRevision, true); validateTimeline(revision.timeline); validateApiMessages(revision.apiMessages) }
  }
}

export function validatePortableData(data: WorkspaceBackupData): void {
  for (const thread of data.conversations) validateConversation(thread)
  for (const bot of data.bots) validateBotRecord(bot)
  for (const project of data.projects) {
    backupFields(project, ['id', 'name', 'memory', 'team', 'mode', 'localOnly', 'tags', 'archivedAt', 'createdAt', 'updatedAt', 'cwd'])
    backupId(project.id); backupText(project.name); optionalText(project, ['memory', 'team', 'cwd']); if (project.cwd !== undefined && project.cwd !== '') invalidRecord()
    if (project.mode !== undefined) enumValue(project.mode, ['solo', 'room']); if (project.localOnly !== undefined) bool(project.localOnly); if (project.tags !== undefined) strings(project.tags)
    optionalNumbers(project, ['createdAt', 'updatedAt']); if (project.archivedAt !== undefined && project.archivedAt !== null) backupNumber(project.archivedAt, true)
  }
  for (const draft of data.drafts) { backupFields(draft, ['id', 'scope', 'snapshot']); backupId(draft.id); validateComposerDraftSnapshot(draft.snapshot); if (draft.scope !== undefined) { const scope = validateComposerDraftScope(draft.scope); if (scope.incognito) invalidRecord() } }
  const settings = data.settings
  backupFields(settings, ['uiPreferences', 'modelHelperDefaults', 'theme', 'themeMode', 'themePreset', 'desktopMode', 'desktopOnboardingComplete', 'activeModel', 'activeProviderId', 'engineModelProfiles', 'notifyRuns', 'showGitInitPrompt', 'confidentialTerms'])
  optionalText(settings, ['themeMode', 'themePreset']); if (settings.theme !== undefined) enumValue(settings.theme, ['light', 'dark', 'system'])
  if (settings.desktopMode !== undefined) enumValue(settings.desktopMode, ['minimal', 'full'])
  for (const key of ['desktopOnboardingComplete', 'notifyRuns', 'showGitInitPrompt']) if (settings[key] !== undefined) bool(settings[key])
  for (const key of ['activeModel', 'activeProviderId']) if (settings[key] !== undefined && settings[key] !== null) backupText(settings[key], 4096, false)
  if (settings.engineModelProfiles !== undefined) validateEngineModelProfileSets(settings.engineModelProfiles)
  if (settings.confidentialTerms !== undefined) { backupList(settings.confidentialTerms, 200); for (const term of settings.confidentialTerms) backupText(term, 4_096) }
  if (settings.uiPreferences !== undefined) { const prefs = backupObject(settings.uiPreferences); backupFields(prefs, ['locale', 'direction', 'chatScale', 'codeScale', 'density', 'replyLanguage', 'sendKey']); enumValue(prefs.locale, LANGUAGE_IDS); if (prefs.sendKey !== undefined) enumValue(prefs.sendKey, ['enter', 'modified-enter']); if (prefs.replyLanguage !== undefined) enumValue(prefs.replyLanguage, ['interface', 'auto']); enumValue(prefs.direction, ['auto', 'ltr', 'rtl']); enumValue(prefs.chatScale, [1, 1.15, 1.3, 1.5]); enumValue(prefs.codeScale, [1, 1.15, 1.3, 1.5]); enumValue(prefs.density, ['comfortable', 'compact']) }
  if (settings.modelHelperDefaults !== undefined) { const defaults = backupObject(settings.modelHelperDefaults); backupFields(defaults, ['naming', 'document_extraction', 'search']); for (const input of Object.values(defaults)) { const route = backupObject(input); backupFields(route, ['providerId', 'accountId', 'model']); backupText(route.providerId, 512, false); backupText(route.model, 4096, false); if (route.accountId !== null) backupText(route.accountId, 512, false) } }
}

/** The original redactor processes at most 200 terms; exceeding that could drop existing guards. */
export function mergedBackupPrivacyTerms(current: string[] | undefined, incoming: unknown): string[] | undefined {
  if (incoming === undefined) return current
  backupList(incoming, 200); backupList(current ?? [], 200)
  const terms = new Map<string, string>()
  for (const value of [...(current ?? []), ...incoming]) { backupText(value, 4_096); terms.set(value.trim().toLowerCase(), value) }
  if (terms.size > 200) throw new Error('Combined confidential terms exceed the redaction limit; review privacy terms before restoring settings.')
  return [...terms.values()]
}
