import { createHash } from 'node:crypto'
import type { BackupJson, BackupRecord, WorkspaceBackupAttachment, WorkspaceBackupCounts, WorkspaceBackupData, WorkspaceBackupSelection } from '../../shared/workspace-backup'
import { redactSecretValues } from '../agent/secret-detection'
import { MAX_BACKUP_JSON_BYTES } from './crypto'
import { validatePortableData } from './record-validation'
import { portableBot, portableConversation, portableImportedRecovery } from './portable-records'

export const MAX_BLOB_BYTES = 16 * 1024 * 1024
export const MAX_TOTAL_BLOB_BYTES = 32 * 1024 * 1024
const HASH = /^[a-f0-9]{64}$/
const COLLECTIONS = ['conversations', 'projects', 'bots', 'drafts', 'attachments'] as const
const SECRET_KEY = /(?:secret|password|passphrase|passwd|api.?key|access.?token|auth.?token|refresh.?token|bot.?token|credential|cookie|authorization|private.?key|profile.?key|^env$|cliEnv)/i
const FORBIDDEN = new Set(['__proto__', 'prototype', 'constructor', 'jobs', 'missions', 'automations', 'pairings', 'profileKey', 'sourcePath', 'appAction', 'question', 'activeRunContract', 'codexResumeCursor', 'claudeResumeCursor', 'openCodeResumeCursor', 'piResumeCursor', 'ompResumeCursor', 'cursorResumeCursor'])
export interface BackupArchive {
  schema: 'tensorfold.workspace-backup/1'
  createdAt: number
  selection: WorkspaceBackupSelection
  data: WorkspaceBackupData
  blobs: Array<{ id: string; base64: string }>
  warnings: string[]
}
export interface BackupSnapshot { revision: string; data: WorkspaceBackupData; blobs: ReadonlyMap<string, Uint8Array> }

export const backupDigest = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex')
function fail(): never { throw new Error('Invalid or unsupported workspace backup data.') }
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail()
  return value as Record<string, unknown>
}
function string(value: unknown, max = 256): string {
  if (typeof value !== 'string' || !value || value.length > max || /\0/u.test(value)) fail()
  return value
}
function identity(value: unknown): string {
  const id = string(value)
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$/u.test(id)) fail()
  return id
}
function fields(value: Record<string, unknown>, allowed: string[]) { if (Object.keys(value).some((key) => !allowed.includes(key))) fail() }

/** Bound structure before cloning/sanitizing; reject dangerous keys and unsupported object types. */
function assertJson(value: unknown, depth = 0, budget = { nodes: 0 }): asserts value is BackupJson {
  if (++budget.nodes > 200_000 || depth > 24) fail()
  if (value === null || typeof value === 'boolean') return
  if (typeof value === 'number') { if (!Number.isFinite(value)) fail(); return }
  if (typeof value === 'string') { if (value.length > 1_000_000 || /\0/u.test(value)) fail(); return }
  if (Array.isArray(value)) { if (value.length > 10_000) fail(); for (const entry of value) assertJson(entry, depth + 1, budget); return }
  const obj = record(value)
  for (const [key, entry] of Object.entries(obj)) { if (key.length > 160 || ['__proto__', 'prototype', 'constructor'].includes(key)) fail(); assertJson(entry, depth + 1, budget) }
}

function sanitize(value: BackupJson, omitted: { fields: number; values: number }): BackupJson {
  if (typeof value === 'string') {
    if (value.startsWith('safe:v1:') || value.startsWith('safe-json:v1:')) { omitted.values++; return '<credential excluded>' }
    const scrubbed = redactSecretValues(value, '<credential excluded>')
    omitted.values += scrubbed.hits.length
    return scrubbed.text
  }
  if (Array.isArray(value)) return value.map((entry) => sanitize(entry, omitted))
  if (value && typeof value === 'object') {
    const output: BackupRecord = {}
    for (const [key, entry] of Object.entries(value)) {
      if (SECRET_KEY.test(key) || FORBIDDEN.has(key) || key === 'token' || key === 'auth') { omitted.fields++; continue }
      output[key] = sanitize(entry, omitted)
    }
    return output
  }
  return value
}
const pick = (value: BackupRecord, allowed: string[]): BackupRecord => Object.fromEntries(Object.entries(value).filter(([key]) => allowed.includes(key)))
const SETTING_FIELDS = ['uiPreferences', 'modelHelperDefaults', 'theme', 'themeMode', 'themePreset', 'desktopMode', 'desktopOnboardingComplete', 'activeModel', 'activeProviderId', 'engineModelProfiles', 'notifyRuns', 'showGitInitPrompt', 'confidentialTerms']
const THREAD_FIELDS = ['id', 'title', 'kind', 'projectId', 'providerId', 'model', 'createdAt', 'updatedAt', 'apiMessages', 'timeline', 'messageEditRevisions', 'conversationRevision', 'documentSourceIds', 'memoryPolicy', 'phoneLocalOnly', 'googleLocalOnly', 'archivedAt', 'notepad', 'annotations', 'parentId', 'forkedAt', 'bot', 'runRecovery']
const PROJECT_FIELDS = ['id', 'name', 'memory', 'team', 'mode', 'localOnly', 'tags', 'archivedAt', 'createdAt', 'updatedAt']
const BOT_FIELDS = ['id', 'name', 'role', 'lane', 'providerId', 'model', 'instructions', 'approvalScope', 'toolScope', 'toolScopeProblem', 'executionTarget', 'createdAt', 'updatedAt', 'archivedAt']

export function validateSelection(value: unknown): WorkspaceBackupSelection {
  const raw = record(value)
  fields(raw, ['settings', 'conversationIds', 'projectIds', 'botIds', 'drafts', 'attachments'])
  const output: WorkspaceBackupSelection = {}
  for (const key of ['settings', 'drafts', 'attachments'] as const) {
    if (raw[key] !== undefined) { if (typeof raw[key] !== 'boolean') fail(); output[key] = raw[key] as boolean }
  }
  for (const key of ['conversationIds', 'projectIds', 'botIds'] as const) {
    if (raw[key] !== undefined) {
      if (!Array.isArray(raw[key]) || raw[key].length > 10_000) fail()
      output[key] = [...new Set(raw[key].map(identity))]
    }
  }
  return output
}

function validateAttachments(input: unknown): WorkspaceBackupAttachment[] {
  if (!Array.isArray(input) || input.length > 10_000) fail()
  const ids = new Set<string>()
  return input.map((entry) => {
    const raw = record(entry)
    fields(raw, ['id', 'name', 'mediaType', 'originalBlobId', 'parsedBlobId'])
    const id = identity(raw.id), name = string(raw.name, 240), mediaType = string(raw.mediaType, 160)
    if (ids.has(id) || /[\\/:\u0000-\u001f\u007f]/u.test(name) || name === '.' || name === '..' || /[\\/:]/u.test(id)) fail()
    ids.add(id)
    for (const key of ['originalBlobId', 'parsedBlobId']) if (raw[key] !== null && (typeof raw[key] !== 'string' || !HASH.test(raw[key] as string))) fail()
    return { id, name, mediaType, originalBlobId: raw.originalBlobId as string | null, parsedBlobId: raw.parsedBlobId as string | null }
  })
}

export function validateData(value: unknown, sourceSnapshot = false): WorkspaceBackupData {
  assertJson(value)
  const raw = record(value)
  fields(raw, ['settings', ...COLLECTIONS])
  record(raw.settings)
  for (const key of COLLECTIONS.filter((key) => key !== 'attachments')) {
    if (!Array.isArray(raw[key]) || (raw[key] as unknown[]).length > 10_000) fail()
    const ids = new Set<string>()
    for (const item of raw[key] as unknown[]) { const id = identity(record(item).id); if (ids.has(id)) fail(); ids.add(id) }
  }
  const data = { ...raw, attachments: validateAttachments(raw.attachments) } as unknown as WorkspaceBackupData
  if (!sourceSnapshot) validatePortableData(data)
  return data
}

export function buildBackupArchive(snapshot: BackupSnapshot, input: WorkspaceBackupSelection, now: number): BackupArchive {
  const selection = validateSelection(input), source = validateData(snapshot.data, true), omitted = { fields: 0, values: 0 }
  const data: WorkspaceBackupData = { settings: {}, conversations: [], projects: [], bots: [], drafts: [], attachments: [] }
  const warnings = ['Connection credentials and the profile encryption key are excluded; providers and integrations may need setup again.', 'Source folders, model caches, VM images and active jobs are not included. Imported chats start without document collections; choose their local sources again.']
  if (selection.settings) {
    data.settings = pick(source.settings, SETTING_FIELDS)
    warnings.push('Settings include appearance, model/helper and onboarding preferences plus confidential redaction terms; service integrations, permission rules, automation and runtime configuration are excluded and need separate setup or review.')
  }
  const selected = (items: BackupRecord[], ids: string[] | undefined) => items.filter((item) => ids?.includes(String(item.id)))
  for (const [collection, ids] of [[source.conversations, selection.conversationIds], [source.projects, selection.projectIds], [source.bots, selection.botIds]] as const) {
    if (ids && selected(collection, ids).length < ids.length) warnings.push('Some selected records are no longer present and could not be included.')
  }
  data.conversations = selected(source.conversations, selection.conversationIds).filter((item) => {
    if (item.incognito === true || item.orchestratorHostId) { warnings.push('An incognito or remote conversation was excluded.'); return false }
    return true
  }).map((item) => ({ ...pick(item, THREAD_FIELDS), cwd: '', status: 'idle' }))
  data.projects = selected(source.projects, selection.projectIds).map((item) => ({ ...pick(item, PROJECT_FIELDS), cwd: '' }))
  data.bots = selected(source.bots, selection.botIds).map((item) => pick(item, BOT_FIELDS))
  if (selection.drafts) data.drafts = source.drafts.filter((item) => !(item.scope && typeof item.scope === 'object' && !Array.isArray(item.scope) && item.scope.incognito === true))
  if (selection.attachments) data.attachments = source.attachments
  else if (data.conversations.length || data.drafts.length) warnings.push('Managed attachment files were not selected; references may need the original files again.')
  if (data.conversations.length || data.projects.length) warnings.push('Restored project folders must be selected again; saved history never grants filesystem access.')
  if (data.bots.length) warnings.push('Restored bots are paused and keep their validated confinement target; review their model, permissions and target before running.')
  const clean = sanitize(data as unknown as BackupJson, omitted) as unknown as WorkspaceBackupData
  clean.conversations = clean.conversations.map(portableConversation)
  clean.bots = clean.bots.map(value => portableBot(value))
  validateData(clean)
  if (clean.conversations.length) warnings.push('Live controller identities and pending actions are excluded; uncertain histories require a new outcome review after restore.')
  if (clean.conversations.some(thread => thread.bot !== undefined)) warnings.push('Restored bot chat captures are read-only with no selected tools; review a paused bot and start a new bot chat to grant capabilities.')
  // These privacy markers are part of the original records and survive sanitization unchanged.
  const blobs: BackupArchive['blobs'] = [], used = new Set<string>()
  let total = 0
  for (const attachment of clean.attachments) {
    for (const key of ['originalBlobId', 'parsedBlobId'] as const) {
      const id = attachment[key], bytes = id ? snapshot.blobs.get(id) : undefined
      if (!id || !bytes) { attachment[key] = null; warnings.push(`Attachment ${attachment.name} has no managed ${key === 'originalBlobId' ? 'original' : 'parsed content'}; it must be selected or processed again.`); continue }
      if (used.has(id)) continue
      if (!(bytes instanceof Uint8Array) || bytes.byteLength > MAX_BLOB_BYTES || backupDigest(bytes) !== id) fail()
      total += bytes.byteLength
      if (total > MAX_TOTAL_BLOB_BYTES || blobs.length >= 512) fail()
      used.add(id); blobs.push({ id, base64: Buffer.from(bytes).toString('base64') })
    }
  }
  if (omitted.fields || omitted.values) warnings.push(`Excluded ${omitted.fields} credential/action fields and redacted ${omitted.values} detected credential values.`)
  return { schema: 'tensorfold.workspace-backup/1', createdAt: now, selection, data: clean, blobs, warnings: [...new Set(warnings)].slice(0, 100) }
}

export function encodeArchive(archive: BackupArchive): Buffer {
  const bytes = Buffer.from(JSON.stringify(archive), 'utf8')
  if (bytes.length > MAX_BACKUP_JSON_BYTES) throw new Error('Backup exceeds the supported archive size.')
  return bytes
}

export function parseArchive(bytes: Uint8Array): { archive: BackupArchive; blobs: Map<string, Uint8Array> } {
  if (bytes.byteLength > MAX_BACKUP_JSON_BYTES) fail()
  let raw: Record<string, unknown>
  try { raw = record(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))) } catch { fail() }
  fields(raw, ['schema', 'createdAt', 'selection', 'data', 'blobs', 'warnings'])
  if (raw.schema !== 'tensorfold.workspace-backup/1' || !Number.isSafeInteger(raw.createdAt) || Number(raw.createdAt) < 0) fail()
  const selection = validateSelection(raw.selection), data = validateData(raw.data)
  if (!Array.isArray(raw.blobs) || raw.blobs.length > 512 || !Array.isArray(raw.warnings) || raw.warnings.length > 100) fail()
  const warnings = raw.warnings.map((entry) => string(entry, 2_000)), blobs = new Map<string, Uint8Array>()
  let total = 0
  for (const entry of raw.blobs) {
    const blob = record(entry); fields(blob, ['id', 'base64'])
    const id = string(blob.id), base64 = blob.base64
    if (typeof base64 !== 'string' || base64.length > Math.ceil(MAX_BLOB_BYTES / 3) * 4) fail()
    if (!HASH.test(id) || blobs.has(id) || base64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(base64)) fail()
    const decoded = Buffer.from(base64, 'base64')
    total += decoded.length
    if (decoded.length > MAX_BLOB_BYTES || total > MAX_TOTAL_BLOB_BYTES || decoded.toString('base64') !== base64 || backupDigest(decoded) !== id) fail()
    blobs.set(id, decoded)
  }
  const referenced = new Set<string>()
  for (const item of data.attachments) for (const id of [item.originalBlobId, item.parsedBlobId]) { if (id && !blobs.has(id)) fail(); if (id) referenced.add(id) }
  if ([...blobs.keys()].some((id) => !referenced.has(id))) fail()
  // An authenticated but crafted archive must satisfy the same secret/action exclusion policy.
  const checked = buildBackupArchive({ revision: 'validation', data, blobs }, selection, Number(raw.createdAt))
  const normalized = portableImportedRecovery(data)
  const ordered = (value: unknown): unknown => Array.isArray(value) ? value.map(ordered) : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => [key, ordered(child)])) : value
  if (JSON.stringify(ordered(checked.data)) !== JSON.stringify(ordered(normalized))) throw new Error('Backup contains unsupported credentials, actions or unsafe workspace fields.')
  return { archive: { schema: raw.schema, createdAt: Number(raw.createdAt), selection, data: normalized, blobs: raw.blobs as BackupArchive['blobs'], warnings }, blobs }
}

export function backupCounts(data: WorkspaceBackupData, blobs: ReadonlyMap<string, Uint8Array>): WorkspaceBackupCounts {
  return { conversations: data.conversations.length, projects: data.projects.length, bots: data.bots.length, drafts: data.drafts.length,
    attachments: data.attachments.length, blobs: blobs.size, blobBytes: [...blobs.values()].reduce((sum, value) => sum + value.byteLength, 0) }
}
