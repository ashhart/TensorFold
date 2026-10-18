import assert from 'node:assert/strict'
import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { BackupRecord, WorkspaceBackupData } from '../../shared/workspace-backup'
import type { RunRecoverySnapshot } from '../../shared/run-recovery'
import type { Settings, Thread } from '../agent/types'
import { assertRecoveryAllowsNewRun } from '../agent/run-outcome-review'
import { isMachineryNotice } from '../../src/lib/turn-blocks'
import { createWorkspaceBackupRepository, type BackupHostState } from '../main/host/workspace-backup/repository'
import { buildBackupArchive, encodeArchive, parseArchive, type BackupArchive } from './archive'
import { encryptWorkspaceBackup } from './crypto'
import { createWorkspaceBackupService } from './service'

const bot = { presetId: 'preset', name: 'Synthetic bot', role: 'Review supplied text', instructions: 'Describe the supplied text.', providerId: 'synthetic', providerAccountId: null, model: 'synthetic', approvalScope: 'review', executionTarget: 'workspace', capturedAt: 1 }
const baseThread: BackupRecord = { id: 'chat', title: 'Synthetic retained chat', kind: 'chat', cwd: '', providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'idle',
  timeline: [{ id: 'notice', kind: 'notice', text: 'Synthetic retained notice' }], apiMessages: [{ role: 'user', content: 'Synthetic retained question' }], bot }
const baseData: WorkspaceBackupData = { settings: {}, conversations: [baseThread], projects: [], bots: [{ id: 'preset', name: 'Synthetic bot', role: 'Review supplied text', lane: 'review', instructions: 'Describe supplied text.', providerId: 'synthetic', model: 'synthetic', approvalScope: 'review', toolScope: { mode: 'selected', names: ['read_file'] }, createdAt: 1, updatedAt: 1 }], drafts: [], attachments: [] }
const archive = buildBackupArchive({ revision: 'synthetic', data: baseData, blobs: new Map() }, { conversationIds: ['chat'], botIds: ['preset'] }, 1)
assert.doesNotThrow(() => parseArchive(encodeArchive(archive)), 'valid retained transcripts remain a positive control')

const attacks: Array<[string, (value: BackupArchive) => void]> = [
  ['timeline-text-object', value => { (value.data.conversations[0].timeline as any)[0].text = { malformed: true } }],
  ['timeline-invalid-kind', value => { (value.data.conversations[0].timeline as any)[0].kind = 'unsupported-kind' }],
  ['timeline-object-id', value => { (value.data.conversations[0].timeline as any)[0].id = { malformed: true } }],
  ['timeline-duplicate-id', value => { (value.data.conversations[0].timeline as any).push(structuredClone((value.data.conversations[0].timeline as any)[0])) }],
  ['api-invalid-role', value => { (value.data.conversations[0].apiMessages as any)[0].role = 'unsupported-role' }],
  ['api-content-object', value => { (value.data.conversations[0].apiMessages as any)[0].content = { malformed: true } }],
  ['api-text-block-object', value => { (value.data.conversations[0].apiMessages as any)[0].content = [{ type: 'text', text: { malformed: true } }] }],
  ['api-toolcall-malformed', value => { (value.data.conversations[0].apiMessages as any)[0].tool_calls = [{ type: 'function', function: { name: { malformed: true }, arguments: [] } }] }],
  ['thread-bot-instructions-object', value => { (value.data.conversations[0].bot as any).instructions = { malformed: true } }],
  ['preset-name-object', value => { value.data.bots[0].name = { malformed: true } }],
  ['preset-toolscope-malformed', value => { value.data.bots[0].toolScope = { mode: 'unsupported', names: [17] } }],
]
const accepted: string[] = []
for (const [name, mutate] of attacks) {
  const crafted = structuredClone(archive); mutate(crafted)
  try { parseArchive(encodeArchive(crafted)); accepted.push(name) } catch { /* Expected rejection. */ }
}

const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-backup-validation-audit-')))
const key = Buffer.alloc(32, 49)
async function writeProtected(file: string, value: unknown) {
  const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key, iv)
  const ciphertext = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()])
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 })
  await fs.writeFile(file, Buffer.concat([iv, cipher.getAuthTag(), ciphertext]), { mode: 0o600 })
}
async function readProtected<T>(file: string, fallback: T): Promise<T> {
  const bytes = await fs.readFile(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!bytes) return fallback
  const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12)); decipher.setAuthTag(bytes.subarray(12, 28))
  return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8'))
}
let state: BackupHostState = { settings: { agentPresets: [] } as unknown as Settings, threads: [], projects: [] }
let committedMalformed = false, rendererFailed = false
try {
  const repository = createWorkspaceBackupRepository({ profileRoot: () => root, getState: () => state, setState: next => { state = next }, validateOwner() {},
    readProtectedJson: readProtected, writeProtectedJson: writeProtected, persist: async next => { committedMalformed ||= typeof next.threads[0]?.timeline[0]?.text === 'object' } })
  const service = createWorkspaceBackupService({ repository })
  const malicious = structuredClone(archive); attacks[0][1](malicious)
  const encrypted = await encryptWorkspaceBackup(encodeArchive(malicious), 'synthetic validation password')
  try {
    const preview = await service.previewRestore({ fileId: 'synthetic-crafted', bytes: encrypted, password: 'synthetic validation password', mode: 'replace' })
    await service.applyRestore({ approvalId: preview.approvalId, fileId: 'synthetic-crafted', bytes: encrypted })
  } catch { /* A validated boundary should reject this before persistence. */ }
  if (state.threads[0]) {
    try { isMachineryNotice(state.threads[0].timeline[0]) } catch { rendererFailed = true }
  }
  const recovery: RunRecoverySnapshot = { runId: 'uncertain-run', state: 'needs_review', reason: 'tool_uncertain', message: 'Synthetic action outcome remains uncertain.', completedTools: 1, completedMutations: 0, pendingOperations: 0, pendingMutations: 0,
    delivery: 'unknown', retry: 'requires_review', retryReason: 'Review the prior outcome before another request.', canStartNewRun: true, updatedAt: 2 }
  const guarded = { ...baseThread, runRecovery: recovery } as unknown as Thread
  assert.throws(() => assertRecoveryAllowsNewRun(guarded), /Review/)
  const recoveryArchive = buildBackupArchive({ revision: 'synthetic', data: { ...baseData, conversations: [guarded as unknown as BackupRecord] }, blobs: new Map() }, { conversationIds: ['chat'] }, 1)
  const restoredRecovery = parseArchive(encodeArchive(recoveryArchive)).archive.data.conversations[0] as unknown as Thread
  let reviewGateLost = false
  try { assertRecoveryAllowsNewRun(restoredRecovery); reviewGateLost = true } catch { /* Expected retained review gate. */ }
  console.log(JSON.stringify({ acceptedMalformedArchives: accepted, committedMalformed, rendererFailed, reviewGateLost }))
  assert.deepEqual({ accepted, committedMalformed, rendererFailed, reviewGateLost }, { accepted: [], committedMalformed: false, rendererFailed: false, reviewGateLost: false }, 'backup input validation must reject unsafe nested records and preserve settled uncertain-run review gates')
  service.dispose()
} finally { await fs.rm(root, { recursive: true, force: true }); key.fill(0) }
