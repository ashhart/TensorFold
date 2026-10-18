import assert from 'node:assert/strict'
import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { BackupRecord, WorkspaceBackupData } from '../../shared/workspace-backup'
import type { RunRecoverySnapshot } from '../../shared/run-recovery'
import type { Settings, Thread } from '../agent/types'
import { assertRecoveryAllowsNewRun } from '../agent/run-outcome-review'
import { botRunApprovalMode } from '../../shared/bot-policy'
import { createWorkspaceBackupRepository, type BackupHostState } from '../main/host/workspace-backup/repository'
import { buildBackupArchive, encodeArchive, parseArchive, validateData } from './archive'
import { createWorkspaceBackupService } from './service'
import { encryptWorkspaceBackup } from './crypto'
import { mergedBackupPrivacyTerms } from './record-validation'

const recovery: RunRecoverySnapshot = { runId: 'settled-uncertain', state: 'needs_review', reason: 'tool_uncertain', message: 'An earlier synthetic action has an uncertain outcome.', completedTools: 2, completedMutations: 1,
  pendingOperations: 0, pendingMutations: 0, delivery: 'unknown', retry: 'requires_review', retryReason: 'Inspect the outcome before another request.', canStartNewRun: true, updatedAt: 3 }
const image = 'data:image/png;base64,AAAA'
const legacy: BackupRecord = {
  id: 'legacy', kind: 'chat', title: 'Supported synthetic legacy transcript', createdAt: 1, updatedAt: 2,
  apiMessages: [
    { role: 'system', content: 'Synthetic system context.' },
    { role: 'user', content: [{ type: 'text', text: 'Synthetic image question.' }, { type: 'image_url', image_url: { url: image, detail: 'auto' } }] },
    { role: 'assistant', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"synthetic.txt"}' } }] },
    { role: 'tool', name: 'read_file', tool_call_id: 'call_1', content: 'Synthetic file content.' },
    { role: 'assistant', content: 'A retained synthetic answer.' },
  ],
  timeline: [
    { id: 'user', kind: 'user', text: 'Synthetic image question.', attachments: [{ id: 'image', kind: 'image', name: 'Synthetic.png', mime: 'image/png', dataUrl: image }] },
    { id: 'tool', kind: 'tool', tool: 'read_file', callId: 'call_1', title: 'Synthetic read', args: { path: 'synthetic.txt' }, state: 'ok', output: 'Synthetic file content.', diff: [{ type: 'ctx', text: 'Synthetic line', oldNo: 1, newNo: 1 }] },
    { id: 'answer', kind: 'assistant', text: 'A retained synthetic answer.' },
    { id: 'chart', kind: 'visualization', visualization: { kind: 'bar', title: 'Synthetic chart', bars: [{ label: 'Synthetic', value: -1, max: 2 }] } },
  ],
  messageEditRevisions: [{ id: 'revision-1', itemId: 'user', editedAt: 2, conversationRevision: 0, timeline: [{ id: 'user', kind: 'user', text: 'An older synthetic prompt.' }], apiMessages: [{ role: 'user', content: 'An older synthetic prompt.' }] }],
  bot: { presetId: 'legacy_bot', name: 'Legacy bot', role: 'Describe supplied text', instructions: 'Be concise.', providerId: 'local', providerAccountId: null, model: 'synthetic', approvalScope: 'readonly', executionTarget: 'vm', toolScope: { mode: 'selected', names: ['read_file'] }, capturedAt: 1 },
  runRecovery: recovery as unknown as BackupRecord, runRecoveryReviewedId: recovery.runId,
}
const base: WorkspaceBackupData = { settings: { theme: 'dark', confidentialTerms: ['Archive private term'], approvalRules: [{ decision: 'allow_always' }], automations: [{ enabled: true }] }, conversations: [legacy], projects: [], bots: [{ id: 'legacy_bot', name: 'Legacy bot', executionTarget: 'vm' }], drafts: [], attachments: [] }
const selection = { settings: true, conversationIds: ['legacy'], botIds: ['legacy_bot'] }
const archive = buildBackupArchive({ revision: 'synthetic', data: base, blobs: new Map() }, selection, 4)
const restored = parseArchive(encodeArchive(archive)).archive.data
const saved = restored.conversations[0] as unknown as Thread
assert.equal(saved.apiMessages[2].content, null)
assert.deepEqual(restored.settings.confidentialTerms, ['Archive private term'])
assert.equal(restored.settings.approvalRules, undefined); assert.equal(restored.settings.automations, undefined)
assert.ok(archive.warnings.some(warning => warning.includes('permission rules') && warning.includes('runtime configuration')))
assert.throws(() => mergedBackupPrivacyTerms(Array.from({ length: 200 }, (_, index) => `Synthetic guard ${index}`), ['New synthetic guard']), /redaction limit/, 'term merging must not push existing guards beyond the original redactor cap')
assert.deepEqual(saved.apiMessages[1], legacy.apiMessages && (legacy.apiMessages as any)[1], 'legacy multimodal image history retains typed content')
assert.equal(saved.apiMessages[2].tool_calls?.[0].function.arguments, '{"path":"synthetic.txt"}')
assert.equal(saved.messageEditRevisions?.[0].timeline[0].text, 'An older synthetic prompt.')
assert.equal(saved.timeline[1].state, 'ok', 'settled legacy tool display stays settled')
assert.equal(saved.runRecovery?.state, 'needs_review')
assert.equal(saved.runRecoveryReviewedId, undefined, 'a restore never silently records a human outcome review')
assert.throws(() => assertRecoveryAllowsNewRun(saved), /Review/)
assert.equal(saved.bot?.executionTarget, 'vm', 'VM confinement cannot become host-workspace permission')
assert.equal(saved.bot?.approvalScope, 'readonly')
assert.deepEqual(saved.bot?.toolScope, { mode: 'selected', names: [] }, 'imported immutable captures cannot activate capabilities through an existing live preset')
assert.equal(restored.bots[0].paused, true)
assert.equal(restored.bots[0].approvalScope, 'readonly', 'missing supported legacy permissions default to readonly')
assert.equal(restored.bots[0].executionTarget, 'vm')
assert.throws(() => botRunApprovalMode(saved, restored.bots as any, [], 'auto'), /paused/)

for (const state of ['running', 'stop_requested', 'stopping'] as const) {
  const active = { ...legacy, runRecovery: { ...recovery, state, pendingOperations: 2, pendingMutations: 1, retry: 'unavailable', canStartNewRun: false } }
  const result = parseArchive(encodeArchive(buildBackupArchive({ revision: 'synthetic', data: { ...base, conversations: [active] }, blobs: new Map() }, selection, 4))).archive.data.conversations[0] as unknown as Thread
  assert.equal(result.runRecovery?.state, 'needs_review'); assert.equal(result.runRecovery?.pendingOperations, 0); assert.equal(result.runRecovery?.pendingMutations, 0)
  assert.equal(result.runRecovery?.completedMutations, 1); assert.equal(result.runRecovery?.canStartNewRun, true)
  assert.throws(() => assertRecoveryAllowsNewRun(result), /Review/)
}
const pending: BackupRecord = { ...legacy, timeline: [{ id: 'pending', kind: 'tool', tool: 'send_message', state: 'awaiting', text: 'Historical pending action', appAction: { kind: 'create_project' }, question: { state: 'pending' } }] }
delete pending.runRecovery
const inert = parseArchive(encodeArchive(buildBackupArchive({ revision: 'synthetic', data: { ...base, conversations: [pending] as BackupRecord[] }, blobs: new Map() }, selection, 4))).archive.data.conversations[0] as unknown as Thread
assert.equal(inert.timeline[0].state, 'denied'); assert.equal(inert.timeline[0].appAction, undefined); assert.equal(inert.timeline[0].question, undefined)
assert.throws(() => assertRecoveryAllowsNewRun(inert), /Review/, 'unsettled legacy actions remain visible and require outcome review')
const incompleteApi: BackupRecord = { ...legacy, timeline: [], apiMessages: [{ role: 'assistant', tool_calls: [{ id: 'unanswered', type: 'function', function: { name: 'read_file', arguments: '{}' } }] }] }
delete incompleteApi.runRecovery
const unfinished = parseArchive(encodeArchive(buildBackupArchive({ revision: 'synthetic', data: { ...base, conversations: [incompleteApi] }, blobs: new Map() }, selection, 4))).archive.data.conversations[0] as unknown as Thread
assert.equal(unfinished.runRecovery?.state, 'needs_review'); assert.throws(() => assertRecoveryAllowsNewRun(unfinished), /Review/)

const malformed = [
  (data: WorkspaceBackupData) => { (data.conversations[0].bot as any).approvalScope = 'auto' },
  (data: WorkspaceBackupData) => { (data.conversations[0].bot as any).executionTarget = 'outside-host' },
  (data: WorkspaceBackupData) => { data.bots[0].toolScope = { mode: 'selected', names: ['delegate_task'] } },
  (data: WorkspaceBackupData) => { (data.conversations[0].timeline as any)[0].attachments[0].dataUrl = 'https://tracking.example.com/pixel.png' },
  (data: WorkspaceBackupData) => { (data.conversations[0].runRecovery as any).controllerId = 'foreign-process' },
  (data: WorkspaceBackupData) => { (data.conversations[0].runRecovery as any).retry = 'safe' },
  (data: WorkspaceBackupData) => { data.conversations[0].runRecoveryReviewedId = recovery.runId },
  (data: WorkspaceBackupData) => { (data.conversations[0].messageEditRevisions as any)[0].timeline[0].text = {} },
]
for (const mutate of malformed) { const data = structuredClone(restored); mutate(data); assert.throws(() => validateData(data)) }

const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-record-validation-'))), key = Buffer.alloc(32, 53)
let writes = 0, persists = 0
let state: BackupHostState = { settings: { agentPresets: [] } as unknown as Settings, threads: [], projects: [] }
async function writeProtected(file: string, value: unknown) {
  writes++; const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key, iv)
  const ciphertext = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()])
  await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, Buffer.concat([iv, cipher.getAuthTag(), ciphertext]), { mode: 0o600 })
}
async function readProtected<T>(file: string, fallback: T): Promise<T> {
  const bytes = await fs.readFile(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!bytes) return fallback
  const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12)); decipher.setAuthTag(bytes.subarray(12, 28))
  return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8'))
}
try {
  const repository = createWorkspaceBackupRepository({ profileRoot: () => root, getState: () => state, setState: next => { state = next }, validateOwner() {}, readProtectedJson: readProtected, writeProtectedJson: writeProtected, persist: async () => { persists++ } })
  const revision = await repository.revision()
  for (const mutate of malformed) {
    const data = structuredClone(restored); mutate(data)
    await assert.rejects(repository.stage({ mode: 'replace', expectedRevision: revision, sourceDigest: 'a'.repeat(64), data, blobs: new Map(), selection }))
    assert.equal(writes, 0); assert.equal(persists, 0, 'invalid nested data is refused before any stage or canonical write')
  }
  assert.deepEqual(await fs.readdir(root), [], 'malformed admission creates no staging directory')
  const missingBlob = { ...restored, attachments: [{ id: 'missing', name: 'Missing.bin', mediaType: 'application/octet-stream', originalBlobId: 'a'.repeat(64), parsedBlobId: null }] }
  await assert.rejects(repository.stage({ mode: 'replace', expectedRevision: revision, sourceDigest: 'a'.repeat(64), data: missingBlob, blobs: new Map(), selection }), /blob is missing/)
  assert.equal(writes, 0, 'invalid blob plans refuse before protected staging writes')
  const staged = await repository.stage({ mode: 'replace', expectedRevision: revision, sourceDigest: 'a'.repeat(64), data: restored, blobs: new Map(), selection })
  const stagedWrites = writes
  ;(staged.plan.data.conversations[0].timeline as any)[0].text = { changedAfterStaging: true }
  await assert.rejects(repository.commit(staged, revision))
  assert.equal(writes, stagedWrites); assert.equal(persists, 0, 'commit revalidates a changed staged plan before persistence')
  await repository.discard(staged)
  const service = createWorkspaceBackupService({ repository })
  const bytes = await encryptWorkspaceBackup(encodeArchive(archive), 'synthetic record password')
  const preview = await service.previewRestore({ fileId: 'synthetic-safe', bytes, password: 'synthetic record password', mode: 'replace' })
  assert.equal((await service.applyRestore({ approvalId: preview.approvalId, fileId: 'synthetic-safe', bytes })).restored, true)
  assert.equal(state.threads[0].apiMessages[1].content && (state.threads[0].apiMessages[1].content as any)[1].image_url.url, image)
  assert.throws(() => assertRecoveryAllowsNewRun(state.threads[0]), /Review/, 'the real host commit preserves the uncertainty gate')
  state.threads[0].phoneLocalOnly = true; state.threads[0].googleLocalOnly = true
  state.threads[0].memoryPolicy = { useExisting: false, contribute: false }
  state.threads[0].runRecovery = { ...recovery, runId: 'new-current-uncertainty' }
  state.settings.confidentialTerms = ['Current private term']
  const replacePreview = await service.previewRestore({ fileId: 'synthetic-safe-again', bytes, password: 'synthetic record password', mode: 'replace' })
  await service.applyRestore({ approvalId: replacePreview.approvalId, fileId: 'synthetic-safe-again', bytes })
  assert.equal(state.threads[0].phoneLocalOnly, true); assert.equal(state.threads[0].googleLocalOnly, true)
  assert.deepEqual(state.threads[0].memoryPolicy, { useExisting: false, contribute: false })
  assert.equal(state.threads[0].runRecovery?.runId, 'new-current-uncertainty', 'an older backup cannot clear a current unreviewed outcome')
  assert.deepEqual(state.settings.confidentialTerms, ['Current private term', 'Archive private term'], 'restored privacy terms cannot weaken existing redaction')
  state.threads[0].incognito = true
  const protectedWrites = writes, protectedPersists = persists
  await assert.rejects(repository.stage({ mode: 'replace', expectedRevision: 'not-read', sourceDigest: 'a'.repeat(64), data: restored, blobs: new Map(), selection }), /incognito/)
  assert.equal(writes, protectedWrites); assert.equal(persists, protectedPersists, 'incognito identity collisions refuse before staging')
  service.dispose()
  console.log('Nested admission, legacy/API image/revision roundtrip, inert bot permissions and uncertainty review tests passed')
} finally { await fs.rm(root, { recursive: true, force: true }); key.fill(0) }
