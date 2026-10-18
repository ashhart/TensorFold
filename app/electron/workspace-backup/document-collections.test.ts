import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { buildBackupArchive, encodeArchive, parseArchive, validateData } from './archive'
import { validatedRestorePlan } from './plan-validation'
import { createWorkspaceBackupRepository, type BackupHostState } from '../main/host/workspace-backup/repository'
import type { WorkspaceBackupData } from '../../shared/workspace-backup'
import type { Settings } from '../agent/types'

const conversation: any = { id: 'imported-chat', title: 'Synthetic imported chat', kind: 'chat', createdAt: 1, updatedAt: 2, cwd: '', status: 'idle', timeline: [], apiMessages: [], documentSourceIds: ['collision'] }
const data: WorkspaceBackupData = { settings: {}, conversations: [conversation], projects: [], bots: [], drafts: [], attachments: [] }
const selection = { conversationIds: [conversation.id] }
const archive = buildBackupArchive({ revision: 'synthetic', data, blobs: new Map() }, selection, 3)
assert.deepEqual(archive.data.conversations[0].documentSourceIds, [], 'Exported conversations do not carry portable collection grants')
assert.deepEqual(parseArchive(encodeArchive(archive)).archive.data.conversations[0].documentSourceIds, [])
assert.ok(archive.warnings.some(warning => warning.includes('without document collections')))
for (const value of ['all', null, ['same', 'same'], [''], ['bad\nname']]) assert.throws(() => validateData({ ...data, conversations: [{ ...conversation, documentSourceIds: value }] }))
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-document-restore-')))
const key = Buffer.alloc(32, 37)
let state: BackupHostState = { settings: { agentPresets: [], documents: { sources: [{ id: 'collision', enabled: true, path: '/synthetic/unrelated-local-folder' }] } } as unknown as Settings, threads: [{ ...conversation, title: 'Local same identity', documentSourceIds: ['collision'] }, { ...conversation, id: 'untouched', documentSourceIds: ['retained'] }], projects: [] }
const repository = createWorkspaceBackupRepository({ profileRoot: () => root, getState: () => state, setState: next => { state = next }, validateOwner() {}, persist: async next => { state = next },
  async readProtectedJson<T>(file: string, fallback: T): Promise<T> { const bytes = await fs.readFile(file).catch(() => null); if (!bytes) return fallback; const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12)); decipher.setAuthTag(bytes.subarray(12, 28)); return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8')) },
  async writeProtectedJson(file, value) { const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key, iv); const ciphertext = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()]); await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, Buffer.concat([iv, cipher.getAuthTag(), ciphertext])) },
})
try {
  const revision = await repository.revision()
  const rawPlan: any = { mode: 'replace', expectedRevision: revision, sourceDigest: 'a'.repeat(64), selection, data, blobs: new Map() }
  assert.deepEqual(validatedRestorePlan(rawPlan).data.conversations[0].documentSourceIds, [], 'Direct restore plans cannot bypass portable normalization')
  const stage = await repository.stage(rawPlan)
  await repository.commit(stage, revision)
  assert.deepEqual(state.threads.find(thread => thread.id === conversation.id)?.documentSourceIds, [], 'Concrete restore binding refuses a foreign/local collection ID collision')
  assert.deepEqual(state.threads.find(thread => thread.id === 'untouched')?.documentSourceIds, ['retained'], 'Unselected local chats keep their owned policy')
  await repository.discard(stage)
  console.log('PASS portable document policy: safe legacy normalization, malformed-grant refusal, direct restore-plan defense, concrete foreign-ID collision isolation and untouched local conversations')
} finally { await fs.rm(root, { recursive: true, force: true }) }
