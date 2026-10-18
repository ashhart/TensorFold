import assert from 'node:assert/strict'
import { backupDigest, buildBackupArchive, encodeArchive, MAX_BLOB_BYTES, parseArchive, validateData, type BackupSnapshot } from './archive'
import { decryptWorkspaceBackup, encryptWorkspaceBackup } from './crypto'

const blank = { settings: {}, conversations: [], projects: [], bots: [], drafts: [], attachments: [] }
const snapshot: BackupSnapshot = { revision: 'synthetic', data: blank, blobs: new Map() }
const archive = buildBackupArchive(snapshot, {}, 1)
assert.deepEqual(parseArchive(encodeArchive(archive)).archive.data, blank)
assert.throws(() => validateData({ ...blank, conversations: [{ id: 'duplicate' }, { id: 'duplicate' }] }), /Invalid/)
assert.throws(() => validateData({ ...blank, conversations: [{ id: '../escape' }] }), /Invalid/)
assert.throws(() => validateData({ ...blank, conversations: Array.from({ length: 10_001 }, (_, i) => ({ id: `synthetic-${i}` })) }), /Invalid/)
assert.throws(() => validateData(JSON.parse('{"settings":{"__proto__":{"polluted":true}},"conversations":[],"projects":[],"bots":[],"drafts":[],"attachments":[]}')), /Invalid/)
const deeplyNested: any = {}
let cursor = deeplyNested
for (let depth = 0; depth < 30; depth++) { cursor.child = {}; cursor = cursor.child }
assert.throws(() => validateData({ ...blank, settings: deeplyNested }), /Invalid/)
const wrongHash = structuredClone(archive)
wrongHash.data.attachments.push({ id: 'attachment', name: 'Synthetic.txt', mediaType: 'text/plain', originalBlobId: 'a'.repeat(64), parsedBlobId: null })
wrongHash.blobs.push({ id: 'a'.repeat(64), base64: Buffer.from('not matching digest').toString('base64') })
assert.throws(() => parseArchive(encodeArchive(wrongHash)), /Invalid/)
const extra = structuredClone(archive)
extra.blobs.push({ id: backupDigest(Buffer.from('unreferenced')), base64: Buffer.from('unreferenced').toString('base64') })
assert.throws(() => parseArchive(encodeArchive(extra)), /Invalid/)
const secret = structuredClone(archive)
secret.selection.settings = true
secret.data.settings = { apiKey: 'synthetic credential' }
assert.throws(() => parseArchive(encodeArchive(secret)), /credentials|unsupported/)
const emptyId = backupDigest(new Uint8Array())
const empty = buildBackupArchive({ revision: 'synthetic', data: { ...blank, attachments: [{ id: 'empty', name: 'Empty.txt', mediaType: 'text/plain', originalBlobId: emptyId, parsedBlobId: null }] }, blobs: new Map([[emptyId, new Uint8Array()]]) }, { attachments: true }, 1)
assert.equal(parseArchive(encodeArchive(empty)).blobs.get(emptyId)?.byteLength, 0)
const largestBlob = Buffer.alloc(MAX_BLOB_BYTES)
const largestId = backupDigest(largestBlob)
const largest = buildBackupArchive({ revision: 'synthetic', data: { ...blank, attachments: [{ id: 'large', name: 'Large.bin', mediaType: 'application/octet-stream', originalBlobId: largestId, parsedBlobId: null }] }, blobs: new Map([[largestId, largestBlob]]) }, { attachments: true }, 1)
assert.equal(parseArchive(encodeArchive(largest)).blobs.get(largestId)?.byteLength, MAX_BLOB_BYTES, 'maximum permitted base64 blob cannot overflow the regexp stack')
const password = 'synthetic crypto password'
const encrypted = await encryptWorkspaceBackup(encodeArchive(archive), password)
const second = await encryptWorkspaceBackup(encodeArchive(archive), password)
assert.notDeepEqual(encrypted, second, 'random salt and nonce differ for identical backups')
assert.deepEqual(parseArchive(await decryptWorkspaceBackup(encrypted, password)).archive.data, blank)
const headerCorruption = Uint8Array.from(encrypted); headerCorruption[0] ^= 1
await assert.rejects(decryptWorkspaceBackup(headerCorruption, password), /header/)
await assert.rejects(encryptWorkspaceBackup(encodeArchive(archive), 'short'), /8 to 1024/)
console.log('Workspace archive structure, paths, secret rejection, blob integrity and crypto tests passed')
