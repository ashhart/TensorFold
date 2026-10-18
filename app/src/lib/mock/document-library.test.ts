import assert from 'node:assert/strict'
import { createMockDocumentLibraryApi } from './document-library'
import { pairedSettingsDefaults } from '../paired-settings'

async function main() {
  const settings = pairedSettingsDefaults()
  const source = { id: 'synthetic', path: '/synthetic/documents', label: 'Synthetic collection', kind: 'dir' as const, enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: 10 }
  settings.documents.sources = [source]
  settings.documents.lastSyncAt = 10
  const api = createMockDocumentLibraryApi(() => settings)
  const list = await api.documentLibraryList()
  assert.equal(list.sources[0].lastScan?.status, 'partial')
  assert.equal(list.files.length, 5)
  const results = await api.documentLibrarySearch({ query: 'violet heron' })
  assert.equal(results.passages.length, 1)
  const ready = results.passages[0]
  const request = { documentId: ready.documentId, revision: ready.revision, passageId: ready.passageId }
  assert.equal((await api.documentLibraryReadEvidence(request)).status, 'ready')
  for (const status of ['changed', 'missing'] as const) {
    const file = list.files.find(item => item.verification === status)!
    const evidence = await api.documentLibraryReadEvidence({ documentId: file.id, revision: file.revision!, passageId: 'demo-passage' })
    assert.equal(evidence.status, status)
    assert.equal('passage' in evidence, false)
  }
  source.enabled = false
  assert.equal((await api.documentLibrarySearch({ query: 'violet heron' })).passages.length, 0)
  assert.equal((await api.documentLibraryReadEvidence(request)).status, 'unavailable')
  source.enabled = true
  await api.documentLibraryPurge(source.id)
  assert.equal((await api.documentLibraryList()).files.length, 0)
  source.lastSyncAt = 20
  assert.equal((await api.documentLibraryList()).files.length, 5, 'A later explicit sync rebuilds the synthetic index')
  settings.documents.sources = []
  assert.equal((await api.documentLibraryReadEvidence(request)).status, 'unavailable')
  console.log('PASS document demo parity: partial inventory, full-text search, evidence, changed/missing originals, disable/remove/purge and resync')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
