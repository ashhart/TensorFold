import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { makeDocumentSource, scanDocumentItems } from './document-connector'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-full-body-'))
  try {
    await fs.writeFile(path.join(root, 'late.md'), '# Planning\n' + 'Early context.\n'.repeat(150) + '\n## Final decision\nThe violet heron marker is near the end.\n')
    const result = await scanDocumentItems({ sources: [makeDocumentSource(root)], maxDocuments: 25, maxDocumentBytes: 16_000, lastSyncAt: null }, { query: 'violet heron marker' })
    assert.equal(result.documents.length, 1, 'A supported fact beyond the 280-character excerpt must be found')
    assert.match(result.documents[0].excerpt, /violet heron marker/)
    console.log('PASS complete-body document query retrieves late fact with matching excerpt')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
