import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { makeDocumentSource, syncDocumentConnector } from './document-connector'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-outcomes-'))
  try {
    await fs.writeFile(path.join(root, 'alpha.txt'), 'Alpha remains readable.')
    await fs.writeFile(path.join(root, 'beta.md'), '# Beta\nBeta remains readable.')
    await fs.writeFile(path.join(root, 'oversized.txt'), 'x'.repeat(9_000))
    const result = await syncDocumentConnector({ sources: [makeDocumentSource(root)], maxDocuments: 25, maxDocumentBytes: 8_192, lastSyncAt: null }, [])
    assert.equal(result.documents.length, 2, 'Both valid files must remain usable')
    const source = result.sources[0] as typeof result.sources[0] & { lastScan?: { status: string; ready: number; failed: number }; lastFileOutcomes?: Array<{ relativePath: string; status: string; error?: string }> }
    assert.equal(source.lastScan?.status, 'partial', 'A folder with an oversized file must not be reported healthy')
    assert.equal(source.lastScan?.ready, 2)
    assert.equal(source.lastScan?.failed, 1)
    assert.match(source.lastError || '', /1.*failed/i)
    assert.equal(source.lastFileOutcomes?.find(file => file.relativePath === 'oversized.txt')?.status, 'failed', 'The rejected file must have a named visible outcome')
    console.log('PASS folder scan: 2 usable files; oversized.txt named failed; source partial')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
