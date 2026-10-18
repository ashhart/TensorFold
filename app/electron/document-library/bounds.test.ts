import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { scanDocumentSnapshot } from './scanner'
import { makeDocumentSource, readDocumentItem } from '../agent/document-connector'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-library-bounds-'))
  try {
    const docs = path.join(root, 'docs')
    await fs.mkdir(docs)
    for (const name of ['alpha', 'beta', 'gamma']) await fs.writeFile(path.join(docs, `${name}.txt`), `${name} body`)
    const source = makeDocumentSource(docs)
    const settings = { sources: [source], maxDocuments: 1, maxDocumentBytes: 8_192, lastSyncAt: null }
    const countBound = await scanDocumentSnapshot(settings)
    assert.equal(countBound.documents.length, 1, 'The configured document count must be honored')
    assert.equal(countBound.sources[0].lastScan?.skipped, 2)
    assert.equal(countBound.sources[0].lastScan?.truncated, true)
    assert.equal(countBound.sources[0].lastScan?.status, 'partial')
    assert.equal(countBound.sources[0].lastFileOutcomes?.length, 3)
    const missing = makeDocumentSource(path.join(root, 'missing'))
    const read = await readDocumentItem({ ...settings, sources: [missing, source], maxDocuments: 10 }, 'beta.txt')
    assert.match(read.content, /beta body/, 'An unavailable earlier collection must not hide a usable configured file')
    const disabled = await scanDocumentSnapshot({ ...settings, sources: [{ ...source, enabled: false }] })
    assert.equal(disabled.documents.length, 0)
    assert.equal(disabled.sources[0].lastScan?.status, 'disabled')
    const linkedRoot = path.join(root, 'linked-root')
    await fs.symlink(docs, linkedRoot)
    const linked = await scanDocumentSnapshot({ ...settings, sources: [makeDocumentSource(linkedRoot)] })
    assert.equal(linked.documents.length, 0)
    assert.match(linked.sources[0].lastError || '', /Symbolic-link document roots/)
    let nested = docs
    for (let depth = 0; depth < 10; depth++) { nested = path.join(nested, `level-${depth}`); await fs.mkdir(nested) }
    await fs.writeFile(path.join(nested, 'unexamined.txt'), 'Never silently claim this was scanned')
    const depthBound = await scanDocumentSnapshot({ ...settings, maxDocuments: 100 })
    assert.equal(depthBound.sources[0].lastScan?.truncated, true)
    assert.ok(depthBound.sources[0].lastFileOutcomes?.some(file => /depth limit/.test(file.error || '')))
    const huge = path.join(root, 'many-files')
    await fs.mkdir(huge)
    for (let batch = 0; batch < 16; batch++) await Promise.all(Array.from({ length: 100 }, (_, i) => fs.writeFile(path.join(huge, `${batch}-${i}.unsupported`), 'not read')))
    const fileBound = await scanDocumentSnapshot({ ...settings, sources: [makeDocumentSource(huge)], maxDocuments: 5_000 })
    assert.equal(fileBound.sources[0].lastFileOutcomes?.length, 1_500)
    assert.equal(fileBound.sources[0].lastScan?.truncated, true)
    assert.equal(fileBound.sources[0].lastScan?.unsupported, 1_500)
    console.log('PASS document bounds: configured count, named skipped files, unavailable collection isolation, disabled and linked roots, depth and file inventory limits')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
