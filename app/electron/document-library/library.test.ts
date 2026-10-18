import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { DocumentConnectorSettings } from '../agent/types'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-library-'))
  process.env.VONTRA_HOME = path.join(root, 'synthetic-profile')
  const { DocumentLibrary } = await import('./index')
  const { makeDocumentSource } = await import('../agent/document-connector')
  const docs = path.join(root, 'documents')
  const outside = path.join(root, 'outside.txt')
  const directory = path.join(root, 'synthetic-profile', 'state', 'document-library')
  await fs.mkdir(docs)
  const original = '# Planning\n' + 'Introductory context.\n'.repeat(240) + '\n## Final decision\nThe violet heron marker is near the end.\n'
  await fs.writeFile(path.join(docs, 'late.md'), original)
  await fs.writeFile(path.join(docs, 'table.csv'), 'Name,Details\nFirst,"two\nlines"\nSecond,crimson egret\n')
  await fs.writeFile(path.join(docs, 'bad.csv'), 'Name,Details\nFirst,"unterminated\n')
  await fs.writeFile(path.join(docs, 'oversized.txt'), 'x'.repeat(40_000))
  await fs.writeFile(path.join(docs, 'paper.pdf'), '%PDF-1.4\n1 0 obj << /Length 35 >>\nstream\nBT (Teal osprey evidence) Tj ET\nendstream\nendobj\n%%EOF')
  await fs.writeFile(path.join(docs, 'book.xlsx'), zipStore([
    ['xl/workbook.xml', '<workbook><sheets><sheet name="Archive" r:id="rId1"/><sheet name="Current" r:id="rId2"/></sheets></workbook>'],
    ['xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>'],
    ['xl/worksheets/sheet1.xml', '<worksheet><sheetData><row r="9"><c r="C9" t="inlineStr"><is><t>Golden ibis token</t></is></c></row></sheetData></worksheet>'],
    ['xl/worksheets/sheet2.xml', '<worksheet><sheetData><row r="21"><c r="B21" t="inlineStr"><is><t>Indigo tern token</t></is></c></row></sheetData></worksheet>'],
  ]))
  await fs.writeFile(path.join(docs, '.env.local.txt'), 'never-index-secret-marker')
  await fs.writeFile(path.join(docs, 'picture.png'), 'never-index-image-marker')
  await fs.writeFile(outside, 'never-index-outside-marker')
  await fs.symlink(outside, path.join(docs, 'shortcut.txt'))
  let settings: DocumentConnectorSettings = { sources: [makeDocumentSource(docs)], maxDocuments: 100, maxDocumentBytes: 32_000, lastSyncAt: null }
  const options = { directory, getSettings: () => settings }
  const library = new DocumentLibrary(options)
  try {
    const scanned = await library.sync()
    assert.equal(scanned.files.filter(file => file.status === 'ready').length, 4)
    assert.equal(scanned.sources[0].lastScan?.status, 'partial')
    assert.equal(scanned.files.find(file => file.relativePath === 'oversized.txt')?.status, 'failed')
    assert.equal(scanned.files.find(file => file.relativePath === 'bad.csv')?.status, 'failed')
    assert.equal(scanned.files.find(file => file.relativePath === 'picture.png')?.status, 'unsupported')
    assert.equal(scanned.files.find(file => file.relativePath === '.env.local.txt')?.status, 'skipped')
    assert.equal(scanned.files.find(file => file.relativePath === 'shortcut.txt')?.status, 'skipped')
    assert.equal((await library.search({ query: 'never-index' })).passages.length, 0)
    const found = await library.search({ query: 'violet heron marker' })
    assert.ok(found.passages.length)
    const passage = found.passages[0]
    assert.equal(passage.location.kind, 'text')
    assert.match(passage.revision, /^[a-f0-9]{64}$/)
    if (passage.location.kind === 'text') {
      assert.equal(original.slice(passage.location.charStart, passage.location.charEnd), passage.text)
      assert.equal(passage.location.lineStart, original.slice(0, passage.location.charStart).split('\n').length)
    }
    const request = { documentId: passage.documentId, revision: passage.revision, passageId: passage.passageId }
    assert.equal((await library.readEvidence(request)).status, 'ready')
    assert.equal((await library.readEvidence({ ...request, passageId: '../outside' })).status, 'refused')
    assert.equal((await library.readEvidence({ ...request, revision: 'bad' })).status, 'refused')
    const table = (await library.search({ query: 'crimson egret' })).passages[0]
    assert.equal(table.location.kind, 'table')
    if (table.location.kind === 'table') { assert.equal(table.location.rowStart, 3); assert.equal(table.location.rowEnd, 3); assert.equal(table.location.columnEnd, 2) }
    const book = (await library.search({ query: 'Golden ibis' })).passages[0]
    assert.equal(book.location.kind, 'table')
    if (book.location.kind === 'table') { assert.equal(book.location.sheet, 'Archive'); assert.equal(book.location.range, 'C9:C9'); assert.equal(book.location.rowStart, 9) }
    const secondSheet = (await library.search({ query: 'Indigo tern' })).passages[0]
    assert.equal(secondSheet.location.kind, 'table')
    if (secondSheet.location.kind === 'table') { assert.equal(secondSheet.location.sheet, 'Current'); assert.equal(secondSheet.location.range, 'B21:B21') }
    const pdf = (await library.search({ query: 'Teal osprey' })).passages[0]
    assert.equal(pdf.location.kind, 'extracted')
    assert.equal('page' in pdf.location, false, 'PDF page coordinates must never be invented')
    if (pdf.location.kind === 'extracted') assert.match(pdf.location.reason, /no verified page mapping/)
    const raw = await fs.readFile(path.join(directory, 'index.json'), 'utf8')
    assert.match(raw, /safe-json:v1:/)
    assert.ok(!raw.includes('violet heron') && !raw.includes('Golden ibis'), 'Cached full text must be encrypted with protected profile storage')
    const restarted = new DocumentLibrary(options)
    assert.ok((await restarted.search({ query: 'violet heron' })).passages.length, 'Indexed evidence survives restart')
    assert.equal((await restarted.readEvidence(request)).status, 'ready')
    await fs.writeFile(path.join(docs, 'late.md'), original + 'Changed bytes.')
    const changed = await restarted.readEvidence(request)
    assert.equal(changed.status, 'changed'); assert.ok(!('passage' in changed), 'Changed evidence must not return cached text')
    const staleSearch = await restarted.search({ query: 'violet heron' })
    assert.equal(staleSearch.passages.length, 0); assert.match(staleSearch.warnings?.[0] ?? '', /late.md.*changed/)
    const staleList = await restarted.list()
    assert.equal(staleList.files.find(file => file.id === passage.documentId)?.verification, 'changed')
    await fs.unlink(path.join(docs, 'late.md'))
    const missing = await restarted.readEvidence(request)
    assert.equal(missing.status, 'missing'); assert.ok(!('passage' in missing))
    assert.equal((await restarted.list()).files.find(file => file.id === passage.documentId)?.verification, 'missing')
    await fs.writeFile(path.join(docs, 'late.md'), original)
    settings = { ...settings, sources: settings.sources.map(source => ({ ...source, enabled: false })) }
    assert.equal((await restarted.search({ query: 'Golden ibis' })).passages.length, 0)
    assert.equal((await restarted.readEvidence(request)).status, 'unavailable')
    assert.ok((await restarted.list()).files.every(file => file.status === 'skipped'))
    settings = { ...settings, sources: settings.sources.map(source => ({ ...source, enabled: true })) }
    assert.equal((await restarted.readEvidence(request)).status, 'ready')
    await fs.unlink(path.join(docs, 'late.md')); await fs.symlink(outside, path.join(docs, 'late.md'))
    assert.equal((await restarted.readEvidence(request)).status, 'unavailable', 'Changed symlinks must be refused')
    await fs.unlink(path.join(docs, 'late.md')); await fs.writeFile(path.join(docs, 'late.md'), original)
    settings = { ...settings, sources: settings.sources.map(source => ({ ...source, path: path.join(root, 'different-source') })) }
    assert.equal((await restarted.list()).files.length, 0, 'Changed source configuration must hide old inventory')
    assert.equal((await restarted.search({ query: 'Golden ibis' })).passages.length, 0)
    await restarted.reconcile()
    settings = { ...settings, sources: [] }
    await restarted.reconcile()
    assert.equal((await new DocumentLibrary(options).list()).files.length, 0, 'Removed sources stay absent after restart')
    settings = { ...settings, sources: [makeDocumentSource(docs)] }
    await restarted.sync()
    await restarted.quiesce()
    await assert.rejects(restarted.search({ query: 'violet heron' }), /paused/)
    const cleared = await restarted.purge()
    assert.equal(cleared.files.length, 0)
    assert.equal(await fs.readFile(path.join(docs, 'late.md'), 'utf8'), original, 'Purge must never touch originals')
    restarted.resume()
    assert.equal((await restarted.search({ query: 'violet heron' })).passages.length, 0)
    assert.equal((await new DocumentLibrary(options).list()).files.length, 0)
    console.log('PASS managed library: full body, text/CSV/multi-sheet coordinates, honest PDF mapping, encrypted restart, changed/missing/disabled/path/symlink refusals, purge and quiesce')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
function zipStore(files: Array<[string, string]>): Buffer {
  return Buffer.concat(files.flatMap(([name, content]) => {
    const filename = Buffer.from(name), data = Buffer.from(content), header = Buffer.alloc(30)
    header.writeUInt32LE(0x04034b50); header.writeUInt16LE(20, 4); header.writeUInt32LE(data.length, 18); header.writeUInt32LE(data.length, 22); header.writeUInt16LE(filename.length, 26)
    return [header, filename, data]
  }))
}
main().catch(error => { console.error(error); process.exitCode = 1 })
