// Run: jiti electron/agent/document-connector.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { deflateRawSync, deflateSync } from 'node:zlib'
import { documentSourceId, documentSourceRef, makeDocumentSource, readDocumentItem, readTableItem, readZipEntries, scanDocumentItems, scanTableItems, summarizeDocuments, summarizeTables, syncDocumentConnector } from './document-connector'
import type { DocumentConnectorSettings } from './types'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-documents-'))
  const docs = path.join(root, 'docs')
  await fs.mkdir(path.join(docs, 'nested'), { recursive: true })
  await fs.writeFile(path.join(docs, 'brief.md'), '# Launch brief\n\nThe launch gate needs reviewer evidence and final sign-off.\n', 'utf8')
  await fs.writeFile(path.join(docs, 'nested', 'notes.txt'), 'Ops notes\n\nCalendar owner should confirm the rollout window.\n', 'utf8')
  await fs.writeFile(path.join(docs, 'metrics.csv'), [
    'Team,Revenue,Cost,Status',
    'Alpha,1200,800,green',
    'Beta,950,500,yellow',
    '"Gamma, East",700,710,red',
  ].join('\n'), 'utf8')
  await writeTinyXlsx(path.join(docs, 'budget.xlsx'), [
    ['Team', 'Budget', 'Actual'],
    ['Alpha', '1000', '900'],
    ['Beta', '800', '850'],
  ])
  await writeTinyPdf(path.join(docs, 'review.pdf'), 'Vendor PDF Review\nRenewal terms require source-backed approval.')
  await fs.writeFile(path.join(docs, '.env'), 'SECRET=not-indexed\n', 'utf8')
  await fs.writeFile(path.join(docs, 'image.png'), 'not text\n', 'utf8')
  // Encrypted PDF that sorts before the valid documents; extraction throws on it.
  await fs.writeFile(path.join(docs, 'aaa-encrypted.pdf'), '%PDF-1.4\ntrailer\n<< /Root 1 0 R/Encrypt 6 0 R >>\nstream\nx\nendstream\n%%EOF\n', 'latin1')

  const compressedBomb = zipDeflated('xl/sharedStrings.xml', 'x'.repeat(250_000))
  assert.throws(() => readZipEntries(compressedBomb, 4_096), /expanded-content limit/, 'compressed XLSX entries cannot expand beyond the document budget')

  const source = makeDocumentSource(docs, 1)
  const settings: DocumentConnectorSettings = {
    sources: [{ ...source, id: 'docs', label: 'Project Docs' }],
    maxDocuments: 20,
    maxDocumentBytes: 100_000,
    lastSyncAt: null,
  }

  const { documents, sources } = await scanDocumentItems(settings, { query: 'reviewer', limit: 10 })
  assert.equal(documents.length, 1)
  assert.equal(documents[0].title, 'Launch brief')
  assert.equal(documents[0].relativePath, 'brief.md')
  assert.match(documents[0].excerpt, /reviewer evidence/)
  assert.match(sources[0].lastError || '', /1 failed/)
  assert.equal(sources[0].lastScan?.status, 'partial')
  assert.ok(sources[0].lastSyncAt)
  assert.equal(documentSourceId(documents[0]), `docs:docs:${documents[0].id}`)
  assert.equal(documentSourceRef(documents[0]).connectorId, 'docs')

  const txt = await scanDocumentItems(settings, { extension: 'txt', limit: 10 })
  assert.equal(txt.documents.length, 1)
  assert.equal(txt.documents[0].relativePath, path.join('nested', 'notes.txt'))

  const pdf = await scanDocumentItems(settings, { extension: 'pdf', limit: 10 })
  assert.equal(pdf.documents.length, 1)
  assert.equal(pdf.documents[0].relativePath, 'review.pdf')
  assert.equal(pdf.documents[0].extension, '.pdf')
  assert.match(pdf.documents[0].excerpt, /Renewal terms/)

  const read = await readDocumentItem(settings, documentSourceId(documents[0]))
  assert.equal(read.id, documents[0].id)
  assert.match(read.content, /final sign-off/)

  const pdfRead = await readDocumentItem(settings, documentSourceId(pdf.documents[0]))
  assert.match(pdfRead.content, /source-backed approval/)

  const byPath = await readDocumentItem(settings, 'brief.md')
  assert.equal(byPath.id, documents[0].id)

  // Regression: an unrelated encrypted PDF sorting before the target must not abort the read.
  const afterEncrypted = await readDocumentItem(settings, documentSourceId(documents[0]))
  assert.equal(afterEncrypted.id, documents[0].id)
  assert.match(afterEncrypted.content, /final sign-off/)

  const summary = await summarizeDocuments(settings, { limit: 10 })
  assert.match(summary.summary, /5 documents matched/)
  assert.match(summary.summary, /Sources:/)
  assert.match(summary.summary, /docs:docs:/)
  assert.equal(summary.sources.length, 5)

  const pdfSummary = await summarizeDocuments(settings, { query: 'Renewal terms', limit: 10 })
  assert.match(pdfSummary.summary, /review\.pdf/)
  assert.match(pdfSummary.summary, /docs:docs:/)
  assert.equal(pdfSummary.sources.length, 1)

  const tables = await scanTableItems(settings, { query: 'Revenue', limit: 10 })
  assert.equal(tables.tables.length, 1)
  assert.equal(tables.tables[0].relativePath, 'metrics.csv')
  assert.equal(tables.tables[0].rowCount, 3)
  assert.equal(tables.tables[0].columnCount, 4)
  assert.deepEqual(tables.tables[0].columns, ['Team', 'Revenue', 'Cost', 'Status'])
  assert.match(tables.tables[0].excerpt, /Revenue/)
  assert.match(tables.tables[0].excerpt, /Alpha/)

  const tableRead = await readTableItem(settings, documentSourceId(tables.tables[0]))
  assert.equal(tableRead.id, tables.tables[0].id)
  assert.deepEqual(tableRead.rows[2], ['Gamma, East', '700', '710', 'red'])
  assert.deepEqual(tableRead.columns, ['Team', 'Revenue', 'Cost', 'Status'])
  // Regression: readTableItem -> readDocumentItem must also survive the encrypted PDF.
  assert.equal((await readTableItem(settings, documentSourceId(tables.tables[0]))).id, tables.tables[0].id)

  const tableSummary = await summarizeTables(settings, { limit: 10 })
  assert.match(tableSummary.summary, /2 tables matched/)
  assert.match(tableSummary.summary, /metrics\.csv/)
  assert.match(tableSummary.summary, /budget\.xlsx/)
  assert.match(tableSummary.summary, /Revenue: min 700, max 1200, avg 950/)
  assert.match(tableSummary.summary, /Budget: min 800, max 1000, avg 900/)
  assert.match(tableSummary.summary, /Sources:/)
  assert.match(tableSummary.summary, /docs:docs:/)
  assert.equal(tableSummary.sources.length, 2)

  const xlsxTables = await scanTableItems(settings, { extension: 'xlsx', limit: 10 })
  assert.equal(xlsxTables.tables.length, 1)
  assert.equal(xlsxTables.tables[0].relativePath, 'budget.xlsx')
  assert.deepEqual(xlsxTables.tables[0].columns, ['Team', 'Budget', 'Actual'])
  const xlsxRead = await readTableItem(settings, documentSourceId(xlsxTables.tables[0]))
  assert.deepEqual(xlsxRead.rows[1], ['Beta', '800', '850'])

  const sync = await syncDocumentConnector(settings, [])
  assert.equal(sync.documents.length, 5)
  assert.equal(sync.events.length, 5)
  assert.equal(sync.events[0].connectorId, 'docs')
  assert.equal(sync.events[0].sources?.[0]?.connectorId, 'docs')
  assert.ok(sync.syncedAt)
  // Regression: every built event must carry a stable payload.messageId so
  // rememberConnectorEvents' insert-time dedup (which only checks payload.messageId) can
  // catch duplicates from two concurrent syncs that both snapshot existingEvents before a
  // slow scan completes. Simulate that race: two syncs of the same new document, neither
  // aware of the other's in-flight events, then dedup by messageId the way the host does.
  for (const event of sync.events) {
    assert.equal(typeof event.payload?.messageId, 'string')
    assert.ok((event.payload!.messageId as string).length > 0)
  }
  const concurrentSyncA = await syncDocumentConnector(settings, [])
  const concurrentSyncB = await syncDocumentConnector(settings, [])
  const messageIdsA = concurrentSyncA.events.map((e) => e.payload?.messageId)
  const messageIdsB = concurrentSyncB.events.map((e) => e.payload?.messageId)
  assert.deepEqual(messageIdsA.sort(), messageIdsB.sort())
  const seenIds = new Set<string>()
  const afterInsertDedup = [...concurrentSyncA.events, ...concurrentSyncB.events].filter((event) => {
    const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
    if (!mid) return true
    if (seenIds.has(mid)) return false
    seenIds.add(mid)
    return true
  })
  assert.equal(afterInsertDedup.length, concurrentSyncA.events.length)

  await fs.rm(root, { recursive: true, force: true })
  console.log('document connector tests passed')
}

void main()

async function writeTinyPdf(file: string, text: string): Promise<void> {
  const content = [
    'BT',
    '/F1 12 Tf',
    '72 720 Td',
    `(${pdfLiteral(text)}) Tj`,
    'ET',
  ].join('\n')
  const stream = deflateSync(Buffer.from(content, 'utf8'))
  const objects: Buffer[] = [
    Buffer.from('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n', 'utf8'),
    Buffer.from('2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n', 'utf8'),
    Buffer.from('3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n', 'utf8'),
    Buffer.concat([
      Buffer.from(`4 0 obj\n<< /Length ${stream.length} /Filter /FlateDecode >>\nstream\n`, 'utf8'),
      stream,
      Buffer.from('\nendstream\nendobj\n', 'utf8'),
    ]),
    Buffer.from('5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n', 'utf8'),
  ]
  const chunks: Buffer[] = [Buffer.from('%PDF-1.4\n%\xE2\xE3\xCF\xD3\n', 'binary')]
  const offsets = [0]
  for (const object of objects) {
    offsets.push(Buffer.concat(chunks).length)
    chunks.push(object)
  }
  const body = Buffer.concat(chunks)
  const xrefOffset = body.length
  const xref = [
    'xref',
    `0 ${offsets.length}`,
    '0000000000 65535 f ',
    ...offsets.slice(1).map((offset) => `${String(offset).padStart(10, '0')} 00000 n `),
    'trailer',
    `<< /Size ${offsets.length} /Root 1 0 R >>`,
    'startxref',
    String(xrefOffset),
    '%%EOF',
    '',
  ].join('\n')
  await fs.writeFile(file, Buffer.concat([body, Buffer.from(xref, 'utf8')]))
}

function pdfLiteral(value: string): string {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
    .replace(/\n/g, '\\n')
}

async function writeTinyXlsx(file: string, rows: string[][]): Promise<void> {
  const shared = [...new Set(rows.flat())]
  const sharedXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="${rows.flat().length}" uniqueCount="${shared.length}">` +
    shared.map((value) => `<si><t>${xmlText(value)}</t></si>`).join('') +
    `</sst>`
  const sheetXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>` +
    rows.map((row, rowIndex) => `<row r="${rowIndex + 1}">` + row.map((value, colIndex) => {
      const ref = `${xlsxCol(colIndex)}${rowIndex + 1}`
      return `<c r="${ref}" t="s"><v>${shared.indexOf(value)}</v></c>`
    }).join('') + `</row>`).join('') +
    `</sheetData></worksheet>`
  const files: Array<[string, string]> = [
    ['[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>'],
    ['_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'],
    ['xl/workbook.xml', '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Budget" sheetId="1" r:id="rId1"/></sheets></workbook>'],
    ['xl/_rels/workbook.xml.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>'],
    ['xl/worksheets/sheet1.xml', sheetXml],
    ['xl/sharedStrings.xml', sharedXml],
  ]
  await fs.writeFile(file, zipStore(files))
}

function zipStore(files: Array<[string, string]>): Buffer {
  const chunks: Buffer[] = []
  for (const [name, content] of files) {
    const nameBytes = Buffer.from(name, 'utf8')
    const data = Buffer.from(content, 'utf8')
    const header = Buffer.alloc(30)
    header.writeUInt32LE(0x04034b50, 0)
    header.writeUInt16LE(20, 4)
    header.writeUInt16LE(0, 6)
    header.writeUInt16LE(0, 8)
    header.writeUInt32LE(0, 10)
    header.writeUInt32LE(0, 14)
    header.writeUInt32LE(data.length, 18)
    header.writeUInt32LE(data.length, 22)
    header.writeUInt16LE(nameBytes.length, 26)
    header.writeUInt16LE(0, 28)
    chunks.push(header, nameBytes, data)
  }
  return Buffer.concat(chunks)
}

function zipDeflated(name: string, content: string): Buffer {
  const nameBytes = Buffer.from(name, 'utf8')
  const data = deflateRawSync(Buffer.from(content, 'utf8'))
  const header = Buffer.alloc(30)
  header.writeUInt32LE(0x04034b50, 0)
  header.writeUInt16LE(20, 4)
  header.writeUInt16LE(0, 6)
  header.writeUInt16LE(8, 8)
  header.writeUInt32LE(data.length, 18)
  header.writeUInt32LE(Buffer.byteLength(content), 22)
  header.writeUInt16LE(nameBytes.length, 26)
  return Buffer.concat([header, nameBytes, data])
}

function xlsxCol(index: number): string {
  let n = index + 1
  let out = ''
  while (n > 0) {
    const rem = (n - 1) % 26
    out = String.fromCharCode(65 + rem) + out
    n = Math.floor((n - 1) / 26)
  }
  return out
}

function xmlText(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
