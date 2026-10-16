import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { deflateSync } from 'node:zlib'

export async function makeEmailSettings() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-email-'))
  await fs.writeFile(path.join(root, 'one.eml'), [
    'Message-ID: <one@example.test>',
    'Date: Fri, 19 Jun 2026 09:00:00 +0000',
    'From: Maya <maya@example.test>',
    'To: User <user@example.test>',
    'Subject: Renewal question',
    '',
    'Can you confirm pricing and rollout dates?',
  ].join('\n'), 'utf8')
  return {
    root,
    settings: {
      sources: [{ id: 'src', label: 'Test Mail', path: root, kind: 'eml_dir' as const, enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
      maxMessages: 20,
      lastSyncAt: null,
    },
  }
}

export async function makeSlackSettings() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-slack-'))
  await fs.writeFile(path.join(root, 'channels.json'), JSON.stringify([{ id: 'C123', name: 'launch-room' }]), 'utf8')
  await fs.writeFile(path.join(root, 'users.json'), JSON.stringify([
    { id: 'U1', name: 'maya', real_name: 'Maya Chen' },
    { id: 'U2', name: 'sam', real_name: 'Sam Patel' },
  ]), 'utf8')
  await fs.mkdir(path.join(root, 'launch-room'), { recursive: true })
  await fs.writeFile(path.join(root, 'launch-room', '2026-06-20.json'), JSON.stringify([
    { type: 'message', user: 'U1', text: 'Auth edge cases need release-note ownership.', ts: '1781978400.000100' },
    { type: 'message', user: 'U2', text: 'Staging data drift is blocking QA.', ts: '1781979000.000200', thread_ts: '1781978400.000100' },
  ]), 'utf8')
  return {
    root,
    settings: {
      sources: [{ id: 'slack', label: 'Launch Slack', path: root, kind: 'export_dir' as const, enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
      maxMessages: 20,
      lastSyncAt: null,
    },
  }
}

export async function makeCalendarSettings() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-calendar-'))
  const file = path.join(root, 'work.ics')
  await fs.writeFile(file, [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'BEGIN:VEVENT',
    'UID:planning@example.test',
    'SUMMARY:Planning review',
    'DTSTART:20260620T100000Z',
    'DTEND:20260620T103000Z',
    'LOCATION:Project room',
    'DESCRIPTION:Review the plan and blockers.',
    'END:VEVENT',
    'END:VCALENDAR',
  ].join('\n'), 'utf8')
  return {
    root,
    settings: {
      sources: [{ id: 'cal', label: 'Work Calendar', path: file, kind: 'ics_file' as const, enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
      maxEvents: 20,
      lastSyncAt: null,
    },
  }
}

export async function makeDocumentSettings() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-documents-'))
  await fs.writeFile(path.join(root, 'brief.md'), '# Launch brief\n\nReviewer evidence and final sign-off are required.\n', 'utf8')
  await fs.writeFile(path.join(root, 'notes.txt'), 'Ops notes\n\nCalendar owner should confirm the rollout window.\n', 'utf8')
  await fs.writeFile(path.join(root, 'metrics.csv'), [
    'Team,Revenue,Cost,Status',
    'Alpha,1200,800,green',
    'Beta,950,500,yellow',
    '"Gamma, East",700,710,red',
  ].join('\n'), 'utf8')
  await writeTinyXlsx(path.join(root, 'budget.xlsx'), [
    ['Team', 'Budget', 'Actual'],
    ['Alpha', '1000', '900'],
    ['Beta', '800', '850'],
  ])
  await writeTinyPdf(path.join(root, 'review.pdf'), 'Vendor PDF Review\nRenewal terms require source-backed approval.')
  return {
    root,
    settings: {
      sources: [{ id: 'docs', label: 'Project Docs', path: root, kind: 'dir' as const, enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
      maxDocuments: 20,
      maxDocumentBytes: 100_000,
      lastSyncAt: null,
    },
  }
}

export async function makeBrowserSettings() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-browser-'))
  const historyPath = path.join(root, 'history.json')
  await fs.writeFile(historyPath, JSON.stringify([
    {
      url: 'https://example.com/launch',
      title: 'Launch notes',
      visitedAt: '2026-06-20T10:00:00Z',
      text: 'Release notes mention auth edge cases and QA evidence.',
    },
  ]), 'utf8')
  return {
    root,
    settings: {
      sources: [
        {
          id: 'tab',
          label: 'Current Tab',
          kind: 'current_tab' as const,
          enabled: true,
          createdAt: 1,
          updatedAt: 1,
          lastSyncAt: null,
          lastError: null,
          url: 'https://docs.example.com/current',
          title: 'Current Tab',
          excerpt: 'Auth launch checklist in the foreground tab.',
          capturedAt: 1_781_979_000_000,
        },
        {
          id: 'history',
          label: 'Browser history export',
          kind: 'history_export' as const,
          enabled: true,
          createdAt: 1,
          updatedAt: 1,
          lastSyncAt: null,
          lastError: null,
          path: historyPath,
        },
      ],
      maxItems: 20,
      lastSyncAt: null,
    },
  }
}

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
