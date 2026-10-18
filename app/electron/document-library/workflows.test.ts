import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { readVerifiedSource } from './source-preview'
import { DocumentLibrary } from './index'
import { documentEventAllowed, documentPolicySignature, scopedDocumentLibrary, scopedDocumentSettings, validateDocumentSourceIds } from './scope'
import { makeDocumentSource } from '../agent/document-connector'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-document-workflows-'))
  process.env.VONTRA_HOME = path.join(root, 'profile')
  try {
    const first = path.join(root, 'first'), second = path.join(root, 'second'); await fs.mkdir(first); await fs.mkdir(second)
    const original = '# Needle context\r\n<script>neverExecute()</script>\r\n' + 'Needle source detail.\n'.repeat(7_000)
    await fs.writeFile(path.join(first, 'original.md'), original)
    await fs.writeFile(path.join(second, 'private.txt'), 'Needle private collection text')
    await fs.writeFile(path.join(first, 'paper.pdf'), '%PDF-1.4\n1 0 obj << /Length 35 >>\nstream\nBT (Needle PDF source) Tj ET\nendstream\nendobj\n%%EOF')
    let settings = { sources: [makeDocumentSource(first), makeDocumentSource(second)], maxDocuments: 20, maxDocumentBytes: 300_000, lastSyncAt: null }
    const library = new DocumentLibrary({ directory: path.join(root, 'profile', 'library'), getSettings: () => settings })
    await library.sync()
    const all = await library.search({ query: 'Needle', limit: 100 }), privatePassage = (await library.search({ query: 'private collection', sourceId: settings.sources[1].id })).passages[0]
    assert.ok(privatePassage)
    const sources = [settings.sources[0].id]
    assert.equal(scopedDocumentSettings(settings, []).sources.length, 0)
    assert.deepEqual(scopedDocumentSettings(settings, sources).sources.map(source => source.id), sources)
    const event = (sourceId: string): any => ({ connectorId: 'docs', threadKey: `docs:${sourceId}:file`, sources: [{ id: `docs:${sourceId}:file`, connectorId: 'docs', sourceId }] })
    assert.equal(documentEventAllowed(event(settings.sources[1].id), settings, sources), false, 'Cached connector excerpts cannot bypass the collection allowlist')
    assert.equal(documentEventAllowed(event(settings.sources[0].id), settings, sources), true)
    assert.equal(documentEventAllowed(event(settings.sources[0].id), settings, []), false)
    assert.equal(documentEventAllowed({ ...event('unknown'), sources: [] }, settings, sources), false)
    let live = sources
    const signature = documentPolicySignature(sources)
    const scoped = scopedDocumentLibrary(library, sources, () => { if (documentPolicySignature(live) !== signature) throw new Error('Source policy changed') })
    const matches = await scoped.search({ query: 'Needle', sourceId: settings.sources[1].id })
    assert.equal(matches.passages.length, 0, 'An explicit other-source query cannot escape the chat allowlist')
    const own = await scoped.search({ query: 'Needle', limit: 100 })
    assert.ok(own.passages.length); assert.ok(own.passages.every(passage => sources.includes(passage.sourceId)))
    const stolen = { documentId: privatePassage.documentId, revision: privatePassage.revision, passageId: privatePassage.passageId }
    assert.equal((await scoped.readEvidence(stolen)).status, 'unavailable', 'Evidence handles do not grant cross-collection access')
    const none = scopedDocumentLibrary(library, [], () => {})
    assert.equal((await none.search({ query: 'Needle' })).passages.length, 0)
    assert.equal((await none.readEvidence(stolen)).status, 'unavailable')
    live = []
    await assert.rejects(scoped.search({ query: 'Needle' }), /Source policy changed/)
    const text = all.passages.find(passage => passage.relativePath === 'original.md')!
    const request = { documentId: text.documentId, revision: text.revision, passageId: text.passageId }
    const preview = await library.readSource(request)
    assert.equal(preview.status, 'ready')
    if (preview.status === 'ready') {
      assert.equal(preview.original.kind, 'text'); assert.equal(preview.extractedTruncated, true)
      assert.ok(preview.extractedText.length <= 120_000)
      if (preview.original.kind === 'text') { assert.ok(preview.original.text.startsWith('# Needle context\r\n<script>')); assert.equal(preview.original.truncated, true) }
      assert.ok(preview.extractedText.startsWith('# Needle context\n<script>'), 'Original bytes and normalized extracted text remain distinguishable')
      assert.equal(preview.passage.revision, text.revision)
    }
    const pdf = (await library.search({ query: 'Needle PDF source' })).passages[0]
    const pdfPreview = await library.readSource({ documentId: pdf.documentId, revision: pdf.revision, passageId: pdf.passageId })
    assert.equal(pdfPreview.status, 'ready')
    if (pdfPreview.status === 'ready') { assert.equal(pdfPreview.original.kind, 'unavailable'); assert.equal(pdfPreview.parser, 'pdf-text'); assert.equal(pdfPreview.passage.location.kind, 'extracted'); assert.equal('page' in pdfPreview.passage.location, false) }
    let checks = 0
    const purgedDuringPreview = await readVerifiedSource({ readEvidence: async () => ++checks === 1 ? { status: 'ready', passage: text, verifiedAt: 1 } : { status: 'unavailable', message: 'Purged while preparing the source preview' } } as any, request, () => settings)
    assert.equal(purgedDuringPreview.status, 'unavailable', 'Purging the indexed citation during an original-file read must discard the prepared preview')
    assert.equal('original' in purgedDuringPreview, false)
    await fs.appendFile(path.join(first, 'original.md'), 'changed')
    assert.equal((await library.readSource(request)).status, 'changed', 'A stale source viewer never displays cached originals')
    settings = { ...settings, sources: settings.sources.map(source => ({ ...source, enabled: false })) }
    assert.equal((await library.readSource(stolen)).status, 'unavailable')
    assert.equal(validateDocumentSourceIds(undefined), undefined); assert.deepEqual(validateDocumentSourceIds([]), [])
    for (const value of [null, 'all', ['x', 'x'], [''], ['bad\nname'], Array.from({ length: 65 }, (_, index) => String(index))]) assert.throws(() => validateDocumentSourceIds(value))
    console.log('PASS document workflows: pre-read collection isolation, stolen-handle refusal, live policy fence, original/extracted previews, bounded text, honest PDF limits and stale/revoked source refusal')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
