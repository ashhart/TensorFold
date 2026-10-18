import type { DocumentLibrary } from './index'
import type { DocumentConnectorSettings } from '../agent/types'
import type { DocumentEvidenceRequest } from '../../shared/document-library'
import type { DocumentSourceResult } from '../../shared/document-processing'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { readFileBytesBounded } from '../agent/document-parser'
import { assertDocumentPath, documentByteLimit, readDocumentFile } from './scanner'

const TEXT_EXTENSIONS = new Set(['.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.jsonl', '.log', '.yaml', '.yml', '.xml', '.html', '.htm', '.rst'])
const MAX_PREVIEW = 120_000
/** Plain strings only: original HTML, scripts, links and PDF actions are never executed in the renderer. */
export async function readVerifiedSource(library: DocumentLibrary, request: DocumentEvidenceRequest, getSettings: () => DocumentConnectorSettings): Promise<DocumentSourceResult> {
  const evidence = await library.readEvidence(request)
  if (evidence.status !== 'ready') return evidence
  const source = getSettings().sources.find(item => item.id === evidence.passage.sourceId && item.enabled)
  if (!source) return { status: 'unavailable', message: 'This document collection is no longer enabled.' }
  const file = source.kind === 'file' ? source.path : path.resolve(source.path, evidence.passage.relativePath)
  const signature = `${source.kind}:${path.resolve(source.path)}`
  const current = () => getSettings().sources.some(item => item.id === source.id && item.enabled && `${item.kind}:${path.resolve(item.path)}` === signature)
  try {
    await assertDocumentPath(source, file)
    const bytes = await readFileBytesBounded(file, documentByteLimit(getSettings()))
    if (createHash('sha256').update(bytes).digest('hex') !== request.revision) return { status: 'changed', message: 'The original changed while its source preview was opening; sync before reading it.' }
    const parsed = await readDocumentFile(source, file, documentByteLimit(getSettings()))
    if (!parsed || parsed.revision !== request.revision || !current()) return { status: 'unavailable', message: 'The source changed or was disabled while its preview was being prepared.' }
    const latest = await library.readEvidence(request)
    if (latest.status !== 'ready') return latest
    if (!current()) return { status: 'unavailable', message: 'The document source was revoked before its preview finished.' }
    const ext = path.extname(file).toLowerCase()
    const text = TEXT_EXTENSIONS.has(ext) ? bytes.toString('utf8') : null
    return {
      status: 'ready', passage: latest.passage, verifiedAt: Date.now(),
      original: text === null ? { kind: 'unavailable', reason: ext === '.pdf' ? 'Executable PDF previews and page anchors are not supported; only verified extracted text is displayed.' : 'This binary document cannot be safely shown inline; only verified extracted text is displayed.' } : { kind: 'text', text: text.slice(0, MAX_PREVIEW), truncated: text.length > MAX_PREVIEW },
      extractedText: parsed.content.slice(0, MAX_PREVIEW), extractedTruncated: parsed.content.length > MAX_PREVIEW,
      parser: ext === '.pdf' ? 'pdf-text' : ext === '.xlsx' ? 'workbook' : ['.docx', '.odt', '.rtf', '.doc'].includes(ext) ? 'office-text' : 'plain-text', locationNote: parsed.locationNote,
    }
  } catch (error) {
    const code = (error as NodeJS.ErrnoException)?.code
    return { status: code === 'ENOENT' || code === 'ENOTDIR' ? 'missing' : 'unavailable', message: `The original could not be safely opened: ${String((error as Error)?.message ?? error).slice(0, 240)}` }
  }
}
