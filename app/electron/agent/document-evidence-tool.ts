import type { DocumentEvidenceRequest, DocumentEvidenceResult, DocumentLibrarySearchOptions, DocumentLibrarySearchResult, DocumentLocation } from '../../shared/document-library'

export interface DocumentEvidencePort {
  search(options: DocumentLibrarySearchOptions): Promise<DocumentLibrarySearchResult>
  readEvidence(request: DocumentEvidenceRequest): Promise<DocumentEvidenceResult>
}
export const evidenceHandle = (request: DocumentEvidenceRequest): string => `evidence:${Buffer.from(JSON.stringify(request)).toString('base64url')}`
function parseHandle(id: string): DocumentEvidenceRequest {
  if (id.length > 2048 || !/^evidence:[\w-]+$/.test(id)) throw new Error('Choose a document evidence handle from search.')
  const request = JSON.parse(Buffer.from(id.slice(9), 'base64url').toString('utf8'))
  if (!request || Object.keys(request).some(key => !['documentId', 'revision', 'passageId'].includes(key))
    || typeof request.documentId !== 'string' || request.documentId.length > 256 || typeof request.passageId !== 'string' || request.passageId.length > 256
    || !/^[a-f0-9]{64}$/.test(request.revision)) throw new Error('Invalid evidence handle.')
  return request
}
function locationLabel(location: DocumentLocation): string {
  if (location.kind === 'text') return `lines ${location.lineStart}–${location.lineEnd}; characters ${location.charStart + 1}–${location.charEnd}`
  if (location.kind === 'table') return `${location.sheet ? `${location.sheet} · ` : ''}${location.range ?? `rows ${location.rowStart}–${location.rowEnd}, columns ${location.columnStart}–${location.columnEnd}`}`
  return `extracted characters ${location.charStart + 1}–${location.charEnd}; page mapping unavailable`
}
/** Returned text is untrusted source material; citation requests contain no filesystem paths. */
export async function documentEvidenceTool(port: DocumentEvidencePort, name: string, args: Record<string, unknown>) {
  if (name === 'document_read') {
    const request = parseHandle(String(args.id ?? ''))
    const result = await port.readEvidence(request)
    if (result.status !== 'ready') return { ok: false, title: 'Document evidence unavailable', output: result.message }
    return { ok: true, title: 'Verified document passage', output: `${result.passage.sourceLabel} / ${result.passage.relativePath}\n${locationLabel(result.passage.location)}\nRevision: ${result.passage.revision}\n\n${result.passage.text}`, documentEvidence: [request] }
  }
  const result = await port.search({ query: String(args.query ?? '').slice(0, 256), limit: Math.max(1, Math.min(20, Number(args.limit) || 10)) })
  const references = result.passages.map(({ documentId, revision, passageId }) => ({ documentId, revision, passageId }))
  const output = result.passages.map((passage, index) => `[${index + 1}] ${passage.sourceLabel} / ${passage.relativePath}\n${locationLabel(passage.location)}\nRead ID: ${evidenceHandle(references[index])}\n${passage.text}`).join('\n\n')
  return { ok: true, title: 'Document passage search', output: `${output || 'No indexed passage matched. Sync the enabled document sources to index their full text.'}${result.truncated ? '\nMore matches exist; narrow your query.' : ''}${result.warnings?.length ? `\nSource warnings:\n${result.warnings.join('\n')}` : ''}`, documentEvidence: references, meta: `${references.length} passages` }
}
