import type { DocumentProcessingRequest } from '../../../shared/document-processing'
import type { DocumentEvidenceRequest, DocumentLibrarySearchOptions } from '../../../shared/document-library'
import { DocumentLibrary } from '../../document-library'
import { vontraHomeDir, vontraPath } from '../../store'
import { store } from '../state'
import { requireSecurityRole } from './audit'
import { handle } from './ipc'

let current: { root: string; library: DocumentLibrary } | null = null
export function documentLibrary(): DocumentLibrary {
  const root = vontraHomeDir()
  if (!current || current.root !== root) {
    if (current) void current.library.quiesce()
    current = { root, library: new DocumentLibrary({ directory: vontraPath('state', 'document-library'), getSettings: () => {
      if (vontraHomeDir() !== root) throw new Error('Document profile changed.')
      return store.settings.documents
    } }) }
  }
  return current.library
}
export async function quiesceDocumentLibrary(): Promise<void> { await current?.library.quiesce() }
export function resumeDocumentLibrary(): void { current?.library.resume() }
export async function purgeDocumentLibrary(): Promise<void> { await documentLibrary().purge() }

export function registerDocumentLibraryRuntime(options: { reserveOperation(): { release(): void } }): void {
  const run = async <T>(role: 'operator' | 'owner', action: () => Promise<T>) => {
    requireSecurityRole(role, 'inspect the local document library')
    const root = vontraHomeDir(), lease = options.reserveOperation()
    try {
      const result = await action()
      requireSecurityRole(role, 'inspect the local document library')
      if (vontraHomeDir() !== root) throw new Error('Document profile changed; try again.')
      return result
    } finally { lease.release() }
  }
  handle('documentLibrary:list', () => run('operator', () => documentLibrary().list()))
  handle('documentLibrary:search', (_event, request: DocumentLibrarySearchOptions) => run('operator', () => {
    if (!request || typeof request.query !== 'string' || request.query.length > 256 || request.sourceId != null && (typeof request.sourceId !== 'string' || request.sourceId.length > 256)) throw new Error('Choose a valid document query.')
    return documentLibrary().search(request)
  }))
  handle('documentLibrary:processing', () => run('operator', () => documentLibrary().processing()))
  handle('documentLibrary:process', (_event, request: DocumentProcessingRequest) => run('operator', () => documentLibrary().process(request)))
  handle('documentLibrary:source', (_event, request: DocumentEvidenceRequest) => run('operator', () => documentLibrary().readSource(request)))
  handle('documentLibrary:evidence', (_event, request: DocumentEvidenceRequest) => run('operator', () => documentLibrary().readEvidence(request)))
  handle('documentLibrary:purge', (_event, sourceId?: string) => run('owner', async () => {
    if (sourceId !== undefined && (typeof sourceId !== 'string' || sourceId.length > 256)) throw new Error('Choose a valid source.')
    return documentLibrary().purge(sourceId)
  }))
}
