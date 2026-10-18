import type { DocumentLibraryApi } from '../../shared/document-library'

export function documentLibraryPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): DocumentLibraryApi {
  return {
    documentLibraryProcessing: () => invoke('documentLibrary:processing'),
    documentLibraryProcess: request => invoke('documentLibrary:process', request),
    documentLibraryReadSource: request => invoke('documentLibrary:source', request),
    documentLibraryList: () => invoke('documentLibrary:list'),
    documentLibrarySearch: request => invoke('documentLibrary:search', request),
    documentLibraryReadEvidence: request => invoke('documentLibrary:evidence', request),
    documentLibraryPurge: sourceId => invoke('documentLibrary:purge', sourceId),
  }
}
