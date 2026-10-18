import type { ConnectorEvent, DocumentConnectorSettings } from '../agent/types'
import type { DocumentEvidencePort } from '../agent/document-evidence-tool'
import type { DocumentLibrary } from './index'

export { validateDocumentSourceIds, documentPolicySignature } from '../../shared/document-collections'
export function scopedDocumentSettings(settings: DocumentConnectorSettings, sourceIds?: readonly string[]): DocumentConnectorSettings {
  return { ...settings, sources: settings.sources.filter(source => source.enabled && (sourceIds === undefined || sourceIds.includes(source.id))).map(source => ({ ...source })) }
}
export function scopedDocumentLibrary(library: DocumentLibrary, sourceIds: readonly string[] | undefined, assertCurrent: () => void): DocumentEvidencePort {
  const allowed = sourceIds === undefined ? undefined : [...sourceIds]
  return {
    search: async options => { assertCurrent(); const found = await library.search(options, allowed); assertCurrent(); return found },
    readEvidence: async request => { assertCurrent(); const found = await library.readEvidence(request, allowed); assertCurrent(); return found },
  }
}

/** Cached connector excerpts retain collection provenance and cannot bypass the active allowlist. */
export function documentEventAllowed(event: ConnectorEvent, settings: DocumentConnectorSettings, sourceIds?: readonly string[]): boolean {
  if (event.connectorId !== 'docs') return true
  const allowed = settings.sources.filter(source => source.enabled && (sourceIds === undefined || sourceIds.includes(source.id)))
  if (event.sources?.length) return event.sources.every(source => source.connectorId === 'docs' && allowed.some(active => active.id === source.sourceId))
  return allowed.some(source => event.threadKey.startsWith(`docs:${source.id}:`))
}
