import { shortPath } from '../../../lib/format'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Input, Segmented, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'

export function DocumentsConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const {
    draft,
    setDraft,
    documentBusy,
    connectorStatus,
    addDocumentSource,
    syncDocuments,
    updateDocumentSource,
    removeDocumentSource,
  } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  return (
    <Section title="Documents connector" desc="Index local documents, PDFs, and CSV/TSV/XLSX tables for source-backed reads and citation-aware summaries.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.documents.sources.length} source{draft.documents.sources.length === 1 ? '' : 's'} configured</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.documents.lastSyncAt ? `Last sync ${new Date(draft.documents.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
            <ConnectorStatusChip status={connectorStatusMap.documents} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="folder" onClick={addDocumentSource} disabled={!!documentBusy}>{documentBusy === 'add' ? 'Adding' : 'Add folder'}</Button>
            <Button variant="primary" size="sm" icon="fileText" onClick={syncDocuments} disabled={!!documentBusy || draft.documents.sources.length === 0}>{documentBusy === 'sync' ? 'Syncing' : 'Sync documents'}</Button>
          </div>
        </div>

        {draft.documents.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add a folder of PDFs, CSV/TSV/XLSX tables, text files, Markdown, JSON, or notes to make Documents available to Assistant and agent tools.
          </div>
        ) : draft.documents.sources.map((source) => (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="docs" size={32} />
              <div className="min-w-0 flex-1">
                <Input
                  value={source.label}
                  onChange={(e) => setDraft((d) => ({ ...d, documents: { ...d.documents, sources: d.documents.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                  onBlur={(e) => updateDocumentSource(source.id, { label: e.target.value })}
                  className="h-8"
                />
                <div className="mt-1 truncate text-[13px] text-ink-3" title={source.path}>{shortPath(source.path, 5)}</div>
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "document source"}`} checked={source.enabled} onChange={(enabled) => updateDocumentSource(source.id, { enabled })} />
              <button onClick={() => removeDocumentSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove document source">
                <Icon name="trash" size={15} />
              </button>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <Segmented
                value={source.kind}
                onChange={(kind) => updateDocumentSource(source.id, { kind })}
                options={[{ value: 'dir', label: 'Folder', icon: 'folder' }, { value: 'file', label: 'File', icon: 'fileText' }]}
              />
              <span className="text-[12px] text-ink-3">{source.lastSyncAt ? `Synced ${new Date(source.lastSyncAt).toLocaleString()}` : 'Not indexed'}</span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
